"""Domain-adaptive pretraining (DAPT): continue IndoBERT's masked-language-model
training on our own financial headlines. This is the project's fine-tuning script.

Why: IndoBERT was pretrained on general Indonesian text (Indo4B: news, Wikipedia,
social media). Financial headlines have their own vocabulary and phrasing
("BI Rate", "IHSG", "rupiah ditutup melemah", "yield SBN"). Gururangan et al.
(2020, "Don't Stop Pretraining") show that a short second phase of MLM on in-domain
text improves downstream tasks. We test whether the domain-adapted encoder gives
better headline features than the general one (brief: domain-specific vs general).

Leakage: trained ONLY on headlines aligned to trading dates <= TEXT_FIT_END (end of
the Task 2 train split). No labels are used. Held-out perplexity is measured on
validation-period headlines (2025-02-25 .. 2025-11-24), which DAPT never sees.

Two phases, because the public indobert-base-p1 checkpoint ships WITHOUT its MLM
head (cls.predictions.* are randomly initialised on load; perplexity ~62,000 ~=
vocabulary size):
  1. head warm-up: encoder frozen, train only the MLM head (HEAD_STEPS steps,
     lr 1e-3). Perplexity after this = the GENERAL IndoBERT baseline.
  2. DAPT: unfreeze everything, 1 epoch of MLM on train-split headlines.
The perplexity drop from phase 1 to phase 2 measures the domain adaptation.

Setup: 15% masking (BERT recipe: 80% [MASK] / 10% random / 10% unchanged),
max_length 48 (headlines are ~10 words), AdamW lr 5e-5, linear warmup/decay,
1 epoch, batch 64, Apple MPS. Saves to cache/models/indobert_dapt (not in git).

Run: python -m src.text.dapt              (then: python -m src.text.encode --encoder indobert_dapt)
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

from src.config import CACHE_DIR, SEED, TEXT_FIT_END
from src.data.articles import load_articles
from src.text.encode import device

BASE = "indobenchmark/indobert-base-p1"
OUT_DIR = CACHE_DIR / "models" / "indobert_dapt"
VAL_START, VAL_END = "2025-02-25", "2025-11-24"
HEAD_STEPS = 600


def batches(texts, tok, collator, batch_size, rng=None):
    order = np.arange(len(texts)) if rng is None else rng.permutation(len(texts))
    for i in range(0, len(order), batch_size):
        enc = tok([texts[j] for j in order[i:i + batch_size]], truncation=True, max_length=48)
        yield collator([{"input_ids": ids} for ids in enc["input_ids"]])


def mlm_loss(model, b):
    """Masked-LM loss computed ONLY at masked positions.

    The stock forward pass projects every token onto the 50k-word vocabulary
    (batch x 48 x 50k logits, ~0.6 GB per batch); only ~15% of tokens are scored,
    so projecting just those cuts memory and time ~7x on a 16 GB laptop.
    Returns (mean loss, number of masked tokens).
    """
    encoder = getattr(model, model.base_model_prefix)
    hidden = encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    sel = b["labels"] != -100
    logits = model.cls(hidden[sel])
    return torch.nn.functional.cross_entropy(logits.float(), b["labels"][sel]), int(sel.sum())


@torch.no_grad()
def perplexity(model, texts, tok, collator, dev, batch_size=64):
    """Masked-LM perplexity on held-out headlines (fixed masking seed -> comparable)."""
    model.eval()
    torch.manual_seed(SEED)
    total, n = 0.0, 0
    for b in batches(texts, tok, collator, batch_size):
        b = {k: v.to(dev) for k, v in b.items()}
        loss, n_masked = mlm_loss(model, b)
        total += float(loss) * n_masked
        n += n_masked
    model.train()
    return math.exp(total / n)


def main(epochs, batch_size, lr, limit):
    torch.manual_seed(SEED)
    dev = device()
    arts = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
    arts = arts[arts["title"].str.len() > 0]
    train = arts.loc[arts["trading_date"] <= TEXT_FIT_END, "title"].tolist()
    val = arts.loc[arts["trading_date"].between(VAL_START, VAL_END), "title"].sample(
        5000, random_state=SEED).tolist()
    if limit:
        train = train[:limit]
    print(f"[dapt] {len(train)} train headlines (<= {TEXT_FIT_END}), {len(val)} held-out val headlines, device={dev}")

    tok = AutoTokenizer.from_pretrained(BASE)
    # eager attention: PyTorch SDPA on MPS does not support attention dropout in training
    model = AutoModelForMaskedLM.from_pretrained(BASE, attn_implementation="eager").to(dev)
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15, return_tensors="pt")

    ppl_random_head = perplexity(model, val, tok, collator, dev)
    print(f"[dapt] held-out perplexity, random MLM head: {ppl_random_head:.2f}")

    # Phase 1: train the MLM head only (encoder + tied embeddings frozen).
    encoder = getattr(model, model.base_model_prefix)
    for p in encoder.parameters():
        p.requires_grad = False
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=1e-3)
    rng, step, t0 = np.random.default_rng(SEED), 0, time.time()
    model.train()
    while step < HEAD_STEPS:
        for b in batches(train, tok, collator, batch_size, rng):
            loss, _ = mlm_loss(model, {k: v.to(dev) for k, v in b.items()})
            loss.backward(); opt.step(); opt.zero_grad()
            step += 1
            if step >= HEAD_STEPS:
                break
    for p in encoder.parameters():
        p.requires_grad = True
    ppl_before = perplexity(model, val, tok, collator, dev)
    print(f"[dapt] held-out perplexity, general IndoBERT (head warmed up, {HEAD_STEPS} steps): "
          f"{ppl_before:.2f} ({time.time() - t0:.0f}s)")

    # Phase 2: domain-adaptive pretraining of the whole model.
    steps = epochs * math.ceil(len(train) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng, step, t0 = np.random.default_rng(SEED), 0, time.time()
    model.train()
    for ep in range(epochs):
        for b in batches(train, tok, collator, batch_size, rng):
            b = {k: v.to(dev) for k, v in b.items()}
            loss, _ = mlm_loss(model, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)

    ppl_after = perplexity(model, val, tok, collator, dev)
    print(f"[dapt] held-out perplexity after: {ppl_after:.2f}  ({time.time() - t0:.0f}s)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(OUT_DIR)   # encoder only (no MLM head) for src.text.encode
    tok.save_pretrained(OUT_DIR)
    log = {"base": BASE, "train_headlines": len(train), "fit_end": TEXT_FIT_END, "epochs": epochs,
           "batch_size": batch_size, "lr": lr, "head_warmup_steps": HEAD_STEPS, "dapt_steps": steps,
           "val_perplexity_random_head": ppl_random_head,
           "val_perplexity_general_indobert": ppl_before, "val_perplexity_dapt": ppl_after}
    (OUT_DIR / "dapt_log.json").write_text(json.dumps(log, indent=1))
    from src.config import RESULTS_DIR
    (RESULTS_DIR / "dapt_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit", type=int, default=0, help="train on the first N headlines only (smoke test)")
    a = ap.parse_args()
    main(a.epochs, a.batch_size, a.lr, a.limit)
