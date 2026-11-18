"""Encode every headline once and cache the vectors (src/text/cache.py contract).

OWNER: Yusuf (see TEAM_TASKS.md). The two generic paths below already work:
  * "sentence-transformers" models (contrastively trained sentence encoders)
  * "hf-mean" models: any HuggingFace encoder, mean-pooled over non-pad tokens
Titles are deduplicated before encoding (wire stories repeat), then expanded back
to one row per article.

Run (from repo root; uses Apple MPS if available, else CUDA, else CPU):
    python -m src.text.encode --encoder e5
    python -m src.text.encode --encoder indobert
    python -m src.text.encode --encoder e5 --limit 2000     # quick smoke test

Encoders marked TODO need their own script first (see each entry).
"""
import argparse
import time

import numpy as np
import torch

from src.data.articles import load_articles
from src.text.cache import save_embeddings

ENCODERS = {
    # General multilingual, contextual, contrastively trained -> good sentence vectors.
    # e5 expects a "query: " prefix for symmetric similarity tasks.
    "e5": {"kind": "sentence-transformers", "model": "intfloat/multilingual-e5-base", "prefix": "query: "},
    # General Indonesian BERT (MLM-pretrained), mean pooling. Raw MLM embeddings are
    # anisotropic -- a point worth discussing in the report vs e5.
    "indobert": {"kind": "hf-mean", "model": "indobenchmark/indobert-base-p1"},
    # Domain-adapted IndoBERT: continued MLM on TRAIN-split headlines.
    # TODO (Yusuf): write src/text/dapt.py that saves the model to this path.
    "indobert_dapt": {"kind": "hf-mean", "model": "cache/models/indobert_dapt"},
    # TODO (Yusuf): static baselines (gensim FastText / Word2Vec trained on TRAIN-split
    # headlines, SIF pooling) -> src/text/static.py, saving with save_embeddings().
    # TODO (Yusuf): sentiment head mdhugol/indonesia-bert-sentiment-classification ->
    # per-headline [P(pos), P(neu), P(neg)] -> src/text/sentiment.py.
}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_sentence_transformers(cfg, titles, batch_size):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["model"], device=device())
    model.max_seq_length = 64       # headlines are ~10 words; 64 tokens is ample
    texts = [cfg.get("prefix", "") + t for t in titles]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)


@torch.no_grad()
def encode_hf_mean(cfg, titles, batch_size):
    from transformers import AutoModel, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModel.from_pretrained(cfg["model"]).to(dev).eval()
    out = []
    for i in range(0, len(titles), batch_size):
        batch = tok(titles[i:i + batch_size], padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(dev)
        hidden = model(**batch).last_hidden_state                  # (b, seq, dim)
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f"  {i:>7d}/{len(titles)}")
    return np.vstack(out)


def main(name, batch_size, limit):
    cfg = ENCODERS[name]
    articles = load_articles(columns=["article_id", "timestamp_wib", "title"])
    if limit:
        articles = articles.iloc[:limit]
    unique, inverse = np.unique(articles["title"].to_numpy(dtype=str), return_inverse=True)
    print(f"[encode] {name}: {len(articles)} articles, {len(unique)} unique titles, device={device()}")
    t0 = time.time()
    fn = encode_sentence_transformers if cfg["kind"] == "sentence-transformers" else encode_hf_mean
    U = fn(cfg, unique.tolist(), batch_size)
    X = U[inverse]
    print(f"[encode] {name}: {X.shape} in {time.time() - t0:.0f}s")
    out_name = name if not limit else f"{name}_limit{limit}"
    save_embeddings(out_name, X, {"encoder": name, **cfg, "n_articles": len(articles)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N articles (smoke test)")
    a = ap.parse_args()
    main(a.encoder, a.batch_size, a.limit)
