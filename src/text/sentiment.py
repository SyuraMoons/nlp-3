"""Transformer sentiment head: per-headline P(positive), P(neutral), P(negative).

Model: mdhugol/indonesia-bert-sentiment-classification (IndoBERT fine-tuned on
Indonesian reviews/tweets; the same model Task 1 used). Label mapping verified by
probing: LABEL_0 = positive, LABEL_1 = neutral, LABEL_2 = negative.

This is the "simple article-level sentiment" baseline that the richer embedding
features have to beat. Note the domain gap: trained on reviews, it scores most
financial headlines as neutral (e.g. "Laba bersih perusahaan melonjak 50 persen"
-> neutral 0.99), which motivates domain-adapted embeddings (src/text/dapt.py).

Output: cache/emb/sentiment.npy, shape (n_articles, 3) = [pos, neu, neg],
in canonical article order (src/data/articles.py).

Run: python -m src.text.sentiment
"""
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.articles import load_articles
from src.text.cache import save_embeddings
from src.text.encode import device

MODEL = "mdhugol/indonesia-bert-sentiment-classification"


@torch.no_grad()
def classify(titles, batch_size=256):
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    order = np.argsort([len(t) for t in titles])
    out = np.zeros((len(titles), 3), dtype=np.float32)
    for i in range(0, len(titles), batch_size):
        idx = order[i:i + batch_size]
        batch = tok([titles[j] for j in idx], padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(dev)
        out[idx] = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
        if (i // batch_size) % 100 == 0:
            print(f"  {i:>7d}/{len(titles)}", flush=True)
    return out


def main():
    articles = load_articles(columns=["article_id", "timestamp_wib", "title"])
    t0 = time.time()
    P = classify(articles["title"].tolist())
    print(f"[sentiment] {P.shape} in {time.time() - t0:.0f}s; mean [pos, neu, neg] = {P.mean(0).round(3)}")
    save_embeddings("sentiment", P, {"encoder": "sentiment", "model": MODEL,
                                     "columns": ["pos", "neu", "neg"], "n_articles": len(articles)})


if __name__ == "__main__":
    main()
