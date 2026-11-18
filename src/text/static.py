"""Static (non-contextual) headline embeddings: Word2Vec and FastText trained on
our own headlines, pooled with SIF weighting.

Why train our own instead of downloading cc.id.300: the pretrained Common Crawl
model is a ~7 GB download, and training on train-split headlines keeps the static
baseline leak-free and in-domain. Static vectors give every word ONE vector
regardless of context ("bunga" = interest rate vs flower), which is exactly the
limitation contextual transformers address -- the comparison the brief asks for.

  word2vec : skip-gram, one vector per vocabulary word (unknown words dropped)
  fasttext : skip-gram over character n-grams (3-6), so unseen or inflected words
             ("melemahnya", "menguatnya") still get a vector -- useful for
             Indonesian affixation

Pooling = SIF (Arora et al., 2017, "A Simple but Tough-to-Beat Baseline for
Sentence Embeddings"): weighted mean with weight a / (a + p(w)) (rare words count
more), then remove the projection on the first principal component (the shared
"common discourse" direction). Word frequencies and the principal component come
from training-period headlines only.

Leakage: trained ONLY on headlines aligned to trading dates <= TEXT_FIT_END.

Run: python -m src.text.static            (writes cache/emb/word2vec.npy and fasttext.npy)
"""
import time
from collections import Counter

import numpy as np
from gensim.models import FastText, Word2Vec

from src.config import CACHE_DIR, SEED, TEXT_FIT_END
from src.data.articles import load_articles
from src.features.nlp_features import preprocess_title
from src.text.cache import save_embeddings

DIM = 100
SIF_A = 1e-3
COMMON = dict(vector_size=DIM, window=5, min_count=3, sg=1, epochs=10, seed=SEED, workers=4)


def sif_embed(tokens_list, wv, freq, total, in_vocab_only):
    X = np.zeros((len(tokens_list), DIM), dtype=np.float32)
    for i, toks in enumerate(tokens_list):
        if in_vocab_only:
            toks = [t for t in toks if t in wv.key_to_index]
        if not toks:
            continue
        w = np.array([SIF_A / (SIF_A + freq.get(t, 0) / total) for t in toks], dtype=np.float32)
        X[i] = (w[:, None] * np.stack([wv[t] for t in toks])).sum(0) / w.sum()
    return X


def remove_first_pc(X, fit_mask):
    """Subtract the projection on the first principal direction of the fit rows."""
    _, _, vt = np.linalg.svd(X[fit_mask], full_matrices=False)
    u = vt[0]
    return X - np.outer(X @ u, u)


def main():
    arts = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
    tokens = [preprocess_title(t) for t in arts["title"]]
    fit_mask = (arts["trading_date"] <= TEXT_FIT_END).to_numpy()
    corpus = [t for t, m in zip(tokens, fit_mask) if m and t]
    freq = Counter(w for toks in corpus for w in toks)
    total = sum(freq.values())
    print(f"[static] corpus: {len(corpus)} train headlines, {total} tokens, {len(freq)} types")

    for name, cls, in_vocab_only in [("word2vec", Word2Vec, True), ("fasttext", FastText, False)]:
        t0 = time.time()
        kw = dict(COMMON, min_n=3, max_n=6) if cls is FastText else COMMON
        model = cls(sentences=corpus, **kw)
        (CACHE_DIR / "models").mkdir(parents=True, exist_ok=True)
        model.save(str(CACHE_DIR / "models" / f"{name}.model"))
        X = remove_first_pc(sif_embed(tokens, model.wv, freq, total, in_vocab_only), fit_mask)
        empty = float(np.mean(~X.any(1)))
        print(f"[static] {name}: vocab {len(model.wv)}, {X.shape}, empty rows {empty:.1%} "
              f"({time.time() - t0:.0f}s); neighbours of 'rupiah': "
              f"{[w for w, _ in model.wv.most_similar('rupiah', topn=5)]}")
        save_embeddings(name, X, {"encoder": name, "dim": DIM, "pooling": "SIF + 1st PC removal",
                                  "fit_end": TEXT_FIT_END, "params": {k: v for k, v in kw.items() if k != "workers"},
                                  "n_articles": len(arts)})


if __name__ == "__main__":
    main()
