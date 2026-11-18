"""Per-headline embedding cache (cache/emb/<encoder>.npy, NOT in git).

Contract: row i of cache/emb/<encoder>.npy is the embedding of the title of article
i of src.data.articles.load_articles() (canonical order). float16, shape
(n_articles, dim). cache/emb/<encoder>.json records the model id, pooling, dim and
n_articles so a stale cache is detected.
"""
import json

import numpy as np

from src.config import CACHE_DIR

EMB_DIR = CACHE_DIR / "emb"


def save_embeddings(name, X, meta):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"{name}.npy", X.astype(np.float16))
    (EMB_DIR / f"{name}.json").write_text(json.dumps({**meta, "shape": list(X.shape)}, indent=1))


def load_embeddings(name, n_expected=None):
    path = EMB_DIR / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run: python -m src.text.encode --encoder {name}")
    X = np.load(path, mmap_mode="r")
    if n_expected is not None and X.shape[0] != n_expected:
        raise ValueError(f"{name}: cache has {X.shape[0]} rows, articles have {n_expected} -- re-encode")
    return X
