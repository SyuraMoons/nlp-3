"""Embedding-space analysis (brief 3.c.i): do news headlines look different on
high-volatility vs low-volatility days?

Volatility groups: |JISDOR log return| of day t, top vs bottom quintile over all
1,168 modelled days. Headlines are those aligned to day t (published before the
fixing that realised the move).

  umap_plot          2-D UMAP of a balanced sample of headline embeddings, coloured
                     by group, + per-group density contours
  topic_lift         share of each BERTopic topic on high- vs low-volatility days
  separability       how well daily mean embeddings separate high/low days
                     (5-fold stratified LogReg AUC on PCA-16), per encoder
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CACHE_DIR, FEATURES_DIR, RESULTS_DIR, SEED
from src.data.articles import load_articles
from src.data.load import load_all
from src.text.cache import load_embeddings


def volatility_days(q=0.2):
    d = pd.concat(load_all().values(), ignore_index=True)
    a = d["target_return"].abs()
    hi, lo = a >= a.quantile(1 - q), a <= a.quantile(q)
    return (pd.Series("high", index=d.loc[hi, "date"]).combine_first(pd.Series("low", index=d.loc[lo, "date"])))


def headline_sample(encoder="e5", n_per_group=3000):
    arts = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
    X = load_embeddings(encoder, n_expected=len(arts))
    vol = volatility_days()
    arts["group"] = arts["trading_date"].map(vol)
    rows = []
    for g in ("high", "low"):
        idx = np.flatnonzero((arts["group"] == g).to_numpy())
        rows.append(np.random.default_rng(SEED).choice(idx, min(n_per_group, len(idx)), replace=False))
    idx = np.sort(np.concatenate(rows))
    return arts.iloc[idx].reset_index(drop=True), np.asarray(X[idx], dtype=np.float32), idx


def umap_plot(encoder="e5", n_per_group=3000, path=None):
    import umap
    arts, X, _ = headline_sample(encoder, n_per_group)
    Z = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for g, c in (("low", "#2563eb"), ("high", "#dc2626")):
        m = (arts["group"] == g).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.25, color=c, label=f"{g}-volatility days")
    ax.legend(markerscale=6, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UMAP of {encoder} headline embeddings ({n_per_group} per group)", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return fig, arts, Z


def topic_lift(path=None, top=12):
    """Share of headlines per topic on high- vs low-volatility days (log2 ratio)."""
    arts = load_articles(columns=["article_id", "timestamp_wib", "trading_date"])
    topics = np.load(CACHE_DIR / "topic_assignments_e5.npy")
    names = pd.read_csv(RESULTS_DIR / "topics_e5.csv").set_index("Topic")["keywords"]
    vol = arts["trading_date"].map(volatility_days())
    t = pd.crosstab(topics, vol, normalize="columns")
    t["log2_lift"] = np.log2(t["high"] / t["low"])
    t["keywords"] = [", ".join(names.get(i, "").split(", ")[:4]) for i in t.index]
    t = t.sort_values("log2_lift")
    show = pd.concat([t.head(top // 2), t.tail(top // 2)])
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(show) + 1))
    ax.barh(range(len(show)), show["log2_lift"], color=["#2563eb" if v < 0 else "#dc2626" for v in show["log2_lift"]])
    ax.set_yticks(range(len(show)), [f"T{i}: {k}" for i, k in zip(show.index, show["keywords"])], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("log2(share on high-vol days / share on low-vol days)")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return t, fig


def separability(tables=("word2vec_mean", "fasttext_mean", "indobert_mean", "indobert_dapt_mean", "e5_mean")):
    vol = volatility_days()
    out = []
    for name in tables:
        p = FEATURES_DIR / f"{name}.parquet"
        if not p.exists():
            continue
        T = pd.read_parquet(p).set_index("date")
        T = T.loc[T.index.intersection(vol.index)]
        y = (vol.loc[T.index] == "high").astype(int).to_numpy()
        clf = make_pipeline(StandardScaler(), PCA(16, random_state=SEED), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        s = cross_val_score(clf, T.to_numpy(), y, cv=cv, scoring="roc_auc")
        out.append({"table": name, "days": len(y), "auc_mean": s.mean(), "auc_sd": s.std()})
    return pd.DataFrame(out)
