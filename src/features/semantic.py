"""Interpretable semantic features from headline embeddings (beyond plain sentiment).

1. Anchor axes ("<encoder>_anchors"). Twelve FX-relevant polar axes, each defined
   by a pair of hand-written anchor headlines (e.g. "Rupiah melemah terhadap dolar
   AS" vs "Rupiah menguat terhadap dolar AS"). For every headline:
       axis score = cos(headline, positive anchor) - cos(headline, negative anchor)
   Positive = pressure towards USD/IDR going UP (weaker rupiah, hawkish Fed, risk-off).
   Per day: the mean axis score over all headlines, and over the day's 20 headlines
   most relevant to that axis (relevance = max of the two cosines), so a few
   on-topic headlines are not drowned by hundreds of unrelated ones.
   Projecting onto a few anchors is a supervised-free, low-dimensional alternative
   to PCA (24 named features instead of 768 anonymous dims) -- important with
   only ~1,000 training days.

2. News dynamics ("<encoder>_dynamics"):
     novelty    = 1 - cos(day t mean vector, mean of the previous 20 trading days)
     dispersion = mean over day t headlines of 1 - cos(headline, day t mean)
   A "regime shift" in news content (new war, surprise policy) shows up as novelty.

3. Transformer sentiment ("sentiment"): daily mean of P(pos) - P(neg), share of
   headlines classified positive / negative, from src/text/sentiment.py. This is
   the article-level sentiment baseline the richer features are compared with.

Run:
    python -m src.features.semantic --encoder e5          # anchors + dynamics
    python -m src.features.semantic --sentiment           # sentiment table
"""
import argparse

import numpy as np
import pandas as pd

from src.data.articles import load_articles
from src.features.daily import daily_mean, daily_sums, write_table
from src.text.cache import load_embeddings

# (axis name, headline pushing USD/IDR UP, headline pushing USD/IDR DOWN)
ANCHOR_AXES = [
    ("rupiah", "Rupiah melemah tajam terhadap dolar AS", "Rupiah menguat tajam terhadap dolar AS"),
    ("dollar", "Dolar AS menguat terhadap mata uang dunia", "Dolar AS melemah terhadap mata uang dunia"),
    ("fed", "The Fed menaikkan suku bunga acuan", "The Fed memangkas suku bunga acuan"),
    ("bi_rate", "Bank Indonesia memangkas BI Rate", "Bank Indonesia menaikkan BI Rate"),
    ("us_inflation", "Inflasi Amerika Serikat naik melebihi perkiraan", "Inflasi Amerika Serikat turun di bawah perkiraan"),
    ("geopolitics", "Perang dan konflik geopolitik memanas", "Gencatan senjata dan perundingan damai tercapai"),
    ("oil", "Harga minyak dunia melonjak", "Harga minyak dunia anjlok"),
    ("trade_war", "Trump menaikkan tarif impor, perang dagang memanas", "Kesepakatan dagang tercapai, tarif impor diturunkan"),
    ("capital_flow", "Investor asing keluar dari pasar modal Indonesia", "Investor asing masuk ke pasar modal Indonesia"),
    ("stocks", "IHSG anjlok di tengah aksi jual", "IHSG menguat ditopang aksi beli"),
    ("growth", "Ekonomi Indonesia melambat, ancaman resesi", "Ekonomi Indonesia tumbuh kuat"),
    ("reserves", "Cadangan devisa Indonesia turun", "Cadangan devisa Indonesia naik"),
]
TOP_K = 20


def encode_texts(encoder, texts):
    # imported lazily: loading torch next to LightGBM/SHAP crashes on macOS (OpenMP clash)
    from src.text.encode import ENCODERS, encode_hf_mean, encode_sentence_transformers
    cfg = ENCODERS[encoder]
    fn = encode_sentence_transformers if cfg["kind"] == "sentence-transformers" else encode_hf_mean
    return fn(cfg, texts, 64)


def unit(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def anchor_scores(X, encoder):
    """Per-headline axis scores and relevances, each (n_articles, n_axes)."""
    pos = unit(encode_texts(encoder, [p for _, p, _ in ANCHOR_AXES]))
    neg = unit(encode_texts(encoder, [n for _, _, n in ANCHOR_AXES]))
    score = np.zeros((X.shape[0], len(ANCHOR_AXES)), dtype=np.float32)
    rel = np.zeros_like(score)
    for s in range(0, X.shape[0], 50_000):
        U = unit(X[s:s + 50_000])
        cp, cn = U @ pos.T, U @ neg.T
        score[s:s + 50_000], rel[s:s + 50_000] = cp - cn, np.maximum(cp, cn)
    return score, rel


def anchors_table(articles, X, encoder):
    score, rel = anchor_scores(X, encoder)
    names = [a for a, _, _ in ANCHOR_AXES]
    df = pd.DataFrame(score, columns=[f"{encoder}_ax_{n}_mean" for n in names])
    df["date"] = articles["trading_date"].to_numpy()
    out = df.groupby("date").mean()
    # mean over the TOP_K most axis-relevant headlines of each day
    for j, n in enumerate(names):
        tmp = pd.DataFrame({"date": df["date"], "s": score[:, j], "r": rel[:, j]})
        top = tmp.sort_values("r", ascending=False).groupby("date").head(TOP_K)
        out[f"{encoder}_ax_{n}_top{TOP_K}"] = top.groupby("date")["s"].mean()
    return out.reset_index()


def dynamics_table(articles, X, encoder, window=20):
    dates, sums, counts = daily_sums(X, articles["trading_date"].to_numpy(), normalize=True)
    M = unit(daily_mean(dates, sums, counts))
    novelty = np.full(len(dates), np.nan)
    for i in range(1, len(dates)):
        prev = unit(M[max(0, i - window):i].mean(0, keepdims=True))[0]
        novelty[i] = 1 - float(M[i] @ prev)
    # dispersion: mean cosine distance of headlines from their day's mean
    codes = pd.factorize(pd.Series(articles["trading_date"].to_numpy()), sort=True)[0]
    disp_sum = np.zeros(len(dates))
    for s in range(0, X.shape[0], 50_000):
        U = unit(X[s:s + 50_000])
        c = codes[s:s + 50_000]
        np.add.at(disp_sum, c, 1 - np.einsum("ij,ij->i", U, M[c]))
    out = pd.DataFrame({"date": dates, f"{encoder}_novelty": novelty,
                        f"{encoder}_dispersion": disp_sum / counts})
    return out.fillna(0.0)


def sentiment_table(articles):
    P = np.asarray(load_embeddings("sentiment", n_expected=len(articles)), dtype=np.float32)
    label = P.argmax(1)
    df = pd.DataFrame({"date": articles["trading_date"].to_numpy(),
                       "sent_score_mean": P[:, 0] - P[:, 2],
                       "sent_pos_share": (label == 0).astype(float),
                       "sent_neg_share": (label == 2).astype(float),
                       "sent_neg_prob_mean": P[:, 2]})
    return df.groupby("date").mean().reset_index()


def main(encoder, sentiment):
    articles = load_articles(columns=["article_id", "timestamp_wib", "trading_date"])
    if sentiment:
        t = sentiment_table(articles)
        write_table("sentiment", t["date"], t.drop(columns="date"))
        return
    X = load_embeddings(encoder, n_expected=len(articles))
    t = anchors_table(articles, X, encoder)
    write_table(f"{encoder}_anchors", t["date"], t.drop(columns="date"))
    t = dynamics_table(articles, X, encoder)
    write_table(f"{encoder}_dynamics", t["date"], t.drop(columns="date"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--sentiment", action="store_true")
    a = ap.parse_args()
    main(a.encoder, a.sentiment)
