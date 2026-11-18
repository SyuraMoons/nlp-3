"""Headline vectors -> one row per trading day (data/features/<name>.parquet).

OWNER: Yusuf (aggregation variants), Harfi (contract). The output follows the
daily-table contract in src/features/groups.py, so a new table becomes an
ablation row by adding "table:<name>" to config.ABLATION_SETS.

Aggregations implemented:
  mean    : average of the headline vectors aligned to day t
  decay   : exponentially decayed average over days <= t (half-life in trading
            days). News from yesterday still counts, but less. Weighted by
            article count, so a busy day weighs more than a quiet one.

TODO (Yusuf), each as its own function writing its own table:
  * within-day recency weighting by timestamp_wib (latest headlines weigh more)
  * anchor projections: cosine similarity of each headline to ~12 hand-written
    anchor sentences (e.g. "Rupiah melemah terhadap dolar AS", "The Fed naikkan
    suku bunga"), daily mean -> "<encoder>_anchors"
  * news novelty: cosine distance between day t's mean vector and the trailing
    20-day mean; dispersion: mean distance of headlines from the day's mean
  * entity channels (keyword gazetteer: USD side / IDR side / geopolitics /
    commodities): per channel volume share, sentiment, anchor projection

Run:
    python -m src.features.daily --encoder e5                 # -> e5_mean
    python -m src.features.daily --encoder e5 --half-life 3   # -> e5_decay3
"""
import argparse

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR
from src.data.articles import load_articles
from src.text.cache import load_embeddings


def daily_sums(X, trading_dates):
    """Sum of vectors and article count per trading date (sorted by date)."""
    codes, dates = pd.factorize(pd.Series(trading_dates), sort=True)
    sums = np.zeros((len(dates), X.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, np.asarray(X, dtype=np.float32))
    counts = np.bincount(codes, minlength=len(dates)).astype(np.float64)
    return np.asarray(dates), sums, counts


def daily_mean(dates, sums, counts):
    return sums / counts[:, None]


def daily_decay(dates, sums, counts, half_life):
    """S_t = sum_{s<=t} w^(t-s) * sum_s  /  sum_{s<=t} w^(t-s) * n_s,  w = 0.5^(1/h).
    t indexes trading dates with news; uses only dates <= t (no look-ahead)."""
    w = 0.5 ** (1.0 / half_life)
    num, den = np.zeros(sums.shape[1]), 0.0
    out = np.zeros_like(sums)
    for i in range(len(dates)):
        num = w * num + sums[i]
        den = w * den + counts[i]
        out[i] = num / den
    return out


def write_table(name, dates, M):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(M.astype(np.float32), columns=[f"{name}_{i}" for i in range(M.shape[1])])
    df.insert(0, "date", dates)
    df.to_parquet(FEATURES_DIR / f"{name}.parquet", index=False)
    print(f"[daily] wrote {name}: {df.shape[0]} days x {M.shape[1]} dims")


def main(encoder, half_life):
    articles = load_articles(columns=["article_id", "timestamp_wib", "trading_date"])
    X = load_embeddings(encoder, n_expected=len(articles))
    dates, sums, counts = daily_sums(X, articles["trading_date"].to_numpy())
    if half_life:
        write_table(f"{encoder}_decay{half_life:g}", dates, daily_decay(dates, sums, counts, half_life))
    else:
        write_table(f"{encoder}_mean", dates, daily_mean(dates, sums, counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--half-life", type=float, default=0, help="0 = plain daily mean")
    a = ap.parse_args()
    main(a.encoder, a.half_life)
