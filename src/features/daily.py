"""Headline vectors -> one row per trading day (data/features/<name>.parquet).

The output follows the daily-table contract in src/features/groups.py, so a new
table becomes an ablation row by adding "table:<name>" to config.ABLATION_SETS.

Temporal aggregation strategies (brief 1.b.v):
  mean     : plain average of the headline vectors aligned to day t
  recency  : within-window recency weighting: each headline weighs
             0.5 ** (hours before the 10:00 WIB fixing / half-life hours), so news
             published just before the fixing counts most and weekend news fades
  decay    : exponentially decayed average across trading days <= t (half-life in
             trading days): yesterday's news still counts, but less. Weighted by
             article count, so a busy day weighs more than a quiet one.
All three only use news aligned to day t or earlier (no look-ahead).

Other daily tables live in their own modules:
  src/features/semantic.py   anchor axes, novelty/dispersion, transformer sentiment
  src/features/entities.py   entity channels (USD side / IDR side / geopolitics / commodities)
  src/features/topics.py     BERTopic topic shares, GDELT theme shares

Run:
    python -m src.features.daily --encoder e5                      # -> e5_mean
    python -m src.features.daily --encoder e5 --half-life 3        # -> e5_decay3
    python -m src.features.daily --encoder e5 --recency-hours 12   # -> e5_recency12
"""
import argparse

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR
from src.data.articles import load_articles
from src.text.cache import load_embeddings


def daily_sums(X, trading_dates, weights=None, normalize=False):
    """Weighted sum of vectors and total weight per trading date (sorted by date).
    normalize=True L2-normalises each headline vector first (for cosine features)."""
    codes, dates = pd.factorize(pd.Series(trading_dates), sort=True)
    w = np.ones(len(codes)) if weights is None else np.asarray(weights, dtype=np.float64)
    sums = np.zeros((len(dates), X.shape[1]), dtype=np.float64)
    for start in range(0, len(codes), 50_000):           # chunked: X may be a float16 memmap
        sl = slice(start, start + 50_000)
        chunk = np.asarray(X[sl], dtype=np.float64)
        if normalize:
            chunk /= np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-8)
        np.add.at(sums, codes[sl], chunk * w[sl, None])
    counts = np.bincount(codes, weights=w, minlength=len(dates))
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


def hours_before_fixing(articles):
    """Hours between each article's timestamp and the 10:00 WIB fixing it is aligned to."""
    fixing = pd.to_datetime(articles["trading_date"]).dt.tz_localize("Asia/Jakarta") + pd.Timedelta(hours=10)
    h = (fixing - articles["timestamp_wib"]).dt.total_seconds() / 3600
    return h.clip(lower=0).to_numpy()


def write_table(name, dates, M, prefix=None):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cols = [f"{prefix or name}_{i}" for i in range(M.shape[1])] if not isinstance(M, pd.DataFrame) else None
    df = M.copy() if isinstance(M, pd.DataFrame) else pd.DataFrame(M.astype(np.float32), columns=cols)
    df.insert(0, "date", np.asarray(dates).astype(str))
    df.to_parquet(FEATURES_DIR / f"{name}.parquet", index=False)
    print(f"[daily] wrote {name}: {df.shape[0]} days x {df.shape[1] - 1} features")


def main(encoder, half_life, recency_hours):
    articles = load_articles(columns=["article_id", "timestamp_wib", "trading_date"])
    X = load_embeddings(encoder, n_expected=len(articles))
    if recency_hours:
        w = 0.5 ** (hours_before_fixing(articles) / recency_hours)
        dates, sums, counts = daily_sums(X, articles["trading_date"].to_numpy(), w)
        write_table(f"{encoder}_recency{recency_hours:g}", dates, daily_mean(dates, sums, counts))
        return
    dates, sums, counts = daily_sums(X, articles["trading_date"].to_numpy())
    if half_life:
        write_table(f"{encoder}_decay{half_life:g}", dates, daily_decay(dates, sums, counts, half_life))
    else:
        write_table(f"{encoder}_mean", dates, daily_mean(dates, sums, counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--half-life", type=float, default=0, help="decay across trading days; 0 = off")
    ap.add_argument("--recency-hours", type=float, default=0, help="within-window recency half-life; 0 = off")
    a = ap.parse_args()
    main(a.encoder, a.half_life, a.recency_hours)
