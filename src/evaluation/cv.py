"""Task 3 evaluation protocol: expanding-window CV on dev, threshold tuning, and a
single final test run. See the protocol comment in src/config.py.

Why not Task 2's single validation split: every trained Task 2 model had val
ROC-AUC < 0.5, so "best on val" picked noise. Pooling 600 out-of-fold days gives a
far more stable selection signal and a second, larger evaluation set than the
176-day test split.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef

from src.config import CV_FOLD_SIZE, CV_N_FOLDS


def expanding_folds(n, n_folds=CV_N_FOLDS, fold_size=CV_FOLD_SIZE, min_train=250):
    """Return [(train_idx, val_idx), ...] over rows 0..n-1 (rows in date order).

    The last n_folds * fold_size rows are cut into consecutive validation blocks;
    the last block absorbs any remainder. Fold k trains on every row before its block.
    """
    start = n - n_folds * fold_size
    if start < min_train:
        raise ValueError(f"only {start} rows before the first CV block (< {min_train})")
    folds = []
    for k in range(n_folds):
        lo = start + k * fold_size
        hi = n if k == n_folds - 1 else lo + fold_size
        folds.append((np.arange(0, lo), np.arange(lo, hi)))
    return folds


def tune_threshold(y, prob, quantiles=np.arange(0.10, 0.901, 0.025)):
    """Threshold on P(UP) that maximises MCC on (y, prob).

    Candidates are quantiles of prob (so each candidate predicts a different UP
    share between 10% and 90%) plus 0.5. Returns (threshold, mcc).
    """
    y, prob = np.asarray(y), np.asarray(prob)
    candidates = np.unique(np.r_[np.quantile(prob, quantiles), 0.5])
    best_t, best_m = 0.5, -np.inf
    for t in candidates:
        m = matthews_corrcoef(y, (prob >= t).astype(int))
        if m > best_m:
            best_t, best_m = float(t), m
    return best_t, best_m


def walk_forward_predictions(y_oof, prob_oof, fold_sizes):
    """Turn OOF probabilities into UP/DOWN predictions without using a threshold
    tuned on the same days: block k uses a threshold tuned on blocks < k, block 0
    uses 0.5. Returns (pred, thresholds_per_fold)."""
    y_oof, prob_oof = np.asarray(y_oof), np.asarray(prob_oof)
    pred = np.zeros(len(prob_oof), dtype=int)
    thresholds, pos = [], 0
    for size in fold_sizes:
        t = 0.5 if pos == 0 else tune_threshold(y_oof[:pos], prob_oof[:pos])[0]
        pred[pos:pos + size] = (prob_oof[pos:pos + size] >= t).astype(int)
        thresholds.append(t)
        pos += size
    return pred, thresholds
