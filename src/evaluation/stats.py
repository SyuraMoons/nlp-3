"""Uncertainty for small test sets: block-bootstrap confidence intervals and paired
model comparisons.

Why a *block* bootstrap: daily FX moves are weakly dependent (volatility clusters),
so resampling single days would understate the variance. A circular block
bootstrap resamples runs of `block` consecutive days instead.

Why *paired*: comparing Model C with Model A on the same days removes the shared
"how hard were these days" noise. The CI of the difference is what tells us whether
text features measurably help, not two overlapping individual CIs.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from src.config import BOOTSTRAP_BLOCK, BOOTSTRAP_N, SEED


def block_bootstrap_indices(n, n_boot=BOOTSTRAP_N, block=BOOTSTRAP_BLOCK, seed=SEED):
    """Yield n_boot index arrays of length n built from circular blocks."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        yield ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def _metric_fns():
    return {
        "roc_auc": lambda y, p, yhat: _auc(y, p),
        "mcc": lambda y, p, yhat: matthews_corrcoef(y, yhat),
        "accuracy": lambda y, p, yhat: float(np.mean(y == yhat)),
    }


def bootstrap_ci(y, prob, pred, alpha=0.05, **kw):
    """95% CIs for ROC-AUC, MCC and accuracy. Returns {metric: (lo, hi)}."""
    y, prob, pred = map(np.asarray, (y, prob, pred))
    fns = _metric_fns()
    draws = {k: [] for k in fns}
    for idx in block_bootstrap_indices(len(y), **kw):
        for k, f in fns.items():
            draws[k].append(f(y[idx], prob[idx], pred[idx]))
    return {k: tuple(np.nanquantile(v, [alpha / 2, 1 - alpha / 2])) for k, v in draws.items()}


def paired_bootstrap_delta(y, prob_a, pred_a, prob_b, pred_b, alpha=0.05, **kw):
    """Difference B - A on the same days, for ROC-AUC, MCC and accuracy.

    Returns {metric: {"delta": point estimate, "lo": .., "hi": .., "p_gt0": share
    of bootstrap draws where B beats A}}.
    """
    y, prob_a, pred_a, prob_b, pred_b = map(np.asarray, (y, prob_a, pred_a, prob_b, pred_b))
    fns = _metric_fns()
    point = {k: f(y, prob_b, pred_b) - f(y, prob_a, pred_a) for k, f in fns.items()}
    draws = {k: [] for k in fns}
    for idx in block_bootstrap_indices(len(y), **kw):
        for k, f in fns.items():
            draws[k].append(f(y[idx], prob_b[idx], pred_b[idx]) - f(y[idx], prob_a[idx], pred_a[idx]))
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        lo, hi = np.nanquantile(v, [alpha / 2, 1 - alpha / 2])
        out[k] = {"delta": point[k], "lo": lo, "hi": hi, "p_gt0": float(np.nanmean(v > 0))}
    return out
