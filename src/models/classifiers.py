"""Trainable classifiers and their (small) hyper-parameter grids.

The SAME model families are used for every ablation row (A, B, C...), so any
difference in results comes from the features, not from a different algorithm.

  logreg   : L2-regularised logistic regression on standardised features --
             simple, interpretable coefficients, hard to overfit.
  xgboost  : gradient-boosted trees -- non-linear interactions
             (e.g. "negative news only matters when volatility is high").
  lightgbm : leaf-wise boosted trees; included because the brief names it for
             early fusion and it is often stronger than XGBoost on small tabular data.

All three are class-balanced (UP is ~55% of days), so P(UP) = 0.5 is a natural
decision threshold and no model wins MCC/F1 by always predicting the majority class.

Every factory takes `seed` so the final test run can average over SEEDS.
Grids stay small on purpose: with ~1,000 dev days a big search mostly fits noise.
Wider Optuna tuning is reserved for the best 1-2 architectures (TODO: src/tune.py).
"""
from itertools import product

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import SEED


def make_logreg(C=1.0, seed=SEED):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed),
    )


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost has no class_weight option; weight samples inversely to class frequency."""

    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kw)


def make_xgboost(max_depth=3, n_estimators=200, learning_rate=0.05, seed=SEED):
    return BalancedXGBClassifier(
        max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="logloss", random_state=seed, n_jobs=1,
    )


def make_lightgbm(num_leaves=8, n_estimators=200, learning_rate=0.05, seed=SEED):
    return LGBMClassifier(
        num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, class_weight="balanced",
        random_state=seed, n_jobs=1, verbose=-1,
    )


MODELS = {
    "logreg": (make_logreg, {"C": [0.01, 0.1, 1.0, 10.0]}),
    "xgboost": (make_xgboost, {"max_depth": [2, 3], "n_estimators": [100, 300],
                               "learning_rate": [0.03, 0.1]}),
    "lightgbm": (make_lightgbm, {"num_leaves": [4, 8], "n_estimators": [100, 300],
                                 "learning_rate": [0.03, 0.1]}),
}


def param_grid(grid):
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))
