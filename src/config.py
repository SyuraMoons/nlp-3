"""Central configuration for Task 3. Every script imports from here, so a design
choice (CV folds, feature sets, encoders) is changed in ONE place.

Run all scripts from the repository root, e.g. ``python -m src.run_ablation``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LEXICON_DIR = DATA_DIR / "lexicon"

# Task 1 output: one row per Bank Indonesia JISDOR trading day, with the news that
# was aligned to that day (all of it published BEFORE that day's 10:00 WIB fixing).
SOURCE_DATASET = DATA_DIR / "raw" / "final_dataset.csv"

# Task 1 article-level data: one row per article (title, timestamp, trading_date,
# GDELT themes, body for 2024). Needed for per-headline embeddings and recency.
ARTICLES_DIR = DATA_DIR / "articles"

# Daily text feature tables (one row per trading date, keyed by `date`).
# Small -> committed to git. Built from the embedding cache by src/features/daily.py.
FEATURES_DIR = DATA_DIR / "features"

# Per-headline embedding cache (hundreds of MB per encoder) -> NOT in git.
CACHE_DIR = ROOT / "cache"

SEED = 42

# ---------------------------------------------------------------------------
# Task formulation (unchanged from Task 2)
# ---------------------------------------------------------------------------
# Row t = trading day t. Target: did the JISDOR rate go UP at fixing t vs. t-1?
#   y_t = 1 if rate_t > rate_{t-1}  (USD up / IDR weaker), 0 if down; flat days dropped.
# Inputs: exchange-rate history up to t-1, and news aligned to day t (published
# before fixing t). Nothing from fixing t itself is ever a feature.
N_LAGS = 5            # return_lag1 .. return_lag5
ROLLING_WINDOWS = (5, 20)

# Chronological split, frozen from Task 2 (train 817 / val 175 / test 176 days).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# Task 3 evaluation protocol (src/evaluation/cv.py). Applied identically to A, B, C.
# ---------------------------------------------------------------------------
#   dev  = train + val (992 days, 2021-09-30 -> 2025-11-24)
#   test = the Task 2 test split (176 days), touched ONCE per final model.
#
# 1. Expanding-window CV on dev: the last N_FOLDS * FOLD_SIZE dev days are cut into
#    consecutive validation blocks; each block is predicted by a model trained on
#    every dev day before it. With 5 x 120 the scored period starts ~2023-04, after
#    the Aug 2022 - Jan 2023 news gap, and gives 600 out-of-fold (OOF) predictions.
# 2. Hyper-parameters are selected by OOF ROC-AUC (threshold-free).
# 3. Every model is class-balanced, and UP is predicted when P(UP) >= 0.5.
#    Secondary metric "mcc_tuned": a threshold tuned for MCC on OOF probabilities
#    (walk-forward: fold k uses a threshold tuned on folds < k; test uses one tuned
#    on all OOF). Kept secondary because absolute thresholds transfer poorly from
#    fold models to the final refit (first run: B predicted UP on ~1% of test days).
# 4. Final model: refit on all of dev (text extractors too), predict test with
#    every seed in SEEDS and average the probabilities.
# 5. Uncertainty: circular block bootstrap (blocks of BOOTSTRAP_BLOCK days) for
#    95% CIs, and paired bootstrap for "model X minus Model A" differences.
CV_N_FOLDS = 5
CV_FOLD_SIZE = 120
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = 5

# ---------------------------------------------------------------------------
# Task 2 NLP features (src/features/nlp_features.py) -- Model B's text features.
# ---------------------------------------------------------------------------
NLP_CONFIG = {
    "use_lexicon": True,          # InSet Indonesian sentiment lexicon on headlines
    "use_tfidf": True,            # TF-IDF on headlines, reduced with TruncatedSVD
    "tfidf_max_features": 5000,
    "tfidf_min_df": 5,
    "svd_components": 20,
}

# ---------------------------------------------------------------------------
# Ablation: feature set name -> list of feature groups (src/features/groups.py).
#   "market"       11 exchange-rate history features
#   "task2_nlp"    Task 2 NLP: volume + GDELT tone + InSet lexicon + TF-IDF/SVD (27)
#   "news_volume"  has_news, log_n_articles, share_weekend_news (no text content)
#   "table:<name>[:pca<k>]"
#                  columns of data/features/<name>.parquet (daily text features),
#                  optionally standardised + PCA to k dims, fitted on the fold's
#                  training days only.
#
# Model C = market + news volume + ONE advanced text group, so any gain over A is
# attributable to that text representation. Add a row per encoder / feature group
# once its daily table exists in data/features/.
# ---------------------------------------------------------------------------
ABLATION_SETS = {
    "A": ["market"],
    "B": ["market", "task2_nlp"],
    # --- Model C variants (uncomment once the daily table exists) ---
    # "C_static":   ["market", "news_volume", "table:fasttext_mean:pca16"],
    "C_e5":         ["market", "news_volume", "table:e5_mean:pca16"],
    # "C_indobert": ["market", "news_volume", "table:indobert_mean:pca16"],
    # "C_dapt":     ["market", "news_volume", "table:indobert_dapt_mean:pca16"],
    # "C_sent":     ["market", "news_volume", "table:sentiment"],
    # "C_anchors":  ["market", "news_volume", "table:e5_anchors"],
    # "C_entities": ["market", "news_volume", "table:entity_channels"],
}

# Models run for every ablation set (keys of src.models.classifiers.MODELS).
ABLATION_MODELS = ["logreg", "xgboost", "lightgbm"]
