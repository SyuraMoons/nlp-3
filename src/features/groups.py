"""Feature groups: the building blocks of every ablation row.

A feature group is an object with fit(df) / transform(df), where df is a slice of
the split table (one row per trading day, with a `date` column). Anything learned
(TF-IDF vocabulary, PCA axes, scalers) is learned in fit(), which the CV runner
only ever calls on the fold's TRAINING days.

Group specs (strings used in config.ABLATION_SETS):
    "market"                 11 exchange-rate history features
    "task2_nlp"              Task 2 NLP features (27)
    "news_volume"            has_news, log_n_articles, share_weekend_news
    "table:<name>"           all columns of data/features/<name>.parquet
    "table:<name>:pca<k>"    same, standardised + PCA to k dims (fit on train days)

Contract for daily tables (data/features/<name>.parquet), written by
src/features/daily.py or any other feature script:
    * a `date` column (YYYY-MM-DD string, same format as data/train.csv)
    * one row per trading date that has news; dates missing from the table are
      filled with 0 (= no news), and the "news_volume" group carries has_news
    * numeric feature columns only, prefixed with the table name
    * built from news aligned to that date OR EARLIER only (no look-ahead)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES_DIR, SEED
from src.features.market_features import MARKET_FEATURES
from src.features.nlp_features import NLPFeatureExtractor, basic_news_features


class Columns:
    """Stateless: select fixed columns of the split table."""

    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, df):
        return self

    def transform(self, df):
        return df[self.columns].astype(float)


class NewsVolume:
    """News volume without any text content (shared by every Model C row)."""

    def fit(self, df):
        return self

    def transform(self, df):
        return basic_news_features(df)[["has_news", "log_n_articles", "share_weekend_news"]]


class DailyTable:
    """Join a precomputed daily text feature table on `date`, optionally + PCA."""

    def __init__(self, name, pca=None):
        self.name, self.pca_k = name, pca
        path = FEATURES_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- build it first (see TEAM_TASKS.md)")
        table = pd.read_parquet(path)
        table["date"] = table["date"].astype(str)
        self.table = table.set_index("date")

    def _lookup(self, df):
        return self.table.reindex(df["date"].astype(str)).fillna(0.0).set_axis(df.index)

    def fit(self, df):
        if self.pca_k:
            X = self._lookup(df).to_numpy(dtype=float)
            self.scaler = StandardScaler().fit(X)
            self.pca = PCA(n_components=self.pca_k, random_state=SEED).fit(self.scaler.transform(X))
        return self

    def transform(self, df):
        X = self._lookup(df)
        if not self.pca_k:
            return X.astype(float)
        Z = self.pca.transform(self.scaler.transform(X.to_numpy(dtype=float)))
        return pd.DataFrame(Z, index=df.index, columns=[f"{self.name}_pc{i}" for i in range(self.pca_k)])


def make_group(spec):
    if spec == "market":
        return Columns(MARKET_FEATURES)
    if spec == "task2_nlp":
        return NLPFeatureExtractor()
    if spec == "news_volume":
        return NewsVolume()
    if spec.startswith("table:"):
        parts = spec.split(":")
        pca = int(parts[2].removeprefix("pca")) if len(parts) > 2 else None
        return DailyTable(parts[1], pca=pca)
    raise ValueError(f"unknown feature group: {spec}")


class FeatureBuilder:
    """A list of groups fitted together; transform() concatenates their columns."""

    def __init__(self, specs):
        self.specs = list(specs)

    def fit(self, df):
        self.groups = [make_group(s).fit(df) for s in self.specs]
        return self

    def transform(self, df):
        X = pd.concat([g.transform(df) for g in self.groups], axis=1)
        if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite features from {self.specs}")
        return X
