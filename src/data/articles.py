"""Load the Task 1 article-level data (data/articles/articles_<year>.parquet).

One row per article. Columns: article_id, timestamp_wib, calendar_date_wib,
trading_date, alignment_case, source, section, url, title, body, has_body, themes,
gdelt_tone. `trading_date` is the first JISDOR fixing (10:00 WIB) at or after the
article's timestamp, i.e. the row of data/train|val|test.csv the article may be
used for -- never earlier (no look-ahead).

The CANONICAL ORDER (sorted by timestamp_wib, then article_id) is the row order of
every embedding cache in cache/emb/, so row i of an embedding matrix is article i
of load_articles().
"""
import pandas as pd

from src.config import ARTICLES_DIR

DEFAULT_COLUMNS = ["article_id", "timestamp_wib", "trading_date", "source", "title", "themes", "gdelt_tone"]


def load_articles(columns=DEFAULT_COLUMNS):
    """All articles 2021-2026 in canonical order. Pass columns=None for every column."""
    files = sorted(ARTICLES_DIR.glob("articles_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no article parquets in {ARTICLES_DIR}")
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
    return df.sort_values(["timestamp_wib", "article_id"], kind="stable").reset_index(drop=True)
