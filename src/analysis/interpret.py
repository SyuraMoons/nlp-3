"""Feature importance & interpretability (brief 3.c.ii): which topics, actors and
news themes drove the model's largest prediction shifts?

PCA components of embeddings are not human-readable, so interpretability uses
the NAMED-feature Model C ("C_combo" in config.ABLATION_SETS): market features +
news volume + anchor axes + novelty/dispersion + entity channels + BERTopic shares
+ transformer sentiment. A LightGBM model is refitted on all dev days and
explained on the test days with TreeSHAP (exact Shapley values for trees).

  shap_summary          mean |SHAP| per feature group and top features
  largest_shifts        test days where the text-aware model moved furthest from
                        Model A, with the headlines behind the dominant text feature
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ABLATION_SETS, RESULTS_DIR
from src.data.articles import load_articles
from src.features.entities import PATTERNS
from src.features.groups import FeatureBuilder
from src.features.semantic import ANCHOR_AXES, anchor_scores
from src.models.classifiers import make_lightgbm
from src.run_ablation import load_dev_test
from src.text.cache import load_embeddings

NAMED_SET = "C_combo"


def feature_group(col):
    if col.startswith("return_") or col in ("up_days_last5", "calendar_days_since_prev"):
        return "market"
    if col in ("has_news", "log_n_articles", "share_weekend_news"):
        return "news volume"
    if "_ax_" in col:
        return "anchor axes"
    if col.endswith(("_novelty", "_dispersion")):
        return "novelty / dispersion"
    if col.startswith("topic_"):
        return "topics"
    if col.startswith("sent_"):
        return "sentiment"
    if col.split("_")[0] in PATTERNS:
        return "entity channels"
    return "other"


def topic_names():
    t = pd.read_csv(RESULTS_DIR / "topics_e5.csv").set_index("Topic")["keywords"]
    return {f"topic_{i}": f"topic {i} ({', '.join(k.split(', ')[:3])})" for i, k in t.items()}


def fit_named_model():
    """LightGBM on the named-feature set, params from the ablation run if available."""
    dev, test = load_dev_test()
    specs = ABLATION_SETS[NAMED_SET]
    fb = FeatureBuilder(specs).fit(dev)
    X_dev, X_test = fb.transform(dev), fb.transform(test)
    params = {"num_leaves": 4, "n_estimators": 300, "learning_rate": 0.03}
    pfile = RESULTS_DIR / "ablation_params.json"
    if pfile.exists():
        params = json.loads(pfile.read_text()).get(f"{NAMED_SET}__lightgbm", {}).get("params", params)
    model = make_lightgbm(**params).fit(X_dev, dev["target"].to_numpy())
    return model, X_dev, X_test, dev, test


def shap_summary(path=None, top=15):
    import shap
    model, X_dev, X_test, dev, test = fit_named_model()
    sv = shap.TreeExplainer(model).shap_values(X_test)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=X_test.columns)
    groups = imp.groupby(imp.index.map(feature_group)).sum().sort_values()
    names = topic_names()
    top_f = imp.sort_values().tail(top)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].barh(groups.index, groups.values, color="#2563eb")
    axes[0].set_title("Mean |SHAP| by feature group (test)", fontsize=9)
    axes[1].barh([names.get(c, c) for c in top_f.index], top_f.values,
                 color=["#6b7280" if feature_group(c) == "market" else "#059669" for c in top_f.index])
    axes[1].set_title(f"Top {top} features (grey = market, green = text)", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    shap_df = pd.DataFrame(sv, columns=X_test.columns, index=test["date"].to_numpy())
    return groups, imp, shap_df, fig


def _day_headlines(articles, date, col, scores, k=3):
    """Headlines behind feature `col` on `date`."""
    day = articles.index[articles["trading_date"] == date]
    if len(day) == 0:
        return []
    if "_ax_" in col:
        axis = col.split("_ax_")[1].rsplit("_", 1)[0]
        j = [a for a, _, _ in ANCHOR_AXES].index(axis)
        s = scores[day, j]
        order = np.argsort(-np.abs(s))[:k]
        return [f"[{s[o]:+.3f}] {articles.at[day[o], 'title']}" for o in order]
    ch = col.split("_")[0]
    if ch in PATTERNS:
        t = articles.loc[day, "title"]
        return t[t.str.contains(PATTERNS[ch])].head(k).tolist()
    return articles.loc[day, "title"].head(k).tolist()


def largest_shifts(shap_df, n=8, a_key="A__lightgbm"):
    """Test days where the named-feature model differs most from Model A, with the
    text feature that contributed most and the headlines behind it."""
    test_pred = pd.read_csv(RESULTS_DIR / "predictions" / "test.csv").set_index("date")
    model, X_dev, X_test, dev, test = fit_named_model()
    p_c = pd.Series(model.predict_proba(X_test)[:, 1], index=test["date"].to_numpy())
    shift = (p_c - test_pred[a_key].reindex(p_c.index)).dropna()
    days = shift.abs().sort_values(ascending=False).index[:n]
    articles = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
    X = load_embeddings("e5", n_expected=len(articles))
    scores, _ = anchor_scores(X, "e5")
    names = topic_names()
    ret = test.set_index("date")["target_return"]
    rows = []
    for d in days:
        text_cols = [c for c in shap_df.columns if feature_group(c) not in ("market", "news volume")]
        top_col = shap_df.loc[d, text_cols].abs().idxmax()
        rows.append({"date": d, "return_%": round(100 * ret[d], 3), "actual": "UP" if ret[d] > 0 else "DOWN",
                     "P_A": round(test_pred.at[d, a_key], 3), "P_C": round(p_c[d], 3),
                     "top_text_feature": names.get(top_col, top_col),
                     "shap": round(shap_df.at[d, top_col], 3),
                     "headlines": _day_headlines(articles, d, top_col, scores)})
    return pd.DataFrame(rows)
