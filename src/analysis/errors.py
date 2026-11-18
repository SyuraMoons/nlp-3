"""Correlation & error analysis (brief 3.c.iii): on which days did text features
help the model avoid a false market signal, and on which did they add noise?

  outcomes            per day: A right / C right -> "both right", "both wrong",
                      "text helped" (C right, A wrong), "text hurt" (A right, C wrong)
  breakdown           helped minus hurt by news volume, move size and geopolitical
                      news share (terciles)
  examples            the biggest-move days in each outcome, with headlines
  feature_correlations  Spearman correlation of named text features with the
                      same-day JISDOR return on dev days with news
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import FEATURES_DIR, RESULTS_DIR
from src.data.articles import load_articles
from src.run_ablation import load_dev_test

TABLES = ["e5_anchors", "e5_dynamics", "entity_channels", "sentiment"]


def day_info():
    dev, test = load_dev_test()
    d = pd.concat([dev, test], ignore_index=True).set_index("date")
    ent = pd.read_parquet(FEATURES_DIR / "entity_channels.parquet").set_index("date")
    d["geo_share"] = ent["geo_share"].reindex(d.index).fillna(0)
    return d


def outcomes(c_key, a_key, ev="test"):
    p = pd.read_csv(RESULTS_DIR / "predictions" / f"{ev}.csv").set_index("date")
    y = p["target"]
    a_ok, c_ok = (p[a_key] >= 0.5).astype(int) == y, (p[c_key] >= 0.5).astype(int) == y
    out = pd.DataFrame({"target": y, "P_A": p[a_key], "P_C": p[c_key]})
    out["outcome"] = np.select([a_ok & c_ok, ~a_ok & ~c_ok, c_ok & ~a_ok], ["both right", "both wrong", "text helped"],
                               "text hurt")
    info = day_info()
    for col in ("target_return", "n_articles", "geo_share"):
        out[col] = info[col].reindex(out.index)
    return out


def breakdown(out, path=None):
    out = out.copy()
    out["|move|"] = pd.qcut(out["target_return"].abs(), 3, labels=["small", "medium", "large"])
    out["news volume"] = pd.qcut(out["n_articles"].rank(method="first"), 3, labels=["low", "medium", "high"])
    out["geo news"] = pd.qcut(out["geo_share"].rank(method="first"), 3, labels=["low", "medium", "high"])
    rows = []
    for by in ("|move|", "news volume", "geo news"):
        for lvl, g in out.groupby(by, observed=True):
            rows.append({"by": by, "level": str(lvl), "days": len(g),
                         "helped": (g["outcome"] == "text helped").sum(),
                         "hurt": (g["outcome"] == "text hurt").sum()})
    t = pd.DataFrame(rows)
    t["net"] = t["helped"] - t["hurt"]
    if path:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["helped"], 0.4, label="text helped (C right, A wrong)", color="#059669")
        ax.bar(x + 0.2, t["hurt"], 0.4, label="text hurt (A right, C wrong)", color="#dc2626")
        short = {"|move|": "|move|", "news volume": "news", "geo news": "geo"}
        ax.set_xticks(x, [f"{short[b]}\n{l}" for b, l in zip(t["by"], t["level"])], fontsize=7)
        ax.set_ylabel("days")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return t


def examples(out, outcome, n=4, k=3):
    arts = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
    from src.features.entities import PATTERNS
    fx = PATTERNS["usd"].pattern + "|" + PATTERNS["idr"].pattern + "|" + PATTERNS["geo"].pattern
    sel = out[out["outcome"] == outcome].assign(m=lambda d: d["target_return"].abs()).sort_values(
        "m", ascending=False).head(n)
    rows = []
    for d, r in sel.iterrows():
        t = arts.loc[arts["trading_date"] == d, "title"]
        rel = t[t.str.contains(fx, case=False, regex=True)]
        rows.append({"date": d, "return_%": round(100 * r["target_return"], 3), "P_A": round(r["P_A"], 3),
                     "P_C": round(r["P_C"], 3), "n_headlines": len(t),
                     "fx_headlines": (rel if len(rel) else t).head(k).tolist()})
    return pd.DataFrame(rows)


def feature_correlations(path=None, top=12):
    dev, _ = load_dev_test()
    dev = dev[dev["n_articles"] > 0].set_index("date")
    rows = []
    for name in TABLES:
        T = pd.read_parquet(FEATURES_DIR / f"{name}.parquet").set_index("date")
        T = T.reindex(dev.index)
        for c in T.columns:
            ok = T[c].notna()
            rho, p = spearmanr(T.loc[ok, c], dev.loc[ok, "target_return"])
            rows.append({"feature": c, "table": name, "rho": rho, "p": p, "n": int(ok.sum())})
    r = pd.DataFrame(rows).sort_values("rho", key=np.abs, ascending=False)
    if path:
        s = r.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 0.2 * top + 1))
        ax.barh(s["feature"], s["rho"], color=["#dc2626" if v > 0 else "#2563eb" for v in s["rho"]])
        ci = 1.96 / np.sqrt(s["n"].iloc[0])
        ax.axvspan(-ci, ci, color="grey", alpha=0.2, label="|rho| < 95% null band")
        ax.set_xlabel("Spearman rho with same-day JISDOR return (dev days with news)")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return r
