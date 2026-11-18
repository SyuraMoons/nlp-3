"""Ablation analysis: Task 2 vs Task 3 comparison table, forest plot of the
difference to Model A, and per-fold stability. Used by notebook/03_ablation.ipynb
and the report (report/figures/)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, ROOT

FIG_DIR = ROOT / "report" / "figures"

GROUP = {  # set -> (family, label) for tables and plots
    "A": ("baseline", "A: market only"),
    "B": ("baseline", "B: market + Task 2 NLP"),
    "C_word2vec": ("representation", "C: Word2Vec (static)"),
    "C_fasttext": ("representation", "C: FastText (static)"),
    "C_indobert": ("representation", "C: IndoBERT (general)"),
    "C_dapt": ("representation", "C: IndoBERT-DAPT (domain)"),
    "C_e5": ("representation", "C: multilingual-e5"),
    "C_sentiment": ("representation", "C: transformer sentiment"),
    "C_e5_recency": ("aggregation", "C: e5, recency-weighted"),
    "C_e5_decay": ("aggregation", "C: e5, decay (h=3 days)"),
    "C_anchors": ("beyond sentiment", "C: anchor axes + novelty"),
    "C_entities": ("beyond sentiment", "C: entity channels"),
    "C_topics": ("beyond sentiment", "C: BERTopic shares"),
    "C_gdelt": ("beyond sentiment", "C: GDELT themes"),
    "C_combo": ("beyond sentiment", "C: all named text features"),
    "C_tuned": ("tuned", "C: Optuna-tuned"),
    "T_e5": ("text only", "T: e5 only"),
    "T_anchors": ("text only", "T: anchors only"),
    "MIL_e5": ("text only", "T: attention over headlines"),
    "L_e5": ("late fusion", "L: A + e5 text model"),
    "L_mil": ("late fusion", "L: A + attention model"),
}


def load():
    return pd.read_csv(RESULTS_DIR / "ablation.csv")


def best_per_set(res=None):
    """For every set, the model family with the best OOF AUC (selection on dev only),
    with its OOF and test metrics side by side."""
    res = load() if res is None else res
    oof = res[res["eval"] == "oof"]
    pick = oof.loc[oof.groupby("set")["roc_auc"].idxmax(), ["set", "model"]]
    rows = []
    for _, p in pick.iterrows():
        r = {"set": p["set"], "model": p["model"]}
        for ev in ("oof", "test"):
            x = res[(res["set"] == p["set"]) & (res["model"] == p["model"]) & (res["eval"] == ev)].iloc[0]
            for k in ("roc_auc", "roc_auc_lo", "roc_auc_hi", "mcc", "accuracy", "f1_macro",
                      "d_roc_auc_vs_A", "d_roc_auc_lo", "d_roc_auc_hi", "d_mcc_vs_A", "p_roc_auc_gt_A"):
                r[f"{ev}_{k}"] = x.get(k, np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["family"] = out["set"].map(lambda s: GROUP.get(s, ("other", s))[0])
    out["label"] = out["set"].map(lambda s: GROUP.get(s, ("other", s))[1])
    order = {s: i for i, s in enumerate(GROUP)}
    return out.sort_values("set", key=lambda c: c.map(order).fillna(99)).reset_index(drop=True)


def task2_vs_task3():
    """Task 2 numbers (single val split, as reported) next to Task 3 (CV protocol)."""
    t2 = pd.read_csv(RESULTS_DIR / "task2" / "metrics.csv")
    t2 = t2[(t2["split"] == "test") & t2["feature_set"].isin(["market", "market+nlp"])]
    rows = [{"variant": f"Task 2 {'A' if r.feature_set == 'market' else 'B'} ({r.model})", "protocol": "Task 2",
             "test_auc": r.roc_auc, "test_mcc": r.mcc, "test_acc": r.accuracy, "test_f1": r.f1_macro,
             "oof_auc": np.nan} for r in t2.itertuples()]
    for r in best_per_set().itertuples():
        rows.append({"variant": f"{r.label} ({r.model})", "protocol": "Task 3",
                     "test_auc": r.test_roc_auc, "test_mcc": r.test_mcc, "test_acc": r.test_accuracy,
                     "test_f1": r.test_f1_macro, "oof_auc": r.oof_roc_auc})
    return pd.DataFrame(rows)


def forest_plot(path=None):
    """ΔAUC vs Model A (paired bootstrap 95% CI), OOF and test, best model per set."""
    b = best_per_set()
    b = b[b["set"] != "A"].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.32 * len(b) + 1.2), sharey=True)
    colors = {"baseline": "#6b7280", "representation": "#2563eb", "aggregation": "#0891b2",
              "beyond sentiment": "#059669", "tuned": "#7c3aed", "text only": "#d97706", "late fusion": "#dc2626"}
    for ax, ev, title in zip(axes, ("oof", "test"), ("Out-of-fold (600 days)", "Test (176 days)")):
        y = np.arange(len(b))
        d, lo, hi = b[f"{ev}_d_roc_auc_vs_A"], b[f"{ev}_d_roc_auc_lo"], b[f"{ev}_d_roc_auc_hi"]
        c = [colors.get(f, "#111") for f in b["family"]]
        ax.hlines(y, lo, hi, color=c, lw=2)
        ax.scatter(d, y, color=c, zorder=3, s=18)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ΔROC-AUC vs Model A")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(np.arange(len(b)), b["label"], fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return fig


def fold_stability(sets=("A", "B"), path=None):
    """Per-fold OOF AUC of the best model of each set (5 consecutive 120-day blocks)."""
    params = json.loads((RESULTS_DIR / "ablation_params.json").read_text())
    b = best_per_set().set_index("set")
    fig, ax = plt.subplots(figsize=(6, 3))
    for s in sets:
        if s not in b.index:
            continue
        key = f"{s}__{b.loc[s, 'model']}"
        ax.plot(range(1, 6), params[key]["fold_auc"], marker="o", label=GROUP.get(s, (0, s))[1])
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("CV block (120 trading days each, 2023-05 → 2025-11)")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return fig
