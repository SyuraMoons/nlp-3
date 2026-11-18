"""Write the LaTeX tables used by report/Task3_Report.tex from results/ (so the
report can never drift from the numbers). Run after src.run_ablation:

    python scripts/make_report_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.ablation import GROUP, best_per_set  # noqa: E402

OUT = ROOT / "report" / "tables"
FEATS = {
    "A": "11 market features", "B": "+ volume, GDELT tone, InSet, TF-IDF/SVD",
    "C_word2vec": "+ Word2Vec SIF, PCA-16", "C_fasttext": "+ FastText SIF, PCA-16",
    "C_indobert": "+ IndoBERT mean, PCA-16", "C_dapt": "+ IndoBERT-DAPT mean, PCA-16",
    "C_e5": "+ e5 mean, PCA-16", "C_sentiment": "+ BERT sentiment (4)",
    "C_e5_recency": "+ e5 recency-weighted, PCA-16", "C_e5_decay": "+ e5 decay $h{=}3$d, PCA-16",
    "C_anchors": "+ 24 anchor axes, novelty", "C_entities": "+ 5 entity channels (20)",
    "C_topics": "+ 30 BERTopic shares", "C_gdelt": "+ 30 GDELT theme shares",
    "C_combo": "+ anchors, entities, topics, sentiment", "C_tuned": "Optuna-tuned C",
    "T_e5": "e5 PCA-16 only", "T_anchors": "anchor axes only", "MIL_e5": "attention over headlines",
    "L_e5": "A $\\oplus$ T-e5 (logit stack)", "L_mil": "A $\\oplus$ attention (logit stack)",
}
SHORT = {"logreg": "LR", "xgboost": "XGB", "lightgbm": "LGBM", "attention": "MIL",
         "stack(logreg)": "stack", "stack(attention)": "stack", "-": "--"}


def ci(v, lo, hi, signed=False):
    if pd.isna(v):
        return "--"
    f = "{:+.3f}" if signed else "{:.3f}"
    s = f"{f.format(v)} {{\\scriptsize[{f.format(lo)}, {f.format(hi)}]}}"
    return s


def ablation_table():
    b = best_per_set()
    lines = []
    fam_prev = None
    for r in b.itertuples():
        if r.set in ("majority_class", "persistence"):
            continue
        fam = GROUP.get(r.set, ("other",))[0]
        if fam_prev is not None and fam != fam_prev:
            lines.append("\\midrule")
        fam_prev = fam
        name = r.set.replace("_", "\\_")
        bold = r.test_d_roc_auc_lo > 0 if not pd.isna(r.test_d_roc_auc_lo) else False
        lines.append(" & ".join([
            f"\\textbf{{{name}}}" if r.set in ("A", "B") else name, FEATS.get(r.set, ""), SHORT.get(r.model, r.model),
            f"{r.oof_roc_auc:.3f}", ci(r.oof_d_roc_auc_vs_A, r.oof_d_roc_auc_lo, r.oof_d_roc_auc_hi, True),
            ci(r.test_roc_auc, r.test_roc_auc_lo, r.test_roc_auc_hi), f"{r.test_mcc:+.3f}",
            ci(r.test_d_roc_auc_vs_A, r.test_d_roc_auc_lo, r.test_d_roc_auc_hi, True)]) + " \\\\")
    head = ("\\begin{tabular}{@{}l l l c c c c c@{}}\n\\toprule\n"
            "\\textbf{Row} & \\textbf{Features} & \\textbf{Model} & \\textbf{OOF AUC} & "
            "\\textbf{OOF $\\Delta$AUC vs A} & \\textbf{Test AUC} & \\textbf{Test MCC} & "
            "\\textbf{Test $\\Delta$AUC vs A} \\\\\n\\midrule\n")
    (OUT / "ablation.tex").write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def task2_rows():
    t2 = pd.read_csv(ROOT / "results" / "task2" / "metrics.csv")
    t2 = t2[(t2["split"] == "test") & t2["feature_set"].isin(["market", "market+nlp"])]
    lines = []
    for r in t2.itertuples():
        name = "A" if r.feature_set == "market" else "B"
        lines.append(f"Task 2 {name} & {SHORT[r.model]} & {r.roc_auc:.3f} & {r.mcc:+.3f} & {r.accuracy:.3f} & "
                     f"{r.pred_up_rate:.2f} \\\\")
    res = pd.read_csv(ROOT / "results" / "ablation.csv")
    for s in ("A", "B"):
        for m in ("logreg", "xgboost"):
            r = res[(res["set"] == s) & (res["model"] == m) & (res["eval"] == "test")].iloc[0]
            lines.append(f"Task 3 {s} & {SHORT[m]} & {r.roc_auc:.3f} & {r.mcc:+.3f} & {r.accuracy:.3f} & "
                         f"{r.pred_up_rate:.2f} \\\\")
    head = ("\\begin{tabular}{@{}l l c c c c@{}}\n\\toprule\n\\textbf{Variant} & \\textbf{Model} & "
            "\\textbf{AUC} & \\textbf{MCC} & \\textbf{Acc} & \\textbf{UP\\%} \\\\\n\\midrule\n")
    (OUT / "task2_vs_task3.tex").write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def numbers():
    """Key scalars as LaTeX macros (\\newcommand) for the text."""
    res = pd.read_csv(ROOT / "results" / "ablation.csv")
    p = res.pivot_table(index=["set", "model"], columns="eval", values="roc_auc").dropna()
    p = p[~p.index.get_level_values(0).isin(["majority_class", "persistence"])]
    from scipy.stats import spearmanr
    rho = spearmanr(p["oof"], p["test"]).statistic
    d = json.loads((ROOT / "results" / "dapt_log.json").read_text())
    lw = json.loads((ROOT / "results" / "late_fusion_weights.json").read_text())
    tuned = json.loads((ROOT / "results" / "tuned_config.json").read_text()) \
        if (ROOT / "results" / "tuned_config.json").exists() else None
    m = {
        "NRows": len(p), "RhoOofTest": f"{rho:+.2f}",
        "NSigWorse": int(((res["eval"] == "test") & (res["d_roc_auc_hi"] < 0)).sum()),
        "NSigBetter": int(((res["d_roc_auc_lo"] > 0)).sum()),
        "PplGeneral": f"{d['val_perplexity_general_indobert']:.1f}", "PplDapt": f"{d['val_perplexity_dapt']:.1f}",
        "DaptHeadlines": f"{d['train_headlines']:,}", "DaptSteps": f"{d['dapt_steps']:,}",
        "WtimeMil": f"{lw['L_mil']['w_time']:+.2f}", "WtextMil": f"{lw['L_mil']['w_text']:+.2f}",
        "WtimeE": f"{lw['L_e5']['w_time']:+.2f}", "WtextE": f"{lw['L_e5']['w_text']:+.2f}",
    }
    m.update({k: "--" for k in ("TunedTrials", "TunedOof", "TunedStart", "TunedModel", "TunedTest", "TunedTestD")})
    if tuned:
        r = res[(res["set"] == "C_tuned")]
        m.update({"TunedTrials": tuned["study"]["n_trials"], "TunedOof": f"{tuned['study']['best_oof_auc']:.3f}",
                  "TunedStart": ", ".join(s.replace("_", "\\_") for s in tuned["study"]["starting_sets"]),
                  "TunedModel": SHORT.get(tuned["model"], tuned["model"]),
                  "TunedTest": f"{r[r['eval'] == 'test']['roc_auc'].iloc[0]:.3f}" if len(r) else "--",
                  "TunedTestD": f"{r[r['eval'] == 'test']['d_roc_auc_vs_A'].iloc[0]:+.3f}" if len(r) else "--"})
    (OUT / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    print(m)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ablation_table()
    task2_rows()
    numbers()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
