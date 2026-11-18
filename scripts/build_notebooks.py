"""Generate the Task 3 analysis notebooks (notebook/03-06) from src/analysis.

The notebooks are thin: every computation lives in src/analysis/*.py so the
numbers in the notebooks, the figures in report/figures and the report text all
come from the same code. Rebuild + execute:

    python scripts/build_notebooks.py
    for nb in notebook/0[3-6]_*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import os, sys
os.chdir('..') if os.path.basename(os.getcwd()) == 'notebook' else None
sys.path.insert(0, os.getcwd())
import pandas as pd
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 200)
FIG = 'report/figures/'"""

NOTEBOOKS = {
    "03_ablation.ipynb": [
        ("md", "# 03 · Ablation study (Task 3)\n\nModel A (market only) vs Model B (market + Task 2 NLP) vs Model C "
               "(market + advanced text features), plus text-only, attention and late-fusion rows. All rows use the "
               "same protocol: expanding-window CV on train+val (600 out-of-fold days), parameters chosen by OOF "
               "ROC-AUC, final refit on train+val, a single evaluation on the 176 test days, 95% block-bootstrap CIs "
               "and a paired bootstrap against Model A. Produced by `python -m src.run_ablation`."),
        ("code", SETUP + "\nfrom src.analysis import ablation as ab"),
        ("md", "## Best model family per row (selected on OOF AUC only)"),
        ("code", "best = ab.best_per_set()\n"
                 "cols = ['label','model','oof_roc_auc','oof_d_roc_auc_vs_A','test_roc_auc','test_roc_auc_lo',"
                 "'test_roc_auc_hi','test_mcc','test_d_roc_auc_vs_A','test_d_roc_auc_lo','test_d_roc_auc_hi']\n"
                 "best[cols].round(3)"),
        ("md", "## Task 2 vs Task 3\n\nTask 2 numbers come from its single-validation-split protocol "
               "(`results/task2/`); Task 3 numbers from the CV protocol."),
        ("code", "ab.task2_vs_task3().round(3)"),
        ("md", "## Difference to Model A with 95% paired-bootstrap CIs"),
        ("code", "fig = ab.forest_plot(FIG + 'ablation_forest.pdf')"),
        ("md", "## Stability across the five CV blocks"),
        ("code", "fig = ab.fold_stability(('A', 'B', best.loc[best['set'].str.startswith('C_') & "
                 "(best['set'] != 'C_tuned')].sort_values('oof_roc_auc').iloc[-1]['set'], 'C_tuned'), "
                 "FIG + 'fold_stability.pdf')"),
        ("md", "## Full table (every model family)"),
        ("code", "res = ab.load()\nres[res['eval'] == 'test'][['set','model','roc_auc','mcc','accuracy','f1_macro',"
                 "'pred_up_rate','d_roc_auc_vs_A','p_roc_auc_gt_A']].round(3)"),
        ("md", "## Hyper-parameter tuning (Optuna) and late-fusion weights"),
        ("code", "import json\nprint(json.dumps(json.load(open('results/tuned_config.json'))['study'], indent=1))\n"
                 "print(json.dumps(json.load(open('results/late_fusion_weights.json')), indent=1))\n"
                 "print(json.dumps(json.load(open('results/dapt_log.json')), indent=1))"),
    ],
    "04_embedding_space.ipynb": [
        ("md", "# 04 · Embedding space: high- vs low-volatility days\n\nHigh/low volatility = top/bottom quintile "
               "of |JISDOR log return| over all 1,168 modelled days. Headlines are those aligned to that day "
               "(published before the fixing)."),
        ("code", SETUP + "\nfrom src.analysis import embedding_space as es"),
        ("md", "## UMAP of e5 headline embeddings"),
        ("code", "fig, arts, Z = es.umap_plot('e5', 3000, FIG + 'umap_volatility.png')"),
        ("md", "## Which topics are over-represented on high-volatility days?"),
        ("code", "lift, fig = es.topic_lift(FIG + 'topic_lift.pdf')\nlift.sort_values('log2_lift').round(3)"),
        ("md", "## Can daily embeddings separate high- from low-volatility days?\n\n5-fold stratified LogReg on "
               "PCA-16 of the daily mean vector, per encoder (a descriptive check, not a forecast)."),
        ("code", "es.separability().round(3)"),
    ],
    "05_interpretability.ipynb": [
        ("md", "# 05 · Feature importance & interpretability\n\nTreeSHAP on the named-feature Model C "
               "(`C_combo`: market + news volume + anchor axes + novelty + entity channels + BERTopic shares + "
               "sentiment), refitted on train+val and explained on the test days."),
        ("code", SETUP + "\nfrom src.analysis import interpret as it"),
        ("code", "groups, imp, shap_df, fig = it.shap_summary(FIG + 'shap_groups.pdf')\ngroups.round(4)"),
        ("code", "names = it.topic_names()\nimp.sort_values(ascending=False).head(20).rename(index=lambda c: names.get(c, c)).round(4)"),
        ("md", "## Largest prediction shifts vs Model A, and the news behind them"),
        ("code", "shifts = it.largest_shifts(shap_df, 8)\nshifts"),
        ("code", "for r in shifts.itertuples():\n    print(r.date, r.actual, f'P_A={r.P_A} P_C={r.P_C}', r.top_text_feature)\n"
                 "    for h in r.headlines: print('   ', h)"),
        ("md", "## What the attention model looked at (test days with the largest moves)"),
        ("code", "att = pd.read_csv('results/predictions/attention_top_headlines_MIL_e5.csv')\n"
                 "from src.analysis.errors import day_info\nret = day_info()['target_return']\n"
                 "big = ret.reindex(att['date'].unique()).abs().sort_values(ascending=False).index[:5]\n"
                 "att[att['date'].isin(big) & (att['rank'] <= 3)]"),
    ],
    "06_error_analysis.ipynb": [
        ("md", "# 06 · Correlation & error analysis\n\nWhere did text features help the model avoid a false "
               "market signal, and where did they add noise? Compares the best Model C (by OOF AUC) with Model A "
               "of the same family, on test and on the 600 OOF days."),
        ("code", SETUP + "\nfrom src.analysis import errors as er, ablation as ab\n"
                 "best = ab.best_per_set()\nc = best[best['set'].str.startswith('C_') & (best['set'] != 'C_tuned')]"
                 ".sort_values('oof_roc_auc').iloc[-1]\nC_KEY = f\"{c['set']}__{c['model']}\"; A_KEY = f\"A__{c['model']}\"\n"
                 "print(C_KEY, 'vs', A_KEY)"),
        ("md", "## Outcome counts"),
        ("code", "for ev in ('oof', 'test'):\n    o = er.outcomes(C_KEY, A_KEY, ev)\n    print(ev, o['outcome'].value_counts().to_dict())\n"
                 "out = er.outcomes(C_KEY, A_KEY, 'oof')"),
        ("md", "## When does text help? (OOF days, terciles)"),
        ("code", "er.breakdown(out, FIG + 'error_breakdown.pdf')"),
        ("md", "## Examples: largest moves where text helped / hurt"),
        ("code", "for kind in ('text helped', 'text hurt'):\n    ex = er.examples(out, kind, 3)\n    print('====', kind)\n"
                 "    for r in ex.itertuples():\n        print(r.date, f'ret={r._2}%', f'P_A={r.P_A} P_C={r.P_C}', f'n={r.n_headlines}')\n"
                 "        for h in r.fx_headlines: print('   ', h)"),
        ("md", "## Correlation of named text features with the same-day return (dev days with news)"),
        ("code", "corr = er.feature_correlations(FIG + 'feature_correlations.pdf')\ncorr.head(15).round(4)"),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                    for kind, src in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, ROOT / "notebook" / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
