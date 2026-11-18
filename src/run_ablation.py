"""Ablation runner: evaluate every feature set x model under the Task 3 protocol.

Protocol (src/config.py has the full rationale):
  1. dev = train + val. Expanding-window CV -> out-of-fold (OOF) P(UP) for every
     hyper-parameter setting. Feature groups are refitted inside every fold.
  2. Pick the setting with the best OOF ROC-AUC. Predictions use P(UP) >= 0.5
     (all models are class-balanced). Secondary: an MCC-tuned threshold
     (walk-forward on OOF; tuned on all OOF for test) reported as mcc_tuned.
  3. Refit on all of dev with each seed in SEEDS, average P(UP) on test,
     evaluate ONCE. 95% CIs from a block bootstrap; every row is also
     compared with Model A (same model family) by a paired bootstrap.

Outputs (results/):
  ablation.csv            one row per (set, model, eval) with metrics, CIs, deltas vs A
  ablation.md             the same as markdown tables (paste into the report)
  predictions/oof.csv     per-day OOF P(UP) of every (set, model) + naive baselines
  predictions/test.csv    per-day test P(UP) and predictions

Run with:
  python -m src.run_ablation                      # every set in ABLATION_SETS
  python -m src.run_ablation --sets A B --models logreg
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import ABLATION_MODELS, ABLATION_SETS, RESULTS_DIR, SEEDS
from src.data.load import load_all
from src.evaluation.cv import expanding_folds, tune_threshold, walk_forward_predictions
from src.evaluation.metrics import evaluate
from src.evaluation.stats import bootstrap_ci, paired_bootstrap_delta
from src.features.groups import FeatureBuilder
from src.models.classifiers import MODELS, param_grid
from src.models.naive import MajorityClass, Persistence

PRED_DIR = RESULTS_DIR / "predictions"


def load_dev_test():
    data = load_all()
    dev = pd.concat([data["train"], data["val"]], ignore_index=True)
    return dev, data["test"].reset_index(drop=True)


def cv_features(specs, dev, folds):
    """Fit the feature groups on each fold's training days; return [(X_tr, X_va)]."""
    out = []
    for tr, va in folds:
        fb = FeatureBuilder(specs).fit(dev.iloc[tr])
        out.append((fb.transform(dev.iloc[tr]), fb.transform(dev.iloc[va])))
    return out


def run_cv(model_name, fold_X, y, folds):
    """Grid search by pooled OOF ROC-AUC. Returns (best_params, oof_prob, per-param log)."""
    factory, grid = MODELS[model_name]
    y_oof = np.concatenate([y[va] for _, va in folds])
    log, best = [], None
    for params in param_grid(grid):
        probs = [factory(**params).fit(X_tr, y[tr]).predict_proba(X_va)[:, 1]
                 for (tr, _), (X_tr, X_va) in zip(folds, fold_X)]
        prob = np.concatenate(probs)
        auc = evaluate(y_oof, (prob >= 0.5).astype(int), prob)["roc_auc"]
        log.append({"params": params, "oof_auc": auc})
        if best is None or auc > best[1]:
            best = (params, auc, prob)
    return best[0], best[2], log


def fit_test(specs, model_name, params, dev, test):
    """Refit on all of dev for every seed; return the seed-averaged test P(UP)."""
    fb = FeatureBuilder(specs).fit(dev)
    X_dev, X_test = fb.transform(dev), fb.transform(test)
    factory, _ = MODELS[model_name]
    probs = [factory(**params, seed=s).fit(X_dev, dev["target"].to_numpy()).predict_proba(X_test)[:, 1]
             for s in SEEDS]
    return np.mean(probs, axis=0), np.std([evaluate(test["target"], (p >= 0.5).astype(int), p)["roc_auc"]
                                           for p in probs])


def score(y, prob, pred):
    m = evaluate(y, pred, prob)
    ci = bootstrap_ci(y, prob, pred)
    for k, (lo, hi) in ci.items():
        m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
    return m


def add_delta(row, y, ref, prob, pred):
    """Paired bootstrap difference vs Model A with the same model family."""
    if ref is None:
        return
    d = paired_bootstrap_delta(y, ref["prob"], ref["pred"], prob, pred)
    for k in ("roc_auc", "mcc"):
        row[f"d_{k}_vs_A"] = d[k]["delta"]
        row[f"d_{k}_lo"], row[f"d_{k}_hi"] = d[k]["lo"], d[k]["hi"]
        row[f"p_{k}_gt_A"] = d[k]["p_gt0"]


def naive_rows(dev, test, folds, rows, oof_cols, test_cols):
    y_dev = dev["target"].to_numpy()
    for name, model in [("majority_class", MajorityClass()), ("persistence", Persistence())]:
        prob = np.concatenate([model.fit(dev.iloc[tr], y_dev[tr]).predict_proba(dev.iloc[va])[:, 1]
                               for tr, va in folds])
        pred = np.concatenate([model.predict(dev.iloc[va]) for _, va in folds])
        y_oof = np.concatenate([y_dev[va] for _, va in folds])
        rows.append({"set": name, "model": "-", "eval": "oof", **score(y_oof, prob, pred)})
        oof_cols[f"{name}"] = prob
        model.fit(dev, y_dev)
        prob, pred = model.predict_proba(test)[:, 1], model.predict(test)
        rows.append({"set": name, "model": "-", "eval": "test", **score(test["target"], prob, pred)})
        test_cols[f"{name}"] = prob


def main(sets, models):
    dev, test = load_dev_test()
    y_dev, y_test = dev["target"].to_numpy(), test["target"].to_numpy()
    folds = expanding_folds(len(dev))
    y_oof = np.concatenate([y_dev[va] for _, va in folds])
    oof_dates = np.concatenate([dev["date"].to_numpy()[va] for _, va in folds])
    print(f"[ablation] dev {len(dev)} days, {len(folds)} folds, OOF {len(y_oof)} days "
          f"({oof_dates[0]} -> {oof_dates[-1]}); test {len(test)} days")

    rows, oof_cols, test_cols, ref, chosen = [], {}, {}, {}, {}
    naive_rows(dev, test, folds, rows, oof_cols, test_cols)

    for set_name in sets:
        specs = ABLATION_SETS[set_name]
        t0 = time.time()
        fold_X = cv_features(specs, dev, folds)
        n_feat = fold_X[0][0].shape[1]
        for model_name in models:
            params, prob_oof, log = run_cv(model_name, fold_X, y_dev, folds)
            pred_oof = (prob_oof >= 0.5).astype(int)
            pred_oof_tuned, _ = walk_forward_predictions(y_oof, prob_oof, [len(va) for _, va in folds])
            threshold, _ = tune_threshold(y_oof, prob_oof)
            prob_test, seed_sd = fit_test(specs, model_name, params, dev, test)
            pred_test = (prob_test >= 0.5).astype(int)
            pred_test_tuned = (prob_test >= threshold).astype(int)

            key = f"{set_name}__{model_name}"
            oof_cols[key], test_cols[key], test_cols[key + "__pred"] = prob_oof, prob_test, pred_test
            chosen[key] = {"features": specs, "params": params, "threshold": threshold, "cv_log": log}
            if set_name == "A":
                ref[model_name] = {"oof": {"prob": prob_oof, "pred": pred_oof},
                                   "test": {"prob": prob_test, "pred": pred_test}}

            base = {"set": set_name, "model": model_name, "n_features": n_feat,
                    "params": json.dumps(params), "threshold": round(threshold, 4)}
            for ev, y, prob, pred, pred_t in [("oof", y_oof, prob_oof, pred_oof, pred_oof_tuned),
                                              ("test", y_test, prob_test, pred_test, pred_test_tuned)]:
                row = {**base, "eval": ev, **score(y, prob, pred)}
                tuned = evaluate(y, pred_t, prob)
                row["mcc_tuned"], row["pred_up_tuned"] = tuned["mcc"], tuned["pred_up_rate"]
                if ev == "test":
                    row["seed_sd_auc"] = seed_sd
                if set_name != "A":
                    add_delta(row, y, ref.get(model_name, {}).get(ev), prob, pred)
                rows.append(row)
            r_oof, r_test = rows[-2], rows[-1]
            print(f"[ablation] {set_name:12s} {model_name:8s} {n_feat:3d} feats  "
                  f"OOF auc={r_oof['roc_auc']:.3f} mcc={r_oof['mcc']:+.3f} | "
                  f"TEST auc={r_test['roc_auc']:.3f} mcc={r_test['mcc']:+.3f}  "
                  f"({time.time() - t0:.0f}s)")

    write_outputs(rows, oof_dates, y_oof, oof_cols, test, test_cols, chosen)


def write_outputs(rows, oof_dates, y_oof, oof_cols, test, test_cols, chosen):
    RESULTS_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "ablation.csv", index=False)
    pd.DataFrame({"date": oof_dates, "target": y_oof, **oof_cols}).to_csv(PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"date": test["date"], "target": test["target"], **test_cols}).to_csv(
        PRED_DIR / "test.csv", index=False)
    (RESULTS_DIR / "ablation_params.json").write_text(json.dumps(chosen, indent=1, default=str))

    def fmt(r, k):
        return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"

    def fmt_d(r, k):
        if pd.isna(r.get(f"d_{k}_vs_A", np.nan)):
            return "–"
        return f"{r[f'd_{k}_vs_A']:+.3f} [{r[f'd_{k}_lo']:+.3f}, {r[f'd_{k}_hi']:+.3f}]"

    md = ["# Ablation results\n",
          "Protocol: expanding-window CV on train+val (OOF = out-of-fold days), "
          "params by OOF ROC-AUC, class-balanced models, UP if P(UP) >= 0.5 "
          "(MCC thr-tuned: threshold tuned for MCC walk-forward on OOF, secondary), final refit on train+val averaged over seeds, test evaluated once. "
          "Brackets: 95% block-bootstrap CI. Δ vs A: paired bootstrap, same model family.\n"]
    for ev in ("oof", "test"):
        sub = res[res["eval"] == ev]
        table = pd.DataFrame({
            "set": sub["set"], "model": sub["model"],
            "ROC-AUC": [fmt(r, "roc_auc") for _, r in sub.iterrows()],
            "MCC": [fmt(r, "mcc") for _, r in sub.iterrows()],
            "acc": sub["accuracy"].round(3), "macro-F1": sub["f1_macro"].round(3),
            "pred UP": sub["pred_up_rate"].round(2),
            "MCC thr-tuned": sub.get("mcc_tuned", pd.Series(np.nan, index=sub.index)).round(3),
            "ΔAUC vs A": [fmt_d(r, "roc_auc") for _, r in sub.iterrows()],
            "ΔMCC vs A": [fmt_d(r, "mcc") for _, r in sub.iterrows()],
        })
        n_days = len(y_oof) if ev == "oof" else len(test)
        md.append(f"\n## {'Out-of-fold (CV on dev)' if ev == 'oof' else 'Test'}: {n_days} days\n")
        md.append(table.to_markdown(index=False))
    (RESULTS_DIR / "ablation.md").write_text("\n".join(md) + "\n")
    print(f"\n[ablation] wrote {RESULTS_DIR / 'ablation.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(ABLATION_SETS))
    ap.add_argument("--models", nargs="+", default=ABLATION_MODELS)
    args = ap.parse_args()
    if "A" not in args.sets:
        args.sets = ["A"] + args.sets    # every row is compared with Model A
    main(args.sets, args.models)
