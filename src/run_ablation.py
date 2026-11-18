"""Ablation runner: evaluate every feature set x model under the Task 3 protocol.

Protocol (src/config.py has the full rationale):
  1. dev = train + val. Expanding-window CV -> out-of-fold (OOF) P(UP) for every
     hyper-parameter setting. Feature groups are refitted inside every fold.
  2. Pick the setting with the best OOF ROC-AUC. Predictions use P(UP) >= 0.5
     (all models are class-balanced). Secondary: an MCC-tuned threshold
     (walk-forward on OOF; tuned on all OOF for test) reported as mcc_tuned.
  3. Refit on all of dev with each seed in SEEDS, average P(UP) on test,
     evaluate ONCE. 95% CIs from a block bootstrap; every row is also compared
     with Model A by a paired bootstrap.

Row types
  early fusion   ABLATION_SETS (A, B, C_*, T_*) x ABLATION_MODELS
  attention      ATTENTION_SETS: attention-pooling MIL over headlines (text only)
  late fusion    LATE_FUSION: logit stacking of a market row and a text row
  tuned          C_tuned: the Optuna-tuned configuration (src/tune.py), if present

Results are UPSERTED: rows/columns of the rows run now replace old ones, other
rows are kept, so a new feature table can be added without re-running everything.
Model A is always (re)computed because every row is compared with it.

Outputs (results/):
  ablation.csv / ablation.md     metrics, CIs, deltas vs A
  ablation_params.json           chosen params, threshold, per-fold OOF AUC per row
  predictions/oof.csv, test.csv  per-day P(UP) of every row
  predictions/attention_top_headlines.csv   top-5 attention headlines per test day
  late_fusion_weights.json       stacker weights per late-fusion row

Run with:
  python -m src.run_ablation                                  # everything
  python -m src.run_ablation --sets C_e5 C_anchors --models logreg --no-mil --no-late
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import (ABLATION_MODELS, ABLATION_SETS, ATTENTION_SETS, FEATURES_DIR, LATE_FUSION,
                        RESULTS_DIR, SEEDS)
from src.data.load import load_all
from src.evaluation.cv import expanding_folds, tune_threshold, walk_forward_predictions
from src.evaluation.metrics import evaluate
from src.evaluation.stats import bootstrap_ci, paired_bootstrap_delta
from src.features.groups import FeatureBuilder
from src.models.classifiers import MODELS, param_grid
from src.models.fusion import late_fusion_oof, late_fusion_test
from src.models.naive import MajorityClass, Persistence

PRED_DIR = RESULTS_DIR / "predictions"
TUNED_FILE = RESULTS_DIR / "tuned_config.json"


def load_dev_test():
    data = load_all()
    dev = pd.concat([data["train"], data["val"]], ignore_index=True)
    return dev, data["test"].reset_index(drop=True)


def missing_tables(specs):
    return [s.split(":")[1] for s in specs
            if s.startswith("table:") and not (FEATURES_DIR / f"{s.split(':')[1]}.parquet").exists()]


def cv_features(specs, dev, folds):
    """Fit the feature groups on each fold's training days; return [(X_tr, X_va)]."""
    out = []
    for tr, va in folds:
        fb = FeatureBuilder(specs).fit(dev.iloc[tr])
        out.append((fb.transform(dev.iloc[tr]), fb.transform(dev.iloc[va])))
    return out


def run_cv(model_name, fold_X, y, folds, grid=None):
    """Grid search by pooled OOF ROC-AUC. Returns (best_params, oof_prob, log)."""
    factory, default_grid = MODELS[model_name]
    y_oof = np.concatenate([y[va] for _, va in folds])
    log, best = [], None
    for params in param_grid(grid or default_grid):
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
    aucs = [evaluate(test["target"], (p >= 0.5).astype(int), p)["roc_auc"] for p in probs]
    return np.mean(probs, axis=0), float(np.std(aucs))


def score(y, prob, pred):
    m = evaluate(y, pred, prob)
    for k, (lo, hi) in bootstrap_ci(y, prob, pred).items():
        m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
    return m


def fold_aucs(y_oof, prob_oof, fold_sizes):
    out, pos = [], 0
    for s in fold_sizes:
        out.append(round(evaluate(y_oof[pos:pos + s], (prob_oof[pos:pos + s] >= .5).astype(int),
                                  prob_oof[pos:pos + s])["roc_auc"], 4))
        pos += s
    return out


class Recorder:
    """Scores a row on OOF and test, compares it with Model A, stores predictions."""

    def __init__(self, y_oof, y_test, fold_sizes):
        self.y_oof, self.y_test, self.fold_sizes = y_oof, y_test, fold_sizes
        self.rows, self.oof_cols, self.test_cols, self.params, self.ref = [], {}, {}, {}, {}

    def add(self, set_name, model_name, prob_oof, prob_test, n_feat=None, info=None, seed_sd=None):
        pred_oof = (prob_oof >= 0.5).astype(int)
        pred_test = (prob_test >= 0.5).astype(int)
        pred_oof_t, _ = walk_forward_predictions(self.y_oof, prob_oof, self.fold_sizes)
        thr, _ = tune_threshold(self.y_oof, prob_oof)
        key = f"{set_name}__{model_name}"
        self.oof_cols[key], self.test_cols[key] = prob_oof, prob_test
        self.params[key] = {**(info or {}), "threshold": thr,
                            "fold_auc": fold_aucs(self.y_oof, prob_oof, self.fold_sizes)}
        if set_name == "A":
            self.ref[model_name] = {"oof": (prob_oof, pred_oof), "test": (prob_test, pred_test)}
        ref = self.ref.get(model_name, self.ref.get("logreg"))   # non-standard models vs A-logreg
        base = {"set": set_name, "model": model_name, "n_features": n_feat,
                "params": json.dumps((info or {}).get("params", {}))}
        for ev, y, prob, pred, pred_t in [("oof", self.y_oof, prob_oof, pred_oof, pred_oof_t),
                                          ("test", self.y_test, prob_test, pred_test, (prob_test >= thr).astype(int))]:
            row = {**base, "eval": ev, **score(y, prob, pred)}
            tuned = evaluate(y, pred_t, prob)
            row["mcc_tuned"], row["pred_up_tuned"] = tuned["mcc"], tuned["pred_up_rate"]
            if ev == "test" and seed_sd is not None:
                row["seed_sd_auc"] = seed_sd
            if set_name != "A" and ref is not None:
                d = paired_bootstrap_delta(y, ref[ev][0], ref[ev][1], prob, pred)
                for k in ("roc_auc", "mcc"):
                    row[f"d_{k}_vs_A"] = d[k]["delta"]
                    row[f"d_{k}_lo"], row[f"d_{k}_hi"] = d[k]["lo"], d[k]["hi"]
                    row[f"p_{k}_gt_A"] = d[k]["p_gt0"]
            self.rows.append(row)
        r_oof, r_test = self.rows[-2], self.rows[-1]
        print(f"[ablation] {set_name:13s} {model_name:22s} OOF auc={r_oof['roc_auc']:.3f} "
              f"mcc={r_oof['mcc']:+.3f} | TEST auc={r_test['roc_auc']:.3f} mcc={r_test['mcc']:+.3f}", flush=True)


def naive_rows(dev, test, folds, rec):
    y_dev = dev["target"].to_numpy()
    for name, model in [("majority_class", MajorityClass()), ("persistence", Persistence())]:
        prob = np.concatenate([model.fit(dev.iloc[tr], y_dev[tr]).predict_proba(dev.iloc[va])[:, 1]
                               for tr, va in folds])
        model.fit(dev, y_dev)
        # persistence outputs hard 0/1 "probabilities"; majority outputs a constant
        rec.add(name, "-", prob, model.predict_proba(test)[:, 1])


def early_fusion_rows(set_name, models, dev, test, folds, rec, grid=None):
    specs = ABLATION_SETS[set_name] if isinstance(set_name, str) and set_name in ABLATION_SETS else set_name
    t0 = time.time()
    fold_X = cv_features(specs, dev, folds)
    y_dev = dev["target"].to_numpy()
    for model_name in models:
        params, prob_oof, log = run_cv(model_name, fold_X, y_dev, folds, grid)
        prob_test, seed_sd = fit_test(specs, model_name, params, dev, test)
        yield model_name, specs, params, prob_oof, prob_test, fold_X[0][0].shape[1], log, seed_sd
    print(f"[ablation]   ({time.time() - t0:.0f}s)")


def main(sets, models, run_mil, run_late, run_tuned):
    dev, test = load_dev_test()
    y_dev, y_test = dev["target"].to_numpy(), test["target"].to_numpy()
    folds = expanding_folds(len(dev))
    fold_sizes = [len(va) for _, va in folds]
    y_oof = np.concatenate([y_dev[va] for _, va in folds])
    oof_dates = np.concatenate([dev["date"].to_numpy()[va] for _, va in folds])
    print(f"[ablation] dev {len(dev)} days, {len(folds)} folds, OOF {len(y_oof)} days "
          f"({oof_dates[0]} -> {oof_dates[-1]}); test {len(test)} days")

    rec = Recorder(y_oof, y_test, fold_sizes)
    naive_rows(dev, test, folds, rec)
    for set_name in ["A"] + [s for s in sets if s != "A"]:
        miss = missing_tables(ABLATION_SETS[set_name])
        if miss:
            print(f"[ablation] skip {set_name}: missing tables {miss}")
            continue
        for model_name, specs, params, p_oof, p_test, n_feat, log, sd in early_fusion_rows(
                set_name, models, dev, test, folds, rec):
            rec.add(set_name, model_name, p_oof, p_test, n_feat,
                    {"features": specs, "params": params, "cv_log": log}, sd)

    if run_tuned and TUNED_FILE.exists():
        tuned = json.loads(TUNED_FILE.read_text())
        grid = {k: [v] for k, v in tuned["params"].items()}
        for model_name, specs, params, p_oof, p_test, n_feat, log, sd in early_fusion_rows(
                tuned["features"], [tuned["model"]], dev, test, folds, rec, grid):
            rec.add("C_tuned", model_name, p_oof, p_test, n_feat,
                    {"features": specs, "params": params, "optuna": tuned.get("study")}, sd)

    if run_mil:
        from src.models.attention_mil import run_mil as mil
        for name, encoder in ATTENTION_SETS.items():
            t0 = time.time()
            p_oof, p_test, top = mil(dev, test, folds, encoder)
            rec.add(name, "attention", p_oof, p_test, None, {"encoder": encoder})
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            top.to_csv(PRED_DIR / f"attention_top_headlines_{name}.csv", index=False)
            print(f"[ablation]   ({time.time() - t0:.0f}s)")

    if run_late:
        prev_oof, prev_test = read_predictions()
        weights = {}
        for name, (k_time, k_text) in LATE_FUSION.items():
            get = lambda k, new, old: new[k] if k in new else (old[k].to_numpy() if old is not None and k in old else None)
            parts = [get(k_time, rec.oof_cols, prev_oof), get(k_text, rec.oof_cols, prev_oof),
                     get(k_time, rec.test_cols, prev_test), get(k_text, rec.test_cols, prev_test)]
            if any(p is None for p in parts):
                print(f"[ablation] skip {name}: base rows {k_time}, {k_text} not computed yet")
                continue
            o_time, o_text, t_time, t_text = parts
            p_oof = late_fusion_oof(y_oof, o_time, o_text, fold_sizes)
            p_test, w = late_fusion_test(y_oof, o_time, o_text, t_time, t_text)
            weights[name] = {"time": k_time, "text": k_text, **w}
            rec.add(name, f"stack({k_text.split('__')[-1]})", p_oof, p_test, None, {"bases": [k_time, k_text], **w})
        old = json.loads((RESULTS_DIR / "late_fusion_weights.json").read_text()) \
            if (RESULTS_DIR / "late_fusion_weights.json").exists() else {}
        (RESULTS_DIR / "late_fusion_weights.json").write_text(json.dumps({**old, **weights}, indent=1))

    write_outputs(rec, oof_dates, test)


def read_predictions():
    try:
        return pd.read_csv(PRED_DIR / "oof.csv"), pd.read_csv(PRED_DIR / "test.csv")
    except FileNotFoundError:
        return None, None


def row_order():
    order = ["majority_class", "persistence"] + list(ABLATION_SETS) + ["C_tuned"] + \
        list(ATTENTION_SETS) + list(LATE_FUSION)
    return {s: i for i, s in enumerate(order)}


def write_outputs(rec, oof_dates, test):
    RESULTS_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    new = pd.DataFrame(rec.rows)
    keys = set(zip(new["set"], new["model"]))
    if (RESULTS_DIR / "ablation.csv").exists():
        old = pd.read_csv(RESULTS_DIR / "ablation.csv")
        old = old[[(s, m) not in keys for s, m in zip(old["set"], old["model"])]]
        new = pd.concat([old, new], ignore_index=True)
    order = row_order()
    new = new.sort_values(["eval", "set", "model"], key=lambda c: c.map(order).fillna(99) if c.name == "set" else c,
                          kind="stable")
    new.to_csv(RESULTS_DIR / "ablation.csv", index=False)

    prev_oof, prev_test = read_predictions()
    oof = pd.DataFrame({"date": oof_dates, "target": rec.y_oof, **rec.oof_cols})
    tst = pd.DataFrame({"date": test["date"], "target": test["target"], **rec.test_cols})
    if prev_oof is not None:
        oof = pd.concat([oof, prev_oof.drop(columns=[c for c in prev_oof if c in oof])], axis=1)
        tst = pd.concat([tst, prev_test.drop(columns=[c for c in prev_test if c in tst])], axis=1)
    oof.to_csv(PRED_DIR / "oof.csv", index=False)
    tst.to_csv(PRED_DIR / "test.csv", index=False)

    pfile = RESULTS_DIR / "ablation_params.json"
    params = json.loads(pfile.read_text()) if pfile.exists() else {}
    params.update(rec.params)
    pfile.write_text(json.dumps(params, indent=1, default=str))
    write_markdown(new, len(rec.y_oof), len(test))


def write_markdown(res, n_oof, n_test):
    def fmt(r, k):
        return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"

    def fmt_d(r, k):
        if pd.isna(r.get(f"d_{k}_vs_A", np.nan)):
            return "–"
        return f"{r[f'd_{k}_vs_A']:+.3f} [{r[f'd_{k}_lo']:+.3f}, {r[f'd_{k}_hi']:+.3f}]"

    md = ["# Ablation results\n",
          "Protocol: expanding-window CV on train+val (OOF = out-of-fold days), params by OOF ROC-AUC, "
          "class-balanced models, UP if P(UP) >= 0.5 (MCC thr-tuned: threshold tuned for MCC walk-forward "
          "on OOF, secondary), final refit on train+val averaged over 5 seeds, test evaluated once. "
          "Brackets: 95% block-bootstrap CI. Δ vs A: paired bootstrap against Model A with the same model "
          "family (attention / late-fusion rows: against A-logreg).\n"]
    for ev, n in (("oof", n_oof), ("test", n_test)):
        sub = res[res["eval"] == ev]
        table = pd.DataFrame({
            "set": sub["set"], "model": sub["model"],
            "ROC-AUC": [fmt(r, "roc_auc") for _, r in sub.iterrows()],
            "MCC": [fmt(r, "mcc") for _, r in sub.iterrows()],
            "acc": sub["accuracy"].round(3), "macro-F1": sub["f1_macro"].round(3),
            "pred UP": sub["pred_up_rate"].round(2),
            "MCC thr-tuned": sub["mcc_tuned"].round(3),
            "ΔAUC vs A": [fmt_d(r, "roc_auc") for _, r in sub.iterrows()],
            "ΔMCC vs A": [fmt_d(r, "mcc") for _, r in sub.iterrows()],
        })
        md.append(f"\n## {'Out-of-fold (CV on dev)' if ev == 'oof' else 'Test'}: {n} days\n")
        md.append(table.to_markdown(index=False))
    (RESULTS_DIR / "ablation.md").write_text("\n".join(md) + "\n")
    print(f"\n[ablation] wrote {RESULTS_DIR / 'ablation.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(ABLATION_SETS))
    ap.add_argument("--models", nargs="+", default=ABLATION_MODELS)
    ap.add_argument("--no-mil", action="store_true")
    ap.add_argument("--no-late", action="store_true")
    ap.add_argument("--no-tuned", action="store_true")
    a = ap.parse_args()
    main(a.sets, a.models, not a.no_mil, not a.no_late, not a.no_tuned)
