"""Hyper-parameter tuning of the best-performing Model C architectures (brief 3.a).

Starting point: the TOP_K Model C rows of results/ablation.csv by OOF ROC-AUC
(run src.run_ablation first). Optuna (TPE sampler) then searches jointly over
  * which of those feature sets to start from,
  * the PCA size of every "table:...:pca<k>" group (4 / 8 / 16 / 32),
  * optional extra feature groups (anchors, entity channels, topics, sentiment),
  * the classifier family (LogReg / LightGBM) and its hyper-parameters,
maximising pooled OOF ROC-AUC on the SAME expanding-window folds as the ablation
(features refitted per fold, as always).

Selection bias: the tuned configuration's OOF score is optimistic because it was
chosen on those very days. Its TEST score is the honest number -- run_ablation
evaluates it once as row "C_tuned" from results/tuned_config.json.

Run: python -m src.tune --trials 60      then: python -m src.run_ablation --sets A --no-mil --no-late
"""
import argparse
import json

import numpy as np
import optuna
import pandas as pd

from src.config import ABLATION_SETS, RESULTS_DIR, SEED
from src.evaluation.cv import expanding_folds
from src.evaluation.metrics import evaluate
from src.models.classifiers import MODELS
from src.run_ablation import TUNED_FILE, cv_features, load_dev_test, missing_tables

TOP_K = 2
EXTRAS = {
    "anchors": ["table:e5_anchors", "table:e5_dynamics"],
    "entities": ["table:entity_channels"],
    "topics": ["table:topics_e5"],
    "sentiment": ["table:sentiment"],
}


def top_sets(k=TOP_K):
    res = pd.read_csv(RESULTS_DIR / "ablation.csv")
    c = res[(res["eval"] == "oof") & res["set"].str.startswith("C_") & (res["set"] != "C_tuned")]
    best = c.groupby("set")["roc_auc"].max().sort_values(ascending=False)
    return best.index[:k].tolist(), best


def with_pca(specs, k):
    return [f"{':'.join(s.split(':')[:2])}:pca{k}" if s.startswith("table:") and ":pca" in s else s for s in specs]


def main(trials):
    dev, test = load_dev_test()
    y = dev["target"].to_numpy()
    folds = expanding_folds(len(dev))
    y_oof = np.concatenate([y[va] for _, va in folds])
    bases, ranking = top_sets()
    print(f"[tune] starting sets (top {TOP_K} by OOF AUC): {bases}\n{ranking.round(3).to_string()}")
    cache = {}

    def features(specs):
        key = tuple(specs)
        if key not in cache:
            cache[key] = cv_features(list(specs), dev, folds)
        return cache[key]

    def objective(trial):
        base = trial.suggest_categorical("base", bases)
        specs = ABLATION_SETS[base]
        if any(":pca" in s for s in specs):
            specs = with_pca(specs, trial.suggest_categorical("pca", [4, 8, 16, 32]))
        for name, groups in EXTRAS.items():
            if any(g in specs for g in groups) or missing_tables(groups):
                continue
            if trial.suggest_categorical(f"add_{name}", [False, True]):
                specs = specs + groups
        family = trial.suggest_categorical("model", ["logreg", "lightgbm"])
        if family == "logreg":
            params = {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
        else:
            params = {"num_leaves": trial.suggest_int("num_leaves", 2, 16),
                      "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
                      "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True)}
        factory, _ = MODELS[family]
        fold_X = features(specs)
        prob = np.concatenate([factory(**params).fit(X_tr, y[tr]).predict_proba(X_va)[:, 1]
                               for (tr, _), (X_tr, X_va) in zip(folds, fold_X)])
        trial.set_user_attr("features", specs)
        trial.set_user_attr("params", params)
        return evaluate(y_oof, (prob >= 0.5).astype(int), prob)["roc_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_trial
    trials_df = study.trials_dataframe()
    trials_df.to_csv(RESULTS_DIR / "tuning_trials.csv", index=False)
    out = {"features": best.user_attrs["features"], "model": best.params["model"],
           "params": best.user_attrs["params"],
           "study": {"n_trials": trials, "best_oof_auc": best.value, "starting_sets": bases,
                     "baseline_oof_auc": {s: float(ranking[s]) for s in bases},
                     "best_trial_params": best.params}}
    TUNED_FILE.write_text(json.dumps(out, indent=1))
    print(f"[tune] best OOF AUC {best.value:.4f} (trial {best.number}): {best.params}")
    print(f"[tune] features: {out['features']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    main(ap.parse_args().trials)
