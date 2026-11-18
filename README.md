# Task 3: Model Refinement & Multimodal Integration

NLP course project: *Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates.*
Task 3 adds transformer/embedding text features, fuses them with exchange-rate history in one model, and runs the A/B/C ablation.

- Predict the daily **direction** (UP/DOWN) of Bank Indonesia's **JISDOR USD/IDR** fixing.
- Inputs: exchange-rate history up to *t−1* plus Indonesian financial news headlines (Bisnis.com, Kontan) aligned to day *t*.
- Builds on [nlp_assignment_1](https://github.com/SyuraMoons/nlp_assignment_1) (data) and [nlp_task_2](https://github.com/SyuraMoons/nlp_task_2) (pipeline + baselines). Task 2 stays frozen.

👉 **Team members: start with [`TEAM_TASKS.md`](TEAM_TASKS.md).**

## Quick start

Python 3.12 is recommended (torch, UMAP and HDBSCAN wheels).

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate     # or: python3.12 -m venv .venv
uv pip install -r requirements.txt                            # or: pip install -r requirements.txt
mkdir -p data/lexicon && for f in positive negative; do curl -sSfL -o data/lexicon/$f.tsv https://raw.githubusercontent.com/fajri91/InSet/master/$f.tsv; done
python -m src.run_ablation          # Model A + B under the Task 3 protocol -> results/ablation.md (~1 min)
```

To build a Model C text table (about 5 minutes on an M4 with MPS for e5):

```bash
python -m src.text.encode --encoder e5      # every headline -> cache/emb/e5.npy (not in git)
python -m src.features.daily --encoder e5   # -> data/features/e5_mean.parquet (committed)
# then enable "C_e5" in src/config.py ABLATION_SETS and re-run src.run_ablation
```

## Repository structure

```
data/
  raw/final_dataset.csv          Task 1 daily dataset
  train.csv val.csv test.csv     frozen Task 2 chronological split (817 / 175 / 176 days)
  articles/articles_<year>.parquet  Task 1 article-level data: 243,844 headlines with timestamp,
                                 trading_date, source, GDELT themes (+ 2024 bodies)
  features/<name>.parquet        daily text feature tables (one row per trading date)
  lexicon/                       InSet lexicon (gitignored, download above)
cache/                           per-headline embeddings + fine-tuned models (gitignored)
src/
  config.py                      every design choice, incl. CV protocol and ABLATION_SETS
  data/                          make_splits.py, load.py, articles.py (article loader)
  features/
    market_features.py           11 exchange-rate history features
    nlp_features.py              Task 2 NLP features (Model B)
    groups.py                    feature-group registry + daily-table contract
    daily.py                     headline vectors -> daily tables (mean, decay)
  text/
    encode.py                    headline encoders (e5, IndoBERT, ...) -> cache/emb/
    cache.py                     embedding cache contract
  models/
    classifiers.py               LogReg, XGBoost, LightGBM (class-balanced) + grids
    fusion.py                    late fusion (TODO)
    naive.py                     majority class, persistence
  evaluation/
    cv.py                        expanding-window CV, threshold tuning
    stats.py                     block bootstrap CIs, paired bootstrap vs Model A
    metrics.py                   accuracy, macro-F1, MCC, ROC-AUC
  run_ablation.py                Task 3 ablation runner
  train.py                       Task 2 protocol (reference only) -> results/task2/
notebook/                        01_eda, 02_results (Task 2); Task 3 notebooks go here
results/
  ablation.md / ablation.csv     ablation table with CIs and deltas vs Model A
  predictions/                   per-day OOF and test probabilities for every model
  task2/                         original Task 2 results
report/                          Task 3 report; report/task2/ holds the Task 2 report
```

## Evaluation protocol (Task 3)

Task 2 tuned on one 175-day validation split. Every trained model there had validation ROC-AUC < 0.5, so "best on validation" was picking noise. Task 3 applies one protocol identically to every ablation row:

1. **dev = train + val** (992 days). **Expanding-window CV**: the last 600 dev days (2023-05-15 → 2025-11-24) are 5 consecutive blocks of 120 days. Each block is predicted by a model trained on all earlier days. Anything learned from text (TF-IDF, PCA, ...) is refitted inside each fold.
2. Hyper-parameters are chosen by **out-of-fold (OOF) ROC-AUC**.
3. All models are **class-balanced**, and UP is predicted when P(UP) ≥ 0.5. An MCC-tuned threshold is reported as a secondary column.
4. **Test once**: refit on all of dev with 5 seeds, average the probabilities, then score the 176 test days.
5. **Uncertainty**: 95% CIs from a circular block bootstrap (5-day blocks). Every row is compared with Model A by a **paired bootstrap** (ΔAUC, ΔMCC).

Two evaluations come out: OOF (600 days, more statistical power) and test (176 days, fully held out).

## Ablation

| Row | Features |
|---|---|
| **A** | market only (11) |
| **B** | market + Task 2 NLP (volume, GDELT tone, InSet lexicon, TF-IDF/SVD: 27) |
| **C-*** | market + news volume + one advanced text representation (static embeddings, e5, IndoBERT, domain-adapted IndoBERT, transformer sentiment, anchors, entity channels, ...) |

Current results are in [`results/ablation.md`](results/ablation.md).
