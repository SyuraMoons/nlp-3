# Task 3: who does what

| Person | Part | Files |
|---|---|---|
| **Muhammad Harfi** | Repo, evaluation protocol, ablation runner, early + late fusion, attention model, tuning | `src/config.py`, `src/evaluation/`, `src/run_ablation.py`, `src/models/`, `src/tune.py` |
| **Yusuf Imantaka** | Text encoders, domain-adaptive pretraining, daily text features (aggregation, anchors, entities, topics) | `src/text/`, `src/features/` |
| **Aufa Sultan** | Analysis notebooks (ablation, UMAP, SHAP, error analysis), fusion schematic, report | `notebook/`, `src/analysis/`, `report/` |

**Status: complete.** Everything below runs end to end, and the outputs are committed: `results/`, `notebook/03–06`, `report/Task3_Report.pdf`. Each member still submits the PDF individually on eLOK. Before the presentation, read the part you own and its section of the report.

## Setup (everyone)

```bash
git clone https://github.com/SyuraMoons/nlp-3.git && cd nlp-3
uv venv --python 3.12 .venv && source .venv/bin/activate      # or python3.12 -m venv .venv
uv pip install -r requirements.txt
mkdir -p data/lexicon && for f in positive negative; do curl -sSfL -o data/lexicon/$f.tsv https://raw.githubusercontent.com/fajri91/InSet/master/$f.tsv; done
python -m src.run_ablation --sets A B --models logreg --no-mil --no-late --no-tuned   # ~30 s sanity check
```

The daily feature tables are in git, so the ablation and most notebooks run without a GPU. Notebook 04 (UMAP) and parts of 05 need the per-headline cache. Rebuild it with `python -m src.text.encode --encoder e5` (~14 min on an M4).

**Rules that kept the ablation honest:**

- **Fit text models on training data only.** Anything learned from text is fitted on training data only. Cheap fits (PCA, scalers, TF-IDF) are refitted inside every CV fold. Expensive fits (DAPT, BERTopic, FastText/Word2Vec) are fitted once on headlines up to 2025-02-24.
- **Never use test days for decisions.** Settings were chosen on out-of-fold results, never on test results.

---

## Harfi: protocol, fusion, tuning ✅

1. ✅ Repo scaffold, article data, Python 3.12 environment.
2. ✅ Protocol:
   - expanding-window CV (`src/evaluation/cv.py`)
   - block bootstrap and paired Δ vs A (`src/evaluation/stats.py`)
   - class-balanced LogReg / XGBoost / LightGBM
3. ✅ Early fusion: the feature-group registry concatenates market + text features (`src/features/groups.py`).
4. ✅ Late fusion: walk-forward logit stacking (`src/models/fusion.py`). Text subnetwork: gated attention over the day's headlines (`src/models/attention_mil.py`).
5. ✅ Ablation runner with text-only, attention, late-fusion and tuned rows (`src/run_ablation.py`).
6. ✅ Optuna tuning around the two best OOF rows (`src/tune.py`, `results/tuned_config.json`).

## Yusuf: text representations ✅

1. ✅ Static baselines: Word2Vec and FastText trained on train headlines, SIF pooling (`src/text/static.py`).
2. ✅ Contextual encoders: IndoBERT and multilingual-e5, masked mean pooling (`src/text/encode.py`).
3. ✅ Domain-adaptive pretraining of IndoBERT (`src/text/dapt.py`). Held-out perplexity fell from 66.4 to 17.0.
4. ✅ Sentiment head (`src/text/sentiment.py`).
5. ✅ Temporal aggregation: daily mean, recency weighting, cross-day decay (`src/features/daily.py`).
6. ✅ Anchor axes, novelty and dispersion (`src/features/semantic.py`).
7. ✅ Entity channels with an NER audit (`src/features/entities.py`, `results/ner_entities.csv`).
8. ✅ BERTopic topics and GDELT themes (`src/features/topics.py`, `results/topics_e5.csv`).

## Aufa: analysis + report ✅

1. ✅ `notebook/03_ablation.ipynb`: ablation matrix, Task 2 vs Task 3, forest plot, fold stability.
2. ✅ `notebook/04_embedding_space.ipynb`: UMAP of high- vs low-volatility days, topic lift, separability.
3. ✅ `notebook/05_interpretability.ipynb`: SHAP by feature group, largest prediction shifts, attention headlines.
4. ✅ `notebook/06_error_analysis.ipynb`: where text helped vs hurt, feature correlations.
5. ✅ Fusion schematic (TikZ, in the report) and `report/Task3_Report.pdf` (5 pages).

## Open items for Task 4

- Formal hypothesis tests on the OOF predictions: Diebold–Mariano and permutation tests.
- Intraday-aligned text: only headlines from the morning before the fixing.
