# Task 3: Model Refinement & Multimodal Integration

NLP course project: *Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates* (team Nasi Ayam).

Task 3 adds transformer and embedding text features, fuses them with exchange-rate history (early and late fusion), tunes the best models, and runs the A/B/C ablation plus exploratory analysis.

- **Target:** the daily direction (UP/DOWN) of Bank Indonesia's **JISDOR USD/IDR** fixing.
- **Inputs:**
  - exchange-rate history up to *t−1*
  - 243,844 Indonesian financial headlines (Bisnis.com, Kontan) aligned to day *t*, published before the 10:00 WIB fixing
- **Builds on:** [nlp_assignment_1](https://github.com/SyuraMoons/nlp_assignment_1) (data) and [nlp_task_2](https://github.com/SyuraMoons/nlp_task_2) (pipeline and baselines).

📄 **Report:** [`report/Task3_Report.pdf`](report/Task3_Report.pdf) · 📊 **Results:** [`results/ablation.md`](results/ablation.md) · 👥 **Who did what:** [`TEAM_TASKS.md`](TEAM_TASKS.md)

## Main result

Across ~50 model variants, no text representation improves on the market-only Model A with a 95% confidence interval that excludes zero. This holds on both the 600 out-of-fold days and the 176 test days. The variants:

- static Word2Vec/FastText
- general IndoBERT
- domain-adapted IndoBERT
- multilingual-e5
- a BERT sentiment head
- temporal decay, anchor axes, entity channels, topics
- late fusion and Optuna tuning

Which representation looks best is not stable from one period to the next: out-of-fold and test AUC are uncorrelated across variants. Exploratory analysis shows the headlines *do* carry FX information. For example, a "rupiah weakens" embedding axis correlates with the same-day move (Spearman ρ = +0.115, p < 0.001), but the signal is too small and unstable to improve day-ahead direction.

## Quick start

Use Python 3.12 (torch, UMAP and HDBSCAN wheels). On Apple Silicon, encoders run on MPS.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate     # or: python3.12 -m venv .venv
uv pip install -r requirements.txt
mkdir -p data/lexicon && for f in positive negative; do curl -sSfL -o data/lexicon/$f.tsv https://raw.githubusercontent.com/fajri91/InSet/master/$f.tsv; done
```

Every daily feature table is committed in `data/features/`, so the ablation runs without any GPU work:

```bash
python -m src.run_ablation           # full ablation -> results/ablation.md (~20 min on CPU)
```

Rebuilding the text features from scratch writes the per-headline caches to `cache/`, which is not in git. Times are for an M4:

```bash
python -m src.text.encode --encoder e5          # ~14 min
python -m src.text.encode --encoder indobert    # ~7 min
python -m src.text.static                       # Word2Vec + FastText, ~1 min
python -m src.text.sentiment                    # ~8 min
python -m src.text.dapt                         # domain-adaptive pretraining, ~25 min
python -m src.text.encode --encoder indobert_dapt
for e in e5 indobert indobert_dapt fasttext word2vec; do python -m src.features.daily --encoder $e; done
python -m src.features.daily --encoder e5 --half-life 3
python -m src.features.daily --encoder e5 --recency-hours 12
python -m src.features.semantic --encoder e5    # anchor axes + novelty
python -m src.features.semantic --sentiment
python -m src.features.entities                 # entity channels (--ner-sample 3000 for the NER audit)
python -m src.features.topics                   # BERTopic + GDELT themes
python -m src.tune --trials 60                  # Optuna on the best Model C rows
python -m src.run_ablation
python scripts/build_notebooks.py && for nb in notebook/0[3-6]_*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
python scripts/make_report_tables.py && cd report && tectonic Task3_Report.tex
```

## Repository structure

```
data/
  raw/final_dataset.csv             Task 1 daily dataset
  train.csv val.csv test.csv        frozen Task 2 chronological split (817 / 175 / 176 days)
  articles/articles_<year>.parquet  Task 1 article-level data (one row per headline)
  features/<name>.parquet           daily text feature tables (one row per trading date)
src/
  config.py                         every design choice: CV protocol, ablation matrix, fusion rows
  data/                             make_splits.py, load.py, articles.py
  text/                             headline-level representations
    encode.py                         e5, IndoBERT, IndoBERT-DAPT (masked mean pooling, MPS)
    static.py                         Word2Vec / FastText trained on train headlines, SIF pooling
    sentiment.py                      IndoBERT sentiment head
    dapt.py                           domain-adaptive MLM pretraining (fine-tuning script)
    cache.py                          embedding cache contract
  features/                         headline vectors -> daily features
    market_features.py                11 exchange-rate history features (Model A)
    nlp_features.py                   Task 2 NLP features (Model B)
    daily.py                          daily mean, recency weighting, cross-day decay
    semantic.py                       anchor axes, novelty/dispersion, sentiment table
    entities.py                       entity channels + NER audit
    topics.py                         BERTopic topic shares, GDELT themes
    groups.py                         feature-group registry (PCA fitted per fold)
  models/
    classifiers.py                    LogReg, XGBoost, LightGBM (class-balanced)
    attention_mil.py                  gated attention over the day's headlines (text subnetwork)
    fusion.py                         late fusion: walk-forward logit stacking
    naive.py                          majority class, persistence
  evaluation/                       cv.py (expanding-window CV), stats.py (bootstrap), metrics.py
  analysis/                         ablation, embedding space, interpretability, error analysis
  run_ablation.py                   ablation runner (early fusion, attention, late fusion, tuned)
  tune.py                           Optuna tuning
  train.py                          Task 2 protocol (reference) -> results/task2/
notebook/
  01_eda, 02_results                Task 2
  03_ablation                       ablation matrix, Task 2 vs Task 3, tuning
  04_embedding_space                UMAP high- vs low-volatility days, topic lift
  05_interpretability               SHAP, largest prediction shifts, attention headlines
  06_error_analysis                 where text helped vs hurt, feature correlations
results/                            ablation.csv/.md, predictions/, tuning, DAPT log, topics, NER audit
report/                             Task3_Report.tex/.pdf, figures/, tables/; report/task2/ = Task 2 report
scripts/                            notebook builder, report table generator
```

## Evaluation protocol

The same protocol applies to every row:

1. **Development set.** dev = train + val (992 days).
2. **Expanding-window CV.** The last 600 dev days (2023-05-15 → 2025-11-24) form five consecutive 120-day blocks. Each block is predicted by a model trained on every earlier day. Anything learned from text is refitted inside the fold.
3. **Model selection.** Hyper-parameters are chosen by out-of-fold ROC-AUC. All models are class-balanced, and UP is predicted when P(UP) ≥ 0.5.
4. **Final test.** The chosen model is refitted on dev with 5 seeds and evaluated once on the 176 test days.
5. **Uncertainty.** 95% circular block-bootstrap CIs, plus a paired bootstrap of ΔAUC and ΔMCC against Model A.
6. **Text-model leakage rule.** Expensive unsupervised text models (Word2Vec/FastText, DAPT, BERTopic) are fitted once on headlines up to 2025-02-24, the end of the train split.
