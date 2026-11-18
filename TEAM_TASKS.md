# Task 3: who does what

| Person | Part | Files |
|---|---|---|
| **Muhammad Harfi** | Repo, evaluation protocol, ablation runner, fusion (early + late), tuning | `src/config.py`, `src/evaluation/`, `src/run_ablation.py`, `src/models/`, `src/tune.py` |
| **Yusuf Imantaka** | Text encoders, domain-adaptive pretraining, daily text features (aggregation, anchors, entities, topics) | `src/text/`, `src/features/daily.py` (+ new feature scripts) |
| **Aufa Sultan** | Analysis notebooks (ablation, UMAP, SHAP, error analysis), fusion schematic, report | `notebook/`, `report/` |

Status: **Phase 0 done.** The runner works end to end, Models A and B have been re-run under the new protocol (`results/ablation.md`), and the e5 encoder path is tested.

**How the parts connect:** the contract is the **daily table**.

- Yusuf's scripts write `data/features/<name>.parquet`: a `date` column plus numeric columns, one row per trading date, built only from news aligned to that date or earlier.
- Harfi's runner uses a table once `"table:<name>"` (or `"table:<name>:pca16"`) is added to `ABLATION_SETS` in `src/config.py`.
- Aufa's notebooks read `results/ablation.csv` and `results/predictions/*.csv`.

Nobody waits on anybody: each person can test against the e5 table that already exists.

## Setup (everyone)

```bash
git clone https://github.com/SyuraMoons/nlp-3.git && cd nlp-3
uv venv --python 3.12 .venv && source .venv/bin/activate      # or python3.12 -m venv .venv
uv pip install -r requirements.txt
mkdir -p data/lexicon && for f in positive negative; do curl -sSfL -o data/lexicon/$f.tsv https://raw.githubusercontent.com/fajri91/InSet/master/$f.tsv; done
python -m src.run_ablation --sets A B --models logreg     # ~20 s sanity check
```

Run everything from the repo root as modules (`python -m src....`). Work on your own branch and open a PR.

**Rules that keep the ablation honest**

- **Fit text models on training data only.** Anything learned from text (vocabularies, PCA, Word2Vec/FastText, BERTopic, DAPT) is fitted on training data only.
  - Cheap fits are refitted per CV fold automatically when they go through `src/features/groups.py`.
  - Expensive fits (DAPT, BERTopic, FastText) are fitted once on headlines with `trading_date <= 2025-02-24` (the end of the Task 2 train split). Say so in the report.
- **Never use test days for decisions.** Choose settings (half-life, PCA size, anchors) by OOF results, never by test results.
- **Keep big files out of git.** Embedding caches (`cache/`) stay out of git; daily tables (`data/features/`) go in.

---

## Harfi: protocol, fusion, tuning

1. ✅ Repo scaffold, Python 3.12 env, article data in `data/articles/`.
2. ✅ Protocol: expanding-window CV (`src/evaluation/cv.py`), block bootstrap + paired Δ vs A (`src/evaluation/stats.py`), class-balanced LogReg / XGBoost / LightGBM, runner `src/run_ablation.py`.
3. ✅ A and B re-run under the new protocol. The original Task 2 numbers are kept in `results/task2/`.
4. **Late fusion** in `src/models/fusion.py`: market model and text-only model combined at the logit level. The weight is fitted on inner OOF predictions, never on the days being scored. It must have the same `fit` / `predict_proba` interface so it plugs into `MODELS`.
5. **Ablation rows**: enable each C variant in `ABLATION_SETS` as Yusuf's tables land. Add the combined rows: best C + anchors + entities + decay.
6. **Tuning**: `src/tune.py`, Optuna with about 50 trials on the best 1–2 configurations, scored by OOF AUC with the same folds. Report how many trials were run.
7. Stretch: attention pooling over each day's headline embeddings (PyTorch, MPS). Its attention weights go to Aufa for the "which headlines mattered" analysis.

## Yusuf: text representations

Each step produces either an embedding cache (`cache/emb/<name>.npy`, rows aligned with `src.data.articles.load_articles()`) or a daily table (`data/features/<name>.parquet`).

1. ✅ `e5` (`intfloat/multilingual-e5-base`): `python -m src.text.encode --encoder e5`, then `python -m src.features.daily --encoder e5`.
2. **IndoBERT** (`indobenchmark/indobert-base-p1`, mean pooling): already wired. Run `--encoder indobert`.
3. **Static baseline**: `src/text/static.py`. Train gensim FastText (and optionally Word2Vec) on train-split headlines, use SIF-weighted pooling per headline, and save with `save_embeddings("fasttext", X, meta)`.
4. **Domain-adaptive pretraining (the required fine-tuning script)**: `src/text/dapt.py`. Continue MLM training of IndoBERT on train-split headlines (1–2 epochs, max_length 64, MPS) and save to `cache/models/indobert_dapt`. Then run `--encoder indobert_dapt`.
5. **Sentiment head**: `src/text/sentiment.py`. `mdhugol/indonesia-bert-sentiment-classification` gives per-headline P(pos), P(neu), P(neg). Write a daily table `sentiment` with the mean of P(pos) − P(neg) and the share of positive and negative headlines.
6. **Aggregation variants** in `src/features/daily.py`:
   - exponential decay is already there (`--half-life 1|2|3|5`)
   - add within-day recency weighting
   - add news novelty (distance from the trailing 20-day mean) and dispersion
7. **Anchor projections**: about 12 hand-written anchor sentences. Ideas:
   - "Rupiah melemah terhadap dolar AS"
   - "The Fed naikkan suku bunga"
   - "Bank Indonesia intervensi pasar"
   - "Konflik Timur Tengah memanas"
   - "Harga minyak melonjak"

   Take the cosine similarity of each headline to each anchor, then the daily mean. This gives table `<encoder>_anchors`.
8. **Entity channels** (keyword gazetteer; `cahya/bert-base-indonesian-NER` is optional for finding more terms):

   | Channel | Terms |
   |---|---|
   | USD side | Fed, FOMC, AS, Trump, tarif |
   | IDR side | Bank Indonesia, BI Rate, rupiah, Kemenkeu |
   | Geopolitics | Rusia, Ukraina, Israel, Iran, China, perang, sanksi |
   | Commodities | minyak, OPEC, emas, batu bara |

   Per channel: volume share, mean sentiment, and anchor projection. This gives table `entity_channels`.
9. Stretch: BERTopic on e5 embeddings of train headlines, giving daily topic shares, compared against GDELT theme counts. Also title vs body embeddings (2024 only).

## Aufa: analysis + report

1. `notebook/03_ablation.ipynb`: the ablation matrix from `results/ablation.csv`, Task 2 vs Task 3 numbers side by side, and forest plots of ΔAUC and ΔMCC vs A with CIs.
2. `notebook/04_embeddings.ipynb`: UMAP of headline embeddings (`cache/emb/e5.npy`) on high- vs low-volatility days (top vs bottom quintile of |return|).
3. `notebook/05_interpretability.ipynb`:
   - SHAP on the best Model C, grouped by feature group (market / anchors / entity channels / PCA)
   - top headlines on the biggest-move days
4. `notebook/06_errors.ipynb`:
   - days where C is right and A is wrong, and the reverse (`results/predictions/`)
   - performance by news volume and by size of move
5. Fusion schematic (how the text and numeric vectors merge, early and late) for the report.
6. Report, **5 pages max**, sections (a) feature extraction and its theory, (b) architecture with the schematic, (c) ablation matrix, (d) exploratory analysis.
