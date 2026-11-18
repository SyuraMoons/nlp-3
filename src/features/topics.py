"""Topic features (brief 1.b.ii): categorise headlines into themes, then use the
daily theme mix as features.

1. BERTopic on e5 embeddings ("topics_e5").
   BERTopic = embed -> reduce -> cluster -> describe clusters with class-based
   TF-IDF. To stay fast and deterministic on 172k training headlines we use
   PCA(50) instead of UMAP and spherical-ish KMeans(N_TOPICS) instead of HDBSCAN
   (every headline gets a topic; no outlier class). Fitted on training-period
   headlines only (trading_date <= TEXT_FIT_END), then applied to all headlines.
   Daily features: share of the day's headlines in each topic.
   Topic keywords are saved to results/topics_e5.csv for interpretation.

2. GDELT theme shares ("gdelt_themes"), the knowledge-based comparison: GDELT's
   own GKG theme tags per article (ECON_*, EPU_*, WB_* ...). The N_THEMES most
   frequent themes in the training period; daily share of headlines tagged with each.

Run: python -m src.features.topics
"""
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from src.config import CACHE_DIR, RESULTS_DIR, SEED, TEXT_FIT_END
from src.data.articles import load_articles
from src.features.daily import write_table
from src.features.nlp_features import INDONESIAN_STOPWORDS
from src.text.cache import load_embeddings

N_TOPICS = 30
N_THEMES = 30
EXTRA_STOP = {"ke", "dari", "ini", "itu", "jadi", "lagi", "usai", "soal", "bakal", "kata", "ada", "tak",
              "tidak", "jelang", "hingga", "masih", "bisa", "akan", "sudah", "ri", "rp", "persen", "begini",
              "berikut", "cek", "simak", "hari", "tahun", "baru", "naik", "turun"}


def bertopic_table(articles):
    X = np.asarray(load_embeddings("e5", n_expected=len(articles)), dtype=np.float32)
    fit = (articles["trading_date"] <= TEXT_FIT_END).to_numpy()
    docs = articles["title"].tolist()
    model = BERTopic(
        umap_model=PCA(n_components=50, random_state=SEED),
        hdbscan_model=KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=4),
        vectorizer_model=CountVectorizer(stop_words=sorted(INDONESIAN_STOPWORDS | EXTRA_STOP),
                                         ngram_range=(1, 2), min_df=5),
        calculate_probabilities=False, verbose=False)
    model.fit([d for d, f in zip(docs, fit) if f], X[fit])
    topics, _ = model.transform(docs, X)
    info = model.get_topic_info()
    info["keywords"] = [", ".join(w for w, _ in model.get_topic(t)[:8]) for t in info["Topic"]]
    info[["Topic", "Count", "keywords"]].to_csv(RESULTS_DIR / "topics_e5.csv", index=False)
    print(info[["Topic", "Count", "keywords"]].to_string(index=False))
    onehot = pd.get_dummies(pd.Series(topics), prefix="topic").astype(float)
    onehot = onehot.reindex(columns=[f"topic_{t}" for t in range(N_TOPICS)], fill_value=0.0)
    onehot["date"] = articles["trading_date"].to_numpy()
    return onehot.groupby("date").mean().reset_index(), topics


def gdelt_table(articles):
    themes = articles["themes"].fillna("").str.split(";")
    fit = (articles["trading_date"] <= TEXT_FIT_END).to_numpy()
    counts = pd.Series([t for ts, f in zip(themes, fit) if f for t in set(ts) if t]).value_counts()
    keep = counts.index[:N_THEMES].tolist()
    M = pd.DataFrame({f"gdelt_{k}": themes.apply(lambda ts, k=k: k in ts).astype(float) for k in keep})
    M["date"] = articles["trading_date"].to_numpy()
    print(f"[topics] GDELT themes kept: {keep}")
    return M.groupby("date").mean().reset_index()


def main():
    articles = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title", "themes"])
    t, topics = bertopic_table(articles)
    write_table("topics_e5", t["date"], t.drop(columns="date"))
    np.save(CACHE_DIR / "topic_assignments_e5.npy", np.asarray(topics, dtype=np.int16))
    t = gdelt_table(articles)
    write_table("gdelt_themes", t["date"], t.drop(columns="date"))


if __name__ == "__main__":
    main()
