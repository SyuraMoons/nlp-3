"""Entity-level filtering (brief 1.b.iii): split each day's headlines into channels
by the actors they mention, then summarise each channel separately.

Why: a day's average over ~230 headlines is dominated by domestic corporate news.
FX-relevant information is concentrated in headlines about a few actors -- the
Fed and US policy (USD side), Bank Indonesia and the rupiah (IDR side),
geopolitical flashpoints, and commodities. Averaging within a channel keeps that
signal from being diluted.

Channels are defined by a gazetteer (regular expressions below). The gazetteer was
built and checked with a transformer NER model (cahya/bert-base-indonesian-NER):
`--ner-sample N` tags N random training-period headlines and writes the most
frequent locations/organisations and the share of them the gazetteer covers to
results/ner_entities.csv (see the report). Running NER on all 244k headlines is
unnecessary for channel assignment and ~10x slower than the regexes.

Per channel and day (table "entity_channels"):
    <ch>_share     share of the day's headlines in the channel
    <ch>_sent      mean P(pos) - P(neg) of channel headlines (transformer sentiment)
    <ch>_rupiah    mean e5 "rupiah weakens vs strengthens" axis score of channel headlines
    <ch>_dollar    mean e5 "dollar strengthens vs weakens" axis score
Days where a channel is silent get 0 (neutral).

Run:
    python -m src.features.entities                  # channel table (needs sentiment + e5 caches)
    python -m src.features.entities --ner-sample 3000
"""
import argparse
import re

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEED, TEXT_FIT_END
from src.data.articles import load_articles
from src.features.daily import write_table
from src.features.semantic import ANCHOR_AXES, anchor_scores
from src.text.cache import load_embeddings

# Case-insensitive unless noted. \b = word boundary.
CHANNELS = {
    "usd": [r"\bthe fed\b", r"\bfed\b", r"\bfomc\b", r"\bpowell\b", r"\bamerika serikat\b", r"\bwall street\b",
            r"\btrump\b", r"\bbiden\b", r"\btreasury\b", r"\bdolar as\b", r"\bgedung putih\b", r"\bwashington\b",
            r"\bus\$", r"(?-i:\bAS\b)"],
    "idr": [r"\bbank indonesia\b", r"\bbi rate\b", r"\bbi-rate\b", r"(?-i:\bBI\b)", r"\brupiah\b", r"\bjisdor\b",
            r"\bkemenkeu\b", r"\bsri mulyani\b", r"\bpurbaya\b", r"\bsbn\b", r"\bcadangan devisa\b", r"\bperry warjiyo\b",
            r"\bojk\b", r"\bapbn\b"],
    "geo": [r"\brusia\b", r"\bukraina\b", r"\bputin\b", r"\bzelensky", r"\bisrael\b", r"\biran\b", r"\bgaza\b",
            r"\bpalestina\b", r"\bhamas\b", r"\bhizbullah\b", r"\bhouthi\b", r"\bperang\b", r"\bsanksi\b",
            r"\bkonflik\b", r"\bmiliter\b", r"\bnato\b", r"\brudal\b", r"\bserangan\b", r"\btaiwan\b",
            r"\bkorea utara\b", r"\bgeopolitik"],
    "china": [r"\bchina\b", r"\btiongkok\b", r"\bbeijing\b", r"\bxi jinping\b", r"\byuan\b"],
    "commodity": [r"\bminyak\b", r"\bopec", r"\bbrent\b", r"\bwti\b", r"\bemas\b", r"\bbatu bara\b", r"\bbatubara\b",
                  r"\bcpo\b", r"\bsawit\b", r"\bnikel\b", r"\bkomoditas\b", r"\bgas alam\b"],
}
PATTERNS = {ch: re.compile("|".join(p), re.IGNORECASE) for ch, p in CHANNELS.items()}


def channel_masks(titles):
    return {ch: titles.str.contains(p).to_numpy() for ch, p in PATTERNS.items()}


def channel_table(articles):
    titles = articles["title"]
    masks = channel_masks(titles)
    P = np.asarray(load_embeddings("sentiment", n_expected=len(articles)), dtype=np.float32)
    sent = P[:, 0] - P[:, 2]
    X = load_embeddings("e5", n_expected=len(articles))
    score, _ = anchor_scores(X, "e5")
    axis = {a: j for j, (a, _, _) in enumerate(ANCHOR_AXES)}
    dates = articles["trading_date"].to_numpy()
    n_day = pd.Series(1, index=dates).groupby(level=0).sum()
    out = pd.DataFrame(index=n_day.index)
    for ch, m in masks.items():
        d = pd.DataFrame({"date": dates[m], "sent": sent[m], "rupiah": score[m, axis["rupiah"]],
                          "dollar": score[m, axis["dollar"]]}).groupby("date")
        out[f"{ch}_share"] = (d.size() / n_day).reindex(out.index).fillna(0.0)
        for col in ("sent", "rupiah", "dollar"):
            out[f"{ch}_{col}"] = d[col].mean().reindex(out.index).fillna(0.0)
        print(f"[entities] {ch:9s}: {m.mean():.1%} of headlines")
    return out.reset_index(names="date")


def ner_audit(articles, n):
    """Tag a sample of training headlines with a transformer NER model and report
    the most frequent locations / organisations and gazetteer coverage."""
    from transformers import pipeline
    from src.text.encode import device
    ner = pipeline("token-classification", model="cahya/bert-base-indonesian-NER",
                   aggregation_strategy="simple", device=device())
    sample = articles[articles["trading_date"] <= TEXT_FIT_END].sample(n, random_state=SEED)
    rows = []
    for title, ents in zip(sample["title"], ner(sample["title"].tolist(), batch_size=64)):
        for e in ents:
            rows.append({"entity": e["word"].lower().strip(), "type": e["entity_group"], "score": e["score"]})
    ents = pd.DataFrame(rows)
    top = (ents[ents["type"].isin(["GPE", "LOC", "ORG", "NOR", "PER"])]
           .groupby(["entity", "type"]).size().rename("count").reset_index()
           .sort_values("count", ascending=False).head(60))
    any_pat = re.compile("|".join(p.pattern for p in PATTERNS.values()), re.IGNORECASE)
    # NER output is lower-cased; the case-sensitive acronyms (AS, BI) are matched upper-case
    hit = lambda e, p: bool(p.search(e) or p.search(e.upper()))
    top["in_gazetteer"] = top["entity"].apply(lambda e: hit(e, any_pat))
    top["channel"] = top["entity"].apply(lambda e: next((c for c, p in PATTERNS.items() if hit(e, p)), ""))
    top.to_csv(RESULTS_DIR / "ner_entities.csv", index=False)
    print(top.head(40).to_string(index=False))


def main(ner_sample):
    articles = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
    if ner_sample:
        ner_audit(articles, ner_sample)
        return
    t = channel_table(articles)
    write_table("entity_channels", t["date"], t.drop(columns="date"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner-sample", type=int, default=0)
    a = ap.parse_args()
    main(a.ner_sample)
