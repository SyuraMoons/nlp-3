"""Multimodal fusion of X_time (market features) and X_text (daily text features).

OWNER: Harfi.

Early fusion (already supported, no code needed here): concatenate the feature
groups of an ablation set ("market" + "news_volume" + "table:...") and feed one
classifier (LogReg / XGBoost / LightGBM). That is how every Model C row runs.

Late fusion (TODO): one model per modality, combined at the logit level:
    logit(p) = a * logit(p_market) + (1 - a) * logit(p_text) + b
  * p_market : the Model A classifier (market features only)
  * p_text   : a text-only classifier (e.g. LogReg on the PCA'd daily embedding,
               or the attention-pooling model over the day's headlines)
  * a, b     : fitted on OOF predictions of the training folds only (stacking),
               so the combiner never sees the days it scores.
Implement as a class with the same fit(X, y) / predict_proba(X) interface as
src.models.classifiers so the ablation runner can treat it like any model.

Stretch (TODO): attention pooling (multiple-instance learning) over the day's
headline embeddings in PyTorch; its attention weights feed the interpretability
analysis (which headlines drove the prediction).
"""
