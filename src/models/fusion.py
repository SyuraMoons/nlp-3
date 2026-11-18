"""Multimodal fusion of X_time (market features) and X_text (text features).

Early fusion (no code needed here): an ablation set lists several feature groups
("market" + "news_volume" + "table:..."); FeatureBuilder concatenates them into one
vector per day and a single classifier (LogReg / XGBoost / LightGBM) learns from
it. Every Model C row is early fusion.

Late fusion (this module): two unimodal models are trained separately --
  * the market model  -> P_time(UP)   (Model A: market features only)
  * a text model      -> P_text(UP)   (text-only LogReg on daily embeddings, or the
                                       attention MIL subnetwork over headlines)
-- and combined at the logit level by a logistic-regression stacker:
      logit P(UP) = w_time * logit(P_time) + w_text * logit(P_text) + b
The stacker never sees the days it scores:
  * OOF evaluation: block k of the CV is combined with weights fitted on the OOF
    predictions of blocks < k (block 0: plain average of the two logits).
  * Test: weights fitted on all OOF predictions, applied to the test predictions
    of the base models refitted on all of dev.
Late fusion lets each modality use its own model class and keeps the market
signal intact when text is noisy (the stacker can shrink w_text towards 0).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _stacker(y, prob_time, prob_text):
    Z = np.column_stack([logit(prob_time), logit(prob_text)])
    return LogisticRegression(C=1.0, class_weight="balanced").fit(Z, y)


def _apply(stacker, prob_time, prob_text):
    return stacker.predict_proba(np.column_stack([logit(prob_time), logit(prob_text)]))[:, 1]


def late_fusion_oof(y_oof, oof_time, oof_text, fold_sizes):
    """Walk-forward fused OOF probabilities."""
    out = np.zeros(len(y_oof))
    pos = 0
    for size in fold_sizes:
        sl = slice(pos, pos + size)
        if pos == 0:
            out[sl] = sigmoid((logit(oof_time[sl]) + logit(oof_text[sl])) / 2)
        else:
            out[sl] = _apply(_stacker(y_oof[:pos], oof_time[:pos], oof_text[:pos]), oof_time[sl], oof_text[sl])
        pos += size
    return out


def late_fusion_test(y_oof, oof_time, oof_text, test_time, test_text):
    """Fused test probabilities and the stacker weights (w_time, w_text, bias)."""
    st = _stacker(y_oof, oof_time, oof_text)
    return _apply(st, test_time, test_text), {"w_time": float(st.coef_[0, 0]),
                                               "w_text": float(st.coef_[0, 1]),
                                               "bias": float(st.intercept_[0])}
