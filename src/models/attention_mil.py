"""Text subnetwork for late fusion: attention pooling over the day's headline stream.

Brief 1.b.v ("attention over headline streams") and 2.a.ii (late fusion with a
separate text subnetwork). Each trading day is a BAG of headlines (multiple-
instance learning): the label (UP/DOWN) belongs to the day, not to any single
headline. Gated attention pooling (Ilse et al., 2018, "Attention-based Deep
Multiple Instance Learning") learns which headlines matter:

    h_i   = ReLU(W_h x_i)                                   x_i: frozen e5 vector (768)
    a_i   = softmax_i( w^T [tanh(V x_i) * sigmoid(U x_i)] ) attention weight per headline
    z     = sum_i a_i h_i                                   day representation (32)
    logit = w_o^T z + b

Only ~50k parameters (the 278M-parameter encoder stays frozen), because ~800
labelled days cannot fine-tune a transformer end to end. Per day we keep the
MAX_HEADLINES headlines published closest to the fixing. Days without news are a
bag with one zero vector (the model learns a base rate for them).

The attention weights are the interpretability hook: for any day we can list the
headlines that drove the prediction (results/predictions/attention_top_headlines.csv).

Training: inputs standardised with train-day statistics; BCE with class balancing;
AdamW (lr 5e-4, weight decay 0.1); input dropout 0.5; 6 epochs; batch 32 days; CPU.
Heavy regularisation is deliberate: with 25 epochs / dropout 0.2 the model reached
train AUC 0.94-0.99 and validation AUC ~0.5 (memorising days). CV: same expanding folds as the
ablation runner; test: refit on all of dev, average over SEEDS.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import SEEDS
from src.data.articles import load_articles
from src.text.cache import load_embeddings

# torch's OpenMP runtime clashes with the one loaded by LightGBM/XGBoost/sklearn on
# macOS (multithreaded torch CPU ops segfault when both are loaded); the model is
# tiny, so a single thread is fine.
torch.set_num_threads(1)

MAX_HEADLINES = 128
HIDDEN = 32
EPOCHS = 6
BATCH = 32


class GatedAttentionMIL(nn.Module):
    def __init__(self, dim, hidden=HIDDEN, att=16, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.h = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.V, self.U, self.w = nn.Linear(dim, att), nn.Linear(dim, att), nn.Linear(att, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        """x: (B, N, dim), mask: (B, N) True for real headlines. Returns logits (B,), weights (B, N)."""
        x = self.drop(x)
        s = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=1)
        z = (a.unsqueeze(-1) * self.h(x)).sum(1)
        return self.out(z).squeeze(-1), a


class HeadlineBags:
    """Padded (n_days, MAX_HEADLINES, dim) tensor of headline vectors per row date."""

    def __init__(self, dates, encoder="e5"):
        arts = load_articles(columns=["article_id", "timestamp_wib", "trading_date", "title"])
        X = load_embeddings(encoder, n_expected=len(arts))
        # most recent first, so head(MAX_HEADLINES) keeps the headlines closest to the fixing
        arts = arts.assign(row=np.arange(len(arts))).sort_values("timestamp_wib", ascending=False)
        groups = arts.groupby("trading_date")
        self.dates = list(dates)
        n, dim = len(self.dates), X.shape[1]
        # stored as numpy float16 (230 MB); batches are converted to torch on the fly
        self.x = np.zeros((n, MAX_HEADLINES, dim), dtype=np.float16)
        self.mask = np.zeros((n, MAX_HEADLINES), dtype=bool)
        self.ids = [[] for _ in range(n)]
        for i, d in enumerate(self.dates):
            if d in groups.groups:
                g = groups.get_group(d).head(MAX_HEADLINES)
                rows = np.sort(g["row"].to_numpy())
                self.x[i, :len(rows)] = np.asarray(X[rows], dtype=np.float16)
                self.mask[i, :len(rows)] = True
                self.ids[i] = g.set_index("row").loc[rows, "title"].tolist()
            else:
                self.mask[i, 0] = True          # empty bag: one zero vector
        self.dim = dim

    def fit_scaler(self, idx):
        """Per-dimension mean/std over the headlines of the TRAINING days only.
        e5 vectors share a large common direction; centring exposes the differences."""
        H = self.x[idx][self.mask[idx]].astype(np.float32)
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-6

    def batch(self, idx):
        x = (self.x[idx].astype(np.float32) - self.mu) / self.sd
        x[~self.mask[idx]] = 0.0
        return torch.from_numpy(x), torch.from_numpy(self.mask[idx])


def train_mil(bags, idx, y, seed):
    bags.fit_scaler(idx)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedAttentionMIL(bags.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-1)
    pos = y[idx].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        for b in np.array_split(rng.permutation(idx), max(1, len(idx) // BATCH)):
            logit, _ = model(*bags.batch(b))
            loss = loss_fn(logit, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def predict_mil(model, bags, idx):
    logit, a = model(*bags.batch(idx))
    return torch.sigmoid(logit).numpy(), a.numpy()


def run_mil(dev, test, folds, encoder="e5", cv_seeds=SEEDS[:3]):
    """OOF P(UP) on the CV blocks and seed-averaged test P(UP), plus test attention."""
    df = pd.concat([dev, test], ignore_index=True)
    bags = HeadlineBags(df["date"].astype(str), encoder)
    y = df["target"].to_numpy()
    oof = []
    for tr, va in folds:
        oof.append(np.mean([predict_mil(train_mil(bags, tr, y, s), bags, va)[0] for s in cv_seeds], axis=0))
    dev_idx, test_idx = np.arange(len(dev)), np.arange(len(dev), len(df))
    probs, atts = [], []
    for s in SEEDS:
        p, a = predict_mil(train_mil(bags, dev_idx, y, s), bags, test_idx)
        probs.append(p); atts.append(a)
    att = np.mean(atts, axis=0)
    top = []
    for j, i in enumerate(test_idx):
        n = int(bags.mask[i].sum())
        order = np.argsort(-att[j, :n])[:5]
        for rank, k in enumerate(order):
            if k < len(bags.ids[i]):
                top.append({"date": df["date"].iloc[i], "rank": rank + 1, "weight": float(att[j, k]),
                            "n_headlines": n, "title": bags.ids[i][k]})
    return np.concatenate(oof), np.mean(probs, axis=0), pd.DataFrame(top)
