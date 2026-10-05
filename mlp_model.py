# ===========================================================================
# 10-fold PyTorch MLP on the same folds and feature view as view_model.py
# (reuses its fold_data, so --win / --orig / --knn work the same way).
# Preprocessing and architecture follow the notebook's MLP; --emb adds
# periodic embeddings of income and commute (a RealMLP-style numeric
# embedding) for a different view of the two continuous axes.
#   usage: .venv/bin/python mlp_model.py NAME [--win] [--emb] [--seeds 42]
# ===========================================================================
import os, sys, time, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--emb", action="store_true")
ap.add_argument("--seeds", default="42")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--wide", action="store_true")         # 512-256-128 body
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--emb_all", action="store_true")      # periodic embedding on every numeric input
margs = ap.parse_known_args()[0]

import view_model as vm
from quick_checks import cat_cols, INC, COM, y

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COUNT_PREFIXES = ("Freq_", "OrigFreq_", "Lift_", "OrigDist_")

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def mlp_matrices(X_fit, X_val, X_tst):
    cats = [c for c in cat_cols if c in X_fit.columns]
    num = [c for c in X_fit.columns if c not in cats]
    def numeric(X):
        M = X[num].astype(np.float64).copy()
        for c in num:
            if c.startswith(COUNT_PREFIXES):
                M[c] = np.log1p(M[c].clip(lower=0))
            elif c.startswith(("TE_", "WIN_", "OrigTE_", "KNN")):
                M[c] = logit(M[c].values)
        return M.values
    def onehot(X):
        blocks = []
        for c in cats:
            codes = X[c].cat.codes.values
            oh = np.zeros((len(X), len(X[c].cat.categories)), dtype=np.float32)
            ok = codes >= 0
            oh[np.arange(len(X))[ok], codes[ok]] = 1.0
            blocks.append(oh)
        return np.hstack(blocks)
    A = [numeric(X) for X in (X_fit, X_val, X_tst)]
    mu, sd = A[0].mean(axis=0), A[0].std(axis=0) + 1e-6
    out = [np.hstack([np.nan_to_num((a - mu) / sd).astype(np.float32), onehot(X)])
           for a, X in zip(A, (X_fit, X_val, X_tst))]
    # raw income / commute (standardised) for the periodic embedding
    raw = [np.column_stack([X[INC].values, X[COM].values]).astype(np.float64) for X in (X_fit, X_val, X_tst)]
    rmu, rsd = raw[0].mean(0), raw[0].std(0)
    raw = [((r - rmu) / rsd).astype(np.float32) for r in raw]
    return out, raw

class PeriodicEmb(nn.Module):
    """x -> [sin(2*pi*c*x), cos(2*pi*c*x)] with learnable frequencies, then linear."""
    def __init__(self, n_feat, k=24, d=16, sigma=3.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_feat, k) * sigma)
        self.lin = nn.Linear(2 * k, d)
    def forward(self, x):
        z = 2 * np.pi * self.c[None] * x[..., None]
        return torch.relu(self.lin(torch.cat([torch.sin(z), torch.cos(z)], -1))).flatten(1)

class Net(nn.Module):
    def __init__(self, d, use_emb, n_num=0):
        super().__init__()
        self.use_emb = use_emb
        self.n_num = n_num if margs.emb_all else 0
        extra = 2 * 16 if use_emb else 0
        if use_emb:
            self.emb = PeriodicEmb(2)
        if self.n_num:
            self.emb_all = PeriodicEmb(self.n_num, k=8, d=4, sigma=1.0)
            extra += 4 * self.n_num
        widths = [512, 256, 128] if margs.wide else [256, 128]
        layers, prev = [], d + extra
        for w in widths:
            layers += [nn.Linear(prev, w), nn.SiLU(), nn.Dropout(0.15 if margs.wide else 0.1)]
            prev = w
        self.body = nn.Sequential(*layers, nn.Linear(prev, 1))
    def forward(self, x, r):
        parts = [x]
        if self.use_emb:
            parts.append(self.emb(r))
        if self.n_num:
            parts.append(self.emb_all(x[:, :self.n_num]))
        return self.body(torch.cat(parts, 1)).squeeze(1)

BATCH = margs.batch

def predict(model, A, R):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(A), 65536):
            xb = torch.from_numpy(A[i:i + 65536]).to(DEVICE); rb = torch.from_numpy(R[i:i + 65536]).to(DEVICE)
            out.append(torch.sigmoid(model(xb, rb)).cpu().numpy())
    return np.concatenate(out)

def train_mlp(A_fit, R_fit, y_fit, A_val, R_val, y_val, A_tst, R_tst, seed, n_num=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(A_fit.shape[1], margs.emb, n_num).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=margs.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=margs.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(A_fit).to(DEVICE); Rt = torch.from_numpy(R_fit).to(DEVICE)
    yt = torch.from_numpy(y_fit.astype(np.float32)).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad = -1, None, 0, 0
    for ep in range(margs.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen).to(DEVICE)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model(Xt[idx], Rt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        s = roc_auc_score(y_val, predict(model, A_val, R_val))
        if s > best:
            best, best_state, best_ep, bad = s, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= margs.patience:
                break
    model.load_state_dict(best_state)
    return predict(model, A_val, R_val), predict(model, A_tst, R_tst), best_ep

if __name__ == "__main__":
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(vm.X_all, y))
    seeds = [int(s) for s in margs.seeds.split(",")]
    t0 = time.time()
    print(f"[{margs.name}] MLP on {DEVICE} | win={vm.args.win} emb={margs.emb} | seeds {seeds}", flush=True)
    oofs, tsts = [], []
    for seed in seeds:
        oof = np.zeros(len(vm.X_all)); tst = np.zeros(len(vm.X_test_all))
        for f, (tr, va) in enumerate(folds):
            X_fit, X_val, X_tst = vm.fold_data(tr, va, seed)
            (A_fit, A_val, A_tst), (R_fit, R_val, R_tst) = mlp_matrices(X_fit, X_val, X_tst)
            n_num = sum(1 for c in X_fit.columns if c not in cat_cols)   # numeric block comes first
            pv, pt, ep = train_mlp(A_fit, R_fit, y[tr], A_val, R_val, y[va], A_tst, R_tst, seed, n_num)
            oof[va] = pv; tst += pt / len(folds)
            print(f"  [{margs.name}] seed {seed} fold {f}: {roc_auc_score(y[va], pv):.5f} ({ep}ep) "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        print(f"[{margs.name}] seed {seed}: {roc_auc_score(y, oof):.5f}", flush=True)
        oofs.append(oof); tsts.append(tst)
    oof, tst = np.mean(oofs, 0), np.mean(tsts, 0)
    np.save(f"oof/{margs.name}_oof.npy", oof); np.save(f"oof/{margs.name}_test.npy", tst)
    print(f"[{margs.name}] FINAL OOF AUC {roc_auc_score(y, oof):.5f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
