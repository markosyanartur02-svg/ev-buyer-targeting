# ===========================================================================
# Non-negative logistic stacking on the prediction pool (pool.py):
#   p = sigmoid(b + sum_k w_k * logit(rank_k)),  w_k >= 0,  small L2 penalty
# fit with L-BFGS-B. Negative weights are what let a 60-model logistic stack
# overfit; forcing w >= 0 prunes the pool to the members that actually help.
# Honest estimate on the same half/half splits as blend.py (seed 0, 4 reps).
#   usage: .venv/bin/python blend_nn.py [--exclude a,b] [--l2 1e-4] [--save out.csv]
# ===========================================================================
import argparse, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from pool import load_pool, y, test_ids, rank01

ap = argparse.ArgumentParser()
ap.add_argument("--exclude", default="")
ap.add_argument("--l2", type=float, default=1e-4)
ap.add_argument("--save", default="")
a = ap.parse_args()

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

oof, tst = load_pool()
names = [n for n in oof if n not in a.exclude.split(",")]
X = np.column_stack([logit(oof[n]) for n in names]).astype(np.float64)
Xt = np.column_stack([logit(tst[n]) for n in names]).astype(np.float64)
mu, sd = X.mean(0), X.std(0)
Xs, Xts = (X - mu) / sd, (Xt - mu) / sd

def fit_nn(Xf, yf):
    n, k = Xf.shape
    def f(theta):
        b, w = theta[0], theta[1:]
        z = b + Xf @ w
        p = 1 / (1 + np.exp(-z))
        loss = np.mean(np.logaddexp(0, z) - yf * z) + a.l2 * np.sum(w ** 2)
        g = p - yf
        return loss, np.concatenate([[g.mean()], Xf.T @ g / n + 2 * a.l2 * w])
    x0 = np.concatenate([[0.0], np.full(k, 1.0 / k)])
    bounds = [(None, None)] + [(0, None)] * k
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    return r.x

t0 = time.time()
best_single = max(names, key=lambda n: roc_auc_score(y, oof[n]))
rng = np.random.RandomState(0)
gains = []
for rep in range(4):
    idx = rng.permutation(len(y)); h = len(y) // 2
    for fit, ev in [(idx[:h], idx[h:]), (idx[h:], idx[:h])]:
        th = fit_nn(Xs[fit], y[fit])
        s = roc_auc_score(y[ev], Xs[ev] @ th[1:])
        gains.append(s - roc_auc_score(y[ev], oof[best_single][ev]))
print(f"{len(names)} models | best single {best_single} | held-out gain (non-neg LR, l2={a.l2}): "
      f"{np.mean(gains):+.5f} (±{np.std(gains):.5f})  [{(time.time() - t0) / 60:.1f} min]", flush=True)

th = fit_nn(Xs, y)
w = th[1:]
z_oof, z_tst = Xs @ w, Xts @ w
print(f"non-neg LR OOF {roc_auc_score(y, z_oof):.5f} | {int((w > 1e-6).sum())} members with weight > 0:")
for n, v in sorted(zip(names, w), key=lambda t: -t[1]):
    if v > 1e-6:
        print(f"   {n:24s} {v:.4f}")
if a.save:
    pd.DataFrame({"id": test_ids, "Will_Buy_EV": rank01(z_tst)}).to_csv(a.save, index=False)
    print(f"saved {a.save}")
