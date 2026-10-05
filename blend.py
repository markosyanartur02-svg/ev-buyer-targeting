# ===========================================================================
# Blend the prediction pool (pool.py).
#   * hill climbing (Caruana, with replacement) on rank-space OOF, AUC objective
#   * logistic-regression stacker on logit(rank) OOF
# Honest estimate: weights fit on one half of the rows, scored on the other
# half (4 random splits, both directions). Final weights fit on all rows.
#   usage: .venv/bin/python blend.py [--exclude name,name] [--save out.csv]
# ===========================================================================
import sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from pool import load_pool, y, test_ids, rank01

def hill_climb(O, yy, names, iters=60, tol=1e-7):
    scores = {n: roc_auc_score(yy, O[n]) for n in names}
    best = max(scores, key=scores.get)
    counts = {n: 0 for n in names}; counts[best] = 1
    cur, cur_auc = O[best].copy(), scores[best]
    for _ in range(iters):
        k = sum(counts.values())
        cand = {n: roc_auc_score(yy, (cur * k + O[n]) / (k + 1)) for n in names}
        n_best = max(cand, key=cand.get)
        if cand[n_best] <= cur_auc + tol:
            break
        counts[n_best] += 1
        cur, cur_auc = (cur * k + O[n_best]) / (k + 1), cand[n_best]
    tot = sum(counts.values())
    return {n: c / tot for n, c in counts.items() if c}, cur_auc

def blend(P, w):
    return sum(w[n] * P[n] for n in w)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def lr_fit(O, yy, names):
    X = np.column_stack([logit(O[n]) for n in names])
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, yy)

def lr_pred(m, P, names):
    return m.decision_function(np.column_stack([logit(P[n]) for n in names]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    oof, tst = load_pool()
    names = [n for n in oof if n not in a.exclude.split(",")]
    if a.only:
        names = [n for n in names if any(n.startswith(p) for p in a.only.split(","))]
    print(f"{len(names)} models: {', '.join(names)}")
    best_single = max(names, key=lambda n: roc_auc_score(y, oof[n]))
    print(f"best single: {best_single} {roc_auc_score(y, oof[best_single]):.5f}")

    # honest half/half estimate
    rng = np.random.RandomState(0)
    hc_gain, lr_gain, eq_gain = [], [], []
    for rep in range(4):
        idx = rng.permutation(len(y)); h = len(y) // 2
        for fit, ev in [(idx[:h], idx[h:]), (idx[h:], idx[:h])]:
            Of = {n: oof[n][fit] for n in names}; Oe = {n: oof[n][ev] for n in names}
            base = roc_auc_score(y[ev], Oe[best_single])
            w, _ = hill_climb(Of, y[fit], names)
            hc_gain.append(roc_auc_score(y[ev], blend(Oe, w)) - base)
            m = lr_fit(Of, y[fit], names)
            lr_gain.append(roc_auc_score(y[ev], lr_pred(m, Oe, names)) - base)
            eq_gain.append(roc_auc_score(y[ev], np.mean([Oe[n] for n in names], 0)) - base)
    print(f"held-out gain vs best single:  hill-climb {np.mean(hc_gain):+.5f} (±{np.std(hc_gain):.5f})"
          f" | logistic {np.mean(lr_gain):+.5f} (±{np.std(lr_gain):.5f})"
          f" | equal-avg {np.mean(eq_gain):+.5f}")

    # final fit on everything
    w, auc = hill_climb(oof, y, names)
    print(f"\nhill-climb OOF {auc:.5f}  weights: " + ", ".join(f"{n} {v:.3f}" for n, v in sorted(w.items(), key=lambda t: -t[1])))
    m = lr_fit(oof, y, names)
    print(f"logistic   OOF {roc_auc_score(y, lr_pred(m, oof, names)):.5f}  coefs: "
          + ", ".join(f"{n} {c:.2f}" for n, c in zip(names, m.coef_[0])))

    if a.save:
        pred = rank01(blend(tst, w))
        pd.DataFrame({"id": test_ids, "Will_Buy_EV": pred}).to_csv(a.save, index=False)
        print(f"saved {a.save} (hill-climb weights)")
