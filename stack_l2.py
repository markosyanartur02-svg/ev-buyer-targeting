# ===========================================================================
# Level-2 stacker on the prediction pool (pool.py), cross-validated on a
# fold split independent of the base models' folds (5-fold, seed 123).
# Compares, on the same outer folds:
#   * logistic regression on logit(rank) OOFs  (current best blender)
#   * shallow LightGBM on logit(rank) OOFs (+ optional raw columns)
#   usage: .venv/bin/python stack_l2.py [--raw] [--save out.csv]
# ===========================================================================
import argparse, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pool import load_pool, y, test_ids, rank01

ap = argparse.ArgumentParser()
ap.add_argument("--raw", action="store_true")
ap.add_argument("--exclude", default="")
ap.add_argument("--save", default="")
a = ap.parse_args()

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

oof, tst = load_pool()
names = [n for n in oof if n not in a.exclude.split(",")]
X = pd.DataFrame({n: logit(oof[n]) for n in names})
Xt = pd.DataFrame({n: logit(tst[n]) for n in names})
if a.raw:
    tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Environmental_Concern_Level", "Age"]:
        X[c] = tr[c].values; Xt[c] = te[c].values
    for c in ["Subsidy_Available", "Range_Anxiety_Level", "City_Type", "Home_Charging_Possible"]:
        cats = sorted(tr[c].unique())
        X[c] = pd.Categorical(tr[c], categories=cats); Xt[c] = pd.Categorical(te[c], categories=cats)

folds = list(StratifiedKFold(5, shuffle=True, random_state=123).split(X, y))
o_lr, o_gb = np.zeros(len(y)), np.zeros(len(y))
t_gb = np.zeros(len(Xt)); t_lr = np.zeros(len(Xt))
P = dict(learning_rate=0.03, max_depth=3, num_leaves=8, min_child_samples=500, reg_lambda=5.0,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_estimators=5000, n_jobs=8)
t0 = time.time()
Xn = X[names].values; Xtn = Xt[names].values
for f, (tr_i, va_i) in enumerate(folds):
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xn[tr_i], y[tr_i])
    o_lr[va_i] = lr.decision_function(Xn[va_i]); t_lr += lr.decision_function(Xtn) / len(folds)
    m = lgb.LGBMClassifier(**P, random_state=f, verbose=-1)
    m.fit(X.iloc[tr_i], y[tr_i], eval_set=[(X.iloc[va_i], y[va_i])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    o_gb[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]; t_gb += m.predict_proba(Xt)[:, 1] / len(folds)
    print(f"fold {f}: LR {roc_auc_score(y[va_i], o_lr[va_i]):.5f} | L2-GBM {roc_auc_score(y[va_i], o_gb[va_i]):.5f} "
          f"({m.best_iteration_}t) [{(time.time() - t0) / 60:.1f} min]", flush=True)
s_lr, s_gb = roc_auc_score(y, o_lr), roc_auc_score(y, o_gb)
s_mix = roc_auc_score(y, rank01(o_lr) + rank01(o_gb))
print(f"\nCV  LR stack {s_lr:.5f} | L2-GBM {s_gb:.5f} | 50/50 {s_mix:.5f}   ({len(names)} models, raw={a.raw})")
if a.save:
    best = max([("lr", s_lr, t_lr), ("gbm", s_gb, t_gb), ("mix", s_mix, rank01(t_lr) + rank01(t_gb))],
               key=lambda t: t[1])
    pd.DataFrame({"id": test_ids, "Will_Buy_EV": rank01(best[2])}).to_csv(a.save, index=False)
    print(f"saved {a.save} ({best[0]}, CV {best[1]:.5f})")
