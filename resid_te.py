# ===========================================================================
# Residual target encoding on top of the finished blend.
# The blend's OOF score is calibrated to a probability q (cross-fitted), then for
# each key (exact income, exact commute, ...) the smoothed mean residual y - q of
# the OTHER folds' rows is added back:  z = logit(q) + beta * R_key / (q(1-q)+eps)
# Evaluated with an outer 5-fold split (seed 7, independent of every base model);
# beta is picked on the outer-training rows only, so the reported gain is honest.
#   usage: .venv/bin/python resid_te.py [--save]
# ===========================================================================
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pool import load_pool, y, rank01, test_ids
from rules import apply_rules

W = {"hj_xgb": .375, "mg_B": .125, "fu_C1_chinzo_lgbm": .125, "fu_D_megayak_liftwin": .125,
     "six_realmlp_g": .125, "lr_v19": .125}                     # blend_seeds hill-climb weights
oof, tst = load_pool()
b_oof = sum(w * rank01(oof[k]) for k, w in W.items())
b_tst = sum(w * rank01(tst[k]) for k, w in W.items())
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv").set_index("id").loc[test_ids].reset_index()
print(f"blend OOF {roc_auc_score(y, b_oof):.6f} | with rules {roc_auc_score(y, apply_rules(b_oof, tr)):.6f}", flush=True)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
lg_oof, lg_tst = logit(np.clip(b_oof, 1e-6, 1 - 1e-6)), logit(np.clip(b_tst, 1e-6, 1 - 1e-6))

KEYS = {"inc": tr.Annual_Income_USD.values, "cmt": tr.Daily_Commute_km.values,
        "inc_sub": (tr.Annual_Income_USD.astype(str) + tr.Subsidy_Available).values}
KEYS_T = {"inc": te.Annual_Income_USD.values, "cmt": te.Daily_Commute_km.values,
          "inc_sub": (te.Annual_Income_USD.astype(str) + te.Subsidy_Available).values}

def calib(fit_idx, apply_idx_list, lg_fit, lg_apply_list):
    m = LogisticRegression(C=1e3, max_iter=1000).fit(lg_fit.reshape(-1, 1), y[fit_idx])
    return [m.predict_proba(l.reshape(-1, 1))[:, 1] for l in lg_apply_list]

def resid_map(keys, r, m):
    s = pd.DataFrame({"k": keys, "r": r}).groupby("k")["r"].agg(["sum", "count"])
    return s["sum"] / (s["count"] + m)                             # shrunk toward 0 residual

def encode(key_fit, r_fit, key_apply, m):
    return pd.Series(key_apply).map(resid_map(key_fit, r_fit, m)).fillna(0.0).values

def score(z_parts, beta):
    base, adds = z_parts
    return base + sum(b * a for b, a in zip(beta, adds))

outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(b_oof, y))
grid = [0.0, 0.5, 1.0, 1.5, 2.0]
for keyset, m in [(["inc"], 20), (["inc"], 50), (["inc", "cmt"], 50), (["inc_sub"], 50), (["inc", "inc_sub"], 50)]:
    gains = []
    for ftr, fva in outer:
        # cross-fitted q on outer-train rows (inner 5 folds), and q for outer-val rows from all outer-train rows
        q_tr = np.empty(len(ftr))
        inner = list(StratifiedKFold(5, shuffle=True, random_state=11).split(ftr, y[ftr]))
        R_tr = {k: np.empty(len(ftr)) for k in keyset}
        for itr, iva in inner:
            a, b = ftr[itr], ftr[iva]
            q_a, q_b = calib(a, None, lg_oof[a], [lg_oof[a], lg_oof[b]])
            q_tr[iva] = q_b
            for k in keyset:
                R_tr[k][iva] = encode(KEYS[k][a], y[a] - q_a, KEYS[k][b], m)
        (q_va,) = calib(ftr, None, lg_oof[ftr], [lg_oof[fva]])
        q_all_tr, = calib(ftr, None, lg_oof[ftr], [lg_oof[ftr]])
        R_va = {k: encode(KEYS[k][ftr], y[ftr] - q_all_tr, KEYS[k][fva], m) for k in keyset}
        # pick one shared beta on outer-train rows
        base_tr = logit(q_tr); base_va = logit(q_va)
        best = max(grid, key=lambda bb: roc_auc_score(y[ftr], base_tr + sum(bb * R_tr[k] / (q_tr * (1 - q_tr) + 0.02) for k in keyset)))
        z_va = base_va + sum(best * R_va[k] / (q_va * (1 - q_va) + 0.02) for k in keyset)
        gains.append((roc_auc_score(y[fva], z_va) - roc_auc_score(y[fva], base_va), best))
    g = np.array([x[0] for x in gains])
    print(f"keys {'+'.join(keyset):12s} m={m:3d}: outer-fold gain {g.mean():+.6f} (per fold {' '.join(f'{x:+.6f}' for x in g)}) "
          f"| betas {[x[1] for x in gains]}", flush=True)
