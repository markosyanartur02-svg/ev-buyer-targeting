# ===========================================================================
# Full 10-fold model on the notebook's folds (StratifiedKFold(10, shuffle,
# seed 42) on train row order), saving OOF + test predictions to oof/ so
# pool.py / blend.py pick them up.
#
# Feature view = notebook variant G (raw + digits + frequencies + nested TE)
#   + optional blocks:  --win   centred-window target rates (income, commute)
#                       --orig  original-dataset features (exp_orig.py blocks)
#                       --knn   nearest-original-row labels
#   usage: .venv/bin/python view_model.py NAME [--win] [--orig] [--knn]
#                                         [--algo lgb|xgb] [--seeds 42,7]
# ===========================================================================
import os, sys, time, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--win", action="store_true")
ap.add_argument("--orig", action="store_true")
ap.add_argument("--knn", action="store_true")
ap.add_argument("--pairs", action="store_true")
ap.add_argument("--nodigits", action="store_true")
ap.add_argument("--pseudo", default="")      # .npy of calibrated test probabilities -> soft-labelled extra rows
ap.add_argument("--algo", default="lgb")
ap.add_argument("--seeds", default="42")
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--leaves", type=int, default=31)
ap.add_argument("--colsample", type=float, default=0.3)
ap.add_argument("--mcs", type=int, default=100)
args = ap.parse_known_args()[0]

if args.orig or args.knn:
    import exp_orig as eo          # builds ORIG / ORIG_TE / KNN columns on the shared frame
import quick_checks as qc
from quick_checks import full, train, RAW, DIGITS, FREQ, INC, COM, y, target_encode, _keys

# --- interaction keys (target-encoded only): combos of the buying recipe -------
PAIRS = [("Subsidy_Available", "Range_Anxiety_Level"),
         ("Environmental_Concern_Level", "Subsidy_Available"),
         ("Environmental_Concern_Level", "Range_Anxiety_Level"),
         ("City_Type", "Home_Charging_Possible"),
         ("Home_Charging_Possible", "Range_Anxiety_Level"),
         ("Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"),
         ("Income_r1000", "Subsidy_Available"), ("Income_r1000", "Range_Anxiety_Level"),
         ("Income_r1000", "Environmental_Concern_Level"),
         ("Commute_r1", "Home_Charging_Possible"),
         ("Age", "Gender"), ("Current_Car_Type", "Number_of_Cars_Owned")]
PAIR_KEYS = []
if args.pairs:
    for combo in PAIRS:
        name = "X_" + "__".join(c[:10] for c in combo)
        key = full[combo[0]].astype(str)
        for c in combo[1:]:
            key = key + "|" + full[c].astype(str)
        full[name] = pd.factorize(key)[0]
        PAIR_KEYS.append(name)

X_all = full.iloc[:len(train)].reset_index(drop=True)
X_test_all = full.iloc[len(train):].reset_index(drop=True)

SPEC_F = qc.SPEC_F
SPEC_G = SPEC_F + [(c, 200) for c in RAW if c not in (INC, COM)]
SPEC_G += [(k, 20) for k in PAIR_KEYS] + [(k, 200) for k in PAIR_KEYS]
FEATS = RAW + ([] if args.nodigits else DIGITS) + FREQ
if args.orig:
    FEATS += eo.ORIG + eo.ORIG_TE
if args.knn:
    FEATS += eo.KNN

# --- centred-window target rates ---------------------------------------------
WIN = {INC: [2, 5, 10, 25, 50, 200, 1000], COM: [0.1, 0.5, 1, 3, 10]}
WIN_PRIOR = 10.0

def _win_rates(v_fit, y_fit, v_q, w, prior):
    u, inv = np.unique(v_fit, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(u)).astype(float)
    pos = np.bincount(inv, weights=y_fit, minlength=len(u))
    C = np.concatenate([[0], np.cumsum(cnt)]); P = np.concatenate([[0], np.cumsum(pos)])
    lo = np.searchsorted(u, v_q - w - 1e-9, "left"); hi = np.searchsorted(u, v_q + w + 1e-9, "right")
    n = C[hi] - C[lo]; p = P[hi] - P[lo]
    return (p + prior * WIN_PRIOR) / (n + WIN_PRIOR)

def window_encode(X_fit, y_fit, X_apply_list, seed):
    prior = y_fit.mean()
    inner = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_fit))
    fit_out, apply_out = {}, [{} for _ in X_apply_list]
    for col, ws in WIN.items():
        v_fit = X_fit[col].values.astype(float)
        for w in ws:
            name = f"WIN_{col[:3]}_{w}"
            enc = np.empty(len(v_fit))
            for itr, iva in inner:
                enc[iva] = _win_rates(v_fit[itr], y_fit[itr], v_fit[iva], w, prior)
            fit_out[name] = enc
            for i, Xa in enumerate(X_apply_list):
                apply_out[i][name] = _win_rates(v_fit, y_fit, Xa[col].values.astype(float), w, prior)
    return pd.DataFrame(fit_out), [pd.DataFrame(d) for d in apply_out]

def fold_data(tr, va, seed):
    X_fit = X_all.iloc[tr].reset_index(drop=True)
    X_val = X_all.iloc[va].reset_index(drop=True)
    te_fit, (te_val, te_tst) = target_encode(X_fit, y[tr], [X_val, X_test_all], SPEC_G, seed)
    parts = [[X_fit[FEATS], te_fit], [X_val[FEATS], te_val],
             [X_test_all[FEATS].reset_index(drop=True), te_tst]]
    if args.win:
        w_fit, (w_val, w_tst) = window_encode(X_fit, y[tr], [X_val, X_test_all], seed)
        parts[0].append(w_fit); parts[1].append(w_val); parts[2].append(w_tst)
    return [pd.concat(p, axis=1) for p in parts]

LGB_PARAMS = dict(learning_rate=args.lr, max_depth=args.depth, num_leaves=args.leaves,
                  min_child_samples=args.mcs, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                  colsample_bytree=args.colsample, n_estimators=60000, n_jobs=8)

PSEUDO = np.load(args.pseudo) if args.pseudo else None

def fit_predict(X_fit, y_fit, X_val, y_val, X_tst, seed):
    if PSEUDO is not None:
        # distillation: test rows join training with the blend's calibrated probabilities
        # as soft targets (cross-entropy objective accepts labels in [0, 1])
        X_aug = pd.concat([X_fit, X_tst], ignore_index=True)
        y_aug = np.concatenate([y_fit.astype(float), PSEUDO])
        m = lgb.LGBMRegressor(**LGB_PARAMS, objective="cross_entropy", random_state=seed, verbose=-1)
        m.fit(X_aug, y_aug, eval_set=[(X_val, y_val.astype(float))], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        return m.predict(X_val), m.predict(X_tst), m.best_iteration_
    if args.algo == "lgb":
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, verbose=-1)
        m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        return m.predict_proba(X_val)[:, 1], m.predict_proba(X_tst)[:, 1], m.best_iteration_
    if args.algo == "cb":
        from catboost import CatBoostClassifier
        cats = [c for c in qc.cat_cols if c in X_fit.columns]
        Xc = [X.copy() for X in (X_fit, X_val, X_tst)]
        for X in Xc:
            for c in cats:
                X[c] = X[c].astype(str)
        m = CatBoostClassifier(iterations=20000, learning_rate=args.lr, depth=args.depth, l2_leaf_reg=3,
                               eval_metric="AUC", od_type="Iter", od_wait=300, random_seed=seed,
                               verbose=0, thread_count=8, border_count=254)
        m.fit(Xc[0], y_fit, cat_features=cats, eval_set=(Xc[1], y_val), use_best_model=True)
        return m.predict_proba(Xc[1])[:, 1], m.predict_proba(Xc[2])[:, 1], m.get_best_iteration()
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=60000, learning_rate=args.lr, max_depth=args.depth, min_child_weight=10,
                          subsample=0.8, colsample_bytree=args.colsample, reg_lambda=2.0, max_bin=512,
                          tree_method="hist", enable_categorical=True, eval_metric="auc",
                          early_stopping_rounds=300, random_state=seed, n_jobs=8)
    m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return m.predict_proba(X_val)[:, 1], m.predict_proba(X_tst)[:, 1], m.best_iteration

if __name__ == "__main__":
    folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42).split(X_all, y))
    seeds = [int(s) for s in args.seeds.split(",")]
    oofs, tsts = [], []
    t0 = time.time()
    print(f"[{args.name}] {len(FEATS)} base feats, {len(SPEC_G)} encodings | win={args.win} "
          f"pairs={args.pairs} nodigits={args.nodigits} orig={args.orig} knn={args.knn} "
          f"| {args.algo} lr {args.lr} | {args.folds} folds | seeds {seeds}", flush=True)
    for seed in seeds:
        oof = np.zeros(len(X_all)); tst = np.zeros(len(X_test_all))
        for f, (tr, va) in enumerate(folds):
            X_fit, X_val, X_tst = fold_data(tr, va, seed)
            pv, pt, it = fit_predict(X_fit, y[tr], X_val, y[va], X_tst, seed)
            oof[va] = pv; tst += pt / len(folds)
            print(f"  [{args.name}] seed {seed} fold {f}: {roc_auc_score(y[va], pv):.5f} ({it}t) "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        print(f"[{args.name}] seed {seed}: {roc_auc_score(y, oof):.5f}", flush=True)
        oofs.append(oof); tsts.append(tst)
    oof, tst = np.mean(oofs, 0), np.mean(tsts, 0)
    os.makedirs(os.path.join(qc.os.path.dirname(os.path.abspath(__file__)), "oof"), exist_ok=True)
    np.save(f"oof/{args.name}_oof.npy", oof); np.save(f"oof/{args.name}_test.npy", tst)
    print(f"[{args.name}] FINAL OOF AUC {roc_auc_score(y, oof):.5f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
