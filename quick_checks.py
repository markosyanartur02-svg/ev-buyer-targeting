# ===========================================================================
# Local quick checks, ported from notebookc6a6a8cb9b (Cell 1 + the Buy_Score
# and recipe-structure cells). 5-fold LightGBM with the notebook's fast
# settings; the variant F baseline is re-run locally so every comparison uses
# the same LightGBM version and folds.
#   usage: .venv/bin/python quick_checks.py [check ...]
#   checks: F buy worry drop buy_worry   (default: all)
# ===========================================================================
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
MIN_GAIN = 0.00005
INC, COM = "Annual_Income_USD", "Daily_Commute_km"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
ID_COL, TARGET_COL = sample_sub.columns[0], sample_sub.columns[1]
RAW = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
cat_cols = train[RAW].select_dtypes(exclude=np.number).columns.tolist()
y = (train[TARGET_COL] == "Yes").astype(int).values

# --- Label-free features (train + test together) ---------------------------
full = pd.concat([train[RAW], test[RAW]], ignore_index=True)
for c in cat_cols:
    full[c] = pd.Categorical(full[c], categories=sorted(train[c].unique()))

inc = full[INC].round().astype(np.int64)
DIGITS = []
for k, name in enumerate(["ones", "tens", "hundreds", "thousands", "ten_thousands", "hundred_thousands"]):
    full[f"Income_{name}"] = (inc // 10**k) % 10
    DIGITS.append(f"Income_{name}")
c10 = (full[COM] * 10).round().astype(np.int64)
for k, name in enumerate(["tenths", "ones", "tens"]):
    full[f"Commute_{name}"] = (c10 // 10**k) % 10
    DIGITS.append(f"Commute_{name}")

FREQ = []
for col in [INC, COM]:
    full[f"Freq_{col}"] = full[col].map(full[col].value_counts()).astype(np.int64)
    FREQ.append(f"Freq_{col}")

full["Income_r10"] = (inc / 10).round() * 10
full["Income_r100"] = (inc / 100).round() * 100
full["Income_r1000"] = (inc / 1000).round() * 1000
full["Commute_r1"] = full[COM].round()
full["Commute_r5"] = (full[COM] / 5).round() * 5

# --- Buy_Score: the original dataset's buying formula ----------------------
anx = full["Range_Anxiety_Level"].astype(str)
sub = full["Subsidy_Available"].astype(str)
full["Buy_Score"] = (1.2 * full[INC] / 100_000
                     + 0.6 * full["Environmental_Concern_Level"]
                     + 2.0 * (sub == "Yes")
                     - 1.0 * (anx == "Medium")
                     - 3.0 * (anx == "High"))

# --- Expected worry: range anxiety predicted from its recipe inputs --------
anx_code = anx.map({"Low": 0, "Medium": 1, "High": 2})
W = pd.DataFrame({
    "commute": full[COM],
    "home_st": full["Charging_Stations_Near_Home"],
    "work_st": full["Charging_Stations_Near_Work"],
    "home_chg": (full["Home_Charging_Possible"].astype(str) == "Yes").astype(int),
})
W = pd.concat([W, pd.get_dummies(full["City_Type"].astype(str), prefix="city", dtype=int)], axis=1)
proba = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(W.values),
                                              anx_code.values).predict_proba(
    StandardScaler().fit_transform(W.values))
full["Worry_Expected"] = proba @ np.array([0, 1, 2])
full["Worry_P_High"] = proba[:, 2]
full["Worry_Surprise"] = anx_code.values - full["Worry_Expected"]
WORRY = ["Worry_Expected", "Worry_P_High", "Worry_Surprise"]

X_all = full.iloc[:len(train)].reset_index(drop=True)
X_test_all = full.iloc[len(train):].reset_index(drop=True)
print(f"train {X_all.shape} | test {X_test_all.shape} | Buy_Score AUC "
      f"{roc_auc_score(y, X_all['Buy_Score']):.5f}", flush=True)

# --- Nested target encoding (labels never reach their own rows) ------------
def _keys(df, col):
    s = df[col]
    return s.cat.codes.values if isinstance(s.dtype, pd.CategoricalDtype) else s.values

def _smoothed(keys, targets, prior, m):
    s = pd.DataFrame({"k": keys, "y": targets}).groupby("k")["y"].agg(["sum", "count"])
    return (s["sum"] + prior * m) / (s["count"] + m)

def target_encode(X_fit, y_fit, X_apply_list, spec, seed):
    prior = y_fit.mean()
    inner = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_fit))
    fit_out, apply_out = {}, [{} for _ in X_apply_list]
    for col, m in spec:
        name = f"TE_{col}_m{m}"
        k_fit = _keys(X_fit, col)
        enc = np.empty(len(k_fit))
        for itr, iva in inner:
            mapping = _smoothed(k_fit[itr], y_fit[itr], prior, m)
            enc[iva] = pd.Series(k_fit[iva]).map(mapping).fillna(prior).values
        fit_out[name] = enc
        mapping = _smoothed(k_fit, y_fit, prior, m)
        for i, Xa in enumerate(X_apply_list):
            apply_out[i][name] = pd.Series(_keys(Xa, col)).map(mapping).fillna(prior).values
    return pd.DataFrame(fit_out), [pd.DataFrame(d) for d in apply_out]

def fold_data(tr, va, feats, spec, seed):
    X_fit = X_all.iloc[tr].reset_index(drop=True)
    X_val = X_all.iloc[va].reset_index(drop=True)
    te_fit, (te_val,) = target_encode(X_fit, y[tr], [X_val], spec, seed)
    return (pd.concat([X_fit[feats], te_fit], axis=1),
            pd.concat([X_val[feats], te_val], axis=1))

SPEC_F = ([(INC, 20), (COM, 20), (INC, 5), (INC, 100), (COM, 5), (COM, 100)]
          + [(c, 20) for c in RAW if c not in (INC, COM)]
          + [("Income_r10", 20), ("Income_r100", 20), ("Income_r1000", 20),
             ("Commute_r1", 20), ("Commute_r5", 20)])
F_FEATS = RAW + DIGITS + FREQ
NOISE = ["Age", "Gender", "Number_of_Cars_Owned", "Current_Car_Type"]
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_all, y))
CHECK_PARAMS = dict(learning_rate=0.05, num_leaves=15, min_child_samples=50, reg_lambda=1.0,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_estimators=5000)

def quick_check(name, feats, spec):
    t0 = time.time()
    oof = np.zeros(len(X_all))
    for fold, (tr, va) in enumerate(FOLDS):
        X_fit, X_val = fold_data(tr, va, feats, spec, SEED)
        m = lgb.LGBMClassifier(**CHECK_PARAMS, random_state=SEED, verbose=-1)
        m.fit(X_fit, y[tr], eval_set=[(X_val, y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(X_val)[:, 1]
        print(f"  [{name}] fold {fold}: {roc_auc_score(y[va], oof[va]):.5f} "
              f"({m.best_iteration_}t, {(time.time() - t0) / 60:.1f} min)", flush=True)
    np.save(os.path.join(OUT_DIR, f"oof_quick_{name}.npy"), oof)
    score = roc_auc_score(y, oof)
    print(f"{name}: {score:.5f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    return score

CHECKS = {
    "F": (F_FEATS, SPEC_F),
    "buy": (F_FEATS + ["Buy_Score"], SPEC_F),
    "worry": (F_FEATS + WORRY, SPEC_F),
    "drop": ([c for c in F_FEATS if c not in NOISE], [(c, m) for c, m in SPEC_F if c not in NOISE]),
    "buy_worry": (F_FEATS + ["Buy_Score"] + WORRY, SPEC_F),
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CHECKS)
    if "F" not in wanted:
        wanted = ["F"] + wanted
    results = {}
    for name in wanted:
        feats, spec = CHECKS[name]
        results[name] = quick_check(name, feats, spec)
    base = results["F"]
    print("\n" + "=" * 50)
    for name, s in results.items():
        verdict = "" if name == "F" else ("  <- keep" if s > base + MIN_GAIN else "")
        print(f"{name:10s} {s:.5f}  ({s - base:+.5f} vs local F){verdict}")
    with open(os.path.join(OUT_DIR, "quick_results.json"), "w") as f:
        json.dump(results, f, indent=2)
