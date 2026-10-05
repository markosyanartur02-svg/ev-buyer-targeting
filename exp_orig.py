# ===========================================================================
# Quick checks for original-dataset features (same 5-fold setup as
# quick_checks.py, compared against local F = 0.94586).
#   orig     : notebook's ORIG_FEATS (orig frequency, lift, novelty, distance)
#   orig_te  : original-data buy rate per value of every raw column
#   knn      : labels of the nearest original rows (k = 1, 5, 25, 100)
#   all_orig : all three
# ===========================================================================
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import quick_checks as qc
from quick_checks import full, INC, COM, RAW, cat_cols, train, F_FEATS, SPEC_F, quick_check

orig = pd.read_csv(qc.os.path.join(qc.DATA_DIR, "original/EV_Adoption_and_Range_Anxiety_Dataset.csv"))
y_orig = (orig["Will_Buy_EV"] == "Yes").astype(int).values

# --- notebook ORIG_FEATS ------------------------------------------------------
ORIG = []
for col in [INC, COM]:
    o = orig[col].dropna()
    o = o.round() if col == INC else o.round(1)
    o_freq = full[col].map(o.value_counts()).fillna(0).values
    full[f"OrigFreq_{col}"] = o_freq
    full[f"Lift_{col}"] = (full[f"Freq_{col}"] / len(full)) / ((o_freq + 1) / (len(o) + 1))
    full[f"Novel_{col}"] = (o_freq == 0).astype(np.int8)
    u = np.sort(o.unique()); v = full[col].values
    idx = np.clip(np.searchsorted(u, v), 1, len(u) - 1)
    full[f"OrigDist_{col}"] = np.minimum(np.abs(v - u[idx - 1]), np.abs(v - u[idx]))
    ORIG += [f"OrigFreq_{col}", f"Lift_{col}", f"Novel_{col}", f"OrigDist_{col}"]

# --- original buy rate per value (smoothed toward the original prior) ---------
ORIG_TE = []
prior = y_orig.mean()
for col in RAW:
    s = pd.DataFrame({"k": orig[col].astype(str), "y": y_orig}).groupby("k")["y"].agg(["sum", "count"])
    rate = (s["sum"] + prior * 5) / (s["count"] + 5)
    full[f"OrigTE_{col}"] = full[col].astype(str).map(rate).astype(float).fillna(prior).values
    ORIG_TE.append(f"OrigTE_{col}")

# --- nearest original rows ----------------------------------------------------
def design(df):
    num = ["Age", INC, COM, "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
           "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
    X = df[num].astype(float).copy()
    for c in cat_cols:
        for v in sorted(train[c].unique()):
            X[f"{c}={v}"] = (df[c].astype(str) == v).astype(float) * 2.0   # weight categoricals
    return X

Do = design(orig)
Do = Do.fillna(Do.median())
mu, sd = Do.mean(), Do.std() + 1e-9
Df = design(full).fillna(Do.median())
nn_ = NearestNeighbors(n_neighbors=100).fit(((Do - mu) / sd).values)
dist, ind = nn_.kneighbors(((Df - mu) / sd).values)
KNN = []
for k in [1, 5, 25, 100]:
    full[f"KNN{k}"] = y_orig[ind[:, :k]].mean(1)
    KNN.append(f"KNN{k}")
full["KNN_dist1"] = dist[:, 0]; KNN.append("KNN_dist1")

qc.X_all = full.iloc[:len(train)].reset_index(drop=True)
qc.X_test_all = full.iloc[len(train):].reset_index(drop=True)
from sklearn.metrics import roc_auc_score
print(f"KNN25 alone AUC on train: {roc_auc_score(qc.y, qc.X_all['KNN25']):.5f}", flush=True)

CHECKS = {
    "orig": F_FEATS + ORIG,
    "orig_te": F_FEATS + ORIG_TE,
    "knn": F_FEATS + KNN,
    "all_orig": F_FEATS + ORIG + ORIG_TE + KNN,
}
if __name__ == "__main__":
    res = {}
    for name in (sys.argv[1:] or list(CHECKS)):
        res[name] = quick_check(name, CHECKS[name], SPEC_F)
    print("\n" + "=" * 50)
    for n, s in res.items():
        print(f"{n:10s} {s:.5f}  ({s - 0.94586:+.5f} vs local F)")
