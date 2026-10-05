# ===========================================================================
# Prediction pool: every out-of-fold (OOF) / test prediction pair available,
# aligned on id and converted to rank space (0..1) so different calibrations
# mix freely (AUC only depends on order).
#   * own runs: oof/<name>_oof.npy + oof/<name>_test.npy (view_model.py, mlp_model.py)
#   * baseline notebook outputs in my_oof/ (optional)
#   * external OOF libraries published by other participants (optional; not
#     redistributed here - see README "External predictions"). Every external
#     source is skipped silently when its files are absent.
# ===========================================================================
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
train_ids = pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["id"])["id"].values
test_ids = pd.read_csv(os.path.join(ROOT, "data/test.csv"), usecols=["id"])["id"].values
y = (pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values

def rank01(v):
    return rankdata(v) / len(v)

def _by_id(path, ids, col=None):
    d = pd.read_csv(path).set_index("id")
    col = col or [c for c in d.columns if c != "id"][0]
    return d.loc[ids, col].values

def load_pool():
    oof, tst = {}, {}
    # own notebook run (10-fold, seed 42 folds)
    my = os.path.join(ROOT, "my_oof")
    for kind in ["lgb", "mlp"]:
        o = [np.load(p) for p in sorted(glob.glob(f"{my}/oof_{kind}_*.npy"))]
        t = [np.load(p) for p in sorted(glob.glob(f"{my}/test_{kind}_*.npy"))]
        if o:
            oof[f"my_{kind}"], tst[f"my_{kind}"] = np.mean(o, 0), np.mean(t, 0)
    if os.path.exists(f"{my}/oof_cb.npy"):
        oof["my_cb"], tst["my_cb"] = np.load(f"{my}/oof_cb.npy"), np.load(f"{my}/test_cb.npy")

    # local runs: oof/<name>_oof.npy + oof/<name>_test.npy
    for p in sorted(glob.glob(os.path.join(ROOT, "oof", "*_oof.npy"))):
        name = os.path.basename(p)[:-8]
        tp = p.replace("_oof.npy", "_test.npy")
        if os.path.exists(tp):
            oof[f"loc_{name}"], tst[f"loc_{name}"] = np.load(p), np.load(tp)

    # extra-seed runs saved as NAME_s2 (2 seeds): merge into NAME, weighted by seed count
    for k in [k for k in oof if k.endswith("_s2") and k[:-3] in oof]:
        base = k[:-3]
        oof[base] = (oof[base] + 2 * oof[k]) / 3; tst[base] = (tst[base] + 2 * tst[k]) / 3
        del oof[k], tst[k]

    # local RealMLP seeds (each with its own fold split): average into one member
    rk = [k for k in oof if k.startswith("loc_R_realmlp_s")]
    if len(rk) > 1:
        oof["loc_R_realmlp"] = np.mean([rank01(oof[k]) for k in rk], 0)
        tst["loc_R_realmlp"] = np.mean([rank01(tst[k]) for k in rk], 0)
        for k in rk:
            del oof[k], tst[k]

    # megayak four views
    mg = os.path.join(ROOT, "public_oof/s6e9-four-feature-views-one-ensemble-oof")
    if os.path.exists(f"{mg}/oof_four_views.csv"):
        for col in ["A_lgbm_triple_te_3seed", "B_xgb_same_features",
                    "C_lgbm_windows_lift_nodigit", "D_lgbm_alt_ladder_noexact"]:
            oof[f"mg_{col[0]}"] = _by_id(f"{mg}/oof_four_views.csv", train_ids, col)
            tst[f"mg_{col[0]}"] = _by_id(f"{mg}/test_four_views.csv", test_ids, col)

    # najiama OOF hub
    nj = os.path.join(ROOT, "public_oof/s6e9-oof")
    pairs = {"nj_lgbV1": ("Pure LGBM_V1_oof.csv", "Pure LGBM_V1_test.csv"),
             "nj_lgbV3": ("Pure LGBM_V3_oof.csv", "Pure LGBM_V3_test.csv"),
             "nj_lgbV5": ("Pure LGBM_V5_oof.csv", "Pure LGBM_V5_test.csv"),
             "nj_lgbV6": ("Pure LGBM_V6_oof.csv", "Pure LGBM_V6_test.csv"),
             "nj_sergey": ("Sergey_LGBM_oof.csv", "Sergey_LGBM_submission.csv"),
             "nj_xgb10": ("XGBoost_Triple_TE_10folds_oof.csv", "XGBoost_Triple_TE_10folds_test.csv"),
             "nj_xgb5": ("XGBoost_Triple_TE_5folds_oof.csv", "XGBoost_Triple_TE_5folds_test.csv")}
    for k, (o, t) in pairs.items():
        if os.path.exists(f"{nj}/{o}"):
            oof[k], tst[k] = _by_id(f"{nj}/{o}", train_ids), _by_id(f"{nj}/{t}", test_ids)

    # public notebook outputs (kout/): base models only, no stacked blends
    ko = os.path.join(ROOT, "kout")
    csv_pairs = {
        "pub_realmlp": ("ps-s6-e9-realmlp-pytorch/oof_preds.csv", "ps-s6-e9-realmlp-pytorch/submission.csv"),
        "pub_histgbm": ("signal-that-matters-eda-histgbm-cv-0-94614/oof_proba.csv",
                        "signal-that-matters-eda-histgbm-cv-0-94614/test_proba.csv"),
        "pub_xgbwin": ("s6e9-xgboost-window-encodings-0-946-cv/oof.csv",
                       "s6e9-xgboost-window-encodings-0-946-cv/submission.csv"),
        "pub_najiV3": ("pure-lgbm-model-cv-0-94607-lb-0-94638/oof_LIGHTGBM.csv",
                       "pure-lgbm-model-cv-0-94607-lb-0-94638/test_LIGHTGBM.csv"),
    }
    for k, (o, t) in csv_pairs.items():
        if os.path.exists(f"{ko}/{o}"):
            oof[k], tst[k] = _by_id(f"{ko}/{o}", train_ids), _by_id(f"{ko}/{t}", test_ids)
    npy_pairs = {
        "pub_kirill_lgb": ("s6e9-lightgbm/oof_predictions_lgb_42_5.npy", "s6e9-lightgbm/test_predictions_lgb_42_5.npy"),
        "pub_dvorkin_xgb": ("s6e9-single-xgb-cv-0-94607/oof_preds_base.npy", "s6e9-single-xgb-cv-0-94607/test_preds_base.npy"),
        "pub_mg_hybrid": ("s6e9-one-lightgbm-from-raw-data-cv-0-9463/oof_hybrid.npy",
                          "s6e9-one-lightgbm-from-raw-data-cv-0-9463/test_hybrid.npy"),
    }
    for k, (o, t) in npy_pairs.items():
        if os.path.exists(f"{ko}/{o}"):
            oof[k], tst[k] = np.load(f"{ko}/{o}"), np.load(f"{ko}/{t}")

    # second harvest of public notebook outputs (kout2/): base models only
    k2 = os.path.join(ROOT, "kout2")
    fus = f"{k2}/ev-quantum-forge-fusion-iv-neural-views-s6e9/fusion_outputs"
    if os.path.exists(f"{fus}/oof_components.csv"):
        fo = pd.read_csv(f"{fus}/oof_components.csv").set_index("id").loc[train_ids]
        ft = pd.read_csv(f"{fus}/test_components.csv").set_index("id").loc[test_ids]
        for c in [c for c in ft.columns if c != "fusion"]:
            oof[f"fu_{c}"], tst[f"fu_{c}"] = fo[c].values, ft[c].values
    csv2 = {
        "k2_cb_naji": ("catboost-triple-te-dynamic-pruning-lb-0-94624/oof_CATBOOST.csv",
                       "catboost-triple-te-dynamic-pruning-lb-0-94624/test_CATBOOST.csv", None),
        "k2_cb_udaken": ("catboost-triple-te/oof_CATBOOST.csv", "catboost-triple-te/test_CATBOOST.csv", None),
        "k2_realmlp_ptk": ("ps-s6-e9-realmlp-pytabkit/oof_preds.csv", "ps-s6-e9-realmlp-pytabkit/submission.csv", None),
        "k2_cb_yhay": ("simple-catboost-baseline-s6e9/catboost_oof.csv", "simple-catboost-baseline-s6e9/submission.csv",
                       "prediction"),
        "k2_cb_tarter": ("ps-s06e09-catboost/predictions/catboost_oof_probs.csv",
                         "ps-s06e09-catboost/predictions/catboost_test_probs.csv", "prob_1"),
        "k2_hgb_tarter": ("ps-s06e09-histgradientboosting/predictions/histgb_oof_probs.csv",
                          "ps-s06e09-histgradientboosting/predictions/histgb_test_probs.csv", "prob_1"),
    }
    for k, (o, t, col) in csv2.items():
        if os.path.exists(f"{k2}/{o}"):
            oof[k] = _by_id(f"{k2}/{o}", train_ids, col)
            tst[k] = _by_id(f"{k2}/{t}", test_ids, col if col == "prob_1" else None)
    npy2 = {"k2_realmlp_exact": ("s6e9-realmlp-exact-value-neural-network-gpu/nn_oof_ensemble.npy",
                                 "s6e9-realmlp-exact-value-neural-network-gpu/nn_test_ensemble.npy"),
            "k2_realmlp_bnzn": ("neural-network-diversity-realmlp-pytorch/oof_nn.npy",
                                "neural-network-diversity-realmlp-pytorch/pred_nn.npy")}
    for k, (o, t) in npy2.items():
        if os.path.exists(f"{k2}/{o}"):
            oof[k], tst[k] = np.load(f"{k2}/{o}"), np.load(f"{k2}/{t}")

    # third harvest: public OOF datasets (public_oof2/)
    p2 = os.path.join(ROOT, "public_oof2")
    six = f"{p2}/s6e9-six-feature-views-oof-library"
    if os.path.exists(f"{six}/oof_six_views.csv"):
        for col, k in [("E_ladder25_250_2500_lift_sm5_50_500", "six_E"), ("F_exact_rate_as_init_score", "six_F")]:
            oof[k] = _by_id(f"{six}/oof_six_views.csv", train_ids, col)
            tst[k] = _by_id(f"{six}/test_six_views.csv", test_ids, col)
        oof["six_realmlp_g"] = _by_id(f"{six}/oof_realmlp_g.csv", train_ids, "G_realmlp_3seed")
        tst["six_realmlp_g"] = _by_id(f"{six}/test_realmlp_g.csv", test_ids, "G_realmlp_3seed")
    # (megayak hybrid 3-seed == four-view A exactly; not added)
    # positional (train.csv / test.csv file order), no ids
    for sub_dir, members in [("s6e9-digit-leak-oof", ["lgb", "cat", "xgb"]),
                             ("s6e9-golem-oof-library", ["a_lgbm", "b_lgbm_deep", "c_xgb", "d_catboost", "e_histgb",
                                                         "f_lgbm_te", "g_mlp", "h_spline_gam", "i_extratrees", "j_logreg"])]:
        tag = "dl" if "digit" in sub_dir else "go"
        for m in members:
            fo, ft = f"{p2}/{sub_dir}/oof_{m}.npy", f"{p2}/{sub_dir}/test_{m}.npy"
            if os.path.exists(fo) and os.path.exists(ft):
                oof[f"{tag}_{m}"], tst[f"{tag}_{m}"] = np.load(fo), np.load(ft)
    mv = f"{p2}/s6e9-medvax-blamerx-oof-predictions"
    if os.path.exists(f"{mv}/oof.csv"):
        oof["mv_blamerx"], tst["mv_blamerx"] = _by_id(f"{mv}/oof.csv", train_ids), _by_id(f"{mv}/submission.csv", test_ids)

    # fourth harvest (kout3/): engines used by lucifer19's EV Grand Prix blend that we lacked
    k3 = os.path.join(ROOT, "kout3")
    hj = f"{k3}/kps6e09-xgb-sample"
    if os.path.exists(f"{hj}/oof/XGB_SAMPLE_OOF.parquet"):
        oof["hj_xgb"] = pd.read_parquet(f"{hj}/oof/XGB_SAMPLE_OOF.parquet").set_index("id").loc[train_ids, "oof_pred"].values
        tst["hj_xgb"] = pd.read_parquet(f"{hj}/test_preds/XGB_SAMPLE_TEST.parquet").set_index("id").loc[test_ids, "test_pred"].values
    for k, d in [("mz_lgb_comp", "s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636"),
                 ("mz_lgb_tfe", "s6e9-leak-free-target-frequency-encoding-lgbm")]:
        if os.path.exists(f"{k3}/{d}/oof_predictions.csv"):
            oof[k] = _by_id(f"{k3}/{d}/oof_predictions.csv", train_ids, "Will_Buy_EV")
            tst[k] = _by_id(f"{k3}/{d}/test_predictions.csv", test_ids, "Will_Buy_EV")
    gm = f"{k3}/row-memory-lgbm"
    if os.path.exists(f"{gm}/oof_row_memory.csv"):
        oof["mm_memory"] = _by_id(f"{gm}/oof_row_memory.csv", train_ids, "oof")
        tst["mm_memory"] = _by_id(f"{gm}/submission.csv", test_ids)
    tf = f"{k3}/s6e9-cv-0-946072-lb-0-94614"            # positional npy, train/test file order
    for m in ["xgb_fp", "xgb_fp_hb", "xgb_fp_kern", "logit_fp", "logit_fp_hb", "nn_fp_wide"]:
        if os.path.exists(f"{tf}/oof_{m}.npy"):
            oof[f"tf_{m}"], tst[f"tf_{m}"] = np.load(f"{tf}/oof_{m}.npy"), np.load(f"{tf}/test_{m}.npy")
    rs = f"{p2}/s6e9-residual-stack-oof"
    for m in ["v19", "yek", "jaz"]:
        if os.path.exists(f"{rs}/{m}_oof.csv"):
            oof[f"lr_{m}"], tst[f"lr_{m}"] = _by_id(f"{rs}/{m}_oof.csv", train_ids), _by_id(f"{rs}/{m}_test.csv", test_ids)

    # extra-seed runs on cloud GPU (cloud_gpu/*_out): averaged in rank space into the
    # original member (each seed has its own fold split, so each OOF is still honest)
    kg = os.path.join(ROOT, "cloud_gpu")
    if "hj_xgb" in oof:
        hs_o, hs_t = [rank01(oof["hj_xgb"])], [rank01(tst["hj_xgb"])]
        for s in (7, 2026):
            po = f"{kg}/hj_out_s{s}/oof/XGB_SAMPLE_OOF.parquet"
            pt = f"{kg}/hj_out_s{s}/test_preds/XGB_SAMPLE_TEST.parquet"
            if os.path.exists(po) and os.path.exists(pt):
                hs_o.append(rank01(pd.read_parquet(po).set_index("id").loc[train_ids, "oof_pred"].values))
                hs_t.append(rank01(pd.read_parquet(pt).set_index("id").loc[test_ids, "test_pred"].values))
        oof["hj_xgb"], tst["hj_xgb"] = np.mean(hs_o, 0), np.mean(hs_t, 0)
    if "pub_xgbwin" in oof:
        bs_o, bs_t = [rank01(oof["pub_xgbwin"])], [rank01(tst["pub_xgbwin"])]
        for s in (7,):
            po, pt = f"{kg}/blamerx_out_s{s}/oof.csv", f"{kg}/blamerx_out_s{s}/submission.csv"
            if os.path.exists(po) and os.path.exists(pt):
                bs_o.append(rank01(_by_id(po, train_ids))); bs_t.append(rank01(_by_id(pt, test_ids)))
        oof["pub_xgbwin"], tst["pub_xgbwin"] = np.mean(bs_o, 0), np.mean(bs_t, 0)

    # TabM run on cloud GPU: OOF from 5-fold CV,
    # test from the notebook's full-data refit
    tb = os.path.join(ROOT, "cloud_gpu/tabm_out")
    if os.path.exists(f"{tb}/oof_tabm.csv") and os.path.exists(f"{tb}/submission.csv"):
        oof["kg_tabm"], tst["kg_tabm"] = _by_id(f"{tb}/oof_tabm.csv", train_ids), _by_id(f"{tb}/submission.csv", test_ids)

    # neural nets drift in calibration between folds: rank their OOF within each fold
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    for k in [k for k in oof if k == "my_mlp" or k.startswith("loc_M")]:
        r = np.empty(len(y))
        for _, va in folds:
            r[va] = rankdata(oof[k][va]) / len(va)
        oof[k] = r

    oof = {k: rank01(v) for k, v in oof.items()}
    tst = {k: rank01(v) for k, v in tst.items()}
    return oof, tst
