import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

from src.data import load_datasets
from src.features import (
    engineer_feature_pack_3,
    CATEGORICAL_COLS,
    BOOLEAN_COLS,
)
from src.mlp_model import train_tabular_mlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("top_tier_pipeline")


def run_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING GRANDMASTER TOP-TIER PIPELINE (EXP-018)")
    logger.info("=" * 70)

    start_total = time.time()
    data_dir = Path("playground-series-s6e11")

    # 1. Load Data
    logger.info("Loading training and testing data...")
    X_train_opt, y_train, X_test_opt, train_ids, test_ids = load_datasets(
        str(data_dir / "train.csv"),
        str(data_dir / "test.csv"),
    )
    y = y_train.values

    # 2. Feature Engineering Pack 3
    logger.info("Engineering Feature Pack 3 (Surge, Anomaly & Exponential Flags)...")
    t0 = time.time()
    train_feat = engineer_feature_pack_3(X_train_opt)
    test_feat = engineer_feature_pack_3(X_test_opt)
    logger.info(f"Feature Pack 3 ready in {time.time() - t0:.2f}s! Features: {train_feat.shape[1]}")

    # Identify feature types
    all_cat_cols = [c for c in train_feat.columns if train_feat[c].dtype == "object" or c in CATEGORICAL_COLS]
    bool_cols_present = [c for c in train_feat.columns if train_feat[c].dtype == "bool" or c in BOOLEAN_COLS]
    num_cols_present = [c for c in train_feat.columns if c not in all_cat_cols and c not in bool_cols_present]

    logger.info(f"Categorical features: {len(all_cat_cols)}, Boolean features: {len(bool_cols_present)}, Numerical: {len(num_cols_present)}")

    # Setup 5-Fold Stratified CV
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_train = len(train_ids)
    n_test = len(test_ids)

    # Prediction containers
    oof_lgb_p3 = np.zeros(n_train, dtype=np.float32)
    test_lgb_p3 = np.zeros(n_test, dtype=np.float32)

    oof_xgb_p3 = np.zeros(n_train, dtype=np.float32)
    test_xgb_p3 = np.zeros(n_test, dtype=np.float32)

    oof_mlp = np.zeros(n_train, dtype=np.float32)
    test_mlp = np.zeros(n_test, dtype=np.float32)

    fold_aucs_lgb = []
    fold_aucs_xgb = []
    fold_aucs_mlp = []

    # Iterate folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_feat, y), 1):
        logger.info("-" * 50)
        logger.info(f"PROCESSING FOLD {fold}/{n_splits}...")
        t_fold = time.time()

        # Split features
        X_tr = train_feat.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = train_feat.iloc[val_idx].copy()
        y_va = y[val_idx]
        X_te = test_feat.copy()

        # --- Tree Preprocessing (Ordinal Encoding) ---
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32)
        X_tr_tree = X_tr.copy()
        X_va_tree = X_va.copy()
        X_te_tree = X_te.copy()

        X_tr_tree[all_cat_cols] = ord_enc.fit_transform(X_tr_tree[all_cat_cols].astype(str))
        X_va_tree[all_cat_cols] = ord_enc.transform(X_va_tree[all_cat_cols].astype(str))
        X_te_tree[all_cat_cols] = ord_enc.transform(X_te_tree[all_cat_cols].astype(str))

        # --- Neural Net Preprocessing (Scaling + One-Hot) ---
        scaler = StandardScaler()
        X_tr_num_scaled = scaler.fit_transform(X_tr_tree[num_cols_present].values.astype(np.float32))
        X_va_num_scaled = scaler.transform(X_va_tree[num_cols_present].values.astype(np.float32))
        X_te_num_scaled = scaler.transform(X_te_tree[num_cols_present].values.astype(np.float32))

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        X_tr_cat_ohe = ohe.fit_transform(X_tr[all_cat_cols].astype(str))
        X_va_cat_ohe = ohe.transform(X_va[all_cat_cols].astype(str))
        X_te_cat_ohe = ohe.transform(X_te[all_cat_cols].astype(str))

        X_tr_bool = X_tr[bool_cols_present].values.astype(np.float32)
        X_va_bool = X_va[bool_cols_present].values.astype(np.float32)
        X_te_bool = X_te[bool_cols_present].values.astype(np.float32)

        X_tr_nn = np.hstack([X_tr_num_scaled, X_tr_bool, X_tr_cat_ohe])
        X_va_nn = np.hstack([X_va_num_scaled, X_va_bool, X_va_cat_ohe])
        X_te_nn = np.hstack([X_te_num_scaled, X_te_bool, X_te_cat_ohe])

        # --- Model 1: Shallow LightGBM (Pack 3) ---
        logger.info(f"  [Fold {fold}] Training Shallow LightGBM...")
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 10,
            "min_child_samples": 50,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "n_estimators": 1200,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_tr_tree,
            y_tr,
            eval_set=[(X_va_tree, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        val_preds_lgb = lgb_model.predict_proba(X_va_tree)[:, 1]
        oof_lgb_p3[val_idx] = val_preds_lgb
        test_lgb_p3 += lgb_model.predict_proba(X_te_tree)[:, 1] / n_splits
        auc_lgb = roc_auc_score(y_va, val_preds_lgb)
        fold_aucs_lgb.append(auc_lgb)
        logger.info(f"  [Fold {fold}] LightGBM AUC: {auc_lgb:.5f}")

        # --- Model 2: Shallow XGBoost (Pack 3) ---
        logger.info(f"  [Fold {fold}] Training Shallow XGBoost...")
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 4,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "n_estimators": 1000,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr_tree,
            y_tr,
            eval_set=[(X_va_tree, y_va)],
            verbose=False,
        )
        val_preds_xgb = xgb_model.predict_proba(X_va_tree)[:, 1]
        oof_xgb_p3[val_idx] = val_preds_xgb
        test_xgb_p3 += xgb_model.predict_proba(X_te_tree)[:, 1] / n_splits
        auc_xgb = roc_auc_score(y_va, val_preds_xgb)
        fold_aucs_xgb.append(auc_xgb)
        logger.info(f"  [Fold {fold}] XGBoost AUC: {auc_xgb:.5f}")

        # --- Model 3: PyTorch Tabular MLP (Pack 3) ---
        logger.info(f"  [Fold {fold}] Training PyTorch Tabular Neural Network (MLP)...")
        val_preds_mlp, test_preds_mlp_fold, auc_mlp = train_tabular_mlp(
            X_tr_nn,
            y_tr,
            X_va_nn,
            y_va,
            X_te_nn,
            epochs=7,
            batch_size=4096,
            lr=2e-3,
            device="cpu",
            seed=42 + fold,
        )
        oof_mlp[val_idx] = val_preds_mlp
        test_mlp += test_preds_mlp_fold / n_splits
        fold_aucs_mlp.append(auc_mlp)
        logger.info(f"  [Fold {fold}] PyTorch MLP AUC: {auc_mlp:.5f}")

        logger.info(f"Fold {fold} finished in {time.time() - t_fold:.1f}s")

    # Evaluate Individual Model OOFs
    total_auc_lgb = roc_auc_score(y, oof_lgb_p3)
    total_auc_xgb = roc_auc_score(y, oof_xgb_p3)
    total_auc_mlp = roc_auc_score(y, oof_mlp)

    logger.info("=" * 70)
    logger.info("5-FOLD CROSS-VALIDATION SUMMARY:")
    logger.info(f"Shallow LightGBM (Pack 3) OOF AUC: {total_auc_lgb:.5f} (Mean: {np.mean(fold_aucs_lgb):.5f} ± {np.std(fold_aucs_lgb):.5f})")
    logger.info(f"Shallow XGBoost  (Pack 3) OOF AUC: {total_auc_xgb:.5f} (Mean: {np.mean(fold_aucs_xgb):.5f} ± {np.std(fold_aucs_xgb):.5f})")
    logger.info(f"PyTorch Tabular MLP       OOF AUC: {total_auc_mlp:.5f} (Mean: {np.mean(fold_aucs_mlp):.5f} ± {np.std(fold_aucs_mlp):.5f})")

    # 4. Save intermediate OOFs & predictions
    output_oof = Path("outputs/oof")
    output_preds = Path("outputs/predictions")
    output_oof.mkdir(parents=True, exist_ok=True)
    output_preds.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": train_ids, "pred": oof_lgb_p3}).to_csv(output_oof / "oof_EXP-018_lgb_p3.csv", index=False)
    pd.DataFrame({"id": test_ids, "is_fraud": test_lgb_p3}).to_csv(output_preds / "preds_EXP-018_lgb_p3.csv", index=False)

    pd.DataFrame({"id": train_ids, "pred": oof_xgb_p3}).to_csv(output_oof / "oof_EXP-018_xgb_p3.csv", index=False)
    pd.DataFrame({"id": test_ids, "is_fraud": test_xgb_p3}).to_csv(output_preds / "preds_EXP-018_xgb_p3.csv", index=False)

    pd.DataFrame({"id": train_ids, "pred": oof_mlp}).to_csv(output_oof / "oof_EXP-018_mlp.csv", index=False)
    pd.DataFrame({"id": test_ids, "is_fraud": test_mlp}).to_csv(output_preds / "preds_EXP-018_mlp.csv", index=False)

    # 5. Grandmaster Stacking & Rank Fusion
    logger.info("=" * 70)
    logger.info("OPTIMIZING GRANDMASTER MULTI-MODEL RANK FUSION...")

    # Load previously verified high-performing diversity models
    oof_lgb_ms = pd.read_csv("outputs/oof/oof_EXP-015_multiseed_lgb.csv")["pred"].values
    test_lgb_ms = pd.read_csv("outputs/predictions/preds_EXP-015_multiseed_lgb.csv")["is_fraud"].values

    oof_cat = pd.read_csv("outputs/oof/oof_EXP-009_shallow_catboost.csv")["pred"].values
    test_cat = pd.read_csv("outputs/predictions/preds_EXP-009_shallow_catboost.csv")["is_fraud"].values

    # Convert to uniform percentile ranks
    r_lgb_p3 = rankdata(oof_lgb_p3) / n_train
    r_xgb_p3 = rankdata(oof_xgb_p3) / n_train
    r_mlp = rankdata(oof_mlp) / n_train
    r_lgb_ms = rankdata(oof_lgb_ms) / n_train
    r_cat = rankdata(oof_cat) / n_train

    rt_lgb_p3 = rankdata(test_lgb_p3) / n_test
    rt_xgb_p3 = rankdata(test_xgb_p3) / n_test
    rt_mlp = rankdata(test_mlp) / n_test
    rt_lgb_ms = rankdata(test_lgb_ms) / n_test
    rt_cat = rankdata(test_cat) / n_test

    models_oof_matrix = np.column_stack([r_lgb_p3, r_xgb_p3, r_mlp, r_lgb_ms, r_cat])
    models_test_matrix = np.column_stack([rt_lgb_p3, rt_xgb_p3, rt_mlp, rt_lgb_ms, rt_cat])
    model_names = ["LGBM_Pack3", "XGB_Pack3", "PyTorch_MLP", "MultiSeed_LGBM", "CatBoost_Shallow"]

    # Nelder-Mead Optimization for weights
    def loss_func(w):
        w_norm = np.clip(w, 0, None)
        if np.sum(w_norm) == 0:
            return 0.0
        w_norm = w_norm / np.sum(w_norm)
        blend = np.dot(models_oof_matrix, w_norm)
        return -roc_auc_score(y, blend)

    init_weights = np.ones(5) / 5.0
    res = minimize(loss_func, init_weights, method="Nelder-Mead", options={"maxiter": 300})
    best_w = np.clip(res.x, 0, None)
    best_w = best_w / np.sum(best_w)

    logger.info("Optimal Ensembling Weights:")
    for name, weight in zip(model_names, best_w):
        logger.info(f"  {name:20s}: {weight:.4f}")

    oof_blend_raw = np.dot(models_oof_matrix, best_w)
    base_ensemble_auc = roc_auc_score(y, oof_blend_raw)
    logger.info(f"Linear Rank Fusion OOF AUC: {base_ensemble_auc:.5f}")

    # Power calibration search (gamma)
    best_gamma = 1.0
    best_calib_auc = base_ensemble_auc
    oof_final = rankdata(oof_blend_raw) / n_train

    for gamma in [0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]:
        calib_preds = (oof_final) ** gamma
        auc_cand = roc_auc_score(y, calib_preds)
        if auc_cand > best_calib_auc:
            best_calib_auc = auc_cand
            best_gamma = gamma

    logger.info(f"Best Calibration Gamma: {best_gamma:.2f} -> Calibrated OOF AUC: {best_calib_auc:.5f}")

    # 6. Generate Final Test Predictions
    test_blend_raw = np.dot(models_test_matrix, best_w)
    test_final = rankdata(test_blend_raw) / n_test
    test_final_calibrated = (test_final) ** best_gamma
    # Normalize to [0, 1]
    test_final_calibrated = (test_final_calibrated - test_final_calibrated.min()) / (test_final_calibrated.max() - test_final_calibrated.min())

    # 7. Create Submission File
    sub_df = pd.DataFrame({"id": test_ids, "is_fraud": test_final_calibrated})
    sub_path = Path("outputs/submissions/submission_v2.csv")
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv("submission_v2.csv", index=False)

    logger.info("=" * 70)
    logger.info(f"FINAL SUBMISSION GENERATED: submission_v2.csv ({len(sub_df):,} rows)")
    logger.info(f"Total Execution Time: {time.time() - start_total:.1f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
