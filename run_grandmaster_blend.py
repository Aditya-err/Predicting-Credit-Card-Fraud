import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("grandmaster_blend")

def main():
    logger.info("=" * 70)
    logger.info("GRANDMASTER MULTI-MODEL RANK FUSION OPTIMIZATION")
    logger.info("=" * 70)

    # 1. Load ground truth target
    train_df = pd.read_csv("playground-series-s6e11/train.csv", usecols=["id", "is_fraud"])
    y = train_df["is_fraud"].values
    n_train = len(y)

    sample_sub = pd.read_csv("playground-series-s6e11/sample_submission.csv")
    test_ids = sample_sub["id"].values
    n_test = len(test_ids)

    # 2. Model inventory definition
    model_configs = [
        {
            "name": "LightGBM_Pack3",
            "oof_path": "outputs/oof/oof_EXP-018_lgb_p3.csv",
            "oof_col": "pred",
            "test_path": "outputs/predictions/preds_EXP-018_lgb_p3.csv",
            "test_col": "is_fraud",
        },
        {
            "name": "XGBoost_Pack3",
            "oof_path": "outputs/oof/oof_EXP-018_xgb_p3.csv",
            "oof_col": "pred",
            "test_path": "outputs/predictions/preds_EXP-018_xgb_p3.csv",
            "test_col": "is_fraud",
        },
        {
            "name": "PyTorch_Tabular_MLP",
            "oof_path": "outputs/oof/oof_EXP-018_mlp.csv",
            "oof_col": "pred",
            "test_path": "outputs/predictions/preds_EXP-018_mlp.csv",
            "test_col": "is_fraud",
        },
        {
            "name": "MultiSeed_LightGBM",
            "oof_path": "outputs/oof/oof_EXP-015_multiseed_lgb.csv",
            "oof_col": "oof_pred",
            "test_path": "outputs/predictions/preds_EXP-015_multiseed_lgb.csv",
            "test_col": "test_pred",
        },
        {
            "name": "Shallow_CatBoost",
            "oof_path": "outputs/oof/oof_EXP-009_shallow_catboost.csv",
            "oof_col": "oof_pred",
            "test_path": "outputs/predictions/preds_EXP-009_shallow_catboost.csv",
            "test_col": "test_pred",
        },
        {
            "name": "Shallow_LightGBM",
            "oof_path": "outputs/oof/oof_EXP-008_shallow_lightgbm.csv",
            "oof_col": "oof_pred",
            "test_path": "outputs/predictions/preds_EXP-008_shallow_lightgbm.csv",
            "test_col": "test_pred",
        },
    ]

    # 3. Load OOF and test predictions, rank transform
    oof_rank_cols = []
    test_rank_cols = []
    model_names = []

    logger.info("Individual Model Out-of-Fold Performance:")
    for cfg in model_configs:
        name = cfg["name"]
        oof_df = pd.read_csv(cfg["oof_path"])
        test_df = pd.read_csv(cfg["test_path"])

        oof_raw = oof_df[cfg["oof_col"]].values
        test_raw = test_df[cfg["test_col"]].values

        auc_ind = roc_auc_score(y, oof_raw)
        logger.info(f"  {name:25s} | OOF ROC AUC: {auc_ind:.5f}")

        # Uniform percentile rank transformation
        r_oof = rankdata(oof_raw) / n_train
        r_test = rankdata(test_raw) / n_test

        oof_rank_cols.append(r_oof)
        test_rank_cols.append(r_test)
        model_names.append(name)

    OOF_M = np.column_stack(oof_rank_cols)
    TEST_M = np.column_stack(test_rank_cols)
    n_models = len(model_names)

    # 4. Nelder-Mead Optimization for Ensemble Weights
    logger.info("-" * 50)
    logger.info(f"Solving optimal blend weights for {n_models} diverse model families...")

    def loss_func(w):
        w_norm = np.clip(w, 0, None)
        if np.sum(w_norm) == 0:
            return 0.0
        w_norm = w_norm / np.sum(w_norm)
        blend = np.dot(OOF_M, w_norm)
        return -roc_auc_score(y, blend)

    init_w = np.ones(n_models) / n_models
    res = minimize(loss_func, init_w, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-5})
    best_w = np.clip(res.x, 0, None)
    best_w = best_w / np.sum(best_w)

    logger.info("Optimal Ensembling Weights:")
    for name, weight in zip(model_names, best_w):
        logger.info(f"  {name:25s}: {weight * 100:6.2f}%")

    oof_blend = np.dot(OOF_M, best_w)
    base_ensemble_auc = roc_auc_score(y, oof_blend)
    logger.info(f"Grandmaster Rank Blend OOF ROC AUC: {base_ensemble_auc:.6f}")

    # 5. Non-linear Power Calibration (R^gamma)
    oof_ranked = rankdata(oof_blend) / n_train
    best_gamma = 1.0
    best_calib_auc = base_ensemble_auc

    for gamma in [0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10]:
        cand_preds = oof_ranked ** gamma
        auc_cand = roc_auc_score(y, cand_preds)
        if auc_cand > best_calib_auc:
            best_calib_auc = auc_cand
            best_gamma = gamma

    logger.info(f"Power Transform Gamma: {best_gamma:.2f} -> Calibrated OOF AUC: {best_calib_auc:.6f}")

    # 6. Generate Test Predictions
    test_blend = np.dot(TEST_M, best_w)
    test_ranked = rankdata(test_blend) / n_test
    test_calib = test_ranked ** best_gamma

    # Normalize to continuous probabilities [0, 1]
    final_test_preds = (test_calib - test_calib.min()) / (test_calib.max() - test_calib.min())

    # 7. Strict Submission Verification
    sub_df = pd.DataFrame({"id": test_ids, "is_fraud": final_test_preds})

    # Save to outputs and root
    sub_path = Path("outputs/submissions/submission_v2.csv")
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv("submission_v2.csv", index=False)

    assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
    assert list(sub_df.columns) == ["id", "is_fraud"], f"Columns mismatch: {sub_df.columns.tolist()}"
    assert sub_df["is_fraud"].isna().sum() == 0, "Submission contains NaN values!"
    assert sub_df["is_fraud"].min() >= 0.0 and sub_df["is_fraud"].max() <= 1.0, "Predictions out of bounds!"
    assert (sub_df["id"].values == sample_sub["id"].values).all(), "ID alignment mismatch!"

    logger.info("=" * 70)
    logger.info("FINAL SUBMISSION VERIFICATION: PASS")
    logger.info(f"  Saved file 1: {sub_path} ({len(sub_df):,} rows)")
    logger.info(f"  Saved file 2: submission_v2.csv ({len(sub_df):,} rows)")
    logger.info(f"  Min prediction: {final_test_preds.min():.6f}, Max: {final_test_preds.max():.6f}")
    logger.info(f"  Previous EXP-017 OOF AUC: 0.947027 (Kaggle Public: 0.94801)")
    logger.info(f"  New Grandmaster Blend OOF AUC: {best_calib_auc:.6f}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
