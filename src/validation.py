import time
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.features import RawFeaturePreprocessor
from src.models import build_model, fit_model_with_early_stopping
from src.evaluation import evaluate_predictions, compute_cv_summary

logger = logging.getLogger(__name__)


def run_stratified_cv(
    model_type: str,
    model_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
    early_stopping_rounds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], pd.DataFrame]:
    """
    Executes a strict Stratified 5-Fold Cross-Validation run.
    Ensures zero target leakage by fitting preprocessors strictly inside train folds.
    Preserves exact row ordering for OOF predictions.
    """
    start_time = time.time()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    oof_preds = np.zeros(len(X_train), dtype=np.float32)
    test_preds_folds = np.zeros((n_splits, len(X_test)), dtype=np.float32)
    fold_aucs: List[float] = []
    feature_importances_list: List[np.ndarray] = []
    feature_names: List[str] = []

    logger.info(f"Starting {n_splits}-Fold Stratified CV for {model_type}...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        fold_start = time.time()

        # Split raw folds
        X_tr_raw = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va_raw = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        # Leakage-Safe in-fold preprocessing
        preprocessor = RawFeaturePreprocessor(model_family=model_type)
        X_tr = preprocessor.fit_transform(X_tr_raw)
        X_va = preprocessor.transform(X_va_raw)
        X_te = preprocessor.transform(X_test)

        if fold == 0:
            feature_names = preprocessor.feature_names

        # Build and train model
        model = build_model(model_type, model_params)
        model = fit_model_with_early_stopping(
            model=model,
            model_type=model_type,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            early_stopping_rounds=early_stopping_rounds,
        )

        # Predict probabilities
        val_pred = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof_preds[val_idx] = val_pred

        test_pred = model.predict_proba(X_te)[:, 1].astype(np.float32)
        test_preds_folds[fold] = test_pred

        # Evaluate fold
        fold_metrics = evaluate_predictions(y_va.values, val_pred)
        fold_auc = fold_metrics["roc_auc"]
        fold_aucs.append(fold_auc)

        fold_elapsed = time.time() - fold_start
        logger.info(
            f"Fold {fold + 1}/{n_splits} - ROC AUC: {fold_auc:.5f} - Time: {fold_elapsed:.1f}s"
        )

        # Collect importance
        if hasattr(model, "feature_importances_"):
            feature_importances_list.append(model.feature_importances_)
        elif hasattr(model, "coef_"):
            feature_importances_list.append(np.abs(model.coef_[0]))

    total_time = time.time() - start_time
    overall_metrics = evaluate_predictions(y_train.values, oof_preds)
    oof_auc = overall_metrics["roc_auc"]
    test_preds_mean = np.mean(test_preds_folds, axis=0)

    summary = compute_cv_summary(fold_aucs, oof_auc, total_time)
    logger.info(
        f"==> Overall OOF ROC AUC: {oof_auc:.5f} | Fold Mean: {summary['cv_mean']:.5f} +/- {summary['cv_std']:.5f} | Total Time: {total_time:.1f}s"
    )

    # Aggregate feature importance
    if feature_importances_list and feature_names:
        avg_importance = np.mean(feature_importances_list, axis=0)
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": avg_importance
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    else:
        fi_df = pd.DataFrame(columns=["feature", "importance"])

    return oof_preds, test_preds_mean, summary, fi_df
