from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates official ROC AUC alongside auxiliary diagnostic metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred)),
    }


def compute_cv_summary(fold_aucs: List[float], oof_auc: float, runtime_sec: float) -> Dict[str, Any]:
    """Aggregates fold-level metrics into a structured summary dict."""
    return {
        "fold_aucs": [round(s, 5) for s in fold_aucs],
        "cv_mean": float(np.mean(fold_aucs)),
        "cv_std": float(np.std(fold_aucs)),
        "cv_min": float(np.min(fold_aucs)),
        "cv_max": float(np.max(fold_aucs)),
        "oof_roc_auc": float(oof_auc),
        "runtime_seconds": round(float(runtime_sec), 2),
    }
