import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def to_rank_percentiles(preds: np.ndarray) -> np.ndarray:
    """Converts continuous predictions into [0, 1] rank percentiles."""
    ranks = stats.rankdata(preds, method="average")
    if len(ranks) > 1:
        return (ranks - 1.0) / (len(ranks) - 1.0)
    return ranks


def compute_weighted_probability_blend(
    pred_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Computes a linear weighted blend of raw probabilities."""
    n_models = len(pred_list)
    if weights is None:
        weights = [1.0 / n_models] * n_models

    w = np.array(weights) / np.sum(weights)
    blend = np.zeros_like(pred_list[0], dtype=np.float64)
    for p, weight in zip(pred_list, w):
        blend += weight * p
    return blend.astype(np.float32)


def compute_rank_average_blend(
    pred_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Computes a weighted percentile rank average across multiple model prediction arrays."""
    n_models = len(pred_list)
    if weights is None:
        weights = [1.0 / n_models] * n_models

    w = np.array(weights) / np.sum(weights)
    rank_blend = np.zeros_like(pred_list[0], dtype=np.float64)

    for p, weight in zip(pred_list, w):
        r = to_rank_percentiles(p)
        rank_blend += weight * r

    # Normalize back to exact rank percentiles
    return to_rank_percentiles(rank_blend).astype(np.float32)


def optimize_ensemble_weights(
    oof_preds_list: List[np.ndarray],
    y_true: np.ndarray,
    method: str = "rank",
) -> Tuple[List[float], float]:
    """
    Optimizes ensemble weights using Nelder-Mead on OOF predictions exclusively.
    Ensures non-negative weights that sum to 1.
    """
    n_models = len(oof_preds_list)
    init_weights = np.ones(n_models) / n_models

    def loss_func(weights):
        # Enforce non-negativity and normalize
        w = np.maximum(weights, 0)
        total = np.sum(w)
        if total == 0:
            return 1.0
        w = w / total

        if method == "rank":
            blend = compute_rank_average_blend(oof_preds_list, w)
        else:
            blend = compute_weighted_probability_blend(oof_preds_list, w)

        # Minimize negative ROC AUC
        return -float(roc_auc_score(y_true, blend))

    bounds = [(0.0, 1.0)] * n_models
    res = minimize(
        loss_func,
        init_weights,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
    )

    opt_weights = np.maximum(res.x, 0)
    opt_weights = (opt_weights / np.sum(opt_weights)).tolist()
    best_auc = -res.fun

    logger.info(f"Optimized weights ({method}): {[round(w, 4) for w in opt_weights]} -> OOF ROC AUC: {best_auc:.5f}")
    return opt_weights, best_auc
