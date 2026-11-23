import pytest
import numpy as np
import pandas as pd
from src.evaluation import evaluate_predictions, compute_cv_summary

def test_evaluate_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["roc_auc"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] > 0.0

def test_compute_cv_summary():
    fold_aucs = [0.952, 0.954, 0.953, 0.951, 0.955]
    summary = compute_cv_summary(fold_aucs, oof_auc=0.953, runtime_sec=12.5)
    
    assert summary["oof_roc_auc"] == 0.953
    assert len(summary["fold_aucs"]) == 5
    assert summary["runtime_seconds"] == 12.5
    assert np.isclose(summary["cv_mean"], 0.953, atol=1e-4)
