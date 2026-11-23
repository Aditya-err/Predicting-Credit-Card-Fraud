import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.validation import run_stratified_cv

def test_stratified_kfold_proportions():
    np.random.seed(42)
    n = 1000
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.98, 0.02]))
    X = pd.DataFrame({"feat1": np.random.randn(n), "feat2": np.random.randn(n)})
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        val_fraud_rate = y.iloc[val_idx].mean()
        tr_fraud_rate = y.iloc[tr_idx].mean()
        assert np.isclose(val_fraud_rate, tr_fraud_rate, atol=0.01)
        assert len(set(tr_idx).intersection(set(val_idx))) == 0
