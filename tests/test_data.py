import os
import pytest
import numpy as np
import pandas as pd
from src.data import optimize_dtypes, load_datasets

def test_optimize_dtypes():
    sample_df = pd.DataFrame({
        "amount_usd": [10.5, 20.0, 99.99],
        "hours_since_last_txn": [1.0, 2.5, 0.5],
        "distance_from_home_km": [5.0, 10.0, 15.0],
        "account_balance_usd": [1000.0, 2000.0, 500.0],
        "velocity_score": [12.0, 15.5, 20.0],
        "merchant_risk_score": [30.0, 45.0, 50.0],
        "txn_count_last_24h": [1, 2, 3],
        "card_age_months": [12, 24, 36],
        "customer_age": [30, 45, 60],
        "cvv_retry_count": [0, 1, 0],
        "time_of_day_hour": [10, 14, 22],
        "day_of_week": [1, 3, 5],
        "prior_disputes": [0, 1, 0],
        "is_foreign_transaction": [False, True, False],
        "is_fraud": [0, 1, 0]
    })
    
    opt_df = optimize_dtypes(sample_df, is_test=False)
    assert opt_df["amount_usd"].dtype == np.float32
    assert opt_df["is_fraud"].dtype == np.int8
    assert opt_df["is_foreign_transaction"].dtype == np.int8
    assert opt_df["customer_age"].dtype in [np.int8, np.int16, np.int32]
