import pytest
import numpy as np
import pandas as pd
from src.features import engineer_feature_pack_1, engineer_feature_pack_2, AdvancedFeaturePreprocessor
from src.ensembling import optimize_ensemble_weights, compute_rank_average_blend, compute_weighted_probability_blend

def test_pipeline_feature_packs_and_preprocessors():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "amount_usd": np.random.uniform(5, 500, n),
        "hours_since_last_txn": np.random.uniform(0.1, 24, n),
        "txn_count_last_24h": np.random.randint(0, 10, n),
        "distance_from_home_km": np.random.uniform(0, 50, n),
        "card_age_months": np.random.randint(12, 60, n),
        "customer_age": np.random.randint(18, 80, n),
        "account_balance_usd": np.random.uniform(100, 10000, n),
        "cvv_retry_count": np.random.randint(0, 3, n),
        "velocity_score": np.random.uniform(0, 60, n),
        "time_of_day_hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "merchant_risk_score": np.random.uniform(0, 100, n),
        "prior_disputes": np.random.randint(0, 4, n),
        "merchant_category": np.random.choice(["Groceries", "Online Retail"], n),
        "card_type": np.random.choice(["Visa", "Mastercard"], n),
        "auth_method": np.random.choice(["PIN", "OTP"], n),
        "channel": np.random.choice(["Online", "POS"], n),
        "device_type": np.random.choice(["iPhone", "Android Phone"], n),
        "is_foreign_transaction": np.random.choice([0, 1], n),
        "is_new_merchant": np.random.choice([0, 1], n),
        "used_vpn": np.random.choice([0, 1], n),
        "ip_country_mismatch": np.random.choice([0, 1], n),
        "billing_shipping_mismatch": np.random.choice([0, 1], n),
        "is_ai_generated_scam_attempt": np.random.choice([0, 1], n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.9, 0.1]))

    df_p1 = engineer_feature_pack_1(df)
    assert "feat_velocity_x_merchant_risk" in df_p1.columns
    assert "feat_risk_flag_count" in df_p1.columns

    df_p2 = engineer_feature_pack_2(df)
    assert "feat_is_round_cents" in df_p2.columns
    assert "feat_log_amount" in df_p2.columns

    prep = AdvancedFeaturePreprocessor(model_family="lightgbm", use_target_encoding=True)
    X_enc = prep.fit_transform(df_p2, y)
    assert "te_merchant_category" in X_enc.columns
    assert X_enc.shape[0] == n

    prep_lr = AdvancedFeaturePreprocessor(model_family="logistic_regression", use_target_encoding=False)
    X_enc_lr = prep_lr.fit_transform(df_p2, y)
    assert X_enc_lr.shape[0] == n
    assert not np.isnan(X_enc_lr.values).any()
