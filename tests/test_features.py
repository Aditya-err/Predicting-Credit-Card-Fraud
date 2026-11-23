import pytest
import numpy as np
import pandas as pd
from src.features import (
    RawFeaturePreprocessor,
    AdvancedFeaturePreprocessor,
    InFoldTargetEncoder,
    engineer_feature_pack_1,
    engineer_feature_pack_2,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    BOOLEAN_COLS,
)

@pytest.fixture
def dummy_train_test():
    np.random.seed(42)
    n = 100
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
        "merchant_category": np.random.choice(["Groceries", "Online Retail", "Electronics"], n),
        "card_type": np.random.choice(["Visa", "Mastercard", "Amex"], n),
        "auth_method": np.random.choice(["PIN", "OTP", "No Authentication"], n),
        "channel": np.random.choice(["Online", "POS", "Contactless"], n),
        "device_type": np.random.choice(["iPhone", "Android Phone", "Windows PC"], n),
        "is_foreign_transaction": np.random.choice([0, 1], n),
        "is_new_merchant": np.random.choice([0, 1], n),
        "used_vpn": np.random.choice([0, 1], n),
        "ip_country_mismatch": np.random.choice([0, 1], n),
        "billing_shipping_mismatch": np.random.choice([0, 1], n),
        "is_ai_generated_scam_attempt": np.random.choice([0, 1], n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.9, 0.1]))
    return df.iloc[:80], df.iloc[80:], y.iloc[:80]

def test_raw_preprocessor_lightgbm(dummy_train_test):
    tr, te, _ = dummy_train_test
    prep = RawFeaturePreprocessor(model_family="lightgbm")
    tr_enc = prep.fit_transform(tr)
    te_enc = prep.transform(te)
    
    assert tr_enc.shape[1] == tr.shape[1]
    assert te_enc.shape[1] == te.shape[1]

def test_feature_pack_1(dummy_train_test):
    tr, te, _ = dummy_train_test
    eng_tr = engineer_feature_pack_1(tr)
    eng_te = engineer_feature_pack_1(te)
    
    for c in ["feat_velocity_x_merchant_risk", "feat_risk_flag_count", "feat_amount_to_balance", "feat_merchant_x_auth", "feat_channel_x_auth"]:
        assert c in eng_tr.columns
        assert c in eng_te.columns

def test_feature_pack_2(dummy_train_test):
    tr, te, _ = dummy_train_test
    eng_tr = engineer_feature_pack_2(tr)
    eng_te = engineer_feature_pack_2(te)
    
    for c in ["feat_is_round_cents", "feat_velocity_per_txn", "feat_high_severity_flags", "feat_log_amount", "feat_log_balance"]:
        assert c in eng_tr.columns
        assert c in eng_te.columns

def test_infold_target_encoder(dummy_train_test):
    tr, te, y = dummy_train_test
    te_enc = InFoldTargetEncoder(target_cols=["merchant_category"], smoothing=10.0)
    te_enc.fit(tr, y)
    out_tr = te_enc.transform(tr)
    out_te = te_enc.transform(te)
    
    assert "te_merchant_category" in out_tr.columns
    assert "te_merchant_category" in out_te.columns
    assert not out_tr.isnull().any().any()
    assert not out_te.isnull().any().any()
