import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUMERICAL_INT_COLS = [
    "txn_count_last_24h",
    "card_age_months",
    "customer_age",
    "cvv_retry_count",
    "time_of_day_hour",
    "day_of_week",
    "prior_disputes",
]

NUMERICAL_FLOAT_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]

BOOLEAN_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

CATEGORICAL_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]


def optimize_dtypes(df: pd.DataFrame, is_test: bool = False, target_col: str = "is_fraud") -> pd.DataFrame:
    """Downcasts numerical and boolean columns to minimize memory consumption."""
    df_opt = df.copy()

    for col in NUMERICAL_FLOAT_COLS:
        if col in df_opt.columns:
            df_opt[col] = df_opt[col].astype(np.float32)

    for col in NUMERICAL_INT_COLS:
        if col in df_opt.columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    for col in BOOLEAN_COLS:
        if col in df_opt.columns:
            df_opt[col] = df_opt[col].astype(np.int8)

    if not is_test and target_col in df_opt.columns:
        df_opt[target_col] = df_opt[target_col].astype(np.int8)

    return df_opt


def load_datasets(
    train_path: str,
    test_path: str,
    target_col: str = "is_fraud",
    id_col: str = "id",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads train and test CSV files safely, optimizes memory, and separates target/IDs.
    Never alters original files.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset file not found at {test_path}")

    logger.info("Reading train and test CSV files...")
    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)

    mem_train_before = raw_train.memory_usage(deep=True).sum() / (1024**2)
    mem_test_before = raw_test.memory_usage(deep=True).sum() / (1024**2)

    # Integrity verification
    assert id_col in raw_train.columns, f"ID column {id_col} missing from train"
    assert id_col in raw_test.columns, f"ID column {id_col} missing from test"
    assert target_col in raw_train.columns, f"Target column {target_col} missing from train"

    train_ids = raw_train[id_col].copy()
    test_ids = raw_test[id_col].copy()
    y_train = raw_train[target_col].copy()

    # Verify target distribution
    unique_targets = set(y_train.dropna().unique())
    assert unique_targets.issubset({0, 1, 0.0, 1.0}), f"Target contains invalid values: {unique_targets}"

    # Extract predictor frames
    X_train_raw = raw_train.drop(columns=[id_col, target_col])
    X_test_raw = raw_test.drop(columns=[id_col])

    # Ensure feature alignment
    assert list(X_train_raw.columns) == list(X_test_raw.columns), "Feature columns mismatch between train and test"

    # Memory downcasting
    X_train_opt = optimize_dtypes(X_train_raw, is_test=False, target_col=target_col)
    X_test_opt = optimize_dtypes(X_test_raw, is_test=True, target_col=target_col)
    y_train = y_train.astype(np.int8)

    mem_train_after = X_train_opt.memory_usage(deep=True).sum() / (1024**2)
    mem_test_after = X_test_opt.memory_usage(deep=True).sum() / (1024**2)

    logger.info(f"Train memory: {mem_train_before:.2f} MB -> {mem_train_after:.2f} MB")
    logger.info(f"Test memory: {mem_test_before:.2f} MB -> {mem_test_after:.2f} MB")
    logger.info(f"Loaded {len(X_train_opt)} train rows and {len(X_test_opt)} test rows.")

    return X_train_opt, y_train, X_test_opt, train_ids, test_ids
