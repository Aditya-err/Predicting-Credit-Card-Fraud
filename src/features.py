import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

BOOLEAN_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

NUMERICAL_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]


class RawFeaturePreprocessor:
    """Preprocesses raw features without creating any engineered features."""

    def __init__(self, model_family: str = "lightgbm"):
        self.model_family = model_family.lower()
        self.encoder: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X = X_train.copy()

        if self.model_family in ["lightgbm", "xgboost", "catboost"]:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int32,
            )
            X[CATEGORICAL_COLS] = self.encoder.fit_transform(X[CATEGORICAL_COLS].astype(str))
            self.feature_names = list(X.columns)
            return X

        elif self.model_family in ["logistic_regression", "ridge"]:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
            encoded_cats = self.encoder.fit_transform(X[CATEGORICAL_COLS].astype(str))
            cat_feature_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))

            self.scaler = StandardScaler()
            scaled_nums = self.scaler.fit_transform(X[NUMERICAL_COLS].astype(np.float32))
            bool_vals = X[BOOLEAN_COLS].values.astype(np.float32)

            all_matrix = np.hstack([scaled_nums, bool_vals, encoded_cats])
            self.feature_names = NUMERICAL_COLS + BOOLEAN_COLS + cat_feature_names

            return pd.DataFrame(all_matrix, columns=self.feature_names, index=X.index)

        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X = X_test.copy()

        if self.model_family in ["lightgbm", "xgboost", "catboost"]:
            assert self.encoder is not None, "Encoder not fitted."
            X[CATEGORICAL_COLS] = self.encoder.transform(X[CATEGORICAL_COLS].astype(str))
            return X

        elif self.model_family in ["logistic_regression", "ridge"]:
            assert self.encoder is not None and self.scaler is not None, "Transformers not fitted."
            encoded_cats = self.encoder.transform(X[CATEGORICAL_COLS].astype(str))
            scaled_nums = self.scaler.transform(X[NUMERICAL_COLS].astype(np.float32))
            bool_vals = X[BOOLEAN_COLS].values.astype(np.float32)

            all_matrix = np.hstack([scaled_nums, bool_vals, encoded_cats])
            return pd.DataFrame(all_matrix, columns=self.feature_names, index=X.index)

        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")


def engineer_feature_pack_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Feature Pack 1 (EXP-005 to EXP-007):
    - feat_velocity_x_merchant_risk: velocity_score * merchant_risk_score
    - feat_risk_flag_count: sum of 6 boolean flags
    - feat_amount_to_balance: amount_usd / (account_balance_usd + 1)
    - feat_merchant_x_auth: merchant_category x auth_method
    - feat_channel_x_auth: channel x auth_method
    """
    X = df.copy()

    # Continuous interactions
    X["feat_velocity_x_merchant_risk"] = (
        X["velocity_score"].astype(np.float32) * X["merchant_risk_score"].astype(np.float32)
    )
    X["feat_risk_flag_count"] = X[BOOLEAN_COLS].sum(axis=1).astype(np.int8)
    X["feat_amount_to_balance"] = (
        X["amount_usd"].astype(np.float32) / (X["account_balance_usd"].astype(np.float32) + 1.0)
    )

    # Categorical crosses
    X["feat_merchant_x_auth"] = (
        X["merchant_category"].astype(str) + "__" + X["auth_method"].astype(str)
    )
    X["feat_channel_x_auth"] = (
        X["channel"].astype(str) + "__" + X["auth_method"].astype(str)
    )

    return X


def engineer_feature_pack_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Feature Pack 2 (EXP-011 to EXP-013) with additional domain features:
    - All features from Feature Pack 1
    - feat_is_round_cents: amount ending in .00
    - feat_velocity_per_txn: velocity_score / (txn_count_last_24h + 1)
    - feat_high_severity_flags: ip_country_mismatch + is_ai_generated_scam_attempt + billing_shipping_mismatch
    - feat_log_amount: log1p(amount_usd)
    - feat_log_balance: log1p(account_balance_usd)
    """
    X = engineer_feature_pack_1(df)

    # Cent forensics
    cents = (X["amount_usd"] * 100).round() % 100
    X["feat_is_round_cents"] = (cents == 0).astype(np.int8)

    # Normalized velocity
    X["feat_velocity_per_txn"] = (
        X["velocity_score"].astype(np.float32) / (X["txn_count_last_24h"].astype(np.float32) + 1.0)
    )

    # High severity flags
    X["feat_high_severity_flags"] = (
        X["ip_country_mismatch"].astype(np.int8)
        + X["is_ai_generated_scam_attempt"].astype(np.int8)
        + X["billing_shipping_mismatch"].astype(np.int8)
    )

    # Log transforms for skewed continuous variables
    X["feat_log_amount"] = np.log1p(X["amount_usd"].astype(np.float32))
    X["feat_log_balance"] = np.log1p(X["account_balance_usd"].astype(np.float32))

    return X


class InFoldTargetEncoder:
    """
    Strictly leakage-safe Out-of-Fold smoothed target encoder.
    Computes smoothed posterior target rates inside the training fold and transforms validation/test.
    """

    def __init__(self, target_cols: List[str], smoothing: float = 10.0):
        self.target_cols = target_cols
        self.smoothing = smoothing
        self.global_mean: float = 0.0
        self.stats_dict: Dict[str, Dict[str, float]] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.global_mean = float(y_train.mean())
        self.stats_dict = {}

        for col in self.target_cols:
            df_temp = pd.DataFrame({col: X_train[col], "target": y_train})
            grouped = df_temp.groupby(col)["target"].agg(["count", "mean"])
            counts = grouped["count"]
            means = grouped["mean"]

            # Empirical Bayes smoothing: (n * mean + m * prior) / (n + m)
            smoothed = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
            self.stats_dict[col] = smoothed.to_dict()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(index=X.index)
        for col in self.target_cols:
            mapping = self.stats_dict.get(col, {})
            encoded_col = f"te_{col}"
            X_out[encoded_col] = (
                X[col].map(mapping).fillna(self.global_mean).astype(np.float32)
            )
        return X_out


class AdvancedFeaturePreprocessor:
    """Preprocessor supporting raw + engineered features with encoding options."""

    def __init__(self, model_family: str = "lightgbm", use_target_encoding: bool = False):
        self.model_family = model_family.lower()
        self.use_target_encoding = use_target_encoding
        self.encoder: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.te_encoder: Optional[InFoldTargetEncoder] = None
        self.all_cat_cols: List[str] = []
        self.feature_names: List[str] = []

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        X = X_train.copy()
        self.all_cat_cols = [c for c in X.columns if X[c].dtype == "object" or c in CATEGORICAL_COLS]

        if self.use_target_encoding and y_train is not None:
            self.te_encoder = InFoldTargetEncoder(target_cols=self.all_cat_cols, smoothing=10.0)
            self.te_encoder.fit(X, y_train)
            te_feats = self.te_encoder.transform(X)
            X = pd.concat([X, te_feats], axis=1)

        if self.model_family in ["lightgbm", "xgboost", "catboost"]:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int32,
            )
            X[self.all_cat_cols] = self.encoder.fit_transform(X[self.all_cat_cols].astype(str))
            self.feature_names = list(X.columns)
            return X

        elif self.model_family in ["logistic_regression", "ridge"]:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
            encoded_cats = self.encoder.fit_transform(X[self.all_cat_cols].astype(str))
            cat_feature_names = list(self.encoder.get_feature_names_out(self.all_cat_cols))

            num_cols_present = [c for c in X.columns if c not in self.all_cat_cols and c not in BOOLEAN_COLS]
            self.scaler = StandardScaler()
            scaled_nums = self.scaler.fit_transform(X[num_cols_present].astype(np.float32))

            bool_cols_present = [c for c in BOOLEAN_COLS if c in X.columns]
            bool_vals = X[bool_cols_present].values.astype(np.float32)

            all_matrix = np.hstack([scaled_nums, bool_vals, encoded_cats])
            self.feature_names = num_cols_present + bool_cols_present + cat_feature_names
            return pd.DataFrame(all_matrix, columns=self.feature_names, index=X.index)

        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X = X_test.copy()

        if self.use_target_encoding and self.te_encoder is not None:
            te_feats = self.te_encoder.transform(X)
            X = pd.concat([X, te_feats], axis=1)

        if self.model_family in ["lightgbm", "xgboost", "catboost"]:
            assert self.encoder is not None, "Encoder not fitted."
            X[self.all_cat_cols] = self.encoder.transform(X[self.all_cat_cols].astype(str))
            return X

        elif self.model_family in ["logistic_regression", "ridge"]:
            assert self.encoder is not None and self.scaler is not None, "Transformers not fitted."
            encoded_cats = self.encoder.transform(X[self.all_cat_cols].astype(str))
            num_cols_present = [c for c in X.columns if c not in self.all_cat_cols and c not in BOOLEAN_COLS]
            scaled_nums = self.scaler.transform(X[num_cols_present].astype(np.float32))
            bool_cols_present = [c for c in BOOLEAN_COLS if c in X.columns]
            bool_vals = X[bool_cols_present].values.astype(np.float32)

            all_matrix = np.hstack([scaled_nums, bool_vals, encoded_cats])
            return pd.DataFrame(all_matrix, columns=self.feature_names, index=X.index)

        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")


def audit_rare_combinations(train_df: pd.DataFrame, test_df: pd.DataFrame, min_freq: int = 50) -> Dict[str, Any]:
    """Analyzes category cardinality and low-frequency intersections across train and test."""
    prep_train = engineer_feature_pack_1(train_df)
    prep_test = engineer_feature_pack_1(test_df)

    audit_out = {}
    for col in ["feat_merchant_x_auth", "feat_channel_x_auth"]:
        tr_counts = prep_train[col].value_counts()
        te_counts = prep_test[col].value_counts()
        rare_in_train = tr_counts[tr_counts < min_freq].to_dict()
        unseen_in_test = list(set(te_counts.index) - set(tr_counts.index))

        audit_out[col] = {
            "total_unique": len(tr_counts),
            "rare_under_threshold": rare_in_train,
            "unseen_in_test": unseen_in_test,
        }

    return audit_out
