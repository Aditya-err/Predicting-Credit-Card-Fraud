import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_submission(
    test_ids: pd.Series,
    test_preds: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """Generates and writes a submission CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sub_df = pd.DataFrame({
        "id": test_ids.values,
        "is_fraud": test_preds.astype(float),
    })

    sub_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} with {len(sub_df)} rows.")
    return sub_df


def verify_submission(
    sub_path: str,
    sample_sub_path: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Rigorously verifies submission compliance with Kaggle competition requirements.
    Never submits automatically.
    """
    if not os.path.exists(sub_path):
        return False, {"error": f"Submission file {sub_path} does not exist"}
    if not os.path.exists(sample_sub_path):
        return False, {"error": f"Sample submission {sample_sub_path} does not exist"}

    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_sub_path)

    checks = {
        "file_exists": True,
        "row_count_matches": len(sub) == len(sample),
        "expected_row_count": len(sample),
        "actual_row_count": len(sub),
        "columns_match": list(sub.columns) == ["id", "is_fraud"],
        "no_nulls": int(sub.isnull().sum().sum()) == 0,
        "ids_match_exact_order": bool((sub["id"] == sample["id"]).all()),
        "values_within_0_and_1": bool((sub["is_fraud"] >= 0.0).all() and (sub["is_fraud"] <= 1.0).all()),
        "pred_min": float(sub["is_fraud"].min()),
        "pred_max": float(sub["is_fraud"].max()),
        "pred_mean": float(sub["is_fraud"].mean()),
        "pred_median": float(sub["is_fraud"].median()),
    }

    is_valid = (
        checks["row_count_matches"]
        and checks["columns_match"]
        and checks["no_nulls"]
        and checks["ids_match_exact_order"]
        and checks["values_within_0_and_1"]
    )

    checks["is_valid"] = is_valid
    return is_valid, checks
