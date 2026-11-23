import os
import datetime
from typing import Dict, Any, Optional
import pandas as pd

LOG_COLUMNS = [
    "experiment_id",
    "model",
    "features",
    "cv_mean",
    "cv_std",
    "oof_roc_auc",
    "runtime_seconds",
    "improvement_over_baseline",
    "status",
    "notes",
    "timestamp",
]


def log_experiment(
    log_path: str,
    experiment_id: str,
    model_name: str,
    feature_config: str,
    summary: Dict[str, Any],
    improvement: float,
    status: str = "COMPLETED",
    notes: str = "",
) -> None:
    """Appends an experiment record to the experiment log CSV."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "experiment_id": experiment_id,
        "model": model_name,
        "features": feature_config,
        "cv_mean": round(summary.get("cv_mean", 0.0), 5),
        "cv_std": round(summary.get("cv_std", 0.0), 5),
        "oof_roc_auc": round(summary.get("oof_roc_auc", 0.0), 5),
        "runtime_seconds": summary.get("runtime_seconds", 0.0),
        "improvement_over_baseline": round(improvement, 5),
        "status": status,
        "notes": notes,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not os.path.exists(log_path):
        df = pd.DataFrame([record], columns=LOG_COLUMNS)
    else:
        df = pd.read_csv(log_path)
        # Avoid duplicate entries for same experiment_id by updating or appending
        if experiment_id in df["experiment_id"].values:
            df.loc[df["experiment_id"] == experiment_id] = record
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df.to_csv(log_path, index=False)
