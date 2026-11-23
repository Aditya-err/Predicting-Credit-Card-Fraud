import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from typing import Dict, Any, Tuple, Optional
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def build_model(model_type: str, params: Dict[str, Any]) -> Any:
    """Instantiates the specified model family with provided hyperparameters."""
    model_type = model_type.lower()

    if model_type == "lightgbm":
        clean_params = params.copy()
        clean_params.setdefault("objective", "binary")
        clean_params.setdefault("metric", "auc")
        clean_params.setdefault("verbose", -1)
        return lgb.LGBMClassifier(**clean_params)

    elif model_type == "catboost":
        clean_params = params.copy()
        clean_params.setdefault("loss_function", "Logloss")
        clean_params.setdefault("eval_metric", "AUC")
        clean_params.setdefault("verbose", False)
        return cb.CatBoostClassifier(**clean_params)

    elif model_type == "xgboost":
        clean_params = params.copy()
        clean_params.setdefault("objective", "binary:logistic")
        clean_params.setdefault("eval_metric", "auc")
        clean_params.setdefault("tree_method", "hist")
        return xgb.XGBClassifier(**clean_params)

    elif model_type in ["logistic_regression", "ridge"]:
        clean_params = params.copy()
        clean_params.setdefault("penalty", "l2")
        clean_params.setdefault("solver", "lbfgs")
        clean_params.setdefault("max_iter", 1000)
        return LogisticRegression(**clean_params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def fit_model_with_early_stopping(
    model: Any,
    model_type: str,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    early_stopping_rounds: int = 100,
) -> Any:
    """Fits model on training fold with validation-based early stopping where applicable."""
    model_type = model_type.lower()

    if model_type == "lightgbm":
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        return model

    elif model_type == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        return model

    elif model_type == "xgboost":
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    elif model_type in ["logistic_regression", "ridge"]:
        model.fit(X_train, y_train)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")
