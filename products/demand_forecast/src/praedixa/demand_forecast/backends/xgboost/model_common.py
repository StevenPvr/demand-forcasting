from __future__ import annotations

from typing import Any, cast

from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_XGBOOST_MAX_CAT_THRESHOLD,
    DEFAULT_XGBOOST_MAX_CAT_TO_ONEHOT,
    DEFAULT_XGBOOST_N_ESTIMATORS,
    DEFAULT_XGBOOST_TUNING_EARLY_STOPPING_ROUNDS,
)


DEFAULT_XGBOOST_MODEL_PARAMS: dict[str, object] = {
    "model_backend": "xgboost",
    "runtime_profile": "local_cpu",
    "device": "cpu",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "enable_categorical": True,
    "n_estimators": DEFAULT_XGBOOST_N_ESTIMATORS,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 64.0,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 1e-2,
    "reg_lambda": 20.0,
    "max_bin": 256,
    "max_cat_to_onehot": DEFAULT_XGBOOST_MAX_CAT_TO_ONEHOT,
    "max_cat_threshold": DEFAULT_XGBOOST_MAX_CAT_THRESHOLD,
    "early_stopping_rounds": DEFAULT_XGBOOST_TUNING_EARLY_STOPPING_ROUNDS,
    "enable_early_stopping": True,
    "enable_dataset_sample_weight": False,
    "verbosity": 0,
    "n_jobs": 1,
    "random_state": 42,
}

XGBOOST_TRAIN_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "objective",
        "eval_metric",
        "device",
        "tree_method",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "max_bin",
        "max_cat_to_onehot",
        "max_cat_threshold",
        "max_leaves",
        "grow_policy",
        "verbosity",
    }
)


def resolve_xgboost_model_params(
    model_params: dict[str, object] | None,
    *,
    default_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Resolve les parametres runtime en ignorant les cles d'orchestration non XGBoost."""

    resolved = {
        **(default_params or DEFAULT_XGBOOST_MODEL_PARAMS),
        **(model_params or {}),
    }
    resolved["model_backend"] = "xgboost"
    return resolved


def xgboost_num_boost_round(model_params: dict[str, object]) -> int:
    return int(cast(Any, model_params["n_estimators"]))


def xgboost_early_stopping_rounds(model_params: dict[str, object]) -> int | None:
    if not bool(model_params.get("enable_early_stopping", False)):
        return None
    return int(cast(Any, model_params["early_stopping_rounds"]))


def xgboost_train_params(model_params: dict[str, object]) -> dict[str, object]:
    train_params = {
        key: value
        for key, value in model_params.items()
        if key in XGBOOST_TRAIN_PARAM_KEYS and value is not None
    }
    train_params["max_depth"] = int(cast(Any, train_params["max_depth"]))
    train_params["max_bin"] = int(cast(Any, train_params["max_bin"]))
    train_params["nthread"] = int(cast(Any, model_params["n_jobs"]))
    train_params["seed"] = int(cast(Any, model_params["random_state"]))
    train_params["alpha"] = float(cast(Any, train_params.pop("reg_alpha")))
    train_params["lambda"] = float(cast(Any, train_params.pop("reg_lambda")))
    if "max_leaves" in train_params:
        train_params["max_leaves"] = int(cast(Any, train_params["max_leaves"]))
    if not bool(model_params.get("enable_categorical", False)):
        train_params.pop("max_cat_to_onehot", None)
        train_params.pop("max_cat_threshold", None)
        return train_params
    train_params["max_cat_to_onehot"] = int(
        cast(Any, train_params["max_cat_to_onehot"])
    )
    train_params["max_cat_threshold"] = int(
        cast(Any, train_params["max_cat_threshold"])
    )
    return train_params
