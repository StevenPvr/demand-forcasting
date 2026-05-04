from __future__ import annotations

from typing import Any, cast

from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_XGBOOST_CUDA_FOLD_WORKERS,
    DEFAULT_XGBOOST_GPU_INPUT_BACKEND,
    DEFAULT_XGBOOST_CUDA_MAX_THREADS,
    DEFAULT_XGBOOST_CUDA_MATRIX_TYPE,
    DEFAULT_XGBOOST_MAX_CAT_THRESHOLD,
    DEFAULT_XGBOOST_MAX_CAT_TO_ONEHOT,
    DEFAULT_XGBOOST_N_ESTIMATORS,
    DEFAULT_XGBOOST_TUNING_EARLY_STOPPING_ROUNDS,
)
from praedixa.demand_forecast.backends.xgboost.runtime import (
    SUPPORTED_XGBOOST_RUNTIME_PROFILES,
    XGBOOST_CUDA_RUNTIME_PROFILES,
    resolve_xgboost_runtime_profile,
    xgboost_uses_cuda,
)


DEFAULT_XGBOOST_MODEL_PARAMS: dict[str, object] = {
    "model_backend": "xgboost",
    "runtime_profile": "local_cpu",
    "device": "cpu",
    "xgboost_matrix_type": "dmatrix",
    "xgboost_gpu_input_backend": "cpu",
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
    "bakery_sample_weight_multiplier": 1.0,
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
    resolved["tree_method"] = "hist"
    runtime_profile = str(resolved.get("runtime_profile", "local_cpu")).lower()
    requested_device = str(resolved.get("device", "")).lower()
    should_resolve_runtime = (
        runtime_profile == "auto"
        or requested_device == "cuda"
        or runtime_profile in XGBOOST_CUDA_RUNTIME_PROFILES
    )
    if runtime_profile not in SUPPORTED_XGBOOST_RUNTIME_PROFILES:
        resolve_xgboost_runtime_profile(runtime_profile)
    if should_resolve_runtime:
        requested_profile = runtime_profile
        if requested_device == "cuda" and runtime_profile in {"", "auto", "local_cpu"}:
            requested_profile = "cuda"
        runtime_resolution = resolve_xgboost_runtime_profile(requested_profile)
        resolved["runtime_profile"] = runtime_resolution.runtime_profile
        resolved["device"] = runtime_resolution.device
        resolved["tree_method"] = runtime_resolution.tree_method
        resolved["accelerator"] = runtime_resolution.accelerator
        resolved["devices"] = runtime_resolution.devices
        resolved["cuda_available"] = runtime_resolution.cuda_available
        resolved["cuda_device_name"] = runtime_resolution.cuda_device_name
        if runtime_resolution.fallback_reason is not None:
            resolved["runtime_fallback_reason"] = runtime_resolution.fallback_reason
    if xgboost_uses_cuda(resolved):
        requested_matrix_type = (model_params or {}).get(
            "xgboost_matrix_type",
            DEFAULT_XGBOOST_CUDA_MATRIX_TYPE,
        )
        requested_gpu_input_backend = (model_params or {}).get(
            "xgboost_gpu_input_backend",
            DEFAULT_XGBOOST_GPU_INPUT_BACKEND,
        )
        if str(requested_gpu_input_backend).lower() != "cudf":
            resolved["enable_categorical"] = False
        requested_runtime_profile = str(resolved.get("runtime_profile", "cuda"))
        if requested_runtime_profile == "local_cpu":
            requested_runtime_profile = "cuda"
        resolved["device"] = "cuda"
        resolved["runtime_profile"] = requested_runtime_profile
        resolved["n_jobs"] = max(
            1,
            min(int(cast(Any, resolved["n_jobs"])), DEFAULT_XGBOOST_CUDA_MAX_THREADS),
        )
        resolved["max_parallel_fold_workers"] = DEFAULT_XGBOOST_CUDA_FOLD_WORKERS
        resolved["xgboost_matrix_type"] = str(requested_matrix_type)
        resolved["xgboost_gpu_input_backend"] = str(requested_gpu_input_backend)
    else:
        resolved["device"] = "cpu"
        resolved["xgboost_matrix_type"] = "dmatrix"
        resolved["xgboost_gpu_input_backend"] = "cpu"
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
