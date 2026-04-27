from __future__ import annotations

import logging
import platform
from typing import Any

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.constants import (
    DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_REMOVED_MODEL_INPUT_COLS,
    DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS,
)
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    reconstruct_absolute_predictions,
)
from praedixa.demand_forecast.backends.tft.model_utils import (
    DEFAULT_TFT_MODEL_PARAMS,
    fit_tft_model,
    predict_quantiles_with_tft_model,
    predict_with_tft_model,
    select_tft_feature_columns,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import (
    TFT_EXPLICIT_ROLE_BY_COLUMN,
)
from praedixa.demand_forecast.backends.xgboost.model_common import (
    DEFAULT_XGBOOST_MODEL_PARAMS,
)
from praedixa.demand_forecast.backends.xgboost.model_fit import (
    FittedXGBoostModel,
    fit_xgboost_model,
    predict_with_xgboost_model,
)

LOGGER = logging.getLogger(__name__)
_native_categorical_eval_override_logged: bool = False


def _model_param_as_int(params: dict[str, object], key: str) -> int:
    value = params[key]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        return int(value)
    raise TypeError(
        f"Model parameter `{key}` must be int-compatible, got {type(value).__name__}."
    )


def compute_equal_dataset_row_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> np.ndarray | None:
    if dataset_source_col not in frame.columns:
        return None
    dataset_sources = (
        frame[dataset_source_col].astype("string").fillna("<NA>").astype(str)
    )
    dataset_counts = dataset_sources.value_counts(dropna=False)
    dataset_count = max(1, len(dataset_counts))
    relative_weights = np.asarray(
        [
            1.0 / (dataset_count * float(dataset_counts.loc[dataset]))
            for dataset in dataset_sources.tolist()
        ],
        dtype=float,
    )
    return relative_weights * float(len(frame))


def drop_constant_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    # Keep constant metadata so evaluation matches the TFT training feature space.
    _ = frame
    return list(feature_cols), []


def select_feature_columns(
    train_frame: pd.DataFrame,
    *,
    learning_target_col: str,
    absolute_target_col: str,
    model_backend: str = "tft",
) -> tuple[list[str], list[str], list[str]]:
    identifier_exclusions = _identifier_feature_exclusions(model_backend)
    selection_frame = _feature_selection_frame(train_frame, model_backend=model_backend)
    numeric_feature_cols = select_tft_feature_columns(
        selection_frame,
        excluded_cols={
            DEFAULT_DATE_COL,
            learning_target_col,
            absolute_target_col,
            *DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS,
            *identifier_exclusions,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
            *DEFAULT_REMOVED_MODEL_INPUT_COLS,
        },
    )
    numeric_feature_cols = _with_xgboost_identifier_features(
        numeric_feature_cols,
        train_frame=train_frame,
        model_backend=model_backend,
    )
    identifier_feature_cols = [
        column
        for column in DEFAULT_IDENTIFIER_FEATURE_COLS
        if column in train_frame.columns and column not in numeric_feature_cols
    ]
    filtered_feature_cols, constant_feature_cols = drop_constant_feature_columns(
        train_frame, numeric_feature_cols
    )
    return filtered_feature_cols, constant_feature_cols, identifier_feature_cols


def _feature_selection_frame(
    train_frame: pd.DataFrame,
    *,
    model_backend: str,
) -> pd.DataFrame:
    if model_backend not in {"xgboost", "chronos2"}:
        return train_frame
    mapped_columns = [
        column
        for column in train_frame.columns
        if column in TFT_EXPLICIT_ROLE_BY_COLUMN
    ]
    return train_frame.loc[:, mapped_columns]


def _identifier_feature_exclusions(model_backend: str) -> tuple[str, ...]:
    if model_backend == "xgboost":
        xgboost_identifiers = set(DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS)
        return tuple(
            column
            for column in DEFAULT_IDENTIFIER_FEATURE_COLS
            if column not in xgboost_identifiers
        )
    return DEFAULT_IDENTIFIER_FEATURE_COLS


def _with_xgboost_identifier_features(
    feature_cols: list[str],
    *,
    train_frame: pd.DataFrame,
    model_backend: str,
) -> list[str]:
    if model_backend != "xgboost":
        return feature_cols
    resolved = list(feature_cols)
    for column in DEFAULT_XGBOOST_IDENTIFIER_FEATURE_COLS:
        if column in train_frame.columns and column not in resolved:
            resolved.append(column)
    return resolved


def fit_evaluation_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[Any, int]:
    learning_target_col = target_contract.learning_target_col
    train_weights = compute_equal_dataset_row_weights(
        train_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL
    )
    valid_weights = compute_equal_dataset_row_weights(
        valid_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL
    )
    train_for_model = train_frame.copy()
    valid_for_model = valid_frame.copy()
    model = fit_tft_model(
        train_for_model,
        feature_cols,
        target_col=learning_target_col,
        model_params=model_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=2000,
        valid_frame=valid_for_model,
        train_weights=train_weights,
        valid_weights=valid_weights,
    )
    best_iteration = getattr(model, "best_iteration", None)
    resolved_rounds = (
        int(best_iteration + 1)
        if best_iteration is not None and best_iteration >= 0
        else 2000
    )
    return model, resolved_rounds


def predict_absolute(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> np.ndarray:
    predictions = predict_with_tft_model(
        model,
        frame,
        feature_cols,
        fallback_policy="raise",
    )
    return reconstruct_absolute_predictions(predictions, frame, target_contract)


def predict_absolute_quantiles(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> pd.DataFrame:
    quantile_predictions = predict_quantiles_with_tft_model(
        model,
        frame,
        feature_cols,
        fallback_policy="raise",
    )
    absolute_quantiles = {
        column: reconstruct_absolute_predictions(
            quantile_predictions[column].to_numpy(dtype=float),
            frame,
            target_contract,
        )
        for column in quantile_predictions.columns
    }
    return pd.DataFrame(absolute_quantiles)


def fit_final_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    num_boost_round: int,
) -> Any:
    learning_target_col = target_contract.learning_target_col
    fit_weights = compute_equal_dataset_row_weights(
        fit_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL
    )
    fit_for_model = fit_frame.copy()
    return fit_tft_model(
        fit_for_model,
        feature_cols,
        target_col=learning_target_col,
        model_params=model_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=num_boost_round,
        train_weights=fit_weights,
    )


def fit_xgboost_evaluation_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[FittedXGBoostModel, int]:
    learning_target_col = target_contract.learning_target_col
    resolved_model_params = _stable_xgboost_evaluation_params(model_params)
    LOGGER.debug(
        "XGBoost evaluation backend fit requested: train_rows=%s valid_rows=%s "
        "feature_count=%s target_col=%s model_params=%s",
        len(train_frame),
        len(valid_frame),
        len(feature_cols),
        learning_target_col,
        _xgboost_eval_param_snapshot(resolved_model_params),
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "XGBoost evaluation backend fit feature diagnostics: train=%s valid=%s",
            _xgboost_eval_frame_summary(
                train_frame,
                feature_cols=feature_cols,
                target_col=learning_target_col,
            ),
            _xgboost_eval_frame_summary(
                valid_frame,
                feature_cols=feature_cols,
                target_col=learning_target_col,
            ),
        )
    train_weights = compute_equal_dataset_row_weights(
        train_frame,
        dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
    )
    LOGGER.debug(
        "XGBoost evaluation backend fit sample weights prepared: %s",
        _sample_weight_summary(train_weights),
    )
    model = fit_xgboost_model(
        train_frame,
        valid_frame,
        feature_cols,
        target_col=learning_target_col,
        model_params=resolved_model_params,
        default_params=DEFAULT_XGBOOST_MODEL_PARAMS,
        train_sample_weight=train_weights,
    )
    best_iteration = getattr(model.model, "best_iteration", None)
    resolved_rounds = (
        int(best_iteration + 1)
        if isinstance(best_iteration, int)
        else _model_param_as_int(model.params, "n_estimators")
    )
    LOGGER.debug(
        "XGBoost evaluation backend fit completed: resolved_rounds=%s native_best_iteration=%s",
        resolved_rounds,
        best_iteration,
    )
    return model, resolved_rounds


def predict_absolute_xgboost(
    model: FittedXGBoostModel,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> np.ndarray:
    _ = feature_cols
    LOGGER.debug(
        "XGBoost evaluation prediction requested: rows=%s feature_count=%s",
        len(frame),
        len(model.feature_spec.feature_cols),
    )
    predictions = predict_with_xgboost_model(model, frame)
    LOGGER.debug("XGBoost evaluation prediction completed: rows=%s", len(predictions))
    return reconstruct_absolute_predictions(predictions, frame, target_contract)


def fit_final_xgboost_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    num_boost_round: int,
) -> FittedXGBoostModel:
    resolved_model_params = _stable_xgboost_evaluation_params(model_params)
    fit_params = {
        **resolved_model_params,
        "n_estimators": int(num_boost_round),
        "enable_early_stopping": False,
    }
    LOGGER.debug(
        "XGBoost final evaluation model fit requested: rows=%s feature_count=%s "
        "num_boost_round=%s target_col=%s",
        len(fit_frame),
        len(feature_cols),
        num_boost_round,
        target_contract.learning_target_col,
    )
    return fit_xgboost_model(
        fit_frame,
        fit_frame.head(0),
        feature_cols,
        target_col=target_contract.learning_target_col,
        model_params=fit_params,
        default_params=DEFAULT_XGBOOST_MODEL_PARAMS,
        train_sample_weight=compute_equal_dataset_row_weights(
            fit_frame,
            dataset_source_col=DEFAULT_DATASET_SOURCE_COL,
        ),
    )


def _stable_xgboost_evaluation_params(
    model_params: dict[str, object],
) -> dict[str, object]:
    resolved = dict(model_params)
    resolved["n_jobs"] = 1
    if platform.system() != "Darwin":
        return resolved
    if _xgboost_params_request_cuda(resolved):
        _log_cuda_evaluation_override()
        resolved["runtime_profile"] = "local_cpu"
        resolved["requested_runtime_profile"] = "local_cpu"
        resolved["device"] = "cpu"
        resolved["accelerator"] = "cpu"
        resolved["devices"] = 0
        resolved["cuda_available"] = False
        resolved["cuda_device_name"] = None
        resolved["xgboost_matrix_type"] = "dmatrix"
        resolved["xgboost_gpu_input_backend"] = "cpu"
    if bool(resolved.get("enable_categorical", False)):
        _log_native_categorical_evaluation_override()
        resolved["enable_categorical"] = False
    return resolved


def _xgboost_params_request_cuda(model_params: dict[str, object]) -> bool:
    runtime_profile = str(model_params.get("runtime_profile", "")).lower()
    requested_runtime_profile = str(
        model_params.get("requested_runtime_profile", "")
    ).lower()
    device = str(model_params.get("device", "")).lower()
    return (
        device == "cuda"
        or runtime_profile in {"cuda", "scaleway_l40s", "nvidia_h100"}
        or requested_runtime_profile in {"cuda", "scaleway_l40s", "nvidia_h100"}
    )


def _log_cuda_evaluation_override() -> None:
    LOGGER.info(
        "Forcing XGBoost evaluation runtime to local_cpu on macOS because tuned "
        "parameters were produced on a CUDA runtime."
    )


def _log_native_categorical_evaluation_override() -> None:
    global _native_categorical_eval_override_logged
    if _native_categorical_eval_override_logged:
        return
    _native_categorical_eval_override_logged = True
    LOGGER.debug(
        "Disabling XGBoost native categorical for evaluation on macOS/Darwin. "
        "Categorical identifiers remain enabled through the stable numeric-code preprocessing path."
    )


def _xgboost_eval_param_snapshot(model_params: dict[str, object]) -> dict[str, object]:
    keys = (
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "max_bin",
        "n_jobs",
        "enable_categorical",
        "tree_method",
        "n_estimators",
        "early_stopping_rounds",
        "enable_early_stopping",
    )
    return {key: model_params[key] for key in keys if key in model_params}


def _xgboost_eval_frame_summary(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, object]:
    present_features = [column for column in feature_cols if column in frame.columns]
    null_counts = frame[present_features].isna().sum()
    top_null_counts = {
        str(column): int(count)
        for column, count in null_counts.sort_values(ascending=False).items()
        if int(count) > 0
    }
    dtype_counts = frame[present_features].dtypes.astype(str).value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "feature_count": int(len(present_features)),
        "target_nulls": int(frame[target_col].isna().sum())
        if target_col in frame.columns
        else None,
        "feature_dtype_counts": {
            str(key): int(value) for key, value in dtype_counts.items()
        },
        "top_feature_null_counts": dict(list(top_null_counts.items())[:12]),
    }


def _sample_weight_summary(sample_weight: np.ndarray | None) -> dict[str, object]:
    if sample_weight is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "rows": int(len(sample_weight)),
        "min": float(np.min(sample_weight)),
        "max": float(np.max(sample_weight)),
        "mean": float(np.mean(sample_weight)),
        "nan_count": int(np.isnan(sample_weight).sum()),
    }
