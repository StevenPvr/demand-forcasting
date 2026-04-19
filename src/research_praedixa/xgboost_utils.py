from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb


DEFAULT_HASH_BUCKET_COUNT = 1_000_003


DEFAULT_XGBOOST_MODEL_PARAMS: dict[str, object] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": os.cpu_count() or 1,
    "verbosity": 0,
}


def select_numeric_feature_columns(
    frame: pd.DataFrame,
    *,
    excluded_cols: Iterable[str],
) -> list[str]:
    excluded = set(excluded_cols)
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _prepare_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    features = frame.loc[:, feature_cols].copy()
    for column in feature_cols:
        if pd.api.types.is_bool_dtype(features[column]):
            # Keep missing values as NaN so XGBoost can handle nullable booleans safely.
            features[column] = features[column].astype("boolean").astype(np.float32)
        elif (
            pd.api.types.is_object_dtype(features[column])
            or pd.api.types.is_string_dtype(features[column])
            or isinstance(features[column].dtype, pd.CategoricalDtype)
        ):
            missing_mask = features[column].isna()
            hashed_values = (
                pd.util.hash_pandas_object(features[column].astype("string"), index=False)
                .to_numpy(dtype=np.uint64)
                % DEFAULT_HASH_BUCKET_COUNT
            ).astype(np.float32)
            hashed_values[missing_mask.to_numpy()] = np.nan
            features[column] = hashed_values
    return features


def prepare_xgboost_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return _prepare_feature_frame(frame, feature_cols)


def _normalize_objective(objective: object) -> object:
    if objective == "regression":
        return "reg:squarederror"
    return objective


def resolve_xgboost_training_config(
    model_params: dict[str, object] | None = None,
    *,
    default_params: dict[str, object] | None = None,
    default_n_estimators: int = 300,
    num_threads_override: int | None = None,
) -> tuple[dict[str, object], int]:
    resolved = {**(default_params or DEFAULT_XGBOOST_MODEL_PARAMS), **(model_params or {})}
    num_boost_round = int(resolved.pop("n_estimators", default_n_estimators))

    if "num_leaves" in resolved and "max_depth" not in resolved:
        num_leaves = max(2, int(resolved.pop("num_leaves")))
        resolved["max_depth"] = max(2, int(math.ceil(math.log2(num_leaves))))
    else:
        resolved.pop("num_leaves", None)

    if "min_child_samples" in resolved and "min_child_weight" not in resolved:
        min_child_samples = max(1, int(resolved.pop("min_child_samples")))
        resolved["min_child_weight"] = float(min_child_samples)
    else:
        resolved.pop("min_child_samples", None)

    if "metric" in resolved and "eval_metric" not in resolved:
        resolved["eval_metric"] = resolved.pop("metric")
    else:
        resolved.pop("metric", None)

    if "random_state" in resolved and "seed" not in resolved:
        resolved["seed"] = int(resolved["random_state"])

    resolved["objective"] = _normalize_objective(resolved.get("objective", "reg:squarederror"))
    resolved.setdefault("eval_metric", "rmse")
    resolved.setdefault("tree_method", "hist")

    verbosity = int(resolved.get("verbosity", 0))
    resolved["verbosity"] = 0 if verbosity < 0 else verbosity

    if num_threads_override is not None:
        resolved["nthread"] = int(num_threads_override)
    elif "n_jobs" in resolved:
        resolved["nthread"] = int(resolved.pop("n_jobs"))
    else:
        resolved["nthread"] = int(os.cpu_count() or 1)

    resolved.pop("n_jobs", None)
    return resolved, num_boost_round


def fit_xgboost_booster(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    default_n_estimators: int = 300,
    valid_frame: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
    num_threads_override: int | None = None,
    train_weights: np.ndarray | None = None,
    valid_weights: np.ndarray | None = None,
) -> xgb.Booster:
    train_matrix = xgb.DMatrix(
        prepare_xgboost_feature_frame(train_frame, feature_cols),
        label=train_frame[target_col].to_numpy(dtype=float),
        feature_names=feature_cols,
        weight=train_weights,
    )
    valid_matrix: xgb.DMatrix | None = None
    if valid_frame is not None:
        valid_matrix = xgb.DMatrix(
            prepare_xgboost_feature_frame(valid_frame, feature_cols),
            label=valid_frame[target_col].to_numpy(dtype=float),
            feature_names=feature_cols,
            weight=valid_weights,
        )
    return train_xgboost_with_matrices(
        train_matrix=train_matrix,
        model_params=model_params,
        default_params=default_params,
        default_n_estimators=default_n_estimators,
        valid_matrix=valid_matrix,
        early_stopping_rounds=early_stopping_rounds,
        num_threads_override=num_threads_override,
    )


def train_xgboost_with_matrices(
    *,
    train_matrix: xgb.DMatrix,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    default_n_estimators: int = 300,
    valid_matrix: xgb.DMatrix | None = None,
    early_stopping_rounds: int | None = None,
    num_threads_override: int | None = None,
) -> xgb.Booster:
    params, num_boost_round = resolve_xgboost_training_config(
        model_params=model_params,
        default_params=default_params,
        default_n_estimators=default_n_estimators,
        num_threads_override=num_threads_override,
    )
    evals: list[tuple[xgb.DMatrix, str]] = []
    if valid_matrix is not None:
        evals.append((valid_matrix, "valid"))
    return xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if valid_matrix is not None else None,
        verbose_eval=False,
    )


def predict_with_xgboost_booster(
    model: xgb.Booster,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    matrix = xgb.DMatrix(
        prepare_xgboost_feature_frame(frame, feature_cols),
        feature_names=feature_cols,
    )
    return predict_with_xgboost_matrix(model, matrix)


def predict_with_xgboost_matrix(
    model: xgb.Booster,
    matrix: xgb.DMatrix,
) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return model.predict(matrix)
    return model.predict(matrix, iteration_range=(0, best_iteration + 1))
