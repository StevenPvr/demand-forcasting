from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


MISSING_CATEGORY = "__missing__"
UNKNOWN_CATEGORY = "__unknown__"


@dataclass(frozen=True)
class XGBoostFeatureSpec:
    feature_cols: list[str]
    categorical_feature_cols: list[str]
    numeric_feature_cols: list[str]
    category_values: dict[str, list[str]]
    native_categorical: bool


@dataclass(frozen=True)
class XGBoostMatrices:
    train_x: pd.DataFrame
    valid_x: pd.DataFrame
    train_y: np.ndarray
    valid_y: np.ndarray
    train_sample_weight: np.ndarray | None
    feature_spec: XGBoostFeatureSpec


@dataclass(frozen=True)
class XGBoostPreparedMatrixData:
    train_x: pd.DataFrame | np.ndarray
    valid_x: pd.DataFrame | np.ndarray
    train_y: np.ndarray
    valid_y: np.ndarray
    train_sample_weight: np.ndarray | None
    feature_spec: XGBoostFeatureSpec


def _is_categorical_feature(series: pd.Series) -> bool:
    return bool(
        isinstance(series.dtype, pd.CategoricalDtype)
        or is_object_dtype(series)
        or is_string_dtype(series)
    )


def _is_numeric_feature(series: pd.Series) -> bool:
    return bool(is_numeric_dtype(series) or is_bool_dtype(series))


def _categorical_values(series: pd.Series) -> list[str]:
    values = series.astype("string").fillna(MISSING_CATEGORY).astype(str)
    unique_values = sorted(set(values.tolist()))
    return [
        MISSING_CATEGORY,
        UNKNOWN_CATEGORY,
        *[
            value
            for value in unique_values
            if value not in {MISSING_CATEGORY, UNKNOWN_CATEGORY}
        ],
    ]


def build_xgboost_feature_spec(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    native_categorical: bool = False,
) -> XGBoostFeatureSpec:
    categorical_feature_cols: list[str] = []
    numeric_feature_cols: list[str] = []
    category_values: dict[str, list[str]] = {}
    for column in feature_cols:
        series = train_frame[column]
        if is_datetime64_any_dtype(series):
            raise ValueError(
                f"XGBoost feature `{column}` is datetime-like; encode it explicitly before training."
            )
        if _is_categorical_feature(series):
            categorical_feature_cols.append(column)
            category_values[column] = _categorical_values(series)
            continue
        if _is_numeric_feature(series):
            numeric_feature_cols.append(column)
            continue
        categorical_feature_cols.append(column)
        category_values[column] = _categorical_values(series)
    return XGBoostFeatureSpec(
        feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
        numeric_feature_cols=numeric_feature_cols,
        category_values=category_values,
        native_categorical=native_categorical,
    )


def _prepared_numeric_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.astype("float32")


def _prepared_categorical_column(
    series: pd.Series,
    *,
    categories: list[str],
    native_categorical: bool,
) -> pd.Series:
    values = series.astype("string").fillna(MISSING_CATEGORY).astype(str)
    known_values = pd.Series(values.isin(categories), index=series.index)
    values = values.where(known_values, UNKNOWN_CATEGORY)
    dtype = pd.CategoricalDtype(categories=categories)
    categorical = pd.Categorical(values, dtype=dtype)
    if native_categorical:
        return pd.Series(categorical, index=series.index)
    codes = np.asarray(categorical.codes, dtype=np.float32)
    return pd.Series(codes, index=series.index, dtype="float32")


def transform_xgboost_features(
    frame: pd.DataFrame,
    feature_spec: XGBoostFeatureSpec,
) -> pd.DataFrame:
    prepared_columns: dict[str, pd.Series] = {}
    for column in feature_spec.feature_cols:
        if column in feature_spec.categorical_feature_cols:
            prepared_columns[column] = _prepared_categorical_column(
                frame[column],
                categories=feature_spec.category_values[column],
                native_categorical=feature_spec.native_categorical,
            )
        else:
            prepared_columns[column] = _prepared_numeric_column(frame[column])
    transformed = pd.DataFrame(prepared_columns, index=frame.index)
    return transformed.reset_index(drop=True)


def to_xgboost_feature_data(
    features: pd.DataFrame | np.ndarray,
    feature_spec: XGBoostFeatureSpec,
) -> pd.DataFrame | np.ndarray:
    if feature_spec.native_categorical:
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "Native categorical XGBoost features must remain a DataFrame."
            )
        return features
    if isinstance(features, np.ndarray):
        return np.require(
            features,
            dtype=np.float32,
            requirements=["C", "A", "O", "W"],
        )
    feature_data = features.to_numpy(dtype=np.float32, copy=True)
    return np.require(
        feature_data,
        dtype=np.float32,
        requirements=["C", "A", "O", "W"],
    )


def _prepared_target_array(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(
        dtype=np.float32,
        copy=True,
    )
    return np.require(
        target,
        dtype=np.float32,
        requirements=["C", "A", "O", "W"],
    )


def _prepared_sample_weight_array(
    sample_weight: np.ndarray | None,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    return np.require(
        np.asarray(sample_weight, dtype=np.float32),
        dtype=np.float32,
        requirements=["C", "A", "O", "W"],
    )


def prepare_xgboost_matrices(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_sample_weight: np.ndarray | None = None,
    native_categorical: bool = False,
) -> XGBoostMatrices:
    feature_spec = build_xgboost_feature_spec(
        train_frame,
        feature_cols,
        native_categorical=native_categorical,
    )
    return XGBoostMatrices(
        train_x=transform_xgboost_features(train_frame, feature_spec),
        valid_x=transform_xgboost_features(valid_frame, feature_spec),
        train_y=_prepared_target_array(train_frame, target_col),
        valid_y=_prepared_target_array(valid_frame, target_col),
        train_sample_weight=_prepared_sample_weight_array(train_sample_weight),
        feature_spec=feature_spec,
    )


def prepare_xgboost_matrix_data(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_sample_weight: np.ndarray | None = None,
    native_categorical: bool = False,
) -> XGBoostPreparedMatrixData:
    matrices = prepare_xgboost_matrices(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        target_col=target_col,
        train_sample_weight=train_sample_weight,
        native_categorical=native_categorical,
    )
    return XGBoostPreparedMatrixData(
        train_x=to_xgboost_feature_data(matrices.train_x, matrices.feature_spec),
        valid_x=to_xgboost_feature_data(matrices.valid_x, matrices.feature_spec),
        train_y=matrices.train_y,
        valid_y=matrices.valid_y,
        train_sample_weight=matrices.train_sample_weight,
        feature_spec=matrices.feature_spec,
    )
