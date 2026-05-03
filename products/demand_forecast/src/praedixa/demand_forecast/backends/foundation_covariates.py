from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MISSING_CATEGORY = "<NA>"
_UNKNOWN_CATEGORY_CODE = -1.0


@dataclass(frozen=True)
class CovariateSchema:
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    categorical_maps: dict[str, dict[str, int]]

    @property
    def covariate_count(self) -> int:
        return len(self.feature_cols)


def build_covariate_schema(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> CovariateSchema:
    present_feature_cols = [
        column for column in feature_cols if column in frame.columns
    ]
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for column in present_feature_cols:
        if is_numeric_like(frame[column]):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return CovariateSchema(
        feature_cols=present_feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categorical_maps={
            column: _category_mapping(frame[column]) for column in categorical_cols
        },
    )


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return True
    non_null_values = series.dropna()
    if non_null_values.empty:
        return True
    numeric_values = pd.to_numeric(non_null_values, errors="coerce")
    return bool(numeric_values.notna().all())


def _category_mapping(series: pd.Series) -> dict[str, int]:
    values = coerce_categorical_covariate(series)
    categories = sorted(values.drop_duplicates().tolist())
    return {category: index for index, category in enumerate(categories)}


def covariate_values(
    frame: pd.DataFrame,
    schema: CovariateSchema,
    column: str,
    *,
    categorical_as_string: bool,
) -> pd.Series:
    if column in schema.numeric_cols:
        return coerce_numeric_covariate(frame[column])
    if categorical_as_string:
        return coerce_categorical_covariate(frame[column])
    return encode_categorical_covariate(
        frame[column],
        schema.categorical_maps[column],
    )


def coerce_numeric_covariate(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        values = series.fillna(False).astype("int8").astype(float)
    else:
        values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.fillna(0.0)


def coerce_categorical_covariate(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(_MISSING_CATEGORY).astype(str)


def encode_categorical_covariate(
    series: pd.Series,
    category_mapping: dict[str, int],
) -> pd.Series:
    values = coerce_categorical_covariate(series)
    encoded = values.map(category_mapping).astype(float)
    return encoded.fillna(_UNKNOWN_CATEGORY_CODE)


def aggregate_cold_start_row(
    frame: pd.DataFrame,
    *,
    date_col: str,
    id_col: str,
    target_col: str,
    cold_start_series_id: str,
    schema: CovariateSchema | None,
) -> pd.DataFrame:
    aggregations: dict[str, str] = {target_col: "mean"}
    if schema is not None:
        for column in schema.feature_cols:
            aggregations[column] = (
                "mean" if column in schema.numeric_cols else "first"
            )
    aggregation_columns = [date_col, *aggregations]
    frame_for_aggregation = frame.loc[:, aggregation_columns].copy()
    grouped = frame_for_aggregation.groupby(date_col, sort=True).agg(aggregations)
    result_columns: dict[str, object] = {
        id_col: [cold_start_series_id] * len(grouped),
        date_col: grouped.index.to_numpy(),
    }
    for column in aggregations:
        result_columns[column] = grouped[column].to_numpy()
    return pd.DataFrame(result_columns, copy=False)


def series_combined_values(
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    series_id: str,
    column: str,
) -> pd.Series:
    context_values = context_frame.loc[
        context_frame[id_col] == series_id, [date_col, column]
    ]
    future_values = future_frame.loc[
        future_frame[id_col] == series_id, [date_col, column]
    ]
    combined = pd.concat([context_values, future_values], ignore_index=True)
    return combined.sort_values(date_col, kind="mergesort")[column].reset_index(
        drop=True
    )


def dynamic_covariate_dict(
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    series_order: list[str],
    columns: list[str],
) -> dict[str, list[list[object]]]:
    return {
        column: [
            series_combined_values(
                context_frame,
                future_frame,
                id_col=id_col,
                date_col=date_col,
                series_id=series_id,
                column=column,
            ).tolist()
            for series_id in series_order
        ]
        for column in columns
    }


def dynamic_real_matrix(
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    series_id: str,
    feature_cols: list[str],
) -> np.ndarray:
    values = [
        series_combined_values(
            context_frame,
            future_frame,
            id_col=id_col,
            date_col=date_col,
            series_id=series_id,
            column=column,
        )
        .astype(float)
        .to_numpy()
        for column in feature_cols
    ]
    return np.vstack(values)
