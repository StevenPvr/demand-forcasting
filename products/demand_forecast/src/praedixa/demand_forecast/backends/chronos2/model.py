from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_REFERENCE_DATE_COL = "date"
_REFERENCE_PRODUCT_COL = "product"
_CHRONOS_ID_COL = "series_id"
_CHRONOS_TIMESTAMP_COL = "timestamp"
_CHRONOS_TARGET_COL = "target"
_CHRONOS_FREQUENCY_ANCHOR_ID = "__praedixa_chronos2_daily_frequency_anchor__"
_DEFAULT_MODEL_PATH = "amazon/chronos-2"
_DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
_PREDICTION_COLUMN_BY_QUANTILE = {
    "0.1": "prediction_p10",
    "0.5": "prediction_p50",
    "0.9": "prediction_p90",
}


@dataclass
class Chronos2ZeroShotModel:
    pipeline: Any
    context_frame: pd.DataFrame
    feature_cols: list[str]
    series_order: list[str]
    covariate_kinds: dict[str, str]
    target_col: str = _CHRONOS_TARGET_COL
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    model_path: str = _DEFAULT_MODEL_PATH
    device_map: str = "cpu"
    batch_size: int | None = None
    context_length: int | None = None
    cross_learning: bool = True
    runtime_profile: str = "chronos2_zero_shot"
    normalization_strategy: dict[str, object] | None = None
    system_info: dict[str, object] | None = None
    artifact_bundle_version: int = 2
    interpretability_payload: dict[str, object] | None = None
    git_sha: str | None = None
    bundle_manifest: dict[str, object] | None = None
    data_hashes: dict[str, str] | None = None


def load_chronos2_pipeline(model_params: dict[str, object]) -> Any:
    return _load_chronos2_pipeline_cached(
        str(model_params.get("model_path", _DEFAULT_MODEL_PATH)),
        str(model_params.get("device_map", "cpu")),
    )


@lru_cache(maxsize=4)
def _load_chronos2_pipeline_cached(model_path: str, device_map: str) -> Any:
    chronos_module = import_module("chronos.chronos2.pipeline")
    chronos_pipeline = getattr(chronos_module, "Chronos2Pipeline")
    return chronos_pipeline.from_pretrained(model_path, device_map=device_map)


def fit_chronos2_zero_shot_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
) -> tuple[Chronos2ZeroShotModel, int]:
    context_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    model_path = str(model_params.get("model_path", _DEFAULT_MODEL_PATH))
    device_map = str(model_params.get("device_map", "cpu"))
    context_frame = _chronos_frame(
        context_frame,
        feature_cols=feature_cols,
        target_col=target_col,
        include_target=True,
    )
    covariate_kinds = _chronos_covariate_kinds(context_frame)
    return (
        Chronos2ZeroShotModel(
            pipeline=load_chronos2_pipeline(model_params),
            context_frame=context_frame,
            feature_cols=_chronos_covariate_columns(feature_cols, context_frame),
            series_order=context_frame[_CHRONOS_ID_COL]
            .drop_duplicates()
            .astype(str)
            .tolist(),
            covariate_kinds=covariate_kinds,
            quantiles=_quantile_levels(model_params),
            model_path=model_path,
            device_map=device_map,
            batch_size=_optional_int(model_params.get("batch_size")),
            context_length=_optional_int(model_params.get("context_length")),
            cross_learning=bool(model_params.get("cross_learning", True)),
            normalization_strategy={"kind": "chronos2_zero_shot_native_scaling"},
            system_info={
                "model_path": model_path,
                "device_map": device_map,
                "zero_shot": True,
            },
            interpretability_payload={
                "available": False,
                "reason": "Chronos-2 zero-shot direct inference does not expose TFT-style interpretability.",
            },
        ),
        0,
    )


def predict_chronos2_quantiles(
    model: Chronos2ZeroShotModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    future_frame = _chronos_frame(
        frame,
        feature_cols=model.feature_cols,
        target_col=model.target_col,
        include_target=False,
        preferred_series_order=model.series_order,
        covariate_kinds=model.covariate_kinds,
    )
    prediction_length = _prediction_length(future_frame)
    future_frame = _with_frequency_anchor_future(
        future_frame,
        context_frame=model.context_frame,
        prediction_length=prediction_length,
    )
    forecast_frame = model.pipeline.predict_df(
        model.context_frame,
        future_df=future_frame,
        prediction_length=prediction_length,
        quantile_levels=list(model.quantiles),
        id_column=_CHRONOS_ID_COL,
        timestamp_column=_CHRONOS_TIMESTAMP_COL,
        target=_CHRONOS_TARGET_COL,
        **_predict_kwargs(model),
    )
    merged = _aligned_forecast(frame, forecast_frame)
    return _quantile_prediction_frame(merged)


def predict_chronos2_median(
    model: Chronos2ZeroShotModel, frame: pd.DataFrame
) -> np.ndarray:
    quantiles = predict_chronos2_quantiles(model, frame)
    return quantiles["prediction_p50"].to_numpy(dtype=float)


def fit_final_chronos2_zero_shot_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
) -> Chronos2ZeroShotModel:
    model, _ = fit_chronos2_zero_shot_model(
        train_frame=fit_frame,
        valid_frame=fit_frame.head(0),
        feature_cols=feature_cols,
        target_col=target_col,
        model_params=model_params,
    )
    return model


def save_chronos2_model_marker(model: Chronos2ZeroShotModel, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model_family": "chronos2",
                "model_path": model.model_path,
                "device_map": model.device_map,
                "zero_shot": True,
                "fine_tuned": False,
                "feature_cols": model.feature_cols,
                "quantiles": list(model.quantiles),
                "context_rows": int(len(model.context_frame)),
                "artifact_note": (
                    "Chronos-2 is loaded from Hugging Face at inference time; "
                    "this file is an evaluation marker, not a serialized checkpoint."
                ),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return output_path


def _chronos_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    include_target: bool,
    preferred_series_order: list[str] | None = None,
    covariate_kinds: dict[str, str] | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            _CHRONOS_ID_COL: _series_ids(frame),
            _CHRONOS_TIMESTAMP_COL: pd.to_datetime(frame[_REFERENCE_DATE_COL]),
        }
    )
    if include_target:
        output[_CHRONOS_TARGET_COL] = frame[target_col].astype(float).to_numpy()
    for column in _chronos_covariate_columns(feature_cols, frame):
        output[column] = _chronos_covariate_series(
            frame[column],
            forced_kind=None
            if covariate_kinds is None
            else covariate_kinds.get(column),
        )
    if include_target:
        output = _with_frequency_anchor_context(output)
    return _sort_chronos_frame(output, preferred_series_order=preferred_series_order)


def _series_ids(frame: pd.DataFrame) -> pd.Series:
    if "client_id" in frame.columns and not frame["client_id"].isna().all():
        return frame["client_id"].astype(str)
    return frame[_REFERENCE_PRODUCT_COL].astype(str)


def _chronos_covariate_columns(
    feature_cols: list[str], frame: pd.DataFrame
) -> list[str]:
    blocked = {
        _REFERENCE_DATE_COL,
        _REFERENCE_PRODUCT_COL,
        _CHRONOS_ID_COL,
        _CHRONOS_TIMESTAMP_COL,
        _CHRONOS_TARGET_COL,
    }
    return [
        column
        for column in feature_cols
        if column in frame.columns and column not in blocked
    ]


def _chronos_covariate_kinds(frame: pd.DataFrame) -> dict[str, str]:
    return {
        column: "numeric"
        if pd.api.types.is_numeric_dtype(frame[column])
        else "categorical"
        for column in frame.columns
        if column not in {_CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL, _CHRONOS_TARGET_COL}
    }


def _chronos_covariate_series(
    series: pd.Series,
    *,
    forced_kind: str | None = None,
) -> pd.Series:
    if forced_kind == "numeric":
        return pd.to_numeric(series, errors="coerce").astype(float)
    if forced_kind == "categorical":
        return series.astype("string").fillna("<NA>").astype(str)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    numeric_values = pd.to_numeric(series, errors="coerce")
    non_null_values = series.dropna()
    if (
        not non_null_values.empty
        and not numeric_values.loc[non_null_values.index].isna().any()
    ):
        return numeric_values.astype(float)
    non_null_values = series.dropna()
    if (
        not non_null_values.empty
        and non_null_values.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    ):
        return series.fillna(False).astype(bool).astype("int8")
    return series.astype("string").fillna("<NA>").astype(str)


def _sort_chronos_frame(
    frame: pd.DataFrame,
    *,
    preferred_series_order: list[str] | None,
) -> pd.DataFrame:
    ordered = frame.copy()
    series_order = preferred_series_order or _series_order_with_inferable_frequency(
        ordered
    )
    order_rank = {series_id: index for index, series_id in enumerate(series_order)}
    ordered["_series_order"] = (
        ordered[_CHRONOS_ID_COL].map(order_rank).fillna(len(order_rank))
    )
    return (
        ordered.sort_values(["_series_order", _CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL])
        .drop(columns="_series_order")
        .reset_index(drop=True)
    )


def _series_order_with_inferable_frequency(frame: pd.DataFrame) -> list[str]:
    base_order = frame[_CHRONOS_ID_COL].drop_duplicates().astype(str).tolist()
    for series_id in base_order:
        timestamps = pd.DatetimeIndex(
            frame.loc[frame[_CHRONOS_ID_COL] == series_id, _CHRONOS_TIMESTAMP_COL]
        )
        if len(timestamps) >= 3 and pd.infer_freq(timestamps) is not None:
            return [series_id, *[other for other in base_order if other != series_id]]
    return base_order


def _with_frequency_anchor_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or _CHRONOS_FREQUENCY_ANCHOR_ID in set(
        frame[_CHRONOS_ID_COL].astype(str)
    ):
        return frame
    start_timestamp = pd.to_datetime(
        frame[_CHRONOS_TIMESTAMP_COL]
    ).min() - pd.Timedelta(days=3)
    anchor_rows: list[dict[str, object]] = []
    for offset in range(3):
        row = _anchor_base_row(frame)
        row[_CHRONOS_TIMESTAMP_COL] = start_timestamp + pd.Timedelta(days=offset)
        row[_CHRONOS_TARGET_COL] = 0.0
        anchor_rows.append(row)
    return pd.concat([pd.DataFrame(anchor_rows), frame], ignore_index=True)


def _with_frequency_anchor_future(
    frame: pd.DataFrame,
    *,
    context_frame: pd.DataFrame,
    prediction_length: int,
) -> pd.DataFrame:
    if frame.empty or _CHRONOS_FREQUENCY_ANCHOR_ID in set(
        frame[_CHRONOS_ID_COL].astype(str)
    ):
        return frame
    anchor_context = context_frame.loc[
        context_frame[_CHRONOS_ID_COL] == _CHRONOS_FREQUENCY_ANCHOR_ID
    ]
    if anchor_context.empty:
        return frame
    last_anchor_timestamp = pd.to_datetime(anchor_context[_CHRONOS_TIMESTAMP_COL]).max()
    anchor_rows: list[dict[str, object]] = []
    for step in range(1, prediction_length + 1):
        row = _anchor_base_row(frame)
        row[_CHRONOS_TIMESTAMP_COL] = last_anchor_timestamp + pd.Timedelta(days=step)
        anchor_rows.append(row)
    return pd.concat([pd.DataFrame(anchor_rows), frame], ignore_index=True)


def _anchor_base_row(frame: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {_CHRONOS_ID_COL: _CHRONOS_FREQUENCY_ANCHOR_ID}
    for column in frame.columns:
        if column in {_CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL}:
            continue
        if column == _CHRONOS_TARGET_COL:
            row[column] = 0.0
            continue
        row[column] = _anchor_covariate_value(frame[column])
    return row


def _anchor_covariate_value(series: pd.Series) -> object:
    if pd.api.types.is_numeric_dtype(series):
        return 0.0
    return "<NA>"


def _prediction_length(future_frame: pd.DataFrame) -> int:
    if future_frame.empty:
        return 0
    return int(
        future_frame.groupby(_CHRONOS_ID_COL, sort=False)[_CHRONOS_TIMESTAMP_COL]
        .nunique()
        .max()
    )


def _predict_kwargs(model: Chronos2ZeroShotModel) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "cross_learning": model.cross_learning,
        "validate_inputs": False,
    }
    if model.batch_size is not None:
        kwargs["batch_size"] = model.batch_size
    if model.context_length is not None:
        kwargs["context_length"] = model.context_length
    return kwargs


def _aligned_forecast(
    source_frame: pd.DataFrame, forecast_frame: pd.DataFrame
) -> pd.DataFrame:
    expected = pd.DataFrame(
        {
            _CHRONOS_ID_COL: _series_ids(source_frame),
            _CHRONOS_TIMESTAMP_COL: pd.to_datetime(source_frame[_REFERENCE_DATE_COL]),
            "_row_order": np.arange(len(source_frame), dtype=np.int64),
        }
    )
    forecast = forecast_frame.copy()
    forecast[_CHRONOS_TIMESTAMP_COL] = pd.to_datetime(forecast[_CHRONOS_TIMESTAMP_COL])
    aligned = expected.merge(
        forecast,
        on=[_CHRONOS_ID_COL, _CHRONOS_TIMESTAMP_COL],
        how="left",
        validate="one_to_one",
    ).sort_values("_row_order")
    if aligned["0.5"].isna().any():
        aligned = _aligned_forecast_by_horizon_order(expected, forecast)
    if aligned["0.5"].isna().any():
        missing_count = int(aligned["0.5"].isna().sum())
        raise ValueError(
            f"Chronos-2 forecast is missing {missing_count} expected rows."
        )
    return aligned.reset_index(drop=True)


def _aligned_forecast_by_horizon_order(
    expected: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    expected_by_horizon = expected.copy()
    forecast_by_horizon = forecast.copy()
    expected_by_horizon["_horizon_step"] = expected_by_horizon.groupby(
        _CHRONOS_ID_COL, sort=False
    ).cumcount()
    forecast_by_horizon["_horizon_step"] = forecast_by_horizon.groupby(
        _CHRONOS_ID_COL, sort=False
    ).cumcount()
    forecast_by_horizon = forecast_by_horizon.drop(columns=[_CHRONOS_TIMESTAMP_COL])
    return (
        expected_by_horizon.merge(
            forecast_by_horizon,
            on=[_CHRONOS_ID_COL, "_horizon_step"],
            how="left",
            validate="one_to_one",
        )
        .sort_values("_row_order")
        .reset_index(drop=True)
    )


def _quantile_prediction_frame(forecast_frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    for chronos_column, prediction_column in _PREDICTION_COLUMN_BY_QUANTILE.items():
        if chronos_column not in forecast_frame.columns:
            raise ValueError(
                f"Chronos-2 forecast missing quantile column `{chronos_column}`."
            )
        output[prediction_column] = (
            forecast_frame[chronos_column].astype(float).to_numpy()
        )
    return output


def _quantile_levels(model_params: dict[str, object]) -> tuple[float, ...]:
    raw_quantiles = model_params.get("quantile_levels", _DEFAULT_QUANTILES)
    if not isinstance(raw_quantiles, (list, tuple)):
        return _DEFAULT_QUANTILES
    quantile_values = cast(tuple[object, ...] | list[object], raw_quantiles)
    parsed_values: list[float] = []
    for value in quantile_values:
        if not isinstance(value, (int, float, str)):
            return _DEFAULT_QUANTILES
        parsed_values.append(float(value))
    parsed = tuple(parsed_values)
    required = {0.1, 0.5, 0.9}
    if not required.issubset(set(parsed)):
        return _DEFAULT_QUANTILES
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    return None
