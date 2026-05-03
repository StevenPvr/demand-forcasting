from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.foundation_covariates import (
    CovariateSchema,
    aggregate_cold_start_row,
    build_covariate_schema,
    covariate_values,
)
from praedixa.demand_forecast.backends.timesfm.covariates import (
    forecast_with_optional_covariates,
)

_REFERENCE_DATE_COL = "date"
_REFERENCE_PRODUCT_COL = "product"
_ID_COL = "series_id"
_DEFAULT_MODEL_PATH = "google/timesfm-2.0-500m-pytorch"
_DEFAULT_CONTEXT_LENGTH = 2048
_DEFAULT_HORIZON_LENGTH = 128
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_BACKEND = "gpu"
_DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
_COLD_START_SERIES_ID = "__praedixa_foundation_cold_start__"
_PREDICTION_COLUMNS = {
    0.1: "prediction_p10",
    0.5: "prediction_p50",
    0.9: "prediction_p90",
}


def _empty_feature_cols() -> list[str]:
    return []


@dataclass
class TimesFMZeroShotModel:
    pipeline: Any
    context_frame: pd.DataFrame
    series_order: list[str]
    target_col: str
    feature_cols: list[str] = field(default_factory=_empty_feature_cols)
    covariate_schema: CovariateSchema | None = None
    model_path: str = _DEFAULT_MODEL_PATH
    backend: str = _DEFAULT_BACKEND
    context_length: int = _DEFAULT_CONTEXT_LENGTH
    horizon_length: int = _DEFAULT_HORIZON_LENGTH
    batch_size: int = _DEFAULT_BATCH_SIZE
    frequency: int = 0
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    xreg_mode: str = "xreg + timesfm"
    xreg_ridge: float = 0.0
    xreg_force_on_cpu: bool = False
    xreg_normalize_target_per_input: bool = True
    cold_start_series_id: str = _COLD_START_SERIES_ID
    runtime_profile: str = "timesfm_zero_shot"
    normalization_strategy: dict[str, object] | None = None
    system_info: dict[str, object] | None = None
    interpretability_payload: dict[str, object] | None = None


def fit_timesfm_zero_shot_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
) -> tuple[TimesFMZeroShotModel, int]:
    raw_context_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    covariate_schema = build_covariate_schema(raw_context_frame, feature_cols)
    context_frame = _with_cold_start_context(
        _context_frame(
            raw_context_frame,
            target_col=target_col,
            covariate_schema=covariate_schema,
        ),
        target_col=target_col,
        covariate_schema=covariate_schema,
    )
    model_path = str(model_params.get("model_path", _DEFAULT_MODEL_PATH))
    backend = str(model_params.get("backend", _DEFAULT_BACKEND))
    context_length = _positive_int(
        model_params.get("context_length"), default=_DEFAULT_CONTEXT_LENGTH
    )
    horizon_length = _positive_int(
        model_params.get("horizon_length"), default=_DEFAULT_HORIZON_LENGTH
    )
    batch_size = _positive_int(
        model_params.get("batch_size"), default=_DEFAULT_BATCH_SIZE
    )
    pipeline = _load_timesfm_pipeline(
        model_path=model_path,
        backend=backend,
        context_length=context_length,
        horizon_length=horizon_length,
        batch_size=batch_size,
    )
    return (
        TimesFMZeroShotModel(
            pipeline=pipeline,
            context_frame=context_frame,
            series_order=_series_order(context_frame),
            target_col=target_col,
            feature_cols=covariate_schema.feature_cols,
            covariate_schema=covariate_schema,
            model_path=model_path,
            backend=backend,
            context_length=context_length,
            horizon_length=horizon_length,
            batch_size=batch_size,
            frequency=_frequency(model_params.get("frequency")),
            quantiles=_quantile_levels(model_params),
            xreg_mode=str(model_params.get("xreg_mode", "xreg + timesfm")),
            xreg_ridge=_float_param(model_params.get("xreg_ridge"), default=0.0),
            xreg_force_on_cpu=bool(model_params.get("xreg_force_on_cpu", False)),
            xreg_normalize_target_per_input=bool(
                model_params.get("xreg_normalize_target_per_input", True)
            ),
            normalization_strategy={"kind": "timesfm_native_scaling"},
            system_info={
                "model_path": model_path,
                "backend": backend,
                "zero_shot": True,
                "covariate_count": covariate_schema.covariate_count,
            },
            interpretability_payload={
                "available": False,
                "reason": "TimesFM zero-shot direct inference does not expose TFT-style interpretability.",
            },
        ),
        0,
    )


def predict_timesfm_quantiles(
    model: TimesFMZeroShotModel, frame: pd.DataFrame
) -> pd.DataFrame:
    future_frame = _future_frame(frame, covariate_schema=model.covariate_schema)
    context_frame = _context_frame_with_cold_started_series(
        model.context_frame,
        future_frame=future_frame,
        cold_start_series_id=model.cold_start_series_id,
    )
    requested_series_order = _requested_series_order(context_frame, future_frame)
    contexts = _timesfm_contexts(
        context_frame,
        series_order=requested_series_order,
        target_col=model.target_col,
    )
    point_forecasts, quantile_forecasts = forecast_with_optional_covariates(
        model,
        contexts=contexts,
        context_frame=context_frame,
        future_frame=future_frame,
        series_order=requested_series_order,
        id_col=_ID_COL,
        date_col=_REFERENCE_DATE_COL,
    )
    forecast_by_series = {
        series_id: (
            np.asarray(point_forecasts[index]),
            _quantile_array(quantile_forecasts, index),
        )
        for index, series_id in enumerate(requested_series_order)
    }
    rows: list[dict[str, float]] = []
    for _, row in future_frame.iterrows():
        series_id = str(row[_ID_COL])
        horizon_step = int(row["_horizon_step"])
        point, quantiles = forecast_by_series[series_id]
        rows.append(_forecast_quantile_row(point, quantiles, horizon_step))
    return pd.DataFrame(rows, columns=list(_PREDICTION_COLUMNS.values()))


def predict_timesfm_median(
    model: TimesFMZeroShotModel, frame: pd.DataFrame
) -> np.ndarray:
    quantiles = predict_timesfm_quantiles(model, frame)
    return quantiles["prediction_p50"].to_numpy(dtype=float)


def fit_final_timesfm_zero_shot_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
) -> TimesFMZeroShotModel:
    model, _ = fit_timesfm_zero_shot_model(
        train_frame=fit_frame,
        valid_frame=fit_frame.head(0),
        feature_cols=feature_cols,
        target_col=target_col,
        model_params=model_params,
    )
    return model


def save_timesfm_model_marker(model: TimesFMZeroShotModel, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model_family": "timesfm",
                "model_path": model.model_path,
                "backend": model.backend,
                "zero_shot": True,
                "fine_tuned": False,
                "multivariate": bool(model.feature_cols),
                "feature_cols": model.feature_cols,
                "numeric_covariate_cols": []
                if model.covariate_schema is None
                else model.covariate_schema.numeric_cols,
                "categorical_covariate_cols": []
                if model.covariate_schema is None
                else model.covariate_schema.categorical_cols,
                "context_length": model.context_length,
                "horizon_length": model.horizon_length,
                "batch_size": model.batch_size,
                "frequency": model.frequency,
                "quantiles": list(model.quantiles),
                "context_rows": int(len(model.context_frame)),
                "artifact_note": (
                    "TimesFM is loaded from Hugging Face at inference time; "
                    "this file is an evaluation marker, not a serialized checkpoint."
                ),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return output_path


@lru_cache(maxsize=4)
def _load_timesfm_pipeline(
    *,
    model_path: str,
    backend: str,
    context_length: int,
    horizon_length: int,
    batch_size: int,
) -> Any:
    timesfm = import_module("timesfm")
    hparams: dict[str, object] = {
        "backend": backend,
        "per_core_batch_size": batch_size,
        "horizon_len": horizon_length,
        "context_len": context_length,
    }
    if "2.0" in model_path:
        hparams["num_layers"] = 50
        hparams["use_positional_embedding"] = False
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(**hparams),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
    )


def _context_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    covariate_schema: CovariateSchema | None,
) -> pd.DataFrame:
    columns: dict[str, object] = {
        _ID_COL: _series_ids(frame),
        _REFERENCE_DATE_COL: _date_values(frame),
        target_col: frame[target_col].astype(float).to_numpy(),
    }
    if covariate_schema is not None:
        for column in covariate_schema.feature_cols:
            columns[column] = covariate_values(
                frame,
                covariate_schema,
                column,
                categorical_as_string=True,
            )
    return pd.DataFrame(columns, copy=False).sort_values(
        [_ID_COL, _REFERENCE_DATE_COL], kind="mergesort"
    )


def _future_frame(
    frame: pd.DataFrame,
    *,
    covariate_schema: CovariateSchema | None,
) -> pd.DataFrame:
    columns: dict[str, object] = {
        _ID_COL: _series_ids(frame),
        _REFERENCE_DATE_COL: _date_values(frame),
        "_row_order": np.arange(len(frame), dtype=np.int64),
    }
    if covariate_schema is not None:
        for column in covariate_schema.feature_cols:
            columns[column] = covariate_values(
                frame,
                covariate_schema,
                column,
                categorical_as_string=True,
            )
    future = pd.DataFrame(columns, copy=False).sort_values(
        [_ID_COL, _REFERENCE_DATE_COL], kind="mergesort"
    ).copy()
    future["_horizon_step"] = future.groupby(_ID_COL, sort=False).cumcount()
    return future.sort_values("_row_order", kind="mergesort").reset_index(drop=True)


def _series_ids(frame: pd.DataFrame) -> pd.Series:
    product_ids = _product_ids(frame)
    if "dataset_source" in frame.columns:
        bakery_mask = frame["dataset_source"].astype(str) == "bakery"
    else:
        bakery_mask = pd.Series(False, index=frame.index)
    if "client_id" not in frame.columns:
        return product_ids
    client_ids = frame["client_id"].astype("string")
    return client_ids.where(~bakery_mask & client_ids.notna(), product_ids).astype(str)


def _product_ids(frame: pd.DataFrame) -> pd.Series:
    if _REFERENCE_PRODUCT_COL in frame.columns:
        return frame[_REFERENCE_PRODUCT_COL].astype(str)
    if "product_id" in frame.columns:
        return frame["product_id"].astype(str)
    raise KeyError(_REFERENCE_PRODUCT_COL)


def _date_values(frame: pd.DataFrame) -> pd.Series:
    if _REFERENCE_DATE_COL in frame.columns:
        return pd.to_datetime(frame[_REFERENCE_DATE_COL])
    if "dt" in frame.columns:
        return pd.to_datetime(frame["dt"])
    raise KeyError(_REFERENCE_DATE_COL)


def _series_order(frame: pd.DataFrame) -> list[str]:
    return frame[_ID_COL].drop_duplicates().astype(str).tolist()


def _requested_series_order(
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
) -> list[str]:
    context_series_order = _series_order(context_frame)
    requested_series = set(future_frame[_ID_COL].astype(str).tolist())
    requested_order = [
        series_id for series_id in context_series_order if series_id in requested_series
    ]
    if not requested_order:
        raise ValueError(
            "TimesFM prediction frame has no series present in model context."
        )
    return requested_order


def _with_cold_start_context(
    frame: pd.DataFrame,
    *,
    target_col: str,
    covariate_schema: CovariateSchema | None,
) -> pd.DataFrame:
    base_frame = frame.loc[frame[_ID_COL] != _COLD_START_SERIES_ID].copy()
    cold_context = aggregate_cold_start_row(
        base_frame,
        date_col=_REFERENCE_DATE_COL,
        id_col=_ID_COL,
        target_col=target_col,
        cold_start_series_id=_COLD_START_SERIES_ID,
        schema=covariate_schema,
    )
    return pd.concat([base_frame, cold_context], ignore_index=True)


def _context_frame_with_cold_started_series(
    frame: pd.DataFrame,
    *,
    future_frame: pd.DataFrame,
    cold_start_series_id: str,
) -> pd.DataFrame:
    requested_series = set(future_frame[_ID_COL].astype(str).tolist())
    known_series = set(frame[_ID_COL].astype(str).tolist())
    missing_series = sorted(requested_series.difference(known_series))
    if not missing_series:
        return frame
    cold_context = frame.loc[frame[_ID_COL] == cold_start_series_id].copy()
    if cold_context.empty:
        return frame
    cold_parts: list[pd.DataFrame] = []
    for series_id in missing_series:
        series_context = cold_context.copy()
        series_context[_ID_COL] = series_id
        cold_parts.append(series_context)
    return pd.concat([frame, *cold_parts], ignore_index=True)


def _timesfm_contexts(
    frame: pd.DataFrame,
    *,
    series_order: list[str],
    target_col: str,
) -> list[np.ndarray]:
    contexts: list[np.ndarray] = []
    for series_id in series_order:
        values = frame.loc[frame[_ID_COL] == series_id, target_col]
        contexts.append(values.astype(float).to_numpy())
    return contexts


def _forecast_quantile_row(
    point_forecast: np.ndarray,
    quantile_forecast: np.ndarray | None,
    horizon_step: int,
) -> dict[str, float]:
    p50 = float(point_forecast[horizon_step])
    if quantile_forecast is None or quantile_forecast.ndim < 2:
        return {"prediction_p10": p50, "prediction_p50": p50, "prediction_p90": p50}
    return {
        "prediction_p10": _quantile_value(quantile_forecast, horizon_step, 0),
        "prediction_p50": p50,
        "prediction_p90": _quantile_value(quantile_forecast, horizon_step, -1),
    }


def _quantile_array(quantile_forecasts: object, index: int) -> np.ndarray | None:
    if quantile_forecasts is None:
        return None
    array = np.asarray(quantile_forecasts)
    if array.ndim < 3:
        return None
    return array[index]


def _quantile_value(
    quantile_forecast: np.ndarray,
    horizon_step: int,
    quantile_index: int,
) -> float:
    return float(quantile_forecast[horizon_step, quantile_index])


def _frequency(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float, str)):
        parsed = int(value)
        return parsed if parsed in {0, 1, 2} else 0
    return 0


def _quantile_levels(model_params: dict[str, object]) -> tuple[float, ...]:
    raw_quantiles = model_params.get("quantile_levels", _DEFAULT_QUANTILES)
    if not isinstance(raw_quantiles, (list, tuple)):
        return _DEFAULT_QUANTILES
    quantile_values = cast(tuple[object, ...] | list[object], raw_quantiles)
    parsed_values: list[float] = []
    for quantile_value in quantile_values:
        if not isinstance(quantile_value, (int, float, str)):
            return _DEFAULT_QUANTILES
        parsed_values.append(float(quantile_value))
    parsed = tuple(parsed_values)
    return parsed if {0.1, 0.5, 0.9}.issubset(set(parsed)) else _DEFAULT_QUANTILES


def _positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float, str)):
        parsed = int(value)
        return parsed if parsed > 0 else default
    return default


def _float_param(value: object, *, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    return default
