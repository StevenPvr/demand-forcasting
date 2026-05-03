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
    dynamic_real_matrix,
)

_REFERENCE_DATE_COL = "date"
_REFERENCE_PRODUCT_COL = "product"
_ID_COL = "series_id"
_DEFAULT_MODEL_PATH = "Salesforce/moirai-2.0-R-small"
_DEFAULT_CONTEXT_LENGTH = 1680
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_NUM_SAMPLES = 100
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
class MoiraiZeroShotModel:
    predictor: Any
    context_frame: pd.DataFrame
    series_order: list[str]
    target_col: str
    feature_cols: list[str] = field(default_factory=_empty_feature_cols)
    covariate_schema: CovariateSchema | None = None
    model_path: str = _DEFAULT_MODEL_PATH
    context_length: int = _DEFAULT_CONTEXT_LENGTH
    batch_size: int = _DEFAULT_BATCH_SIZE
    num_samples: int = _DEFAULT_NUM_SAMPLES
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    cold_start_series_id: str = _COLD_START_SERIES_ID
    runtime_profile: str = "moirai_zero_shot"
    normalization_strategy: dict[str, object] | None = None
    system_info: dict[str, object] | None = None
    interpretability_payload: dict[str, object] | None = None


def fit_moirai_zero_shot_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
) -> tuple[MoiraiZeroShotModel, int]:
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
    prediction_length = _positive_int(model_params.get("prediction_length"), default=1)
    context_length = _positive_int(
        model_params.get("context_length"), default=_DEFAULT_CONTEXT_LENGTH
    )
    batch_size = _positive_int(
        model_params.get("batch_size"), default=_DEFAULT_BATCH_SIZE
    )
    model_path = str(model_params.get("model_path", _DEFAULT_MODEL_PATH))
    predictor = _load_moirai_predictor(
        model_path=model_path,
        prediction_length=prediction_length,
        context_length=context_length,
        covariate_count=covariate_schema.covariate_count,
        patch_size=model_params.get("patch_size", "auto"),
        num_samples=_positive_int(
            model_params.get("num_samples"), default=_DEFAULT_NUM_SAMPLES
        ),
        batch_size=batch_size,
    )
    return (
        MoiraiZeroShotModel(
            predictor=predictor,
            context_frame=context_frame,
            series_order=_series_order(context_frame),
            target_col=target_col,
            feature_cols=covariate_schema.feature_cols,
            covariate_schema=covariate_schema,
            model_path=model_path,
            context_length=context_length,
            batch_size=batch_size,
            num_samples=_positive_int(
                model_params.get("num_samples"), default=_DEFAULT_NUM_SAMPLES
            ),
            quantiles=_quantile_levels(model_params),
            normalization_strategy={"kind": "moirai_native_scaling"},
            system_info={
                "model_path": model_path,
                "zero_shot": True,
                "covariate_count": covariate_schema.covariate_count,
            },
            interpretability_payload={
                "available": False,
                "reason": "Moirai zero-shot direct inference does not expose TFT-style interpretability.",
            },
        ),
        0,
    )


def predict_moirai_quantiles(
    model: MoiraiZeroShotModel, frame: pd.DataFrame
) -> pd.DataFrame:
    future_frame = _future_frame(frame, covariate_schema=model.covariate_schema)
    context_frame = _context_frame_with_cold_started_series(
        model.context_frame,
        future_frame=future_frame,
        cold_start_series_id=model.cold_start_series_id,
    )
    requested_series_order = _requested_series_order(context_frame, future_frame)
    forecast_by_series = dict(
        zip(
            requested_series_order,
            model.predictor.predict(
                _moirai_instances(
                    context_frame,
                    future_frame,
                    series_order=requested_series_order,
                    target_col=model.target_col,
                    feature_cols=model.feature_cols,
                )
            ),
            strict=False,
        )
    )
    rows: list[dict[str, float]] = []
    for _, row in future_frame.iterrows():
        series_id = str(row[_ID_COL])
        horizon_step = int(row["_horizon_step"])
        forecast = forecast_by_series[series_id]
        rows.append(_forecast_quantile_row(forecast, horizon_step, model.quantiles))
    return pd.DataFrame(rows, columns=list(_PREDICTION_COLUMNS.values()))


def predict_moirai_median(
    model: MoiraiZeroShotModel, frame: pd.DataFrame
) -> np.ndarray:
    quantiles = predict_moirai_quantiles(model, frame)
    return quantiles["prediction_p50"].to_numpy(dtype=float)


def fit_final_moirai_zero_shot_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object],
) -> MoiraiZeroShotModel:
    model, _ = fit_moirai_zero_shot_model(
        train_frame=fit_frame,
        valid_frame=fit_frame.head(0),
        feature_cols=feature_cols,
        target_col=target_col,
        model_params=model_params,
    )
    return model


def save_moirai_model_marker(model: MoiraiZeroShotModel, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model_family": "moirai",
                "model_path": model.model_path,
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
                "batch_size": model.batch_size,
                "num_samples": model.num_samples,
                "quantiles": list(model.quantiles),
                "context_rows": int(len(model.context_frame)),
                "artifact_note": (
                    "Moirai is loaded from Hugging Face at inference time; "
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
def _load_moirai_predictor(
    *,
    model_path: str,
    prediction_length: int,
    context_length: int,
    covariate_count: int,
    patch_size: object,
    num_samples: int,
    batch_size: int,
) -> Any:
    if "moirai-2.0" in model_path:
        module = import_module("uni2ts.model.moirai2")
        forecast_cls = getattr(module, "Moirai2Forecast")
        module_cls = getattr(module, "Moirai2Module")
        forecast = forecast_cls(
            module=module_cls.from_pretrained(model_path),
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=covariate_count,
            past_feat_dynamic_real_dim=0,
        )
        return forecast.create_predictor(batch_size=batch_size)
    module = import_module("uni2ts.model.moirai")
    forecast_cls = getattr(module, "MoiraiForecast")
    module_cls = getattr(module, "MoiraiModule")
    forecast = forecast_cls(
        module=module_cls.from_pretrained(model_path),
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=covariate_count,
        past_feat_dynamic_real_dim=0,
    )
    return forecast.create_predictor(batch_size=batch_size)


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
                categorical_as_string=False,
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
                categorical_as_string=False,
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
            "Moirai prediction frame has no series present in model context."
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


def _moirai_instances(
    frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    series_order: list[str],
    target_col: str,
    feature_cols: list[str],
) -> list[dict[str, object]]:
    instances: list[dict[str, object]] = []
    for series_id in series_order:
        series_frame = frame.loc[frame[_ID_COL] == series_id]
        instance: dict[str, object] = {
            "start": pd.Period(
                pd.to_datetime(series_frame[_REFERENCE_DATE_COL]).min(),
                freq="D",
            ),
            "target": series_frame[target_col].astype(float).to_numpy(),
        }
        if feature_cols:
            instance["feat_dynamic_real"] = _moirai_dynamic_real_matrix(
                frame,
                future_frame,
                series_id=series_id,
                feature_cols=feature_cols,
            )
        instances.append(instance)
    return instances


def _moirai_dynamic_real_matrix(
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    series_id: str,
    feature_cols: list[str],
) -> np.ndarray:
    return dynamic_real_matrix(
        context_frame,
        future_frame,
        id_col=_ID_COL,
        date_col=_REFERENCE_DATE_COL,
        series_id=series_id,
        feature_cols=feature_cols,
    )


def _forecast_quantile_row(
    forecast: Any,
    horizon_step: int,
    quantiles: tuple[float, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for quantile in quantiles:
        output_col = _PREDICTION_COLUMNS.get(float(quantile))
        if output_col is None:
            continue
        values[output_col] = _forecast_quantile_value(forecast, quantile, horizon_step)
    if "prediction_p50" not in values:
        values["prediction_p50"] = _forecast_mean_value(forecast, horizon_step)
    values.setdefault("prediction_p10", values["prediction_p50"])
    values.setdefault("prediction_p90", values["prediction_p50"])
    return values


def _forecast_quantile_value(
    forecast: Any, quantile: float, horizon_step: int
) -> float:
    if hasattr(forecast, "quantile"):
        return float(np.asarray(forecast.quantile(quantile))[horizon_step])
    if hasattr(forecast, "samples"):
        return float(
            np.quantile(np.asarray(forecast.samples), quantile, axis=0)[horizon_step]
        )
    return _forecast_mean_value(forecast, horizon_step)


def _forecast_mean_value(forecast: Any, horizon_step: int) -> float:
    if hasattr(forecast, "mean"):
        return float(np.asarray(forecast.mean)[horizon_step])
    return float(np.asarray(forecast)[horizon_step])


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
