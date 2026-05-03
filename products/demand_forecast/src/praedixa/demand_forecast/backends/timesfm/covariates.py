from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.foundation_covariates import (
    dynamic_covariate_dict,
)


def forecast_with_optional_covariates(
    model: Any,
    *,
    contexts: list[np.ndarray],
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    series_order: list[str],
    id_col: str,
    date_col: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    frequency = [model.frequency for _ in contexts]
    if model.covariate_schema is None or not model.feature_cols:
        return model.pipeline.forecast(contexts, freq=frequency)
    if not hasattr(model.pipeline, "forecast_with_covariates"):
        raise RuntimeError(
            "TimesFM multivariate mode requires `forecast_with_covariates`; "
            "install a TimesFM version that exposes covariate inference."
        )
    result = model.pipeline.forecast_with_covariates(
        inputs=contexts,
        dynamic_numerical_covariates=dynamic_covariate_dict(
            context_frame,
            future_frame,
            id_col=id_col,
            date_col=date_col,
            series_order=series_order,
            columns=model.covariate_schema.numeric_cols,
        ),
        dynamic_categorical_covariates=dynamic_covariate_dict(
            context_frame,
            future_frame,
            id_col=id_col,
            date_col=date_col,
            series_order=series_order,
            columns=model.covariate_schema.categorical_cols,
        ),
        static_numerical_covariates={},
        static_categorical_covariates={},
        freq=frequency,
        xreg_mode=model.xreg_mode,
        ridge=model.xreg_ridge,
        force_on_cpu=model.xreg_force_on_cpu,
        normalize_xreg_target_per_input=model.xreg_normalize_target_per_input,
    )
    return _forecast_result_tuple(result)


def _forecast_result_tuple(result: object) -> tuple[np.ndarray, np.ndarray | None]:
    if not isinstance(result, tuple):
        raise TypeError("TimesFM forecast result must be a tuple of forecasts.")
    result_items = cast(tuple[object, ...], result)
    if len(result_items) < 2:
        raise TypeError("TimesFM forecast result must include quantile forecasts.")
    point_forecast = np.asarray(result_items[0], dtype=float)
    quantile_forecast = (
        None if result_items[1] is None else np.asarray(result_items[1], dtype=float)
    )
    return point_forecast, quantile_forecast
