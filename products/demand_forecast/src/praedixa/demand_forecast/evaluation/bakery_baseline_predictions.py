from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_baseline_shared import (
    BlendForecaster,
    FIELD_BASELINE_NAME,
    HistoryForecaster,
)
from praedixa.demand_forecast.evaluation.bakery_metrics import (
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    REFERENCE_TARGET_COL,
    mae_score,
    mase_score,
    rmse_score,
    smape,
)


def _array_mean(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.mean(values))


def _array_median(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.median(values))


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("date cannot be NaT")
    return timestamp


def _safe_last(history: list[float]) -> float:
    return float(history[-1])


def _lag_forecaster(lag: int) -> HistoryForecaster:
    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_lag = max(1, lag)
        if len(history) < resolved_lag:
            return _safe_last(history)
        return float(history[-resolved_lag])

    return forecast


def _rolling_mean_forecaster(window: int) -> HistoryForecaster:
    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_window = max(1, min(window, len(history)))
        return float(np.mean(history[-resolved_window:]))

    return forecast


def _rolling_median_forecaster(window: int) -> HistoryForecaster:
    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_window = max(1, min(window, len(history)))
        return float(np.median(history[-resolved_window:]))

    return forecast


def _ewm_forecaster(span: int) -> HistoryForecaster:
    def forecast(history: list[float], _: pd.Timestamp) -> float:
        history_series = pd.Series(history, dtype=float)
        return float(history_series.ewm(span=max(1, span), adjust=False).mean().iloc[-1])

    return forecast


def _expanding_mean_forecaster(history: list[float], _: pd.Timestamp) -> float:
    return float(np.mean(history))


def _expanding_median_forecaster(history: list[float], _: pd.Timestamp) -> float:
    return float(np.median(history))


def _same_weekday_history(history_df: pd.DataFrame, forecast_date: pd.Timestamp) -> list[float]:
    date_series = history_df[REFERENCE_DATE_COL]
    mask = date_series.dt.dayofweek == forecast_date.dayofweek
    return [float(value) for value in history_df.loc[mask, REFERENCE_TARGET_COL].astype(float).tolist()]


def _trimmed_mean_forecaster(window: int, trim_ratio: float = 0.2) -> HistoryForecaster:
    def forecast(history: list[float], _: pd.Timestamp) -> float:
        resolved_window = max(1, min(window, len(history)))
        values = np.sort(np.asarray(history[-resolved_window:], dtype=float))
        trim_count = int(np.floor(len(values) * trim_ratio))
        if (len(values) - (2 * trim_count)) <= 0:
            return float(np.mean(values))
        trimmed = values[trim_count : len(values) - trim_count]
        return float(np.mean(trimmed))

    return forecast


def _blend_forecaster(weights: dict[str, float]) -> BlendForecaster:
    def forecast(predictions: dict[str, float]) -> float:
        return float(sum(predictions[name] * weight for name, weight in weights.items()))

    return forecast


def _year_ago_date_candidates(forecast_date: pd.Timestamp) -> list[pd.Timestamp]:
    return [
        forecast_date - pd.Timedelta(days=365),
        forecast_date - pd.Timedelta(days=364),
        forecast_date - pd.Timedelta(days=366),
    ]


def _same_day_last_year_value(history_df: pd.DataFrame, forecast_date: pd.Timestamp) -> float | None:
    indexed_history = history_df.set_index(REFERENCE_DATE_COL)
    for candidate_date in _year_ago_date_candidates(forecast_date):
        if candidate_date in indexed_history.index:
            return float(cast(float, indexed_history.at[candidate_date, REFERENCE_TARGET_COL]))
    return None


def _append_future_row(row: dict[Hashable, Any]) -> pd.DataFrame:
    normalized_row = {str(key): value for key, value in row.items()}
    appended_row = pd.DataFrame([normalized_row])
    appended_row[REFERENCE_DATE_COL] = pd.to_datetime(appended_row[REFERENCE_DATE_COL])
    return appended_row


def predict_baseline_series(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor: HistoryForecaster | BlendForecaster,
    *,
    precomputed_names: list[str] | None = None,
    prediction_cache: dict[str, list[float]] | None = None,
) -> pd.Series:
    observed_history_df = history_df.copy()
    observed_values = observed_history_df[REFERENCE_TARGET_COL].astype(float).tolist()
    predictions: list[float] = []
    future_records = test_df.to_dict(orient="records")
    for row in future_records:
        forecast_date = _coerce_timestamp(row[REFERENCE_DATE_COL])
        if precomputed_names is None:
            direct_predictor = cast(HistoryForecaster, predictor)
            predicted_value = float(direct_predictor(observed_values, forecast_date))
        else:
            if prediction_cache is None:
                raise RuntimeError("prediction_cache is required for blended baselines")
            component_predictions = {name: prediction_cache[name][len(predictions)] for name in precomputed_names}
            blended_predictor = cast(BlendForecaster, predictor)
            predicted_value = float(blended_predictor(component_predictions))
        predictions.append(predicted_value)
        observed_history_df = pd.concat([observed_history_df, _append_future_row(row)], ignore_index=True)
        observed_values.append(float(row[REFERENCE_TARGET_COL]))
    return pd.Series(predictions, dtype=float)


def _same_weekday_predictor(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reducer: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    window: int | None,
) -> pd.Series:
    observed_history_df = history_df.copy()
    predictions: list[float] = []
    future_records = test_df.to_dict(orient="records")
    for row in future_records:
        forecast_date = _coerce_timestamp(row[REFERENCE_DATE_COL])
        weekday_history = _same_weekday_history(observed_history_df, forecast_date)
        if not weekday_history:
            predictions.append(float(observed_history_df[REFERENCE_TARGET_COL].astype(float).iloc[-1]))
        else:
            predictions.append(_reduced_weekday_prediction(weekday_history, reducer, window))
        observed_history_df = pd.concat([observed_history_df, _append_future_row(row)], ignore_index=True)
    return pd.Series(predictions, dtype=float)


def _reduced_weekday_prediction(
    weekday_history: list[float],
    reducer: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    window: int | None,
) -> float:
    resolved_history = weekday_history if window is None else weekday_history[-window:]
    return float(reducer(np.asarray(resolved_history, dtype=float)))


def _same_day_last_year_predictions(history_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    observed_history_df = history_df.copy()
    predictions: list[float] = []
    future_records = test_df.to_dict(orient="records")
    for row in future_records:
        forecast_date = _coerce_timestamp(row[REFERENCE_DATE_COL])
        year_ago_value = _same_day_last_year_value(observed_history_df, forecast_date)
        fallback_value = float(observed_history_df[REFERENCE_TARGET_COL].astype(float).iloc[-1])
        predictions.append(fallback_value if year_ago_value is None else year_ago_value)
        observed_history_df = pd.concat([observed_history_df, _append_future_row(row)], ignore_index=True)
    return pd.Series(predictions, dtype=float)


def baseline_metrics(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    insample_series: pd.Series,
) -> dict[str, float | str]:
    errors = y_true.astype(float).reset_index(drop=True) - y_pred.astype(float).reset_index(drop=True)
    return {
        "name": name,
        "mae": mae_score(y_true, y_pred),
        "rmse": rmse_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase_score(y_true, y_pred, insample_series, seasonal_period=7),
        "bias_mean_error": float(errors.mean()),
        "mean_prediction": float(y_pred.mean()),
        "std_prediction": float(np.std(np.asarray(y_pred, dtype=float), ddof=1)),
    }


def prediction_rows_payload(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> list[dict[str, float | str]]:
    target_dates = pd.to_datetime(test_df[REFERENCE_DATE_COL]).dt.strftime("%Y-%m-%d")
    payload_df = pd.DataFrame(
        {
            "product": test_df[REFERENCE_PRODUCT_COL].astype(str),
            "origin_date": (pd.to_datetime(test_df[REFERENCE_DATE_COL]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
            "target_date": target_dates,
            "actual": y_true.astype(float),
            "prediction": y_pred.astype(float),
        }
    )
    payload_df["absolute_error"] = (payload_df["actual"] - payload_df["prediction"]).abs()
    return [
        {
            "product": str(row["product"]),
            "origin_date": str(row["origin_date"]),
            "target_date": str(row["target_date"]),
            "actual": float(row["actual"]),
            "prediction": float(row["prediction"]),
            "absolute_error": float(row["absolute_error"]),
        }
        for row in payload_df.to_dict(orient="records")
    ]


def baseline_definitions() -> tuple[
    dict[str, HistoryForecaster],
    dict[str, tuple[list[str], BlendForecaster]],
]:
    direct_baselines: dict[str, HistoryForecaster] = {
        "naive_lag_1": _lag_forecaster(1),
        "seasonal_naive_lag_7": _lag_forecaster(7),
        "lag_14": _lag_forecaster(14),
        "rolling_mean_3": _rolling_mean_forecaster(3),
        "rolling_mean_7": _rolling_mean_forecaster(7),
        "rolling_mean_14": _rolling_mean_forecaster(14),
        "rolling_median_7": _rolling_median_forecaster(7),
        "rolling_median_14": _rolling_median_forecaster(14),
        "ewm_span_3": _ewm_forecaster(3),
        "ewm_span_7": _ewm_forecaster(7),
        "ewm_span_14": _ewm_forecaster(14),
        "expanding_mean": _expanding_mean_forecaster,
        "expanding_median": _expanding_median_forecaster,
        "trimmed_mean_7": _trimmed_mean_forecaster(7),
    }
    blended_baselines: dict[str, tuple[list[str], BlendForecaster]] = {
        FIELD_BASELINE_NAME: (
            ["naive_lag_1", "seasonal_naive_lag_7"],
            _blend_forecaster({"naive_lag_1": 0.5, "seasonal_naive_lag_7": 0.5}),
        ),
        "blend_roll_mean_7_ewm_7_50_50": (
            ["rolling_mean_7", "ewm_span_7"],
            _blend_forecaster({"rolling_mean_7": 0.5, "ewm_span_7": 0.5}),
        ),
    }
    return direct_baselines, blended_baselines


def same_weekday_baselines(history_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "same_weekday_mean_4": _same_weekday_predictor(history_df, test_df, _array_mean, 4),
        "same_weekday_median_4": _same_weekday_predictor(history_df, test_df, _array_median, 4),
        "same_weekday_mean_expanding": _same_weekday_predictor(history_df, test_df, _array_mean, None),
        "same_weekday_median_expanding": _same_weekday_predictor(history_df, test_df, _array_median, None),
        "same_day_last_year": _same_day_last_year_predictions(history_df, test_df),
    }


def ranked_rows(metrics_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    ranked = sorted(metrics_rows, key=lambda row: (float(row["mae"]), float(row["rmse"]), str(row["name"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank_mae"] = rank
    return ranked


__all__ = [
    "baseline_definitions",
    "baseline_metrics",
    "predict_baseline_series",
    "prediction_rows_payload",
    "ranked_rows",
    "same_weekday_baselines",
]
