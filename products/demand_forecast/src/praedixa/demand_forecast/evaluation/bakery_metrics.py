from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import polars as pl

from praedixa.platform.utils.memory import downcast_pandas_frame


REFERENCE_DATE_COL = "date"
REFERENCE_PRODUCT_COL = "product"
REFERENCE_TARGET_COL = "quantity"
_FLOAT_COMPARISON_EPSILON = 1e-8


def load_reference_split(csv_path: str | Path) -> pd.DataFrame:
    frame = (
        pl.read_csv(str(csv_path))
        .with_columns(
            [
                pl.col(REFERENCE_DATE_COL).str.to_datetime(format="%Y-%m-%d", strict=True),
                pl.col(REFERENCE_PRODUCT_COL).cast(pl.Utf8, strict=False),
                pl.col(REFERENCE_TARGET_COL).cast(pl.Float64, strict=False),
            ]
        )
        .sort([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL])
    )
    return downcast_pandas_frame(frame.to_pandas())


def mae_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if len(actual) == 0:
        return 0.0
    diff = actual - predicted
    max_abs_diff = float(np.max(np.abs(diff)))
    if not np.isfinite(max_abs_diff):
        return float(max_abs_diff)
    if abs(max_abs_diff) <= _FLOAT_COMPARISON_EPSILON:
        return 0.0
    scaled_diff = diff / max_abs_diff
    return float(max_abs_diff * np.sqrt(np.mean(np.square(scaled_diff))))


def smape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    if len(actual) == 0:
        return 0.0
    scaled_errors = np.zeros_like(actual, dtype=float)
    valid_mask = denominator > 0.0
    scaled_errors[valid_mask] = (2.0 * np.abs(actual[valid_mask] - predicted[valid_mask])) / denominator[valid_mask]
    return float(np.mean(scaled_errors))


def mase_score(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    insample_series: pd.Series | np.ndarray,
    seasonal_period: int = 1,
) -> float:
    resolved_period = max(int(seasonal_period), 1)
    insample_values = np.asarray(insample_series, dtype=float)
    if len(insample_values) <= resolved_period:
        return math.inf
    naive_errors = np.abs(insample_values[resolved_period:] - insample_values[:-resolved_period])
    denominator = float(np.mean(naive_errors))
    if abs(denominator) <= _FLOAT_COMPARISON_EPSILON:
        return math.inf
    return mae_score(y_true, y_pred) / denominator


def lag_baseline_predictions(
    history_values: pd.Series | np.ndarray,
    future_values: pd.Series | np.ndarray,
    lag: int,
) -> pd.Series:
    resolved_lag = max(int(lag), 1)
    observed_values = list(np.asarray(history_values, dtype=float))
    predictions: list[float] = []
    for actual_value in np.asarray(future_values, dtype=float):
        if len(observed_values) < resolved_lag:
            predictions.append(float(observed_values[-1]))
        else:
            predictions.append(float(observed_values[-resolved_lag]))
        observed_values.append(float(actual_value))
    return pd.Series(predictions, dtype=float)


def interval_coverage(
    y_true: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> float | None:
    actual = np.asarray(y_true, dtype=float)
    lower_bound = np.asarray(lower, dtype=float)
    upper_bound = np.asarray(upper, dtype=float)
    if len(actual) == 0 or np.isnan(lower_bound).all() or np.isnan(upper_bound).all():
        return None
    covered = (actual >= lower_bound) & (actual <= upper_bound)
    return float(np.mean(covered))


def interval_width(
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> float | None:
    lower_bound = np.asarray(lower, dtype=float)
    upper_bound = np.asarray(upper, dtype=float)
    if len(lower_bound) == 0 or np.isnan(lower_bound).all() or np.isnan(upper_bound).all():
        return None
    return float(np.nanmean(upper_bound - lower_bound))


def pinball_loss(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    quantile: float,
) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = actual - predicted
    resolved_quantile = float(quantile)
    return float(np.mean(np.maximum(resolved_quantile * errors, (resolved_quantile - 1.0) * errors)))


def _prediction_quantile_columns(predictions_df: pd.DataFrame) -> list[tuple[str, float]]:
    resolved_columns: list[tuple[str, float]] = []
    for column in predictions_df.columns:
        match = re.fullmatch(r"prediction_p([0-9_]+)", column)
        if match is None:
            continue
        percentage_value = float(match.group(1).replace("_", "."))
        resolved_columns.append((column, percentage_value / 100.0))
    return sorted(resolved_columns, key=lambda item: item[1])


def _pinball_metric_key(quantile: float) -> str:
    return f"pinball_loss_p{format(quantile * 100.0, 'g').replace('.', '_')}"


def build_predictions_frame(
    reference_test_df: pd.DataFrame,
    *,
    actual: pd.Series | np.ndarray,
    prediction_raw: pd.Series | np.ndarray,
    train_rows_used: int,
    quantile_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output = _base_predictions_frame(
        reference_test_df=reference_test_df,
        actual=actual,
        prediction_raw=prediction_raw,
        train_rows_used=train_rows_used,
    )
    if quantile_predictions is None:
        return output.reset_index(drop=True)
    if len(quantile_predictions) != len(output):
        raise ValueError("Quantile prediction rows must align with the reference scoring frame.")

    return _apply_quantile_predictions(output, quantile_predictions.reset_index(drop=True).copy())


def compute_metrics_payload(
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict[str, object]:
    history_by_product: dict[str, pd.DataFrame] = {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in history_df.groupby(REFERENCE_PRODUCT_COL, sort=True)
    }
    per_product_metrics: dict[str, dict[str, float | int | None]] = {}
    for product_name, product_predictions in predictions_df.groupby("product", sort=True):
        per_product_metrics[str(product_name)] = _per_product_metrics_payload(
            history_df=history_by_product[str(product_name)],
            product_predictions_df=product_predictions,
        )

    actual_series = predictions_df["actual"].astype(float)
    predicted_series = predictions_df["prediction_raw"].astype(float)
    overall_lower_80 = predictions_df["lower_80"]
    overall_upper_80 = predictions_df["upper_80"]
    overall_lower_95 = predictions_df["lower_95"]
    overall_upper_95 = predictions_df["upper_95"]
    product_maes = [float(payload["mae"]) for payload in per_product_metrics.values() if payload["mae"] is not None]
    overall_metrics = _overall_metrics_payload(
        predictions_df=predictions_df,
        actual_series=actual_series,
        predicted_series=predicted_series,
        overall_lower_80=overall_lower_80,
        overall_upper_80=overall_upper_80,
        overall_lower_95=overall_lower_95,
        overall_upper_95=overall_upper_95,
        product_maes=product_maes,
    )
    return {
        "product_count": int(len(per_product_metrics)),
        "overall_metrics": overall_metrics,
        "per_product_metrics": per_product_metrics,
    }


def _base_predictions_frame(
    *,
    reference_test_df: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    prediction_raw: pd.Series | np.ndarray,
    train_rows_used: int,
) -> pd.DataFrame:
    reference_dates = pd.to_datetime(reference_test_df[REFERENCE_DATE_COL])
    point_predictions = np.asarray(prediction_raw, dtype=float)
    return pd.DataFrame(
        {
            "origin_date": (reference_dates - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
            "target_date": reference_dates.dt.strftime("%Y-%m-%d"),
            "actual": np.asarray(actual, dtype=float),
            "prediction_raw": point_predictions,
            "prediction_rounded": np.round(point_predictions, 0),
            "lower_80": np.nan,
            "upper_80": np.nan,
            "lower_95": np.nan,
            "upper_95": np.nan,
            "train_rows_used": int(train_rows_used),
            "product": reference_test_df[REFERENCE_PRODUCT_COL].astype(str).tolist(),
        }
    )


def _supported_prediction_quantiles(quantile_frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["prediction_p2_5", "prediction_p10", "prediction_p50", "prediction_p90", "prediction_p97_5"]
        if column in quantile_frame.columns
    ]


def _apply_quantile_predictions(output: pd.DataFrame, quantile_frame: pd.DataFrame) -> pd.DataFrame:
    for column in _supported_prediction_quantiles(quantile_frame):
        output[column] = quantile_frame[column].to_numpy(dtype=float)
    _apply_median_prediction(output)
    _apply_interval_columns(output)
    return output.reset_index(drop=True)


def _apply_median_prediction(output: pd.DataFrame) -> None:
    if "prediction_p50" not in output.columns:
        return
    output["prediction_raw"] = output["prediction_p50"].to_numpy(dtype=float)
    output["prediction_rounded"] = np.round(output["prediction_raw"].to_numpy(dtype=float), 0)


def _apply_interval_columns(output: pd.DataFrame) -> None:
    if "prediction_p10" in output.columns and "prediction_p90" in output.columns:
        output["lower_80"] = output["prediction_p10"].to_numpy(dtype=float)
        output["upper_80"] = output["prediction_p90"].to_numpy(dtype=float)
    if "prediction_p2_5" in output.columns and "prediction_p97_5" in output.columns:
        output["lower_95"] = output["prediction_p2_5"].to_numpy(dtype=float)
        output["upper_95"] = output["prediction_p97_5"].to_numpy(dtype=float)


def _per_product_metrics_payload(
    *,
    history_df: pd.DataFrame,
    product_predictions_df: pd.DataFrame,
) -> dict[str, float | int | None]:
    history_values = history_df[REFERENCE_TARGET_COL].astype(float)
    y_true = product_predictions_df["actual"].astype(float)
    y_pred = product_predictions_df["prediction_raw"].astype(float)
    lower_80 = product_predictions_df["lower_80"]
    upper_80 = product_predictions_df["upper_80"]
    lower_95 = product_predictions_df["lower_95"]
    upper_95 = product_predictions_df["upper_95"]
    lag_1_baseline = lag_baseline_predictions(history_values, y_true, lag=1)
    seasonal_baseline = lag_baseline_predictions(history_values, y_true, lag=7)
    return {
        "test_rows": int(len(product_predictions_df)),
        "mae": mae_score(y_true, y_pred),
        "rmse": rmse_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase_score(y_true, y_pred, history_values, seasonal_period=7),
        "naive_lag_1_mae": mae_score(y_true, lag_1_baseline),
        "seasonal_naive_lag_7_mae": mae_score(y_true, seasonal_baseline),
        "coverage_80": interval_coverage(y_true, lower_80, upper_80),
        "coverage_95": interval_coverage(y_true, lower_95, upper_95),
    }


def _overall_metrics_payload(
    *,
    predictions_df: pd.DataFrame,
    actual_series: pd.Series,
    predicted_series: pd.Series,
    overall_lower_80: pd.Series,
    overall_upper_80: pd.Series,
    overall_lower_95: pd.Series,
    overall_upper_95: pd.Series,
    product_maes: list[float],
) -> dict[str, float | int | None]:
    return {
        "test_rows": int(len(predictions_df)),
        "mae": mae_score(actual_series, predicted_series),
        "rmse": rmse_score(actual_series, predicted_series),
        "smape": smape(actual_series, predicted_series),
        "coverage_80": interval_coverage(actual_series, overall_lower_80, overall_upper_80),
        "coverage_95": interval_coverage(actual_series, overall_lower_95, overall_upper_95),
        "mean_product_mae": float(np.mean(product_maes)),
    }


def compute_probabilistic_metrics_payload(predictions_df: pd.DataFrame) -> dict[str, object]:
    quantile_columns = _prediction_quantile_columns(predictions_df)
    actual_series = predictions_df["actual"].astype(float)
    overall_metrics: dict[str, float | None] = {
        "coverage_80": interval_coverage(actual_series, predictions_df["lower_80"], predictions_df["upper_80"]),
        "coverage_95": interval_coverage(actual_series, predictions_df["lower_95"], predictions_df["upper_95"]),
        "interval_width_80": interval_width(predictions_df["lower_80"], predictions_df["upper_80"]),
        "interval_width_95": interval_width(predictions_df["lower_95"], predictions_df["upper_95"]),
    }
    for column, quantile in quantile_columns:
        overall_metrics[_pinball_metric_key(quantile)] = pinball_loss(
            actual_series,
            predictions_df[column].astype(float),
            quantile,
        )

    per_product_metrics: dict[str, dict[str, float | None]] = {}
    for product_name, product_predictions in predictions_df.groupby("product", sort=True):
        product_predictions_df = product_predictions
        y_true = product_predictions_df["actual"].astype(float)
        product_metrics: dict[str, float | None] = {
            "coverage_80": interval_coverage(y_true, product_predictions_df["lower_80"], product_predictions_df["upper_80"]),
            "coverage_95": interval_coverage(y_true, product_predictions_df["lower_95"], product_predictions_df["upper_95"]),
            "interval_width_80": interval_width(product_predictions_df["lower_80"], product_predictions_df["upper_80"]),
            "interval_width_95": interval_width(product_predictions_df["lower_95"], product_predictions_df["upper_95"]),
        }
        for column, quantile in quantile_columns:
            product_metrics[_pinball_metric_key(quantile)] = pinball_loss(
                y_true,
                product_predictions_df[column].astype(float),
                quantile,
            )
        per_product_metrics[str(product_name)] = product_metrics

    return {
        "quantile_columns": [column for column, _ in quantile_columns],
        "overall_metrics": overall_metrics,
        "per_product_metrics": per_product_metrics,
    }
