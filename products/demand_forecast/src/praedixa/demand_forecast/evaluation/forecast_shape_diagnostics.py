from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_metrics import mae_score


STD_RATIO_LOW_VARIANCE_THRESHOLD = 0.50
HIGH_ACTUAL_QUANTILE = 0.80
LOW_ACTUAL_QUANTILE = 0.20
_FLOAT_COMPARISON_EPSILON = 1e-8


def build_forecast_shape_diagnostics(predictions_df: pd.DataFrame) -> dict[str, Any]:
    if len(predictions_df) == 0:
        return _empty_shape_diagnostics()
    working_df = _shape_working_frame(predictions_df)
    per_product = _per_product_shape_payloads(working_df)
    return {
        "purpose": "Detect forecasts that win aggregate MAE by over-smoothing toward product-level central values.",
        "thresholds": {
            "low_variance_std_ratio": float(STD_RATIO_LOW_VARIANCE_THRESHOLD),
            "high_actual_quantile": float(HIGH_ACTUAL_QUANTILE),
            "low_actual_quantile": float(LOW_ACTUAL_QUANTILE),
        },
        "overall": _overall_shape_payload(working_df, per_product),
        "daily": _daily_shape_payload(working_df),
        "actual_regime": _actual_regime_payload(working_df),
        "per_product": per_product,
    }


def _empty_shape_diagnostics() -> dict[str, Any]:
    return {
        "purpose": "Detect forecasts that win aggregate MAE by over-smoothing toward product-level central values.",
        "thresholds": {
            "low_variance_std_ratio": float(STD_RATIO_LOW_VARIANCE_THRESHOLD),
            "high_actual_quantile": float(HIGH_ACTUAL_QUANTILE),
            "low_actual_quantile": float(LOW_ACTUAL_QUANTILE),
        },
        "overall": {},
        "daily": {},
        "actual_regime": {},
        "per_product": {},
    }


def _shape_working_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    frame = predictions_df.copy()
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce").astype(float)
    frame["prediction_raw"] = pd.to_numeric(
        frame["prediction_raw"],
        errors="coerce",
    ).astype(float)
    frame["product"] = frame["product"].astype(str)
    frame["model_error"] = frame["prediction_raw"] - frame["actual"]
    frame["model_abs_error"] = frame["model_error"].abs()
    frame["product_mean_prediction"] = frame.groupby("product")["prediction_raw"].transform("mean")
    frame["product_median_prediction"] = frame.groupby("product")["prediction_raw"].transform("median")
    frame["product_mean_prediction_abs_error"] = (
        frame["product_mean_prediction"] - frame["actual"]
    ).abs()
    frame["product_median_prediction_abs_error"] = (
        frame["product_median_prediction"] - frame["actual"]
    ).abs()
    frame["product_actual_q80"] = frame.groupby("product")["actual"].transform(
        lambda series: float(series.quantile(HIGH_ACTUAL_QUANTILE))
    )
    frame["product_actual_q20"] = frame.groupby("product")["actual"].transform(
        lambda series: float(series.quantile(LOW_ACTUAL_QUANTILE))
    )
    if "best_statistical_baseline_prediction" in frame.columns:
        frame["baseline_prediction"] = pd.to_numeric(
            frame["best_statistical_baseline_prediction"],
            errors="coerce",
        ).astype(float)
        frame["baseline_error"] = frame["baseline_prediction"] - frame["actual"]
        frame["baseline_abs_error"] = frame["baseline_error"].abs()
    return frame


def _per_product_shape_payloads(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    per_product: dict[str, dict[str, Any]] = {}
    for product_name, product_frame in frame.groupby("product", sort=True):
        per_product[str(product_name)] = _product_shape_payload(product_frame)
    return per_product


def _product_shape_payload(product_frame: pd.DataFrame) -> dict[str, Any]:
    actual = product_frame["actual"].astype(float)
    predicted = product_frame["prediction_raw"].astype(float)
    model_mae = mae_score(actual, predicted)
    constant_mean_mae = float(product_frame["product_mean_prediction_abs_error"].mean())
    constant_median_mae = float(product_frame["product_median_prediction_abs_error"].mean())
    actual_std = _series_std(actual)
    predicted_std = _series_std(predicted)
    std_ratio = _safe_ratio(predicted_std, actual_std)
    payload: dict[str, Any] = {
        "rows": int(len(product_frame)),
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predicted.mean()),
        "actual_std": actual_std,
        "prediction_std": predicted_std,
        "prediction_actual_std_ratio": std_ratio,
        "actual_prediction_correlation": _safe_correlation(actual, predicted),
        "model_mae": model_mae,
        "constant_product_mean_mae": constant_mean_mae,
        "constant_product_median_mae": constant_median_mae,
        "mae_gain_vs_constant_product_mean": float(constant_mean_mae - model_mae),
        "mae_gain_vs_constant_product_median": float(constant_median_mae - model_mae),
        "constant_product_mean_beats_model": bool(constant_mean_mae < model_mae),
        "constant_product_median_beats_model": bool(constant_median_mae < model_mae),
    }
    return payload


def _overall_shape_payload(
    frame: pd.DataFrame,
    per_product: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    std_ratios = _numeric_payload_values(per_product, "prediction_actual_std_ratio")
    correlations = _numeric_payload_values(per_product, "actual_prediction_correlation")
    model_mae = float(frame["model_abs_error"].mean())
    constant_mean_mae = float(frame["product_mean_prediction_abs_error"].mean())
    constant_median_mae = float(frame["product_median_prediction_abs_error"].mean())
    return {
        "row_count": int(len(frame)),
        "product_count": int(len(per_product)),
        "model_mae": model_mae,
        "constant_product_mean_mae": constant_mean_mae,
        "constant_product_median_mae": constant_median_mae,
        "mae_gain_vs_constant_product_mean": float(constant_mean_mae - model_mae),
        "mae_gain_vs_constant_product_median": float(constant_median_mae - model_mae),
        "products_where_constant_mean_beats_model": _count_bool(
            per_product,
            "constant_product_mean_beats_model",
        ),
        "products_where_constant_median_beats_model": _count_bool(
            per_product,
            "constant_product_median_beats_model",
        ),
        "products_low_variance_std_ratio_count": int(
            sum(value < STD_RATIO_LOW_VARIANCE_THRESHOLD for value in std_ratios)
        ),
        "product_std_ratio": _summary_stats(std_ratios),
        "product_actual_prediction_correlation": _summary_stats(correlations),
    }


def _daily_shape_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if "target_date" not in frame.columns:
        return {}
    grouped = frame.groupby("target_date", sort=True).agg(
        actual_sum=("actual", "sum"),
        prediction_sum=("prediction_raw", "sum"),
        model_mae=("model_abs_error", "mean"),
    )
    payload: dict[str, Any] = {
        "day_count": int(len(grouped)),
        "daily_model_mae_mean": float(grouped["model_mae"].mean()),
        "daily_actual_prediction_sum_correlation": _safe_correlation(
            grouped["actual_sum"],
            grouped["prediction_sum"],
        ),
        "daily_model_total_abs_error_mean": float(
            (grouped["prediction_sum"] - grouped["actual_sum"]).abs().mean()
        ),
    }
    if "baseline_prediction" in frame.columns:
        baseline_grouped = frame.groupby("target_date", sort=True).agg(
            baseline_sum=("baseline_prediction", "sum"),
            baseline_mae=("baseline_abs_error", "mean"),
        )
        merged = grouped.join(baseline_grouped, how="inner")
        payload.update(
            {
                "daily_baseline_mae_mean": float(merged["baseline_mae"].mean()),
                "days_model_mae_beats_baseline": int(
                    (merged["model_mae"] < merged["baseline_mae"]).sum()
                ),
                "days_baseline_mae_beats_model": int(
                    (merged["baseline_mae"] < merged["model_mae"]).sum()
                ),
                "days_model_mae_equals_baseline": int(
                    (merged["baseline_mae"] == merged["model_mae"]).sum()
                ),
                "daily_baseline_total_abs_error_mean": float(
                    (merged["baseline_sum"] - merged["actual_sum"]).abs().mean()
                ),
                "daily_actual_baseline_sum_correlation": _safe_correlation(
                    merged["actual_sum"],
                    merged["baseline_sum"],
                ),
            }
        )
    return payload


def _actual_regime_payload(frame: pd.DataFrame) -> dict[str, Any]:
    high_frame = frame.loc[frame["actual"] >= frame["product_actual_q80"]].copy()
    low_frame = frame.loc[frame["actual"] <= frame["product_actual_q20"]].copy()
    return {
        "high_actual_days": _regime_metrics(high_frame),
        "low_actual_days": _regime_metrics(low_frame),
    }


def _regime_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "model_mae": _mean_or_none(frame["model_abs_error"]),
        "model_bias": _mean_or_none(frame["model_error"]),
    }
    if "baseline_abs_error" in frame.columns:
        metrics["baseline_mae"] = _mean_or_none(frame["baseline_abs_error"])
        metrics["baseline_bias"] = _mean_or_none(frame["baseline_error"])
    return metrics


def _series_std(series: pd.Series) -> float:
    return float(series.astype(float).std(ddof=0))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= _FLOAT_COMPARISON_EPSILON:
        return None
    return float(numerator / denominator)


def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    left_std = _series_std(left)
    right_std = _series_std(right)
    if abs(left_std) <= _FLOAT_COMPARISON_EPSILON or abs(right_std) <= _FLOAT_COMPARISON_EPSILON:
        return None
    return float(left.astype(float).corr(right.astype(float)))


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    numeric_values = np.asarray(values, dtype=float)
    return {
        "count": int(len(numeric_values)),
        "mean": float(np.mean(numeric_values)),
        "median": float(np.median(numeric_values)),
        "min": float(np.min(numeric_values)),
        "max": float(np.max(numeric_values)),
    }


def _numeric_payload_values(
    payload: dict[str, dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for item in payload.values():
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _count_bool(payload: dict[str, dict[str, Any]], key: str) -> int:
    return int(sum(bool(item.get(key)) for item in payload.values()))


def _mean_or_none(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(series.astype(float).mean())
