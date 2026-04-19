from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd


REFERENCE_DATE_COL = "date"
REFERENCE_PRODUCT_COL = "product"
REFERENCE_TARGET_COL = "quantity"
FIELD_BASELINE_NAME = "blend_lag_1_lag_7_50_50"
DEFAULT_UNIT_COST_EUR = 1.0
DEFAULT_UNIT_SALE_PRICE_EUR = 1.0
DEFAULT_PRODUCTION_COST_RATIO = 0.35
DEFAULT_RAW_SALES_CSV = Path("bakery_sales/data/Bakery sales.csv")
INVALID_ARTICLES: tuple[str, ...] = (
    "COUPON",
    "DECOUVERTE",
    "ARTICLE ANNULER",
    "MERCI DE VOTRE VISITE",
)

HistoryForecaster = Callable[[list[float], pd.Timestamp], float]


def load_reference_split(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame[REFERENCE_DATE_COL] = pd.to_datetime(frame[REFERENCE_DATE_COL], format="%Y-%m-%d")
    return frame.sort_values([REFERENCE_PRODUCT_COL, REFERENCE_DATE_COL]).reset_index(drop=True)


def mae_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


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
    if denominator == 0.0:
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


def build_predictions_frame(
    reference_test_df: pd.DataFrame,
    *,
    actual: pd.Series | np.ndarray,
    prediction_raw: pd.Series | np.ndarray,
    train_rows_used: int,
) -> pd.DataFrame:
    target_dates = cast(pd.Series, reference_test_df[REFERENCE_DATE_COL]).dt.strftime("%Y-%m-%d")
    output = pd.DataFrame(
        {
            "origin_date": (cast(pd.Series, reference_test_df[REFERENCE_DATE_COL]) - pd.Timedelta(days=1)).dt.strftime(
                "%Y-%m-%d"
            ),
            "target_date": target_dates,
            "actual": np.asarray(actual, dtype=float),
            "prediction_raw": np.asarray(prediction_raw, dtype=float),
            "prediction_rounded": np.round(np.asarray(prediction_raw, dtype=float), 0),
            "lower_80": np.nan,
            "upper_80": np.nan,
            "lower_95": np.nan,
            "upper_95": np.nan,
            "train_rows_used": int(train_rows_used),
            "product": cast(pd.Series, reference_test_df[REFERENCE_PRODUCT_COL]).astype(str).tolist(),
        }
    )
    return output.reset_index(drop=True)


def compute_metrics_payload(
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict[str, Any]:
    history_by_product = {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in history_df.groupby(REFERENCE_PRODUCT_COL, sort=True)
    }
    per_product_metrics: dict[str, Any] = {}
    for product_name, product_predictions_df in predictions_df.groupby("product", sort=True):
        history_values = cast(pd.Series, history_by_product[str(product_name)][REFERENCE_TARGET_COL]).astype(float)
        y_true = cast(pd.Series, product_predictions_df["actual"]).astype(float)
        y_pred = cast(pd.Series, product_predictions_df["prediction_raw"]).astype(float)
        lag_1_baseline = lag_baseline_predictions(history_values, y_true, lag=1)
        seasonal_baseline = lag_baseline_predictions(history_values, y_true, lag=7)
        per_product_metrics[str(product_name)] = {
            "test_rows": int(len(product_predictions_df)),
            "mae": mae_score(y_true, y_pred),
            "rmse": rmse_score(y_true, y_pred),
            "smape": smape(y_true, y_pred),
            "mase": mase_score(y_true, y_pred, history_values, seasonal_period=7),
            "naive_lag_1_mae": mae_score(y_true, lag_1_baseline),
            "seasonal_naive_lag_7_mae": mae_score(y_true, seasonal_baseline),
            "coverage_80": interval_coverage(y_true, product_predictions_df["lower_80"], product_predictions_df["upper_80"]),
            "coverage_95": interval_coverage(y_true, product_predictions_df["lower_95"], product_predictions_df["upper_95"]),
        }

    actual_series = cast(pd.Series, predictions_df["actual"]).astype(float)
    predicted_series = cast(pd.Series, predictions_df["prediction_raw"]).astype(float)
    overall_metrics = {
        "test_rows": int(len(predictions_df)),
        "mae": mae_score(actual_series, predicted_series),
        "rmse": rmse_score(actual_series, predicted_series),
        "smape": smape(actual_series, predicted_series),
        "coverage_80": interval_coverage(actual_series, predictions_df["lower_80"], predictions_df["upper_80"]),
        "coverage_95": interval_coverage(actual_series, predictions_df["lower_95"], predictions_df["upper_95"]),
        "mean_product_mae": float(np.mean([payload["mae"] for payload in per_product_metrics.values()])),
    }
    return {
        "product_count": int(len(per_product_metrics)),
        "overall_metrics": overall_metrics,
        "per_product_metrics": per_product_metrics,
    }


def _array_mean(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.mean(values))


def _array_median(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.median(values))


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("date cannot be NaT")
    return cast(pd.Timestamp, timestamp)


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
    date_series = cast(pd.Series, history_df[REFERENCE_DATE_COL])
    mask = date_series.dt.dayofweek == forecast_date.dayofweek
    values = cast(pd.Series, history_df.loc[mask, REFERENCE_TARGET_COL]).astype(float).tolist()
    return values


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


def _blend_forecaster(weights: dict[str, float]) -> Callable[[dict[str, float]], float]:
    def forecast(predictions: dict[str, float]) -> float:
        return float(sum(predictions[name] * weight for name, weight in weights.items()))

    return forecast


def _year_ago_date_candidates(forecast_date: pd.Timestamp) -> list[pd.Timestamp]:
    return [
        cast(pd.Timestamp, forecast_date - pd.Timedelta(days=365)),
        cast(pd.Timestamp, forecast_date - pd.Timedelta(days=364)),
        cast(pd.Timestamp, forecast_date - pd.Timedelta(days=366)),
    ]


def _same_day_last_year_value(history_df: pd.DataFrame, forecast_date: pd.Timestamp) -> float | None:
    indexed_history = history_df.set_index(REFERENCE_DATE_COL)
    for candidate_date in _year_ago_date_candidates(forecast_date):
        if candidate_date in indexed_history.index:
            return float(cast(float, indexed_history.at[candidate_date, REFERENCE_TARGET_COL]))
    return None


def _predict_baseline(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor: HistoryForecaster | Callable[[dict[str, float]], float],
    *,
    precomputed_names: list[str] | None = None,
    prediction_cache: dict[str, list[float]] | None = None,
) -> pd.Series:
    observed_history_df = history_df.copy()
    observed_values = cast(pd.Series, observed_history_df[REFERENCE_TARGET_COL]).astype(float).tolist()
    predictions: list[float] = []
    future_records = cast(list[dict[str, Any]], test_df.to_dict(orient="records"))
    for row in future_records:
        forecast_date = _coerce_timestamp(row[REFERENCE_DATE_COL])
        if precomputed_names is None:
            direct_predictor = cast(HistoryForecaster, predictor)
            predicted_value = float(direct_predictor(observed_values, forecast_date))
        else:
            if prediction_cache is None:
                raise RuntimeError("prediction_cache is required for blended baselines")
            component_predictions = {
                name: prediction_cache[name][len(predictions)]
                for name in precomputed_names
            }
            blended_predictor = cast(Callable[[dict[str, float]], float], predictor)
            predicted_value = float(blended_predictor(component_predictions))
        predictions.append(predicted_value)
        appended_row = pd.DataFrame([row]).assign(
            **{REFERENCE_DATE_COL: lambda df: pd.to_datetime(df[REFERENCE_DATE_COL])}
        )
        observed_history_df = pd.concat([observed_history_df, appended_row], ignore_index=True)
        observed_values.append(float(cast(float, row[REFERENCE_TARGET_COL])))
    return pd.Series(predictions, dtype=float)


def _same_weekday_predictor(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reducer: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    window: int | None,
) -> pd.Series:
    observed_history_df = history_df.copy()
    predictions: list[float] = []
    future_records = cast(list[dict[str, Any]], test_df.to_dict(orient="records"))
    for row in future_records:
        forecast_date = _coerce_timestamp(row[REFERENCE_DATE_COL])
        weekday_history = _same_weekday_history(observed_history_df, forecast_date)
        if not weekday_history:
            predictions.append(
                float(cast(pd.Series, observed_history_df[REFERENCE_TARGET_COL]).astype(float).iloc[-1])
            )
        else:
            if window is not None:
                weekday_history = weekday_history[-window:]
            predictions.append(float(reducer(np.asarray(weekday_history, dtype=float))))
        appended_row = pd.DataFrame([row]).assign(
            **{REFERENCE_DATE_COL: lambda df: pd.to_datetime(df[REFERENCE_DATE_COL])}
        )
        observed_history_df = pd.concat([observed_history_df, appended_row], ignore_index=True)
    return pd.Series(predictions, dtype=float)


def _same_day_last_year_predictions(history_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    observed_history_df = history_df.copy()
    predictions: list[float] = []
    future_records = cast(list[dict[str, Any]], test_df.to_dict(orient="records"))
    for row in future_records:
        forecast_date = _coerce_timestamp(row[REFERENCE_DATE_COL])
        year_ago_value = _same_day_last_year_value(observed_history_df, forecast_date)
        if year_ago_value is None:
            year_ago_value = float(cast(pd.Series, observed_history_df[REFERENCE_TARGET_COL]).astype(float).iloc[-1])
        predictions.append(year_ago_value)
        appended_row = pd.DataFrame([row]).assign(
            **{REFERENCE_DATE_COL: lambda df: pd.to_datetime(df[REFERENCE_DATE_COL])}
        )
        observed_history_df = pd.concat([observed_history_df, appended_row], ignore_index=True)
    return pd.Series(predictions, dtype=float)


def _baseline_metrics(
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


def _prediction_rows_payload(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> list[dict[str, float | str]]:
    target_dates = cast(pd.Series, test_df[REFERENCE_DATE_COL]).dt.strftime("%Y-%m-%d")
    payload_df = pd.DataFrame(
        {
            "product": cast(pd.Series, test_df[REFERENCE_PRODUCT_COL]).astype(str),
            "origin_date": (cast(pd.Series, test_df[REFERENCE_DATE_COL]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
            "target_date": target_dates,
            "actual": y_true.astype(float),
            "prediction": y_pred.astype(float),
        }
    )
    payload_df["absolute_error"] = (payload_df["actual"] - payload_df["prediction"]).abs()
    return cast(list[dict[str, float | str]], payload_df.to_dict(orient="records"))


def _baseline_definitions() -> tuple[
    dict[str, HistoryForecaster],
    dict[str, tuple[list[str], Callable[[dict[str, float]], float]]],
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
    blended_baselines: dict[str, tuple[list[str], Callable[[dict[str, float]], float]]] = {
        "blend_lag_1_lag_7_50_50": (
            ["naive_lag_1", "seasonal_naive_lag_7"],
            _blend_forecaster({"naive_lag_1": 0.5, "seasonal_naive_lag_7": 0.5}),
        ),
        "blend_roll_mean_7_ewm_7_50_50": (
            ["rolling_mean_7", "ewm_span_7"],
            _blend_forecaster({"rolling_mean_7": 0.5, "ewm_span_7": 0.5}),
        ),
    }
    return direct_baselines, blended_baselines


def _same_weekday_baselines(history_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "same_weekday_mean_4": _same_weekday_predictor(history_df, test_df, _array_mean, 4),
        "same_weekday_median_4": _same_weekday_predictor(history_df, test_df, _array_median, 4),
        "same_weekday_mean_expanding": _same_weekday_predictor(history_df, test_df, _array_mean, None),
        "same_weekday_median_expanding": _same_weekday_predictor(history_df, test_df, _array_median, None),
        "same_day_last_year": _same_day_last_year_predictions(history_df, test_df),
    }


def _ranked_rows(metrics_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    ranked_rows = sorted(metrics_rows, key=lambda row: (float(row["mae"]), float(row["rmse"]), str(row["name"])))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_mae"] = rank
    return ranked_rows


def _product_payload(
    product_name: str,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.Series], pd.Series, pd.Series]:
    y_true = cast(pd.Series, test_df[REFERENCE_TARGET_COL].astype(float).reset_index(drop=True))
    insample_series = cast(pd.Series, history_df[REFERENCE_TARGET_COL].astype(float).reset_index(drop=True))
    direct_baselines, blended_baselines = _baseline_definitions()
    prediction_cache: dict[str, list[float]] = {}
    prediction_series_by_name: dict[str, pd.Series] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for name, predictor in direct_baselines.items():
        predictions = _predict_baseline(history_df, test_df, predictor)
        prediction_cache[name] = predictions.tolist()
        prediction_series_by_name[name] = predictions
        metrics_rows.append(_baseline_metrics(name, y_true, predictions, insample_series))

    for name, predictions in _same_weekday_baselines(history_df, test_df).items():
        prediction_cache[name] = predictions.tolist()
        prediction_series_by_name[name] = predictions
        metrics_rows.append(_baseline_metrics(name, y_true, predictions, insample_series))

    for name, (component_names, predictor) in blended_baselines.items():
        predictions = _predict_baseline(
            history_df=history_df,
            test_df=test_df,
            predictor=predictor,
            precomputed_names=component_names,
            prediction_cache=prediction_cache,
        )
        prediction_series_by_name[name] = predictions
        metrics_rows.append(_baseline_metrics(name, y_true, predictions, insample_series))

    ranked_rows = _ranked_rows(metrics_rows)
    best_baseline_name = str(ranked_rows[0]["name"])
    best_baseline_predictions = prediction_series_by_name[best_baseline_name]
    best_payload = {
        **ranked_rows[0],
        "product": product_name,
        "prediction_rows": _prediction_rows_payload(test_df=test_df, y_true=y_true, y_pred=best_baseline_predictions),
    }
    field_baseline_predictions = prediction_series_by_name[FIELD_BASELINE_NAME]
    field_metrics_row = next(row for row in ranked_rows if str(row["name"]) == FIELD_BASELINE_NAME)
    field_payload = {
        **field_metrics_row,
        "product": product_name,
        "prediction_rows": _prediction_rows_payload(test_df=test_df, y_true=y_true, y_pred=field_baseline_predictions),
    }
    return best_payload, field_payload, prediction_series_by_name, y_true, insample_series


def build_statistical_baselines_payload(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    history_products = {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in history_df.groupby(REFERENCE_PRODUCT_COL, sort=True)
    }
    test_products = {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in test_df.groupby(REFERENCE_PRODUCT_COL, sort=True)
    }

    aggregate_predictions: dict[str, list[pd.DataFrame]] = {}
    aggregate_insample: dict[str, list[pd.Series]] = {}
    per_product_best_baselines: dict[str, Any] = {}
    per_product_field_baselines: dict[str, Any] = {}

    for product_name, product_test_df in test_products.items():
        (
            product_best_payload,
            product_field_payload,
            prediction_series_by_name,
            y_true,
            insample_series,
        ) = _product_payload(
            product_name=product_name,
            history_df=history_products[product_name],
            test_df=product_test_df,
        )
        per_product_best_baselines[product_name] = product_best_payload
        per_product_field_baselines[product_name] = product_field_payload
        for baseline_name, y_pred in prediction_series_by_name.items():
            aggregate_predictions.setdefault(baseline_name, []).append(
                pd.DataFrame(
                    {
                        REFERENCE_PRODUCT_COL: [product_name] * len(y_pred),
                        "y_true": y_true.astype(float).to_list(),
                        "y_pred": y_pred.astype(float).to_list(),
                    }
                )
            )
            aggregate_insample.setdefault(baseline_name, []).append(insample_series)

    aggregate_metrics_rows: list[dict[str, float | str]] = []
    for baseline_name, prediction_frames in aggregate_predictions.items():
        merged_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        y_true = cast(pd.Series, merged_predictions_df["y_true"]).astype(float)
        y_pred = cast(pd.Series, merged_predictions_df["y_pred"]).astype(float)
        insample_values = pd.concat(
            [series.reset_index(drop=True) for series in aggregate_insample[baseline_name]],
            ignore_index=True,
        )
        aggregate_metrics_rows.append(
            _baseline_metrics(
                baseline_name,
                y_true=y_true,
                y_pred=y_pred,
                insample_series=cast(pd.Series, insample_values.astype(float)),
            )
        )

    ranked_rows = _ranked_rows(aggregate_metrics_rows)
    best_baseline_name = str(ranked_rows[0]["name"])
    best_prediction_frames = pd.concat(aggregate_predictions[best_baseline_name], ignore_index=True)
    best_baseline_payload = {
        **ranked_rows[0],
        "prediction_rows": _prediction_rows_payload(
            test_df=test_df.reset_index(drop=True),
            y_true=cast(pd.Series, best_prediction_frames["y_true"]).astype(float).reset_index(drop=True),
            y_pred=cast(pd.Series, best_prediction_frames["y_pred"]).astype(float).reset_index(drop=True),
        ),
    }

    return {
        "date_column": REFERENCE_DATE_COL,
        "product_column": REFERENCE_PRODUCT_COL,
        "target_column": REFERENCE_TARGET_COL,
        "history_rows": int(len(history_df)),
        "test_rows": int(len(test_df)),
        "product_count": int(len(test_products)),
        "history_range": {
            "start": str(cast(pd.Timestamp, cast(pd.Series, history_df[REFERENCE_DATE_COL]).min()).date()),
            "end": str(cast(pd.Timestamp, cast(pd.Series, history_df[REFERENCE_DATE_COL]).max()).date()),
        },
        "test_range": {
            "start": str(cast(pd.Timestamp, cast(pd.Series, test_df[REFERENCE_DATE_COL]).min()).date()),
            "end": str(cast(pd.Timestamp, cast(pd.Series, test_df[REFERENCE_DATE_COL]).max()).date()),
        },
        "baseline_count": int(len(ranked_rows)),
        "field_baseline_name": FIELD_BASELINE_NAME,
        "best_baseline": best_baseline_payload,
        "baselines_ranked_by_mae": ranked_rows,
        "per_product_best_baselines": per_product_best_baselines,
        "per_product_field_baselines": per_product_field_baselines,
    }


def enrich_predictions_with_best_baseline(
    predictions_df: pd.DataFrame,
    statistical_baselines_payload: dict[str, Any] | None,
    *,
    unit_cost_eur: float = DEFAULT_UNIT_COST_EUR,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if statistical_baselines_payload is None:
        return predictions_df, None
    best_baseline = cast(dict[str, Any] | None, statistical_baselines_payload.get("best_baseline"))
    if not best_baseline:
        return predictions_df, None
    prediction_rows = cast(list[dict[str, Any]], best_baseline.get("prediction_rows", []))
    if len(prediction_rows) != len(predictions_df):
        return predictions_df, None
    baseline_df = pd.DataFrame(prediction_rows)
    if (
        baseline_df["origin_date"].astype(str).tolist() != predictions_df["origin_date"].astype(str).tolist()
        or baseline_df["target_date"].astype(str).tolist() != predictions_df["target_date"].astype(str).tolist()
        or baseline_df["product"].astype(str).tolist() != predictions_df["product"].astype(str).tolist()
    ):
        return predictions_df, None
    enriched_df = predictions_df.copy()
    actual_series = cast(pd.Series, enriched_df["actual"]).astype(float).reset_index(drop=True)
    model_prediction_series = cast(pd.Series, enriched_df["prediction_raw"]).astype(float).reset_index(drop=True)
    model_abs_error = (actual_series - model_prediction_series).abs()
    baseline_prediction = baseline_df["prediction"].astype(float).reset_index(drop=True)
    baseline_abs_error = baseline_df["absolute_error"].astype(float).reset_index(drop=True)
    per_product_best_baselines = cast(dict[str, Any], statistical_baselines_payload.get("per_product_best_baselines", {}))
    model_mae_value = mae_score(actual_series, model_prediction_series)
    best_baseline_mae = float(best_baseline["mae"])
    rounded_model_mae = int(round(model_mae_value))
    rounded_best_baseline_mae = int(round(best_baseline_mae))
    rounded_mae_gap_units = rounded_best_baseline_mae - rounded_model_mae
    test_day_count = int(len(predictions_df))
    estimated_savings_eur = float(rounded_mae_gap_units * float(unit_cost_eur) * test_day_count)
    enriched_df["best_statistical_baseline_name"] = str(best_baseline["name"])
    enriched_df["best_statistical_baseline_prediction"] = baseline_prediction
    enriched_df["best_statistical_baseline_abs_error"] = baseline_abs_error
    enriched_df["best_statistical_baseline_product_mae"] = [
        float(cast(dict[str, Any], per_product_best_baselines.get(str(product_name), {})).get("mae", best_baseline_mae))
        for product_name in cast(pd.Series, enriched_df["product"]).astype(str)
    ]
    enriched_df["model_abs_error"] = model_abs_error
    enriched_df["absolute_error_saved_vs_best_baseline"] = baseline_abs_error - model_abs_error
    savings_series = cast(pd.Series, enriched_df["absolute_error_saved_vs_best_baseline"]).astype(float)
    enriched_df["relative_error_reduction_vs_best_baseline"] = np.where(
        baseline_abs_error > 0.0,
        enriched_df["absolute_error_saved_vs_best_baseline"] / baseline_abs_error,
        0.0,
    )
    savings_payload: dict[str, Any] = {
        "best_statistical_baseline_name": str(best_baseline["name"]),
        "best_statistical_baseline_mae": best_baseline_mae,
        "model_mae": float(model_mae_value),
        "absolute_mae_saved_vs_best_baseline": float(best_baseline_mae - model_mae_value),
        "rounded_best_statistical_baseline_mae_baguettes": rounded_best_baseline_mae,
        "rounded_model_mae_baguettes": rounded_model_mae,
        "rounded_mae_gap_baguettes_vs_best_baseline": rounded_mae_gap_units,
        "baguette_unit_cost_eur": float(unit_cost_eur),
        "test_day_count": test_day_count,
        "estimated_savings_eur_vs_best_baseline": estimated_savings_eur,
        "total_absolute_error_saved_vs_best_baseline": float(savings_series.sum()),
        "rows_better_than_best_baseline": int((savings_series > 0.0).sum()),
        "rows_equal_to_best_baseline": int((savings_series == 0.0).sum()),
        "rows_worse_than_best_baseline": int((savings_series < 0.0).sum()),
        "relative_mae_improvement_vs_best_baseline": float(
            0.0 if best_baseline_mae <= 0.0 else ((best_baseline_mae - model_mae_value) / best_baseline_mae)
        ),
    }
    return enriched_df, savings_payload


def build_simple_economic_gain_payload(
    predictions_df: pd.DataFrame,
    metrics_payload: dict[str, Any],
    *,
    raw_sales_csv: str | Path | None = DEFAULT_RAW_SALES_CSV,
    production_cost_ratio: float = DEFAULT_PRODUCTION_COST_RATIO,
) -> dict[str, Any]:
    per_product_metrics = cast(dict[str, Any], metrics_payload["per_product_metrics"])
    comparison_df = _economic_comparison_frame(predictions_df)
    if len(comparison_df) == 0:
        return {
            "model_family": "FOUNDATION_XGBOOST",
            "product_count": 0,
            "default_unit_sale_price_eur": float(DEFAULT_UNIT_SALE_PRICE_EUR),
            "production_cost_ratio": float(production_cost_ratio),
            "field_baseline_name": FIELD_BASELINE_NAME,
            "per_product_gain": {},
            "total_model_loss_eur": 0.0,
            "total_baseline_loss_eur": 0.0,
            "total_absolute_mae_saved_vs_best_baselines": 0.0,
            "total_estimated_savings_eur_vs_best_baselines": 0.0,
        }

    unit_sale_prices = _test_window_unit_prices(
        predictions_df=predictions_df,
        raw_sales_csv=raw_sales_csv,
    )
    priced_comparison_df = _apply_loss_model(
        comparison_df=comparison_df,
        unit_sale_prices=unit_sale_prices,
        production_cost_ratio=production_cost_ratio,
    )

    per_product_gain: dict[str, Any] = {}
    for product_name, metrics in per_product_metrics.items():
        product_comparison_df = priced_comparison_df.loc[
            cast(pd.Series, priced_comparison_df[REFERENCE_PRODUCT_COL]).astype(str) == str(product_name)
        ].copy()
        if len(product_comparison_df) == 0:
            continue
        per_product_gain[str(product_name)] = _product_gain_payload(
            product_df=product_comparison_df,
            product_metrics=cast(dict[str, Any], metrics),
        )

    return {
        "model_family": "FOUNDATION_XGBOOST",
        "product_count": int(len(per_product_gain)),
        "default_unit_sale_price_eur": float(DEFAULT_UNIT_SALE_PRICE_EUR),
        "production_cost_ratio": float(production_cost_ratio),
        "field_baseline_name": FIELD_BASELINE_NAME,
        "per_product_gain": per_product_gain,
        "total_model_loss_eur": float(
            sum(cast(float, payload["model_total_loss_eur"]) for payload in per_product_gain.values())
        ),
        "total_baseline_loss_eur": float(
            sum(cast(float, payload["baseline_total_loss_eur"]) for payload in per_product_gain.values())
        ),
        "total_absolute_mae_saved_vs_best_baselines": float(
            sum(cast(float, payload["absolute_mae_saved_vs_best_baseline"]) for payload in per_product_gain.values())
        ),
        "total_estimated_savings_eur_vs_best_baselines": float(
            sum(cast(float, payload["estimated_savings_eur_vs_best_baseline"]) for payload in per_product_gain.values())
        ),
    }


def _parse_unit_price_series(price_series: pd.Series) -> pd.Series:
    normalized_series = (
        price_series.astype("string")
        .fillna("0")
        .str.replace("€", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    numeric_series = cast(pd.Series, pd.to_numeric(normalized_series, errors="coerce"))
    return cast(pd.Series, numeric_series.fillna(0.0).astype(float))


def _load_sales_dataset(raw_sales_csv: str | Path) -> pd.DataFrame:
    sales_df = pd.read_csv(raw_sales_csv)
    keep_columns = [
        str(column)
        for column in sales_df.columns
        if str(column).strip() and not str(column).lower().startswith("unnamed:")
    ]
    if keep_columns and keep_columns[0].lower() == "index":
        keep_columns = keep_columns[1:]
    sales_df = sales_df.loc[:, keep_columns].copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    sales_df["article"] = sales_df["article"].astype("string").str.strip()
    sales_df["Quantity"] = sales_df["Quantity"].astype(float)
    sales_df["unit_price"] = _parse_unit_price_series(cast(pd.Series, sales_df["unit_price"]))
    return sales_df.loc[:, ["date", "article", "Quantity", "unit_price"]]


def _cancel_negative_sales_for_article(article_df: pd.DataFrame) -> pd.DataFrame:
    working_df = article_df.copy()
    positive_indices: list[int] = []
    for row_index in working_df.index:
        quantity = float(working_df.at[row_index, "Quantity"])
        if quantity > 0.0:
            positive_indices.append(row_index)
            continue
        remaining_to_cancel = abs(quantity)
        while remaining_to_cancel > 0.0 and positive_indices:
            previous_index = positive_indices[-1]
            previous_quantity = float(working_df.at[previous_index, "Quantity"])
            cancelled_quantity = min(previous_quantity, remaining_to_cancel)
            working_df.at[previous_index, "Quantity"] = previous_quantity - cancelled_quantity
            remaining_to_cancel -= cancelled_quantity
            if float(working_df.at[previous_index, "Quantity"]) <= 0.0:
                positive_indices.pop()
        working_df.at[row_index, "Quantity"] = 0.0
    return working_df


def _remove_negative_cancellations(sales_df: pd.DataFrame) -> pd.DataFrame:
    preserved_columns = [str(column) for column in sales_df.columns]
    ordered_df = sales_df.copy()
    ordered_df["row_order"] = range(len(ordered_df))
    ordered_df = ordered_df.sort_values(["article", "date", "row_order"])
    cleaned_articles = [
        _cancel_negative_sales_for_article(article_df)
        for _, article_df in ordered_df.groupby("article", sort=False)
    ]
    cleaned_df = pd.concat(cleaned_articles, ignore_index=True)
    cleaned_df = cleaned_df.loc[cleaned_df["Quantity"] > 0].copy()
    cleaned_df = cleaned_df.sort_values(["date", "row_order"]).reset_index(drop=True)
    return cleaned_df.loc[:, preserved_columns]


def _test_window_unit_prices(
    *,
    predictions_df: pd.DataFrame,
    raw_sales_csv: str | Path | None,
) -> dict[str, float]:
    if raw_sales_csv is None or not Path(raw_sales_csv).exists():
        return {}

    test_dates = set(cast(pd.Series, predictions_df["target_date"]).astype(str).tolist())
    test_products = set(cast(pd.Series, predictions_df["product"]).astype(str).tolist())
    sales_df = _load_sales_dataset(raw_sales_csv)
    sales_df = sales_df.loc[~cast(pd.Series, sales_df["article"]).isin(INVALID_ARTICLES)].copy()
    sales_df = _remove_negative_cancellations(sales_df)
    filtered_sales_df = sales_df.loc[
        cast(pd.Series, sales_df["date"]).astype(str).isin(test_dates)
        & cast(pd.Series, sales_df["article"]).astype(str).isin(test_products)
    ].copy()
    if len(filtered_sales_df) == 0:
        return {}

    filtered_sales_df["line_revenue"] = (
        cast(pd.Series, filtered_sales_df["unit_price"]).astype(float)
        * cast(pd.Series, filtered_sales_df["Quantity"]).astype(float)
    )
    grouped_df = filtered_sales_df.groupby("article", as_index=False).agg(
        total_quantity=("Quantity", "sum"),
        total_revenue=("line_revenue", "sum"),
    )
    grouped_df = grouped_df.loc[cast(pd.Series, grouped_df["total_quantity"]).astype(float) > 0.0].copy()
    grouped_df["unit_sale_price_eur"] = (
        cast(pd.Series, grouped_df["total_revenue"]).astype(float)
        / cast(pd.Series, grouped_df["total_quantity"]).astype(float)
    )
    return {
        str(row["article"]): float(row["unit_sale_price_eur"])
        for row in grouped_df.to_dict(orient="records")
    }


def _rounded_units(values: pd.Series) -> pd.Series:
    rounded_values = values.astype(float).round().clip(lower=0.0)
    return cast(pd.Series, rounded_values.astype(int))


def _economic_comparison_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "product",
        "target_date",
        "actual",
        "prediction_raw",
        "prediction_rounded",
        "best_statistical_baseline_name",
        "best_statistical_baseline_prediction",
        "best_statistical_baseline_abs_error",
    }
    if not required_columns.issubset(predictions_df.columns):
        return pd.DataFrame()

    comparison_df = predictions_df.loc[
        :,
        [
            "product",
            "target_date",
            "actual",
            "prediction_raw",
            "prediction_rounded",
            "best_statistical_baseline_name",
            "best_statistical_baseline_prediction",
            "best_statistical_baseline_abs_error",
            "best_statistical_baseline_product_mae",
        ],
    ].copy()
    comparison_df = comparison_df.rename(
        columns={
            "actual": "model_actual",
            "prediction_raw": "model_prediction_raw",
            "prediction_rounded": "model_prediction_rounded",
            "best_statistical_baseline_name": "baseline_name",
            "best_statistical_baseline_prediction": "baseline_prediction_raw",
            "best_statistical_baseline_product_mae": "best_baseline_mae",
            "best_statistical_baseline_abs_error": "baseline_absolute_error",
        }
    )
    comparison_df["actual_units"] = _rounded_units(cast(pd.Series, comparison_df["model_actual"]))
    comparison_df["model_prediction_units"] = _rounded_units(
        cast(pd.Series, comparison_df["model_prediction_rounded"])
    )
    comparison_df["baseline_prediction_units"] = _rounded_units(
        cast(pd.Series, comparison_df["baseline_prediction_raw"])
    )
    return comparison_df


def _apply_loss_model(
    *,
    comparison_df: pd.DataFrame,
    unit_sale_prices: dict[str, float],
    production_cost_ratio: float,
) -> pd.DataFrame:
    priced_df = comparison_df.copy()
    priced_df["unit_sale_price_eur"] = [
        float(unit_sale_prices.get(str(product_name), DEFAULT_UNIT_SALE_PRICE_EUR))
        for product_name in cast(pd.Series, priced_df[REFERENCE_PRODUCT_COL]).astype(str)
    ]
    priced_df["unit_production_cost_eur"] = (
        cast(pd.Series, priced_df["unit_sale_price_eur"]).astype(float) * float(production_cost_ratio)
    )
    for prefix in ("model", "baseline"):
        predicted_units = cast(pd.Series, priced_df[f"{prefix}_prediction_units"]).astype(int)
        actual_units = cast(pd.Series, priced_df["actual_units"]).astype(int)
        priced_df[f"{prefix}_overproduction_units"] = (predicted_units - actual_units).clip(lower=0)
        priced_df[f"{prefix}_underproduction_units"] = (actual_units - predicted_units).clip(lower=0)
        priced_df[f"{prefix}_overproduction_loss_eur"] = (
            cast(pd.Series, priced_df[f"{prefix}_overproduction_units"]).astype(float)
            * cast(pd.Series, priced_df["unit_production_cost_eur"]).astype(float)
        )
        priced_df[f"{prefix}_underproduction_loss_eur"] = (
            cast(pd.Series, priced_df[f"{prefix}_underproduction_units"]).astype(float)
            * cast(pd.Series, priced_df["unit_sale_price_eur"]).astype(float)
        )
        priced_df[f"{prefix}_total_loss_eur"] = (
            cast(pd.Series, priced_df[f"{prefix}_overproduction_loss_eur"]).astype(float)
            + cast(pd.Series, priced_df[f"{prefix}_underproduction_loss_eur"]).astype(float)
        )
    return priced_df


def _product_gain_payload(
    *,
    product_df: pd.DataFrame,
    product_metrics: dict[str, Any],
) -> dict[str, Any]:
    best_baseline_name = str(cast(pd.Series, product_df["baseline_name"]).iloc[0])
    unit_sale_price = float(cast(pd.Series, product_df["unit_sale_price_eur"]).iloc[0])
    unit_production_cost = float(cast(pd.Series, product_df["unit_production_cost_eur"]).iloc[0])
    model_mae = float(product_metrics["mae"])
    baseline_mae = float(cast(pd.Series, product_df["best_baseline_mae"]).iloc[0])
    model_total_loss = float(cast(pd.Series, product_df["model_total_loss_eur"]).sum())
    baseline_total_loss = float(cast(pd.Series, product_df["baseline_total_loss_eur"]).sum())
    estimated_savings_eur = float(baseline_total_loss - model_total_loss)
    return {
        "best_baseline_name": best_baseline_name,
        "model_mae": model_mae,
        "best_baseline_mae": baseline_mae,
        "absolute_mae_saved_vs_best_baseline": float(baseline_mae - model_mae),
        "relative_mae_improvement_vs_best_baseline": float(
            0.0 if baseline_mae <= 0.0 else ((baseline_mae - model_mae) / baseline_mae)
        ),
        "test_day_count": int(product_metrics["test_rows"]),
        "unit_sale_price_eur": unit_sale_price,
        "unit_production_cost_eur": unit_production_cost,
        "model_overproduction_units": float(cast(pd.Series, product_df["model_overproduction_units"]).sum()),
        "model_underproduction_units": float(cast(pd.Series, product_df["model_underproduction_units"]).sum()),
        "baseline_overproduction_units": float(cast(pd.Series, product_df["baseline_overproduction_units"]).sum()),
        "baseline_underproduction_units": float(cast(pd.Series, product_df["baseline_underproduction_units"]).sum()),
        "model_total_loss_eur": model_total_loss,
        "baseline_total_loss_eur": baseline_total_loss,
        "estimated_realistic_savings_eur_vs_best_baseline": estimated_savings_eur,
        "estimated_savings_eur_vs_best_baseline": estimated_savings_eur,
    }
