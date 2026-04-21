from __future__ import annotations

from typing import Any

import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_baseline_predictions import (
    baseline_definitions,
    baseline_metrics,
    predict_baseline_series,
    prediction_rows_payload,
    ranked_rows,
    same_weekday_baselines,
)
from praedixa.demand_forecast.evaluation.bakery_baseline_shared import FIELD_BASELINE_NAME
from praedixa.demand_forecast.evaluation.bakery_metrics import (
    REFERENCE_PRODUCT_COL,
    REFERENCE_TARGET_COL,
)


def _add_direct_baseline_rows(
    *,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    insample_series: pd.Series,
    prediction_cache: dict[str, list[float]],
    prediction_series_by_name: dict[str, pd.Series],
    metrics_rows: list[dict[str, float | str]],
) -> None:
    direct_baselines, _ = baseline_definitions()
    for name, predictor in direct_baselines.items():
        predictions = predict_baseline_series(history_df, test_df, predictor)
        prediction_cache[name] = predictions.tolist()
        prediction_series_by_name[name] = predictions
        metrics_rows.append(baseline_metrics(name, y_true, predictions, insample_series))


def _add_same_weekday_rows(
    *,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    insample_series: pd.Series,
    prediction_cache: dict[str, list[float]],
    prediction_series_by_name: dict[str, pd.Series],
    metrics_rows: list[dict[str, float | str]],
) -> None:
    for name, predictions in same_weekday_baselines(history_df, test_df).items():
        prediction_cache[name] = predictions.tolist()
        prediction_series_by_name[name] = predictions
        metrics_rows.append(baseline_metrics(name, y_true, predictions, insample_series))


def _add_blended_rows(
    *,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    insample_series: pd.Series,
    prediction_cache: dict[str, list[float]],
    prediction_series_by_name: dict[str, pd.Series],
    metrics_rows: list[dict[str, float | str]],
) -> None:
    _, blended_baselines = baseline_definitions()
    for name, (component_names, blend_predictor) in blended_baselines.items():
        predictions = predict_baseline_series(
            history_df=history_df,
            test_df=test_df,
            predictor=blend_predictor,
            precomputed_names=component_names,
            prediction_cache=prediction_cache,
        )
        prediction_series_by_name[name] = predictions
        metrics_rows.append(baseline_metrics(name, y_true, predictions, insample_series))


def _product_ranked_rows(
    *,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    insample_series: pd.Series,
    prediction_cache: dict[str, list[float]],
    prediction_series_by_name: dict[str, pd.Series],
) -> list[dict[str, float | str]]:
    metrics_rows: list[dict[str, float | str]] = []
    _add_direct_baseline_rows(
        history_df=history_df,
        test_df=test_df,
        y_true=y_true,
        insample_series=insample_series,
        prediction_cache=prediction_cache,
        prediction_series_by_name=prediction_series_by_name,
        metrics_rows=metrics_rows,
    )
    _add_same_weekday_rows(
        history_df=history_df,
        test_df=test_df,
        y_true=y_true,
        insample_series=insample_series,
        prediction_cache=prediction_cache,
        prediction_series_by_name=prediction_series_by_name,
        metrics_rows=metrics_rows,
    )
    _add_blended_rows(
        history_df=history_df,
        test_df=test_df,
        y_true=y_true,
        insample_series=insample_series,
        prediction_cache=prediction_cache,
        prediction_series_by_name=prediction_series_by_name,
        metrics_rows=metrics_rows,
    )
    return ranked_rows(metrics_rows)


def _baseline_payload(
    *,
    ranked_row: dict[str, float | str],
    product_name: str,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    return {
        **ranked_row,
        "product": product_name,
        "prediction_rows": prediction_rows_payload(test_df=test_df, y_true=y_true, y_pred=y_pred),
    }


def _product_payload(
    product_name: str,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.Series], pd.Series, pd.Series]:
    y_true = test_df[REFERENCE_TARGET_COL].astype(float).reset_index(drop=True)
    insample_series = history_df[REFERENCE_TARGET_COL].astype(float).reset_index(drop=True)
    prediction_cache: dict[str, list[float]] = {}
    prediction_series_by_name: dict[str, pd.Series] = {}
    ranked = _product_ranked_rows(
        history_df=history_df,
        test_df=test_df,
        y_true=y_true,
        insample_series=insample_series,
        prediction_cache=prediction_cache,
        prediction_series_by_name=prediction_series_by_name,
    )
    best_name = str(ranked[0]["name"])
    best_payload = _baseline_payload(
        ranked_row=ranked[0],
        product_name=product_name,
        test_df=test_df,
        y_true=y_true,
        y_pred=prediction_series_by_name[best_name],
    )
    field_payload = _field_baseline_payload(
        ranked_rows=ranked,
        product_name=product_name,
        test_df=test_df,
        y_true=y_true,
        prediction_series_by_name=prediction_series_by_name,
    )
    return best_payload, field_payload, prediction_series_by_name, y_true, insample_series


def _field_baseline_payload(
    *,
    ranked_rows: list[dict[str, float | str]],
    product_name: str,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    prediction_series_by_name: dict[str, pd.Series],
) -> dict[str, Any]:
    field_predictions = prediction_series_by_name[FIELD_BASELINE_NAME]
    field_metrics = next(row for row in ranked_rows if str(row["name"]) == FIELD_BASELINE_NAME)
    return _baseline_payload(
        ranked_row=field_metrics,
        product_name=product_name,
        test_df=test_df,
        y_true=y_true,
        y_pred=field_predictions,
    )


def _group_product_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(product_name): product_df.reset_index(drop=True)
        for product_name, product_df in frame.groupby(REFERENCE_PRODUCT_COL, sort=True)
    }


def _append_aggregate_predictions(
    *,
    aggregate_predictions: dict[str, list[pd.DataFrame]],
    aggregate_insample: dict[str, list[pd.Series]],
    product_name: str,
    prediction_series_by_name: dict[str, pd.Series],
    y_true: pd.Series,
    insample_series: pd.Series,
) -> None:
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


def _aggregate_metrics_rows(
    *,
    aggregate_predictions: dict[str, list[pd.DataFrame]],
    aggregate_insample: dict[str, list[pd.Series]],
) -> list[dict[str, float | str]]:
    metrics_rows: list[dict[str, float | str]] = []
    for baseline_name, prediction_frames in aggregate_predictions.items():
        merged_predictions = pd.concat(prediction_frames, ignore_index=True)
        insample_values = pd.concat(
            [series.reset_index(drop=True) for series in aggregate_insample[baseline_name]],
            ignore_index=True,
        )
        metrics_rows.append(
            baseline_metrics(
                baseline_name,
                y_true=merged_predictions["y_true"].astype(float),
                y_pred=merged_predictions["y_pred"].astype(float),
                insample_series=insample_values.astype(float),
            )
        )
    return metrics_rows


def _best_baseline_payload(
    *,
    ranked_rows: list[dict[str, float | str]],
    aggregate_predictions: dict[str, list[pd.DataFrame]],
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    best_name = str(ranked_rows[0]["name"])
    best_predictions = pd.concat(aggregate_predictions[best_name], ignore_index=True)
    return {
        **ranked_rows[0],
        "prediction_rows": prediction_rows_payload(
            test_df=test_df.reset_index(drop=True),
            y_true=best_predictions["y_true"].astype(float).reset_index(drop=True),
            y_pred=best_predictions["y_pred"].astype(float).reset_index(drop=True),
        ),
    }


def _summary_payload(
    *,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_products: dict[str, pd.DataFrame],
    ranked_rows: list[dict[str, float | str]],
    aggregate_predictions: dict[str, list[pd.DataFrame]],
    per_product_best_baselines: dict[str, Any],
    per_product_field_baselines: dict[str, Any],
) -> dict[str, Any]:
    return {
        "date_column": "date",
        "product_column": "product",
        "target_column": "quantity",
        "history_rows": int(len(history_df)),
        "test_rows": int(len(test_df)),
        "product_count": int(len(test_products)),
        "history_range": _date_range_payload(history_df),
        "test_range": _date_range_payload(test_df),
        "baseline_count": int(len(ranked_rows)),
        "field_baseline_name": FIELD_BASELINE_NAME,
        "best_baseline": _best_baseline_payload(
            ranked_rows=ranked_rows,
            aggregate_predictions=aggregate_predictions,
            test_df=test_df,
        ),
        "baselines_ranked_by_mae": ranked_rows,
        "per_product_best_baselines": per_product_best_baselines,
        "per_product_field_baselines": per_product_field_baselines,
    }


def _date_range_payload(frame: pd.DataFrame) -> dict[str, str]:
    dates = pd.to_datetime(frame["date"])
    return {
        "start": str(pd.Timestamp(dates.min()).date()),
        "end": str(pd.Timestamp(dates.max()).date()),
    }


def build_statistical_baselines_payload(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    history_products = _group_product_frames(history_df)
    test_products = _group_product_frames(test_df)
    aggregate_predictions: dict[str, list[pd.DataFrame]] = {}
    aggregate_insample: dict[str, list[pd.Series]] = {}
    per_product_best_baselines: dict[str, Any] = {}
    per_product_field_baselines: dict[str, Any] = {}
    for product_name, product_test_df in test_products.items():
        best_payload, field_payload, prediction_series, y_true, insample_series = _product_payload(
            product_name=product_name,
            history_df=history_products[product_name],
            test_df=product_test_df,
        )
        per_product_best_baselines[product_name] = best_payload
        per_product_field_baselines[product_name] = field_payload
        _append_aggregate_predictions(
            aggregate_predictions=aggregate_predictions,
            aggregate_insample=aggregate_insample,
            product_name=product_name,
            prediction_series_by_name=prediction_series,
            y_true=y_true,
            insample_series=insample_series,
        )
    ranked = ranked_rows(
        _aggregate_metrics_rows(
            aggregate_predictions=aggregate_predictions,
            aggregate_insample=aggregate_insample,
        )
    )
    return _summary_payload(
        history_df=history_df,
        test_df=test_df,
        test_products=test_products,
        ranked_rows=ranked,
        aggregate_predictions=aggregate_predictions,
        per_product_best_baselines=per_product_best_baselines,
        per_product_field_baselines=per_product_field_baselines,
    )
