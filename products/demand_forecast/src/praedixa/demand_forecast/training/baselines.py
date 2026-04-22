from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.training.baseline_vectorized import (
    evaluate_polars_baseline_macro,
)
from praedixa.demand_forecast.feature_screening.pipeline import compute_wape
from praedixa.demand_forecast.contracts.targets import TargetContract


LOGGER = logging.getLogger(__name__)


BASELINE_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "naive_lag_1": ("sale_amount_lag_1", "lag_1"),
    "seasonal_naive_lag_7": ("target_lag_7", "sale_amount_lag_7", "lag_7", "seasonal_naive_d7"),
    "trailing_mean_7": ("sale_amount_rolling_mean_7", "rolling_mean_7"),
    "trailing_mean_28": ("sale_amount_rolling_mean_28", "rolling_mean_28", "moving_average_28"),
    "same_weekday_mean_4": ("target_same_dow_mean_4w", "same_dow_mean_4w"),
}

_DEFAULT_BASELINE_DATE_COL = "dt"


def _safe_last(history: list[float]) -> float:
    return float(history[-1])


def _append_observed_history_row(
    observed_history: pd.DataFrame,
    *,
    forecast_date: pd.Timestamp,
    observed_value: float,
    date_col: str,
    absolute_target_col: str,
) -> pd.DataFrame:
    return pd.concat(
        [
            observed_history,
            pd.DataFrame([{date_col: forecast_date, absolute_target_col: observed_value}]),
        ],
        ignore_index=True,
    )


def _same_weekday_mean_prediction(
    observed_history: pd.DataFrame,
    *,
    forecast_date: pd.Timestamp,
    absolute_target_col: str,
    date_col: str,
    fallback_value: float,
) -> float:
    weekday_history = observed_history.loc[
        observed_history[date_col].dt.dayofweek == forecast_date.dayofweek,
        absolute_target_col,
    ].astype(float).tolist()
    if not weekday_history:
        return fallback_value
    return float(np.mean(weekday_history[-min(4, len(weekday_history)) :]))


def _history_based_prediction_value(
    *,
    baseline_name: str,
    history_values: list[float],
    observed_history: pd.DataFrame,
    forecast_date: pd.Timestamp,
    absolute_target_col: str,
    date_col: str,
) -> float:
    fallback_value = _safe_last(history_values)
    if baseline_name == "naive_lag_1":
        return fallback_value
    if baseline_name == "seasonal_naive_lag_7":
        return float(history_values[-7]) if len(history_values) >= 7 else fallback_value
    if baseline_name == "trailing_mean_7":
        return float(np.mean(history_values[-min(7, len(history_values)) :]))
    if baseline_name == "trailing_mean_28":
        return float(np.mean(history_values[-min(28, len(history_values)) :]))
    if baseline_name == "same_weekday_mean_4":
        return _same_weekday_mean_prediction(
            observed_history,
            forecast_date=forecast_date,
            absolute_target_col=absolute_target_col,
            date_col=date_col,
            fallback_value=fallback_value,
        )
    raise ValueError(f"Unsupported baseline `{baseline_name}`.")


def _history_based_baseline_predictions(
    history_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    baseline_name: str,
    absolute_target_col: str,
    date_col: str = _DEFAULT_BASELINE_DATE_COL,
) -> pd.Series:
    observed_history = history_frame[[date_col, absolute_target_col]].copy()
    observed_history[date_col] = pd.to_datetime(observed_history[date_col])
    observed_history = observed_history.sort_values(date_col).reset_index(drop=True)
    future_rows = future_frame[[date_col, absolute_target_col]].copy()
    future_rows[date_col] = pd.to_datetime(future_rows[date_col])
    future_rows = future_rows.sort_values(date_col).reset_index(drop=True)

    predictions: list[float] = []
    for row in future_rows.itertuples(index=False):
        history_values = observed_history[absolute_target_col].astype(float).tolist()
        if not history_values:
            raise ValueError("Cannot score a statistical baseline without any training history.")
        forecast_date = pd.Timestamp(getattr(row, date_col))
        predicted_value = _history_based_prediction_value(
            baseline_name=baseline_name,
            history_values=history_values,
            observed_history=observed_history,
            forecast_date=forecast_date,
            absolute_target_col=absolute_target_col,
            date_col=date_col,
        )
        predictions.append(predicted_value)
        observed_history = _append_observed_history_row(
            observed_history,
            forecast_date=forecast_date,
            observed_value=float(getattr(row, absolute_target_col)),
            date_col=date_col,
            absolute_target_col=absolute_target_col,
        )
    return pd.Series(predictions, index=future_frame.sort_values(date_col).index, dtype=float)


def compute_equal_dataset_row_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> np.ndarray:
    return compute_equal_dataset_row_weights_from_values(frame[dataset_source_col])


def compute_equal_dataset_row_weights_from_values(dataset_sources: pd.Series | np.ndarray) -> np.ndarray:
    source_series = pd.Series(dataset_sources, copy=False)
    source_keys = source_series.astype("string").fillna("<NA>").astype(str)
    dataset_counts = source_keys.value_counts(dropna=False)
    dataset_count = max(1, len(dataset_counts))
    relative_weights = np.asarray(
        [1.0 / (dataset_count * float(dataset_counts.loc[dataset])) for dataset in source_keys.tolist()],
        dtype=float,
    )
    return relative_weights * float(len(source_keys))


def summarize_dataset_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> dict[str, dict[str, float | int]]:
    dataset_counts = frame[dataset_source_col].value_counts(dropna=False).sort_index()
    dataset_count = max(1, len(dataset_counts))
    scale_factor = float(len(frame))
    summary: dict[str, dict[str, float | int]] = {}
    for dataset_source, row_count in dataset_counts.items():
        relative_row_weight = 1.0 / (dataset_count * float(row_count))
        effective_row_weight = relative_row_weight * scale_factor
        summary[str(dataset_source)] = {
            "rows": int(row_count),
            "row_weight": float(effective_row_weight),
            "total_weight": float(relative_row_weight * float(row_count)),
            "effective_total_weight": float(effective_row_weight * float(row_count)),
        }
    return summary


def compute_dataset_macro_wape(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    dataset_source_col: str,
    target_col: str,
) -> float:
    scored = frame[[dataset_source_col, target_col]].copy()
    scored["prediction"] = predictions
    dataset_wapes: list[float] = []
    for _, dataset_group in scored.groupby(dataset_source_col, sort=False):
        dataset_frame = dataset_group
        dataset_wapes.append(
            compute_wape(
                dataset_frame[target_col],
                dataset_frame["prediction"],
            )
        )
    if not dataset_wapes:
        raise ValueError("Unable to compute dataset-macro WAPE with an empty validation frame.")
    return float(np.mean(dataset_wapes))


def _baseline_column_candidates(
    baseline_name: str,
    *,
    target_contract: TargetContract | None,
) -> tuple[str, ...]:
    if baseline_name != "seasonal_naive_lag_7":
        return BASELINE_COLUMN_CANDIDATES[baseline_name]
    anchor_candidates: list[str] = []
    if target_contract is not None and target_contract.reconstruction_anchor_col is not None:
        anchor_candidates.append(target_contract.reconstruction_anchor_col)
    return (*anchor_candidates, *BASELINE_COLUMN_CANDIDATES[baseline_name])


def _resolve_first_available_baseline_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return None


def _resolve_first_available_baseline_column_name(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _resolve_baseline_column_name(
    frame: pd.DataFrame,
    baseline_name: str,
    *,
    target_contract: TargetContract | None = None,
) -> str | None:
    if baseline_name in BASELINE_COLUMN_CANDIDATES:
        return _resolve_first_available_baseline_column_name(
            frame,
            _baseline_column_candidates(
                baseline_name,
                target_contract=target_contract,
            ),
        )
    raise ValueError(f"Unsupported baseline `{baseline_name}`.")


def resolve_baseline_prediction(
    frame: pd.DataFrame,
    baseline_name: str,
    *,
    target_contract: TargetContract | None = None,
) -> pd.Series | None:
    if baseline_name in BASELINE_COLUMN_CANDIDATES:
        return _resolve_first_available_baseline_column(
            frame,
            _baseline_column_candidates(
                baseline_name,
                target_contract=target_contract,
            ),
    )
    raise ValueError(f"Unsupported baseline `{baseline_name}`.")


def _fold_frame(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    index_key: str,
) -> pd.DataFrame:
    return frame.iloc[np.asarray(fold[index_key], dtype=np.int32)]


def _score_baseline_fold(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    baseline_name: str,
    absolute_target_col: str,
    target_contract: TargetContract | None,
    baseline_column_name: str | None = None,
) -> dict[str, object] | None:
    valid_frame = _fold_frame(frame=frame, fold=fold, index_key="valid_idx")
    if baseline_column_name is not None:
        target_values = pd.to_numeric(valid_frame[absolute_target_col], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        prediction_values = pd.to_numeric(
            valid_frame[baseline_column_name],
            errors="coerce",
        ).to_numpy(dtype=float, copy=False)
        valid_mask = np.isfinite(target_values) & np.isfinite(prediction_values)
        if not np.any(valid_mask):
            return None
        return {
            "dataset_source": str(fold["dataset_source"]),
            "fold": int(cast(Any, fold["fold"])),
            "wape": float(compute_wape(target_values[valid_mask], prediction_values[valid_mask])),
            "rows_scored": int(np.count_nonzero(valid_mask)),
        }
    predictions = resolve_baseline_prediction(valid_frame, baseline_name, target_contract=target_contract)
    if predictions is None or pd.isna(predictions).all():
        train_frame = _fold_frame(frame=frame, fold=fold, index_key="train_idx").sort_values(
            _DEFAULT_BASELINE_DATE_COL
        )
        valid_frame = valid_frame.sort_values(_DEFAULT_BASELINE_DATE_COL)
        predictions = _history_based_baseline_predictions(
            train_frame,
            valid_frame,
            baseline_name=baseline_name,
            absolute_target_col=absolute_target_col,
        )
    scored_frame = valid_frame[[absolute_target_col]].copy()
    scored_frame["prediction"] = pd.to_numeric(predictions, errors="coerce")
    target_values = scored_frame[absolute_target_col]
    prediction_values = scored_frame["prediction"]
    scored_frame = scored_frame[target_values.notna() & prediction_values.notna()].copy()
    if scored_frame.empty:
        return None
    return {
        "dataset_source": str(fold["dataset_source"]),
        "fold": int(cast(Any, fold["fold"])),
        "wape": float(
            compute_wape(
                scored_frame[absolute_target_col],
                scored_frame["prediction"],
            )
        ),
        "rows_scored": int(len(scored_frame)),
    }


def _dataset_macro_scores(fold_results: list[dict[str, object]]) -> dict[str, float]:
    dataset_mean_wape: dict[str, float] = {}
    for dataset_source in sorted({str(result["dataset_source"]) for result in fold_results}):
        dataset_scores = [
            float(cast(Any, result["wape"]))
            for result in fold_results
            if str(result["dataset_source"]) == dataset_source
        ]
        dataset_mean_wape[dataset_source] = float(np.mean(dataset_scores))
    return dataset_mean_wape


def evaluate_statistical_baselines_macro(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    *,
    absolute_target_col: str,
    target_contract: TargetContract | None = None,
    logger: logging.Logger = LOGGER,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_names = (
        "naive_lag_1",
        "seasonal_naive_lag_7",
        "trailing_mean_7",
        "trailing_mean_28",
        "same_weekday_mean_4",
    )
    baseline_column_names = {
        baseline_name: _resolve_baseline_column_name(
            frame,
            baseline_name,
            target_contract=target_contract,
        )
        for baseline_name in baseline_names
    }
    max_workers = min(max(1, os.cpu_count() or 1), max(1, len(baseline_names) * len(folds)))
    logger.info(
        "Executing statistical baselines in parallel: baselines=%s fold_evaluations=%s workers=%s",
        len(baseline_names),
        len(baseline_names) * len(folds),
        max_workers,
    )
    baseline_started_at = {baseline_name: time.perf_counter() for baseline_name in baseline_names}
    report_rows: list[dict[str, object]] = []
    threaded_baselines: list[str] = []
    for baseline_name in baseline_names:
        baseline_column_name = baseline_column_names[baseline_name]
        if baseline_column_name is not None:
            polars_baseline_row = evaluate_polars_baseline_macro(
                frame=frame,
                folds=folds,
                baseline_name=baseline_name,
                baseline_column_name=baseline_column_name,
                absolute_target_col=absolute_target_col,
                dataset_source_col="dataset_source",
                logger=logger,
            )
            if polars_baseline_row is not None:
                report_rows.append(polars_baseline_row)
            continue
        logger.info(
            "Starting statistical baseline evaluation: baseline=%s fold_evaluations=%s engine=python_fallback",
            baseline_name,
            len(folds),
        )
        threaded_baselines.append(baseline_name)
    if not threaded_baselines:
        return _finalize_baseline_rows(report_rows, logger=logger)
    baseline_results: dict[str, list[dict[str, object]]] = {baseline_name: [] for baseline_name in baseline_names}
    failed_baselines: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_baseline = {
            executor.submit(
                _score_baseline_fold,
                frame=frame,
                fold=fold,
                baseline_name=baseline_name,
                absolute_target_col=absolute_target_col,
                target_contract=target_contract,
                baseline_column_name=baseline_column_names[baseline_name],
            ): baseline_name
            for baseline_name in threaded_baselines
            for fold in folds
        }
        for future, baseline_name in future_to_baseline.items():
            fold_result = future.result()
            if fold_result is None:
                failed_baselines.add(baseline_name)
                continue
            baseline_results[baseline_name].append(fold_result)
    report_rows: list[dict[str, object]] = []
    for baseline_name in baseline_names:
        fold_results = baseline_results[baseline_name]
        if baseline_name in failed_baselines or len(fold_results) != len(folds):
            continue
        dataset_mean_wape = _dataset_macro_scores(fold_results)
        baseline_row: dict[str, object] = {
            "baseline_name": baseline_name,
            "mean_wape": float(np.mean(list(dataset_mean_wape.values()))),
            "dataset_mean_wape": dataset_mean_wape,
            "fold_wape_scores": fold_results,
        }
        logger.info(
            "Finished statistical baseline evaluation: baseline=%s mean_wape=%.6f duration_seconds=%.3f",
            baseline_name,
            baseline_row["mean_wape"],
            time.perf_counter() - baseline_started_at[baseline_name],
        )
        report_rows.append(baseline_row)
    return _finalize_baseline_rows(report_rows, logger=logger)


def _finalize_baseline_rows(
    report_rows: list[dict[str, object]],
    *,
    logger: logging.Logger,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not report_rows:
        raise ValueError("No statistical baseline could be evaluated with the available columns.")
    best_row = min(report_rows, key=lambda row: float(cast(Any, row["mean_wape"])))
    logger.info(
        "Best statistical baseline under macro-dataset WAPE: name=%s mean_wape=%.6f",
        best_row["baseline_name"],
        float(cast(Any, best_row["mean_wape"])),
    )
    return report_rows, best_row
