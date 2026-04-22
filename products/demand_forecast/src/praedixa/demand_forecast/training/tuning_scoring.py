from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from threading import Lock
import time
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd

from praedixa.demand_forecast.backends.tft.frame_utils import (
    attach_group_and_time_columns,
    build_combined_frame,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    lazy_import_tft_dependencies,
)
from praedixa.demand_forecast.backends.tft.model_utils import (
    DEFAULT_TFT_MODEL_PARAMS,
    fit_tft_model,
    predict_quantiles_with_tft_model,
    predict_with_tft_model,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import (
    select_explicit_tft_group_id_columns,
)
from praedixa.demand_forecast.backends.tft.training_dataset import (
    TrainingDatasetArtifacts,
    build_training_dataset_artifacts,
)
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.contracts.targets import reconstruct_absolute_predictions
from praedixa.demand_forecast.feature_screening.pipeline import compute_wape
from praedixa.demand_forecast.training.baselines import compute_equal_dataset_row_weights_from_values
from praedixa.demand_forecast.training.constants import (
    DEFAULT_DATE_COL,
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_TARGET_TRANSFORM,
)
from praedixa.demand_forecast.training.tuning_policy import resolve_fold_execution_plan


@dataclass(frozen=True)
class CachedFoldArtifacts:
    fold: dict[str, object]
    fold_train_frame: pd.DataFrame
    fold_valid_frame: pd.DataFrame
    valid_indices: np.ndarray
    fold_train_weights: np.ndarray
    fold_valid_weights: np.ndarray
    dataset_artifacts: TrainingDatasetArtifacts


_FOLD_DATASET_CACHE: dict[tuple[object, ...], CachedFoldArtifacts] = {}
_FOLD_DATASET_CACHE_LOCK = Lock()


def _build_shared_tuning_context(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    target_transform: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    learning_target_col = target_contract.learning_target_col
    required_frame_cols: list[str] = []
    for column in [DEFAULT_DATE_COL, *DEFAULT_IDENTIFIER_FEATURE_COLS, *feature_cols]:
        if column in train_frame.columns and column not in required_frame_cols:
            required_frame_cols.append(column)
    shared_columns = [*required_frame_cols, learning_target_col]
    shared_frame = pd.concat(
        [
            train_frame.loc[:, shared_columns],
            tuning_frame.loc[:, shared_columns],
        ],
        axis=0,
        ignore_index=True,
    )
    if target_transform == "log1p":
        if (shared_frame[learning_target_col] < 0).any():
            raise ValueError("Negative targets are incompatible with log1p target transformation.")
        shared_frame[learning_target_col] = np.log1p(shared_frame[learning_target_col].astype(float))
    shared_dataset_sources = pd.concat(
        [train_frame[DEFAULT_DATASET_SOURCE_COL].reset_index(drop=True), tuning_frame[DEFAULT_DATASET_SOURCE_COL].reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    ).to_numpy(copy=False)
    shared_absolute_target = pd.concat(
        [train_frame[target_contract.absolute_target_col].reset_index(drop=True), tuning_frame[target_contract.absolute_target_col].reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    ).to_numpy(dtype=np.float32, copy=False)
    shared_reconstruction_anchor = None
    if target_contract.reconstruction_anchor_col is not None:
        shared_reconstruction_anchor = pd.concat(
            [train_frame[target_contract.reconstruction_anchor_col].reset_index(drop=True), tuning_frame[target_contract.reconstruction_anchor_col].reset_index(drop=True)],
            axis=0,
            ignore_index=True,
        ).to_numpy(dtype=np.float32, copy=False)
    return shared_frame, np.arange(len(train_frame), dtype=np.int32), shared_dataset_sources, shared_absolute_target, shared_reconstruction_anchor


def _reconstruct_absolute_predictions(
    *,
    raw_predictions: np.ndarray | pd.Series,
    valid_indices: np.ndarray,
    target_contract: TargetContract,
    shared_reconstruction_anchor: np.ndarray | None,
) -> np.ndarray:
    prediction_array = np.asarray(raw_predictions, dtype=np.float32)
    if target_contract.target_mode == "delta_log_wow":
        if shared_reconstruction_anchor is None:
            raise ValueError("WoW delta reconstruction requires an anchor array in the scoring frame.")
        anchor = shared_reconstruction_anchor[valid_indices]
        absolute_predictions = np.expm1(prediction_array + np.log1p(np.clip(anchor, 0.0, None)))
    elif target_contract.target_mode == "log1p":
        absolute_predictions = np.expm1(prediction_array)
    else:
        absolute_predictions = prediction_array
    return np.asarray(np.clip(absolute_predictions, 0.0, None), dtype=np.float32)


def _dataset_fold_results(
    *,
    fold: dict[str, object],
    valid_indices: np.ndarray,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    predictions: np.ndarray,
    quantile_predictions: pd.DataFrame | None,
    absolute_target_col: str,
    feature_cols: list[str],
) -> list[dict[str, object]]:
    scored = pd.DataFrame(
        {
            DEFAULT_DATASET_SOURCE_COL: shared_dataset_sources[valid_indices],
            absolute_target_col: shared_absolute_target[valid_indices],
            "prediction": predictions,
        }
    )
    fold_results: list[dict[str, object]] = []
    if quantile_predictions is not None:
        scored = pd.concat([scored.reset_index(drop=True), quantile_predictions.reset_index(drop=True)], axis=1)
    for dataset_source, dataset_frame in scored.groupby(DEFAULT_DATASET_SOURCE_COL, sort=False):
        fold_results.append(
            {
                "dataset_source": str(dataset_source),
                "fold": int(cast(Any, fold["fold"])),
                "wape": float(compute_wape(dataset_frame[absolute_target_col], dataset_frame["prediction"])),
                "bias": float((dataset_frame["prediction"] - dataset_frame[absolute_target_col]).mean()),
                "abs_bias": float(abs((dataset_frame["prediction"] - dataset_frame[absolute_target_col]).mean())),
                "coverage_80": _interval_coverage(
                    dataset_frame,
                    actual_col=absolute_target_col,
                    lower_col="prediction_p10",
                    upper_col="prediction_p90",
                ),
                "coverage_95": _interval_coverage(
                    dataset_frame,
                    actual_col=absolute_target_col,
                    lower_col="prediction_p2_5",
                    upper_col="prediction_p97_5",
                ),
                "feature_count": int(len(feature_cols)),
                "rows_scored": int(len(dataset_frame)),
            }
        )
    return fold_results


def _interval_coverage(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    lower_col: str,
    upper_col: str,
) -> float | None:
    if lower_col not in frame.columns or upper_col not in frame.columns:
        return None
    lower = frame[lower_col].astype(float)
    upper = frame[upper_col].astype(float)
    actual = frame[actual_col].astype(float)
    valid_mask = lower.notna() & upper.notna()
    if not valid_mask.any():
        return None
    covered = ((actual[valid_mask] >= lower[valid_mask]) & (actual[valid_mask] <= upper[valid_mask])).mean()
    return float(covered)


def _reconstruct_absolute_quantiles(
    *,
    quantile_predictions: pd.DataFrame,
    frame: pd.DataFrame,
    target_contract: TargetContract,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: reconstruct_absolute_predictions(
                quantile_predictions[column].to_numpy(dtype=float),
                frame,
                target_contract,
            )
            for column in quantile_predictions.columns
        }
    )


def filter_predictable_validation_rows(
    *,
    fold_train_frame: pd.DataFrame,
    fold_valid_frame: pd.DataFrame,
    valid_indices: np.ndarray,
    max_encoder_length: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    group_cols = select_explicit_tft_group_id_columns(fold_train_frame)
    history = fold_train_frame.loc[:, [*group_cols, DEFAULT_DATE_COL]].copy()
    history[DEFAULT_DATE_COL] = pd.to_datetime(history[DEFAULT_DATE_COL])
    valid_probe = fold_valid_frame.loc[:, [*group_cols, DEFAULT_DATE_COL]].copy()
    valid_probe[DEFAULT_DATE_COL] = pd.to_datetime(valid_probe[DEFAULT_DATE_COL])
    valid_probe["__row_pos__"] = np.arange(len(valid_probe), dtype=np.int32)
    history_dates_by_group: dict[tuple[str, ...], np.ndarray] = {}
    for group_key, group_history in history.groupby(group_cols, sort=False):
        normalized_group_key = group_key if isinstance(group_key, tuple) else (str(group_key),)
        history_dates_by_group[normalized_group_key] = np.sort(
            group_history[DEFAULT_DATE_COL].to_numpy(dtype="datetime64[ns]")
        )
    keep_positions: list[int] = []
    for _, group_frame in valid_probe.groupby(group_cols, sort=False):
        group_key = tuple(str(group_frame.iloc[0][column]) for column in group_cols)
        history_dates = history_dates_by_group.get(group_key)
        if history_dates is None or history_dates.size == 0:
            continue
        valid_dates = group_frame[DEFAULT_DATE_COL].to_numpy(dtype="datetime64[ns]")
        eligible_rows = np.searchsorted(history_dates, valid_dates, side="left") >= max_encoder_length
        keep_positions.extend(group_frame.loc[eligible_rows, "__row_pos__"].tolist())
    if not keep_positions:
        raise ValueError("No predictable validation rows remain after enforcing TFT encoder history requirements.")
    keep_index = np.asarray(sorted(keep_positions), dtype=np.int32)
    return fold_valid_frame.iloc[keep_index].copy(), valid_indices[keep_index]


def _fold_dataset_cache_key(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_contract: TargetContract,
    max_encoder_length: int,
) -> tuple[object, ...]:
    fold_signature = (
        int(cast(Any, fold["fold"])),
        tuple(np.asarray(fold["train_idx"], dtype=np.int32).tolist()),
        tuple(np.asarray(fold["valid_idx"], dtype=np.int32).tolist()),
    )
    return (
        id(train_frame),
        id(tuning_frame),
        fold_signature,
        tuple(feature_cols),
        target_contract.learning_target_col,
        target_contract.absolute_target_col,
        target_contract.target_mode,
        target_contract.reconstruction_anchor_col,
        max_encoder_length,
    )


def _resolve_fold_preparation_workers(
    *,
    folds: list[dict[str, object]],
    execution_plan: dict[str, object],
) -> int:
    if bool(cast(Any, execution_plan.get("gpu_safe_mode", False))):
        return 1
    total_threads = int(cast(Any, execution_plan["total_threads"]))
    if len(folds) <= 1 or total_threads <= 1:
        return 1
    return max(1, min(len(folds), max(1, total_threads // 2)))


def _build_cached_fold_artifacts(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    shared_frame: pd.DataFrame,
    base_train_indices: np.ndarray,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> CachedFoldArtifacts:
    prep_start = time.perf_counter()
    logger.info(
        "TFT tuning fold artifact prep started: fold=%s train_candidates=%s valid_candidates=%s encoder_length=%s",
        int(cast(Any, fold["fold"])),
        len(cast(Any, fold["train_idx"])),
        len(cast(Any, fold["valid_idx"])),
        int(cast(Any, resolved_params["max_encoder_length"])),
    )
    fold_train_idx = np.asarray(fold["train_idx"], dtype=np.int32)
    fold_valid_idx = np.asarray(fold["valid_idx"], dtype=np.int32)
    if fold_train_idx.size == 0 or fold_valid_idx.size == 0:
        raise ValueError(f"Target contract `{target_contract.target_mode}` produced an empty grouped fold {fold['fold']}.")
    train_row_count = len(base_train_indices)
    train_indices = np.concatenate([base_train_indices, train_row_count + fold_train_idx]).astype(np.int32)
    valid_indices = train_row_count + fold_valid_idx
    logger.info("TFT tuning fold frame slicing started: fold=%s", int(cast(Any, fold["fold"])))
    fold_train_frame = shared_frame.iloc[train_indices].copy()
    fold_valid_frame = shared_frame.iloc[valid_indices].copy()
    logger.info(
        "TFT tuning fold frame slicing completed: fold=%s duration_seconds=%.3f",
        int(cast(Any, fold["fold"])),
        time.perf_counter() - prep_start,
    )
    filter_start = time.perf_counter()
    logger.info("TFT tuning fold predictable-row filter started: fold=%s", int(cast(Any, fold["fold"])))
    fold_valid_frame, valid_indices = filter_predictable_validation_rows(
        fold_train_frame=fold_train_frame,
        fold_valid_frame=fold_valid_frame,
        valid_indices=valid_indices,
        max_encoder_length=int(cast(Any, resolved_params["max_encoder_length"])),
    )
    logger.info(
        "TFT tuning fold predictable-row filter completed: fold=%s duration_seconds=%.3f kept_valid_rows=%s",
        int(cast(Any, fold["fold"])),
        time.perf_counter() - filter_start,
        len(fold_valid_frame),
    )
    fold_train_weights = compute_equal_dataset_row_weights_from_values(shared_dataset_sources[train_indices])
    fold_valid_weights = compute_equal_dataset_row_weights_from_values(shared_dataset_sources[valid_indices])
    imports = lazy_import_tft_dependencies()
    combined_start = time.perf_counter()
    logger.info("TFT tuning fold combined frame build started: fold=%s", int(cast(Any, fold["fold"])))
    prepared_frame = attach_group_and_time_columns(
        build_combined_frame(
            fold_train_frame,
            fold_valid_frame,
            train_weights=fold_train_weights,
            valid_weights=fold_valid_weights,
        ),
        feature_cols,
    )
    logger.info(
        "TFT tuning fold combined frame build completed: fold=%s duration_seconds=%.3f prepared_rows=%s",
        int(cast(Any, fold["fold"])),
        time.perf_counter() - combined_start,
        len(prepared_frame),
    )
    dataset_start = time.perf_counter()
    logger.info("TFT tuning fold TimeSeriesDataSet build started: fold=%s", int(cast(Any, fold["fold"])))
    dataset_artifacts = build_training_dataset_artifacts(
        imports,
        prepared_frame=prepared_frame,
        feature_cols=feature_cols,
        target_col=target_contract.learning_target_col,
        resolved_params=resolved_params,
    )
    logger.info(
        "TFT tuning fold TimeSeriesDataSet build completed: fold=%s duration_seconds=%.3f",
        int(cast(Any, fold["fold"])),
        time.perf_counter() - dataset_start,
    )
    logger.info(
        "TFT tuning fold artifact prep completed: fold=%s train_rows=%s valid_rows=%s total_duration_seconds=%.3f",
        int(cast(Any, fold["fold"])),
        len(fold_train_frame),
        len(fold_valid_frame),
        time.perf_counter() - prep_start,
    )
    return CachedFoldArtifacts(
        fold=fold,
        fold_train_frame=fold_train_frame,
        fold_valid_frame=fold_valid_frame,
        valid_indices=valid_indices,
        fold_train_weights=fold_train_weights,
        fold_valid_weights=fold_valid_weights,
        dataset_artifacts=dataset_artifacts,
    )


def _cached_fold_artifacts(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    execution_plan: dict[str, object],
    shared_frame: pd.DataFrame,
    base_train_indices: np.ndarray,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> dict[int, CachedFoldArtifacts]:
    max_encoder_length = int(cast(Any, resolved_params["max_encoder_length"]))
    results_by_fold_id: dict[int, CachedFoldArtifacts] = {}
    pending: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for fold in folds:
        cache_key = _fold_dataset_cache_key(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            fold=fold,
            feature_cols=feature_cols,
            target_contract=target_contract,
            max_encoder_length=max_encoder_length,
        )
        with _FOLD_DATASET_CACHE_LOCK:
            cached = _FOLD_DATASET_CACHE.get(cache_key)
        if cached is not None:
            results_by_fold_id[id(fold)] = cached
            continue
        pending.append((cache_key, fold))
    if not pending:
        return results_by_fold_id
    prep_workers = _resolve_fold_preparation_workers(
        folds=[fold for _, fold in pending],
        execution_plan=execution_plan,
    )
    if prep_workers == 1:
        for cache_key, fold in pending:
            cached = _build_cached_fold_artifacts(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                fold=fold,
                shared_frame=shared_frame,
                base_train_indices=base_train_indices,
                shared_dataset_sources=shared_dataset_sources,
                shared_absolute_target=shared_absolute_target,
                shared_reconstruction_anchor=shared_reconstruction_anchor,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
            )
            with _FOLD_DATASET_CACHE_LOCK:
                _FOLD_DATASET_CACHE[cache_key] = cached
            results_by_fold_id[id(fold)] = cached
        return results_by_fold_id
    with ThreadPoolExecutor(max_workers=prep_workers) as executor:
        future_map = {
            executor.submit(
                _build_cached_fold_artifacts,
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                fold=fold,
                shared_frame=shared_frame,
                base_train_indices=base_train_indices,
                shared_dataset_sources=shared_dataset_sources,
                shared_absolute_target=shared_absolute_target,
                shared_reconstruction_anchor=shared_reconstruction_anchor,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
            ): (cache_key, fold)
            for cache_key, fold in pending
        }
        for future in as_completed(future_map):
            cache_key, fold = future_map[future]
            cached = future.result()
            with _FOLD_DATASET_CACHE_LOCK:
                _FOLD_DATASET_CACHE[cache_key] = cached
            results_by_fold_id[id(fold)] = cached
    return results_by_fold_id


def _fit_single_tuning_fold(
    *,
    cached_fold: CachedFoldArtifacts,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> dict[str, object]:
    fold_number = int(cast(Any, cached_fold.fold["fold"]))
    logger.info(
        "TFT tuning fold fit started: fold=%s train_rows=%s valid_rows=%s",
        fold_number,
        len(cached_fold.fold_train_frame),
        len(cached_fold.fold_valid_frame),
    )
    model = fit_tft_model(
        cached_fold.fold_train_frame,
        feature_cols,
        target_col=target_contract.learning_target_col,
        model_params=resolved_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=2000,
        valid_frame=cached_fold.fold_valid_frame,
        train_weights=cached_fold.fold_train_weights,
        valid_weights=cached_fold.fold_valid_weights,
        dataset_artifacts=cached_fold.dataset_artifacts,
    )
    logger.info("TFT tuning fold fit completed: fold=%s", fold_number)
    logger.info("TFT tuning fold point prediction started: fold=%s", fold_number)
    predictions = _reconstruct_absolute_predictions(
        raw_predictions=cast(Any, predict_with_tft_model(model, cached_fold.fold_valid_frame, feature_cols)),
        valid_indices=cached_fold.valid_indices,
        target_contract=target_contract,
        shared_reconstruction_anchor=shared_reconstruction_anchor,
    )
    logger.info("TFT tuning fold quantile prediction started: fold=%s", fold_number)
    quantile_predictions = _reconstruct_absolute_quantiles(
        quantile_predictions=predict_quantiles_with_tft_model(model, cached_fold.fold_valid_frame, feature_cols),
        frame=cached_fold.fold_valid_frame,
        target_contract=target_contract,
    )
    return {
        "fold": int(cast(Any, cached_fold.fold["fold"])),
        "dataset_results": _dataset_fold_results(
            fold=cached_fold.fold,
            valid_indices=cached_fold.valid_indices,
            shared_dataset_sources=shared_dataset_sources,
            shared_absolute_target=shared_absolute_target,
            predictions=predictions,
            quantile_predictions=quantile_predictions,
            absolute_target_col=target_contract.absolute_target_col,
            feature_cols=feature_cols,
        ),
    }


def _execute_tuning_folds(
    *,
    folds: list[dict[str, object]],
    cached_folds: dict[int, CachedFoldArtifacts],
    fold_workers: int,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> list[dict[str, object]]:
    if fold_workers == 1:
        return [
            _fit_single_tuning_fold(
                cached_fold=cached_folds[id(fold)],
                shared_dataset_sources=shared_dataset_sources,
                shared_absolute_target=shared_absolute_target,
                shared_reconstruction_anchor=shared_reconstruction_anchor,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
            )
            for fold in folds
        ]
    with ThreadPoolExecutor(max_workers=fold_workers) as executor:
        future_to_index = {
            executor.submit(
                _fit_single_tuning_fold,
                cached_fold=cached_folds[id(fold)],
                shared_dataset_sources=shared_dataset_sources,
                shared_absolute_target=shared_absolute_target,
                shared_reconstruction_anchor=shared_reconstruction_anchor,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
            ): index
            for index, fold in enumerate(folds)
        }
        results_by_index = {future_to_index[future]: future.result() for future in as_completed(future_to_index)}
    return [results_by_index[index] for index in range(len(folds))]


def _dataset_macro_scores_from_fold_results(fold_results: list[dict[str, object]]) -> dict[str, float]:
    dataset_mean_wape: dict[str, float] = {}
    for dataset_source in sorted({str(result["dataset_source"]) for result in fold_results}):
        dataset_scores = [float(cast(Any, result["wape"])) for result in fold_results if str(result["dataset_source"]) == dataset_source]
        dataset_mean_wape[dataset_source] = float(np.mean(dataset_scores))
    return dataset_mean_wape


def _mean_optional_metric(
    fold_results: list[dict[str, object]],
    *,
    metric_key: str,
) -> float | None:
    metric_values = [
        float(cast(Any, result[metric_key]))
        for result in fold_results
        if result.get(metric_key) is not None
    ]
    if not metric_values:
        return None
    return float(np.mean(metric_values))


def _log_fold_completion(
    *,
    logger: logging.Logger,
    fold_number: int,
    total_folds: int,
    dataset_mean_wape: dict[str, float],
) -> None:
    macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
    logger.info(
        "TFT tuning fold completed: fold=%s/%s macro_mean_wape=%.6f dataset_mean_wape=%s",
        fold_number,
        total_folds,
        macro_mean_wape,
        dataset_mean_wape,
    )


def _iterative_trial_scoring(
    *,
    logger: logging.Logger,
    trial: optuna.trial.Trial,
    folds: list[dict[str, object]],
    cached_folds: dict[int, CachedFoldArtifacts],
    shared_frame: pd.DataFrame,
    base_train_indices: np.ndarray,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
) -> tuple[list[dict[str, object]], int]:
    fold_results: list[dict[str, object]] = []
    folds_completed = 0
    for folds_completed, fold in enumerate(folds, start=1):
        grouped_fold_result = _fit_single_tuning_fold(
            cached_fold=cached_folds[id(fold)],
            shared_dataset_sources=shared_dataset_sources,
            shared_absolute_target=shared_absolute_target,
            shared_reconstruction_anchor=shared_reconstruction_anchor,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
        )
        fold_results.extend(cast(list[dict[str, object]], grouped_fold_result["dataset_results"]))
        dataset_mean_wape = _dataset_macro_scores_from_fold_results(fold_results)
        interim_macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
        _log_fold_completion(
            logger=logger,
            fold_number=folds_completed,
            total_folds=len(folds),
            dataset_mean_wape=dataset_mean_wape,
        )
        trial.report(-interim_macro_mean_wape, step=folds_completed)
        if trial.should_prune():
            trial.set_user_attr("fold_results", fold_results)
            trial.set_user_attr("dataset_mean_wape", dataset_mean_wape)
            trial.set_user_attr("folds_completed", folds_completed)
            raise optuna.TrialPruned(f"Trial pruned after fold {folds_completed} with interim_wape={interim_macro_mean_wape:.6f}")
    return fold_results, folds_completed


def _score_all_folds(
    *,
    folds: list[dict[str, object]],
    cached_folds: dict[int, CachedFoldArtifacts],
    execution_plan: dict[str, object],
    logger: logging.Logger,
    shared_frame: pd.DataFrame,
    base_train_indices: np.ndarray,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    trial: optuna.trial.Trial | None,
) -> tuple[list[dict[str, object]], int]:
    if trial is not None and int(cast(Any, execution_plan["fold_workers"])) == 1:
        return _iterative_trial_scoring(
            logger=logger,
            trial=trial,
            folds=folds,
            cached_folds=cached_folds,
            shared_frame=shared_frame,
            base_train_indices=base_train_indices,
            shared_dataset_sources=shared_dataset_sources,
            shared_absolute_target=shared_absolute_target,
            shared_reconstruction_anchor=shared_reconstruction_anchor,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
        )
    grouped_fold_results = _execute_tuning_folds(
        folds=folds,
        cached_folds=cached_folds,
        fold_workers=int(cast(Any, execution_plan["fold_workers"])),
        shared_dataset_sources=shared_dataset_sources,
        shared_absolute_target=shared_absolute_target,
        shared_reconstruction_anchor=shared_reconstruction_anchor,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
    )
    fold_results = [result for grouped in grouped_fold_results for result in cast(list[dict[str, object]], grouped["dataset_results"])]
    return fold_results, len(folds)


def fit_and_score_tft_model_on_tuning(
    train_frame: pd.DataFrame, tuning_frame: pd.DataFrame, folds: list[dict[str, object]], feature_cols: list[str],
    target_contract: TargetContract, *, logger: logging.Logger, model_params: dict[str, object] | None = None,
    total_threads: int | None = None, target_transform: str = DEFAULT_TARGET_TRANSFORM,
    trial: optuna.trial.Trial | None = None,
) -> dict[str, object]:
    resolved_params = {**DEFAULT_TFT_MODEL_PARAMS, **(model_params or {})}
    execution_plan = resolve_fold_execution_plan(
        folds=folds,
        resolved_params=resolved_params,
        total_threads=total_threads,
        logger=logger,
    )
    shared_frame, base_train_indices, shared_dataset_sources, shared_absolute_target, shared_reconstruction_anchor = _build_shared_tuning_context(
        train_frame=train_frame, tuning_frame=tuning_frame, feature_cols=feature_cols, target_contract=target_contract, target_transform=target_transform
    )
    cached_folds = _cached_fold_artifacts(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        execution_plan=execution_plan,
        shared_frame=shared_frame,
        base_train_indices=base_train_indices,
        shared_dataset_sources=shared_dataset_sources,
        shared_absolute_target=shared_absolute_target,
        shared_reconstruction_anchor=shared_reconstruction_anchor,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
    )
    fold_results, folds_completed = _score_all_folds(
        folds=folds,
        cached_folds=cached_folds,
        execution_plan=execution_plan,
        logger=logger,
        shared_frame=shared_frame,
        base_train_indices=base_train_indices,
        shared_dataset_sources=shared_dataset_sources,
        shared_absolute_target=shared_absolute_target,
        shared_reconstruction_anchor=shared_reconstruction_anchor,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        trial=trial,
    )
    dataset_mean_wape = _dataset_macro_scores_from_fold_results(fold_results)
    macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
    mean_abs_bias = _mean_optional_metric(fold_results, metric_key="abs_bias")
    mean_coverage_80 = _mean_optional_metric(fold_results, metric_key="coverage_80")
    mean_coverage_95 = _mean_optional_metric(fold_results, metric_key="coverage_95")
    if trial is not None:
        trial.report(-macro_mean_wape, step=max(1, folds_completed))
    return {
        "macro_mean_wape": macro_mean_wape,
        "dataset_mean_wape": dataset_mean_wape,
        "mean_abs_bias": mean_abs_bias,
        "mean_coverage_80": mean_coverage_80,
        "mean_coverage_95": mean_coverage_95,
        "fold_results": fold_results,
        "folds_completed": folds_completed,
        "execution_policy": execution_plan,
    }
