from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import logging
from threading import Lock
import time
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
import polars as pl

from praedixa.demand_forecast.backends.tft.dataset_core import (
    TimeSeriesDatasetCore,
    build_training_dataset_core,
    build_validation_dataset_core,
    clone_training_dataset_from_core,
    clone_validation_dataset_from_core,
)
from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    PREDICTION_ROW_ID_COL,
    SPLIT_COL,
    TIME_IDX_COL,
    WEIGHT_COL,
    attach_group_and_time_columns,
    build_combined_frame,
    build_group_identifier,
    cast_categorical_columns,
    defragment_frame,
    resolve_layout,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    lazy_import_tft_dependencies,
    resolve_model_params,
    suppress_tft_runtime_noise,
)
from praedixa.demand_forecast.backends.tft.model_fit import (
    calibrate_tft_learning_rate,
)
from praedixa.demand_forecast.backends.tft.model_utils import (
    DEFAULT_TFT_MODEL_PARAMS,
    fit_tft_model,
    predict_quantiles_with_tft_model,
    predict_with_tft_model,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import (
    select_explicit_tft_categorical_columns,
    select_explicit_tft_group_id_columns,
)
from praedixa.demand_forecast.backends.tft.training_dataset import (
    TrainingDatasetArtifacts,
    build_categorical_encoders,
    filter_groups_with_sufficient_history,
    fit_real_feature_scalers,
)
from praedixa.demand_forecast.backends.tft.training_runtime import seed_tft_runtime
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.training.shared.metrics import compute_wape
from praedixa.demand_forecast.training.baselines import (
    compute_equal_dataset_row_weights_from_values,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATE_COL,
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
    DEFAULT_TARGET_TRANSFORM,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    TRAINING_ELIGIBILITY_COLUMNS,
    filter_training_eligible_rows,
)
from praedixa.demand_forecast.training.tft.policy import resolve_fold_execution_plan

_ = clone_validation_dataset_from_core

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedFoldArtifacts:
    fold: dict[str, object]
    fit_reference_frame: pd.DataFrame
    train_row_count: int
    fold_valid_frame: pd.DataFrame
    valid_indices: np.ndarray
    fold_valid_weights: np.ndarray
    dataset_artifacts: TrainingDatasetArtifacts


@dataclass(frozen=True)
class CachedFoldCoreArtifacts:
    fold: dict[str, object]
    train_row_count: int
    train_group_counts: dict[str, int]
    fit_reference_frame: pd.DataFrame
    fold_valid_candidate_frame: pd.DataFrame
    valid_candidate_indices: np.ndarray
    layout: Any
    feature_scalers: dict[str, Any]
    training_core: TimeSeriesDatasetCore


@dataclass(frozen=True)
class SharedTuningContext:
    train_frame: pd.DataFrame
    tuning_frame: pd.DataFrame
    train_dataset_sources: np.ndarray
    train_dataset_source_counts: dict[str, int]
    tuning_dataset_sources: np.ndarray
    tuning_absolute_target: np.ndarray
    tuning_reconstruction_anchor: np.ndarray | None


@dataclass(frozen=True)
class FoldFramePreparationPlan:
    mode: str
    reason: str
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame | None


_SHARED_TUNING_CONTEXT_CACHE: dict[tuple[object, ...], SharedTuningContext] = {}
_SHARED_TUNING_CONTEXT_CACHE_LOCK = Lock()
_FOLD_CORE_CACHE: dict[tuple[object, ...], CachedFoldCoreArtifacts] = {}
_FOLD_CORE_CACHE_LOCK = Lock()
_FOLD_DATASET_CACHE: dict[tuple[object, ...], CachedFoldArtifacts] = {}
_FOLD_DATASET_CACHE_LOCK = Lock()


def clear_tuning_fold_caches() -> None:
    with _SHARED_TUNING_CONTEXT_CACHE_LOCK:
        _SHARED_TUNING_CONTEXT_CACHE.clear()
    with _FOLD_CORE_CACHE_LOCK:
        _FOLD_CORE_CACHE.clear()
    with _FOLD_DATASET_CACHE_LOCK:
        _FOLD_DATASET_CACHE.clear()


def _take_frame_rows(
    frame: pd.DataFrame,
    row_selector: slice | np.ndarray,
) -> pd.DataFrame:
    selected = frame.iloc[row_selector]
    return selected.copy()


def _build_shared_tuning_context(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    target_transform: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    dict[str, int],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    train_frame = filter_training_eligible_rows(
        train_frame,
        label="optuna_shared_train",
        logger=LOGGER,
    )
    learning_target_col = target_contract.learning_target_col
    required_frame_cols: list[str] = []
    identifier_cols: list[str] = [
        str(column) for column in DEFAULT_IDENTIFIER_FEATURE_COLS
    ]
    for column in [
        GROUP_COL,
        TIME_IDX_COL,
        DEFAULT_DATE_COL,
        DEFAULT_DATASET_SOURCE_COL,
        *identifier_cols,
        *TRAINING_ELIGIBILITY_COLUMNS,
        *feature_cols,
    ]:
        if column in train_frame.columns and column not in required_frame_cols:
            required_frame_cols.append(column)
    train_projected = train_frame.loc[
        :, [*required_frame_cols, learning_target_col]
    ].copy()
    tuning_projected = tuning_frame.loc[
        :, [*required_frame_cols, learning_target_col]
    ].copy()
    _ = target_transform
    train_dataset_sources = (
        train_frame[DEFAULT_DATASET_SOURCE_COL]
        .reset_index(drop=True)
        .to_numpy(copy=False)
    )
    train_dataset_source_counts = {
        str(dataset_source): int(count)
        for dataset_source, count in pd.Series(train_dataset_sources)
        .value_counts(sort=False)
        .items()
    }
    tuning_dataset_sources = (
        tuning_frame[DEFAULT_DATASET_SOURCE_COL]
        .reset_index(drop=True)
        .to_numpy(copy=False)
    )
    tuning_absolute_target = (
        tuning_frame[target_contract.absolute_target_col]
        .reset_index(drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )
    tuning_reconstruction_anchor = None
    if target_contract.reconstruction_anchor_col is not None:
        tuning_reconstruction_anchor = (
            tuning_frame[target_contract.reconstruction_anchor_col]
            .reset_index(drop=True)
            .to_numpy(dtype=np.float32, copy=False)
        )
    return (
        train_projected,
        tuning_projected,
        train_dataset_sources,
        train_dataset_source_counts,
        tuning_dataset_sources,
        tuning_absolute_target,
        tuning_reconstruction_anchor,
    )


def _shared_tuning_context_cache_key(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    target_transform: str,
) -> tuple[object, ...]:
    return (
        id(train_frame),
        id(tuning_frame),
        tuple(feature_cols),
        target_contract,
        target_transform,
    )


def _cached_shared_tuning_context(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    target_transform: str,
) -> SharedTuningContext:
    cache_key = _shared_tuning_context_cache_key(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        target_transform=target_transform,
    )
    with _SHARED_TUNING_CONTEXT_CACHE_LOCK:
        cached = _SHARED_TUNING_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    shared_context = SharedTuningContext(
        *_build_shared_tuning_context(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
            target_transform=target_transform,
        )
    )
    with _SHARED_TUNING_CONTEXT_CACHE_LOCK:
        _SHARED_TUNING_CONTEXT_CACHE[cache_key] = shared_context
    return shared_context


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
            raise ValueError(
                "WoW delta reconstruction requires an anchor array in the scoring frame."
            )
        anchor = shared_reconstruction_anchor[valid_indices]
        absolute_predictions = np.expm1(
            prediction_array + np.log1p(np.clip(anchor, 0.0, None))
        )
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
        scored = pd.concat(
            [
                scored.reset_index(drop=True),
                quantile_predictions.reset_index(drop=True),
            ],
            axis=1,
        )
    for dataset_source, dataset_frame in scored.groupby(
        DEFAULT_DATASET_SOURCE_COL, sort=False
    ):
        fold_results.append(
            {
                "dataset_source": str(dataset_source),
                "fold": int(cast(Any, fold["fold"])),
                "wape": float(
                    compute_wape(
                        dataset_frame[absolute_target_col], dataset_frame["prediction"]
                    )
                ),
                "bias": float(
                    (
                        dataset_frame["prediction"] - dataset_frame[absolute_target_col]
                    ).mean()
                ),
                "abs_bias": float(
                    abs(
                        (
                            dataset_frame["prediction"]
                            - dataset_frame[absolute_target_col]
                        ).mean()
                    )
                ),
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
                "pinball_loss_p50": _pinball_loss(
                    dataset_frame,
                    actual_col=absolute_target_col,
                    prediction_col="prediction_p50",
                    quantile=0.50,
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
    covered = (
        (actual[valid_mask] >= lower[valid_mask])
        & (actual[valid_mask] <= upper[valid_mask])
    ).mean()
    return float(covered)


def _pinball_loss(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    prediction_col: str,
    quantile: float,
) -> float | None:
    if prediction_col not in frame.columns:
        return None
    actual = frame[actual_col].astype(float)
    predicted = frame[prediction_col].astype(float)
    valid_mask = actual.notna() & predicted.notna()
    if not valid_mask.any():
        return None
    errors = actual[valid_mask] - predicted[valid_mask]
    resolved_quantile = float(quantile)
    return float(
        np.maximum(
            resolved_quantile * errors,
            (resolved_quantile - 1.0) * errors,
        ).mean()
    )


def _reconstruct_absolute_quantiles(
    *,
    quantile_predictions: pd.DataFrame,
    valid_indices: np.ndarray,
    target_contract: TargetContract,
    shared_reconstruction_anchor: np.ndarray | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: _reconstruct_absolute_predictions(
                raw_predictions=quantile_predictions[column].to_numpy(dtype=float),
                valid_indices=valid_indices,
                target_contract=target_contract,
                shared_reconstruction_anchor=shared_reconstruction_anchor,
            )
            for column in quantile_predictions.columns
        }
    )


def filter_predictable_validation_rows(
    *,
    fold_train_frame: pd.DataFrame | None,
    fold_valid_frame: pd.DataFrame,
    valid_indices: np.ndarray,
    max_encoder_length: int,
    train_group_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if train_group_counts is not None:
        if GROUP_COL in fold_valid_frame.columns:
            valid_group_ids = (
                fold_valid_frame[GROUP_COL].astype("string").fillna("<NA>")
            )
        else:
            valid_group_cols = select_explicit_tft_group_id_columns(fold_valid_frame)
            valid_group_ids = build_group_identifier(fold_valid_frame, valid_group_cols)
        valid_group_sizes = valid_group_ids.map(train_group_counts).fillna(0)
        keep_mask = valid_group_sizes.to_numpy(dtype=np.int32) >= max_encoder_length
        if not keep_mask.any():
            raise ValueError(
                "No predictable validation rows remain after enforcing TFT encoder history requirements."
            )
        keep_index = np.flatnonzero(keep_mask).astype(np.int32)
        return _take_frame_rows(fold_valid_frame, keep_index), valid_indices[keep_index]
    if (
        fold_train_frame is not None
        and GROUP_COL in fold_train_frame.columns
        and GROUP_COL in fold_valid_frame.columns
    ):
        train_group_sizes = (
            fold_train_frame[GROUP_COL]
            .astype("string")
            .value_counts(sort=False)
            .to_dict()
        )
        valid_group_sizes = (
            fold_valid_frame[GROUP_COL]
            .astype("string")
            .map(train_group_sizes)
            .fillna(0)
        )
        keep_mask = valid_group_sizes.to_numpy(dtype=np.int32) >= max_encoder_length
        if not keep_mask.any():
            raise ValueError(
                "No predictable validation rows remain after enforcing TFT encoder history requirements."
            )
        keep_index = np.flatnonzero(keep_mask).astype(np.int32)
        return _take_frame_rows(fold_valid_frame, keep_index), valid_indices[keep_index]
    if fold_train_frame is None:
        raise ValueError(
            "fold_train_frame or train_group_counts is required to filter predictable validation rows."
        )
    group_cols = select_explicit_tft_group_id_columns(fold_train_frame)
    history = fold_train_frame.loc[:, [*group_cols, DEFAULT_DATE_COL]].copy()
    history[DEFAULT_DATE_COL] = pd.to_datetime(history[DEFAULT_DATE_COL])
    valid_probe = fold_valid_frame.loc[:, [*group_cols, DEFAULT_DATE_COL]].copy()
    valid_probe[DEFAULT_DATE_COL] = pd.to_datetime(valid_probe[DEFAULT_DATE_COL])
    valid_probe["__row_pos__"] = np.arange(len(valid_probe), dtype=np.int32)
    history_dates_by_group: dict[tuple[str, ...], np.ndarray] = {}
    for group_key, group_history in history.groupby(group_cols, sort=False):
        normalized_group_key = tuple(str(value) for value in group_key)
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
        eligible_rows = (
            np.searchsorted(history_dates, valid_dates, side="left")
            >= max_encoder_length
        )
        keep_positions.extend(group_frame.loc[eligible_rows, "__row_pos__"].tolist())
    if not keep_positions:
        raise ValueError(
            "No predictable validation rows remain after enforcing TFT encoder history requirements."
        )
    keep_index = np.asarray(sorted(keep_positions), dtype=np.int32)
    return _take_frame_rows(fold_valid_frame, keep_index), valid_indices[keep_index]


def _fold_signature(fold: dict[str, object]) -> tuple[object, ...]:
    return (
        int(cast(Any, fold["fold"])),
        tuple(np.asarray(fold["train_idx"], dtype=np.int32).tolist()),
        tuple(np.asarray(fold["valid_idx"], dtype=np.int32).tolist()),
    )


def _fold_core_cache_key(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_contract: TargetContract,
    core_max_encoder_length: int,
) -> tuple[object, ...]:
    return (
        id(train_frame),
        id(tuning_frame),
        _fold_signature(fold),
        tuple(feature_cols),
        target_contract.learning_target_col,
        target_contract.absolute_target_col,
        target_contract.target_mode,
        target_contract.reconstruction_anchor_col,
        core_max_encoder_length,
    )


def _fold_dataset_cache_key(
    *,
    core_cache_key: tuple[object, ...],
    max_encoder_length: int,
) -> tuple[object, ...]:
    return (
        core_cache_key,
        max_encoder_length,
    )


def _dataset_weight_by_source(
    *,
    train_dataset_source_counts: dict[str, int],
    train_extension_sources: np.ndarray,
) -> dict[str, float]:
    combined_counts = dict(train_dataset_source_counts)
    if train_extension_sources.size > 0:
        for dataset_source, count in (
            pd.Series(train_extension_sources).value_counts(sort=False).items()
        ):
            combined_counts[str(dataset_source)] = combined_counts.get(
                str(dataset_source), 0
            ) + int(count)
    if not combined_counts:
        raise ValueError(
            "Cannot compute fold dataset weights without any dataset sources."
        )
    dataset_total_weight = 1.0 / float(len(combined_counts))
    return {
        dataset_source: dataset_total_weight / float(row_count)
        for dataset_source, row_count in combined_counts.items()
    }


def _resolve_fold_preparation_workers(
    *,
    folds: list[dict[str, object]],
    execution_plan: dict[str, object],
) -> int:
    _ = folds
    _ = execution_plan
    return 1


def _can_prepare_fold_frame_with_polars(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
) -> bool:
    if GROUP_COL not in train_frame.columns or TIME_IDX_COL not in train_frame.columns:
        return False
    if valid_frame is None:
        return True
    return GROUP_COL in valid_frame.columns and TIME_IDX_COL in valid_frame.columns


def _drop_tft_support_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[GROUP_COL, TIME_IDX_COL], errors="ignore")


def _resolve_fold_frame_preparation_plan(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
) -> FoldFramePreparationPlan:
    if _can_prepare_fold_frame_with_polars(
        train_frame=train_frame,
        valid_frame=valid_frame,
    ):
        return FoldFramePreparationPlan(
            mode="polars_precomputed",
            reason="precomputed_support_columns",
            train_frame=train_frame,
            valid_frame=valid_frame,
        )
    if valid_frame is None:
        return FoldFramePreparationPlan(
            mode="pandas_recomputed",
            reason="training_frame_missing_precomputed_support_columns",
            train_frame=_drop_tft_support_columns(train_frame),
            valid_frame=None,
        )
    support_columns = (GROUP_COL, TIME_IDX_COL)
    has_mixed_precomputed_support = any(
        (column in train_frame.columns) != (column in valid_frame.columns)
        for column in support_columns
    )
    reason = (
        "mixed_precomputed_support_columns"
        if has_mixed_precomputed_support
        else "support_columns_missing_from_both_frames"
    )
    return FoldFramePreparationPlan(
        mode="pandas_recomputed",
        reason=reason,
        train_frame=_drop_tft_support_columns(train_frame),
        valid_frame=_drop_tft_support_columns(valid_frame),
    )


def _polars_split_frame(
    *,
    frame: pd.DataFrame,
    split_name: str,
    weights: np.ndarray | None,
) -> pl.DataFrame:
    split_frame = pl.from_pandas(frame, include_index=False)
    columns: list[pl.Expr | pl.Series] = [pl.lit(split_name).alias(SPLIT_COL)]
    if weights is not None:
        if len(weights) != len(frame):
            raise ValueError("Sample weights length must match frame length.")
        columns.append(
            pl.Series(name=WEIGHT_COL, values=np.asarray(weights, dtype=np.float64))
        )
    return split_frame.with_columns(columns)


def _build_combined_frame_with_polars(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    train_weights: np.ndarray | None,
    valid_weights: np.ndarray | None,
    feature_cols: list[str],
) -> pd.DataFrame:
    split_frames = [
        _polars_split_frame(
            frame=train_frame,
            split_name="train",
            weights=train_weights,
        )
    ]
    if valid_frame is not None:
        split_frames.append(
            _polars_split_frame(
                frame=valid_frame,
                split_name="valid",
                weights=valid_weights,
            )
        )
    combined = (
        pl.concat(split_frames, how="diagonal_relaxed", rechunk=True)
        .lazy()
        .with_columns(
            [
                pl.col(DEFAULT_DATE_COL).cast(pl.Datetime, strict=False),
                pl.col(GROUP_COL).cast(pl.Utf8, strict=False).fill_null("<NA>"),
                pl.col(TIME_IDX_COL).cast(pl.Int32, strict=False),
            ]
        )
        .sort([GROUP_COL, TIME_IDX_COL, DEFAULT_DATE_COL])
        .collect()
    )
    prepared = combined.to_pandas()
    prepared[TIME_IDX_COL] = (
        prepared.groupby(GROUP_COL, sort=False).cumcount().astype(np.int32)
    )
    categorical_cols = select_explicit_tft_categorical_columns(feature_cols)
    return defragment_frame(
        cast_categorical_columns(prepared, [*categorical_cols, GROUP_COL])
    )


def _prepare_fold_frame(
    *,
    fold_number: int,
    stage_name: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    train_weights: np.ndarray | None,
    valid_weights: np.ndarray | None,
    feature_cols: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    preparation_plan = _resolve_fold_frame_preparation_plan(
        train_frame=train_frame,
        valid_frame=valid_frame,
    )
    if preparation_plan.mode == "polars_precomputed":
        started_at = time.perf_counter()
        logger.info(
            "TFT tuning fold frame preparation started: fold=%s stage=%s engine=polars threads=%s",
            fold_number,
            stage_name,
            pl.thread_pool_size(),
        )
        try:
            prepared = _build_combined_frame_with_polars(
                train_frame=preparation_plan.train_frame,
                valid_frame=preparation_plan.valid_frame,
                train_weights=train_weights,
                valid_weights=valid_weights,
                feature_cols=feature_cols,
            )
        except Exception as exc:
            raise RuntimeError(
                "Polars fold frame preparation failed for a frame that already satisfied the "
                "precomputed TFT support-column contract."
            ) from exc
        else:
            logger.info(
                "TFT tuning fold frame preparation completed: fold=%s stage=%s duration_seconds=%.3f engine=polars",
                fold_number,
                stage_name,
                time.perf_counter() - started_at,
            )
            return prepared
    started_at = time.perf_counter()
    logger.info(
        "TFT tuning fold frame preparation started: fold=%s stage=%s engine=pandas reason=%s",
        fold_number,
        stage_name,
        preparation_plan.reason,
    )
    prepared = attach_group_and_time_columns(
        build_combined_frame(
            preparation_plan.train_frame,
            preparation_plan.valid_frame,
            train_weights=train_weights,
            valid_weights=valid_weights,
        ),
        feature_cols,
    )
    logger.info(
        "TFT tuning fold frame preparation completed: fold=%s stage=%s duration_seconds=%.3f engine=pandas reason=%s",
        fold_number,
        stage_name,
        time.perf_counter() - started_at,
        preparation_plan.reason,
    )
    return prepared


def _build_core_validation_frame(
    *,
    fold_number: int,
    fold_train_frame: pd.DataFrame,
    fold_valid_candidate_frame: pd.DataFrame,
    fold_train_weights: np.ndarray | None,
    feature_cols: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    return _prepare_fold_frame(
        fold_number=fold_number,
        stage_name="validation_core",
        train_frame=fold_train_frame,
        valid_frame=fold_valid_candidate_frame,
        train_weights=fold_train_weights,
        valid_weights=np.ones(len(fold_valid_candidate_frame), dtype=float),
        feature_cols=feature_cols,
        logger=logger,
    )


def _build_core_training_frame(
    *,
    fold_number: int,
    fold_train_frame: pd.DataFrame,
    fold_train_weights: np.ndarray,
    feature_cols: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    return _prepare_fold_frame(
        fold_number=fold_number,
        stage_name="training_core",
        train_frame=fold_train_frame,
        valid_frame=None,
        train_weights=fold_train_weights,
        valid_weights=None,
        feature_cols=feature_cols,
        logger=logger,
    )


def _history_tail_frame(
    frame: pd.DataFrame,
    *,
    max_encoder_length: int,
) -> pd.DataFrame:
    return (
        frame.groupby(GROUP_COL, sort=False)
        .tail(max_encoder_length)
        .copy()
        .reset_index(drop=True)
    )


def _build_cached_fold_core_artifacts(
    *,
    fold: dict[str, object],
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    train_dataset_source_counts: dict[str, int],
    feature_cols: list[str],
    target_contract: TargetContract,
    core_max_encoder_length: int,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> CachedFoldCoreArtifacts:
    prep_start = time.perf_counter()
    fold_number = int(cast(Any, fold["fold"]))
    logger.info(
        "TFT tuning fold core prep started: fold=%s train_candidates=%s valid_candidates=%s core_encoder_length=%s",
        fold_number,
        len(cast(Any, fold["train_idx"])),
        len(cast(Any, fold["valid_idx"])),
        core_max_encoder_length,
    )
    fold_train_idx = np.asarray(fold["train_idx"], dtype=np.int32)
    fold_valid_idx = np.asarray(fold["valid_idx"], dtype=np.int32)
    if fold_train_idx.size == 0 or fold_valid_idx.size == 0:
        raise ValueError(
            f"Target contract `{target_contract.target_mode}` produced an empty grouped fold {fold['fold']}."
        )
    ordered_train_extension_idx = np.sort(fold_train_idx)
    ordered_valid_candidate_idx = np.sort(fold_valid_idx)
    logger.info("TFT tuning fold frame slicing started: fold=%s", fold_number)
    fold_train_extension_frame = _take_frame_rows(
        tuning_frame, ordered_train_extension_idx
    )
    fold_train_extension_frame = filter_training_eligible_rows(
        fold_train_extension_frame,
        label=f"optuna_fold_{fold_number}:tuning_train_extension",
        logger=logger,
    )
    fold_valid_candidate_frame = _take_frame_rows(
        tuning_frame, ordered_valid_candidate_idx
    )
    fold_train_frame = pd.concat(
        [train_frame, fold_train_extension_frame], ignore_index=True
    )
    fold_valid_candidate_frame[PREDICTION_ROW_ID_COL] = np.arange(
        len(fold_valid_candidate_frame),
        dtype=np.int32,
    )
    logger.info(
        "TFT tuning fold frame slicing completed: fold=%s duration_seconds=%.3f",
        fold_number,
        time.perf_counter() - prep_start,
    )
    dataset_weight_by_source = _dataset_weight_by_source(
        train_dataset_source_counts=train_dataset_source_counts,
        train_extension_sources=fold_train_extension_frame[
            DEFAULT_DATASET_SOURCE_COL
        ].to_numpy(copy=False),
    )
    fold_train_weights = (
        pd.Series(fold_train_frame[DEFAULT_DATASET_SOURCE_COL], copy=False)
        .map(dataset_weight_by_source)
        .to_numpy(dtype=np.float32, copy=False)
    )
    imports = lazy_import_tft_dependencies()
    with suppress_tft_runtime_noise():
        seed_tft_runtime(imports, resolved_params)
    training_frame = _build_core_training_frame(
        fold_number=fold_number,
        fold_train_frame=fold_train_frame,
        fold_train_weights=fold_train_weights,
        feature_cols=feature_cols,
        logger=logger,
    )
    training_frame = filter_groups_with_sufficient_history(
        training_frame,
        max_encoder_length=core_max_encoder_length,
    )
    fit_reference_frame = _history_tail_frame(
        training_frame,
        max_encoder_length=core_max_encoder_length,
    )
    train_row_count = int(len(training_frame))
    train_group_counts = {
        str(group_id): int(count)
        for group_id, count in training_frame[GROUP_COL]
        .astype("string")
        .value_counts(sort=False)
        .items()
    }
    del fold_train_extension_frame
    del fold_train_frame
    del fold_train_weights
    gc.collect()
    layout = resolve_layout(training_frame, feature_cols)
    categorical_encoders = build_categorical_encoders(imports, layout=layout)
    feature_scalers = fit_real_feature_scalers(training_frame, layout=layout)
    training_core = build_training_dataset_core(
        imports,
        training_frame=training_frame,
        target_col=target_contract.learning_target_col,
        max_encoder_length=core_max_encoder_length,
        weight_col=WEIGHT_COL,
        layout=layout,
        categorical_encoders=categorical_encoders,
        real_feature_scalers=feature_scalers or None,
    )
    del training_frame
    gc.collect()
    logger.info(
        "TFT tuning fold core prep completed: fold=%s train_rows=%s valid_candidates=%s total_duration_seconds=%.3f",
        fold_number,
        train_row_count,
        len(fold_valid_candidate_frame),
        time.perf_counter() - prep_start,
    )
    return CachedFoldCoreArtifacts(
        fold=fold,
        train_row_count=train_row_count,
        train_group_counts=train_group_counts,
        fit_reference_frame=fit_reference_frame,
        fold_valid_candidate_frame=fold_valid_candidate_frame,
        valid_candidate_indices=ordered_valid_candidate_idx.astype(np.int32),
        layout=layout,
        feature_scalers=feature_scalers,
        training_core=training_core,
    )


def _dataset_artifacts_from_core(
    *,
    cached_core: CachedFoldCoreArtifacts,
    shared_dataset_sources: np.ndarray,
    max_encoder_length: int,
    feature_cols: list[str],
    logger: logging.Logger,
) -> CachedFoldArtifacts:
    fallback_train_frame = None
    if GROUP_COL not in cached_core.fold_valid_candidate_frame.columns:
        fallback_train_frame = cached_core.fit_reference_frame
    fold_valid_frame, valid_indices = filter_predictable_validation_rows(
        fold_train_frame=fallback_train_frame,
        fold_valid_frame=cached_core.fold_valid_candidate_frame,
        valid_indices=cached_core.valid_candidate_indices,
        max_encoder_length=max_encoder_length,
        train_group_counts=cached_core.train_group_counts,
    )
    fold_valid_weights = compute_equal_dataset_row_weights_from_values(
        shared_dataset_sources[valid_indices]
    )
    training_slice = _history_tail_frame(
        cached_core.fit_reference_frame,
        max_encoder_length=max_encoder_length,
    )
    history_train_frame = training_slice.drop(
        columns=[SPLIT_COL, GROUP_COL, TIME_IDX_COL],
        errors="ignore",
    )
    history_train_weights = (
        history_train_frame[WEIGHT_COL].to_numpy(dtype=float, copy=False)
        if WEIGHT_COL in history_train_frame.columns
        else None
    )
    validation_frame = _build_core_validation_frame(
        fold_number=int(cast(Any, cached_core.fold["fold"])),
        fold_train_frame=history_train_frame,
        fold_valid_candidate_frame=fold_valid_frame,
        fold_train_weights=np.asarray(history_train_weights, dtype=float)
        if history_train_weights is not None
        else None,
        feature_cols=feature_cols,
        logger=logger,
    )
    validation_frame = filter_groups_with_sufficient_history(
        validation_frame,
        max_encoder_length=max_encoder_length,
    )
    validation_core = build_validation_dataset_core(
        training_core=cached_core.training_core,
        validation_frame=validation_frame,
        min_prediction_idx=max_encoder_length,
    )
    validation_dataset = getattr(validation_core, "dataset", validation_core)
    dataset_artifacts = TrainingDatasetArtifacts(
        training_slice=training_slice,
        training_dataset=clone_training_dataset_from_core(
            core=cached_core.training_core,
            max_encoder_length=max_encoder_length,
        ),
        validation_dataset=validation_dataset,
        layout=cached_core.layout,
        feature_scalers=cached_core.feature_scalers,
        normalization_strategy={
            "kind": "group_normalizer",
            "method": "standard",
            "groups": [GROUP_COL],
        },
    )
    return CachedFoldArtifacts(
        fold=cached_core.fold,
        fit_reference_frame=cached_core.fit_reference_frame,
        train_row_count=cached_core.train_row_count,
        fold_valid_frame=fold_valid_frame,
        valid_indices=valid_indices,
        fold_valid_weights=fold_valid_weights,
        dataset_artifacts=dataset_artifacts,
    )


def _cached_fold_cores(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    execution_plan: dict[str, object],
    train_dataset_source_counts: dict[str, int],
    feature_cols: list[str],
    target_contract: TargetContract,
    core_max_encoder_length: int,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> dict[int, CachedFoldCoreArtifacts]:
    results_by_fold_id: dict[int, CachedFoldCoreArtifacts] = {}
    pending: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for fold in folds:
        cache_key = _fold_core_cache_key(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            fold=fold,
            feature_cols=feature_cols,
            target_contract=target_contract,
            core_max_encoder_length=core_max_encoder_length,
        )
        with _FOLD_CORE_CACHE_LOCK:
            cached = _FOLD_CORE_CACHE.get(cache_key)
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
    _ = prep_workers
    for cache_key, fold in pending:
        cached = _build_cached_fold_core_artifacts(
            fold=fold,
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            train_dataset_source_counts=train_dataset_source_counts,
            feature_cols=feature_cols,
            target_contract=target_contract,
            core_max_encoder_length=core_max_encoder_length,
            resolved_params=resolved_params,
            logger=logger,
        )
        with _FOLD_CORE_CACHE_LOCK:
            _FOLD_CORE_CACHE[cache_key] = cached
        results_by_fold_id[id(fold)] = cached
    return results_by_fold_id


def _cached_fold_artifacts(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    execution_plan: dict[str, object],
    train_dataset_source_counts: dict[str, int],
    tuning_dataset_sources: np.ndarray,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> dict[int, CachedFoldArtifacts]:
    max_encoder_length = int(cast(Any, resolved_params["max_encoder_length"]))
    core_max_encoder_length = int(
        cast(
            Any,
            resolved_params.get("dataset_core_max_encoder_length", max_encoder_length),
        )
    )
    cached_cores = _cached_fold_cores(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        execution_plan=execution_plan,
        train_dataset_source_counts=train_dataset_source_counts,
        feature_cols=feature_cols,
        target_contract=target_contract,
        core_max_encoder_length=core_max_encoder_length,
        resolved_params=resolved_params,
        logger=logger,
    )
    results_by_fold_id: dict[int, CachedFoldArtifacts] = {}
    for fold in folds:
        core_cache_key = _fold_core_cache_key(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            fold=fold,
            feature_cols=feature_cols,
            target_contract=target_contract,
            core_max_encoder_length=core_max_encoder_length,
        )
        cache_key = _fold_dataset_cache_key(
            core_cache_key=core_cache_key,
            max_encoder_length=max_encoder_length,
        )
        with _FOLD_DATASET_CACHE_LOCK:
            cached = _FOLD_DATASET_CACHE.get(cache_key)
        if cached is None:
            cached = _dataset_artifacts_from_core(
                cached_core=cached_cores[id(fold)],
                shared_dataset_sources=tuning_dataset_sources,
                max_encoder_length=max_encoder_length,
                feature_cols=feature_cols,
                logger=logger,
            )
            with _FOLD_DATASET_CACHE_LOCK:
                _FOLD_DATASET_CACHE[cache_key] = cached
        results_by_fold_id[id(fold)] = cached
    return results_by_fold_id


def _resolved_tft_tuning_inputs(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    logger: logging.Logger,
    model_params: dict[str, object] | None,
    total_threads: int | None,
    target_transform: str,
    hpo_mode: bool,
) -> tuple[dict[str, object], dict[str, object], SharedTuningContext]:
    resolved_params = resolve_model_params(
        model_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=2000,
    )
    if hpo_mode:
        resolved_params.update(
            {
                "use_learning_rate_finder": False,
                "enable_csv_logger": False,
                "enable_lr_monitor": False,
                "enable_validation_metric_logging": False,
                "enable_business_validation_metrics": False,
                "enable_device_stats_monitor": False,
                "validation_monitor_metric": "val_loss",
                "loss_patience": 0,
                "log_every_n_steps": 50,
                "tensorboard_logdir": None,
            }
        )
        if str(cast(Any, resolved_params.get("accelerator", "cpu"))) == "gpu":
            resolved_params["determinism_mode"] = "off"
    resolved_params.setdefault(
        "dataset_core_max_encoder_length",
        int(cast(Any, resolved_params["max_encoder_length"])),
    )
    execution_plan = resolve_fold_execution_plan(
        folds=folds,
        resolved_params=resolved_params,
        total_threads=total_threads,
        logger=logger,
    )
    shared_context = _cached_shared_tuning_context(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        target_transform=target_transform,
    )
    return resolved_params, execution_plan, shared_context


def prewarm_tft_fold_cores_for_optuna(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    logger: logging.Logger,
    model_params: dict[str, object] | None = None,
    total_threads: int | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> dict[str, object]:
    if model_params is None or (
        "dataset_core_max_encoder_length" not in model_params
        and "max_encoder_length" not in model_params
    ):
        logger.info(
            "Skipping TFT fold core prewarm before Optuna because encoder length is not fixed; "
            "trial-specific core caches will be built on demand."
        )
        return {
            "cached_cores": 0,
            "core_max_encoder_length": None,
            "duration_seconds": 0.0,
            "execution_policy": {
                "prewarm_skipped": True,
                "reason": "variable_encoder_length",
            },
        }
    resolved_params, execution_plan, shared_context = _resolved_tft_tuning_inputs(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        logger=logger,
        model_params=model_params,
        total_threads=total_threads,
        target_transform=target_transform,
        hpo_mode=True,
    )
    if _should_use_memory_safe_sequential_fold_processing(
        folds=folds,
        execution_plan=execution_plan,
    ):
        logger.info(
            "Skipping TFT fold core prewarm before Optuna because sequential multi-fold execution is running in memory-safe mode."
        )
        return {
            "cached_cores": 0,
            "core_max_encoder_length": None,
            "duration_seconds": 0.0,
            "execution_policy": {
                **execution_plan,
                "prewarm_skipped": True,
                "reason": "memory_safe_sequential_folds",
            },
        }
    core_max_encoder_length = int(
        cast(Any, resolved_params["dataset_core_max_encoder_length"])
    )
    warmup_start = time.perf_counter()
    logger.info(
        "Prewarming TFT fold core cache before Optuna: folds=%s feature_count=%s core_encoder_length=%s train_rows=%s tuning_rows=%s",
        len(folds),
        len(feature_cols),
        core_max_encoder_length,
        len(shared_context.train_frame),
        len(shared_context.tuning_frame),
    )
    cached_cores = _cached_fold_cores(
        train_frame=shared_context.train_frame,
        tuning_frame=shared_context.tuning_frame,
        folds=folds,
        execution_plan=execution_plan,
        train_dataset_source_counts=shared_context.train_dataset_source_counts,
        feature_cols=feature_cols,
        target_contract=target_contract,
        core_max_encoder_length=core_max_encoder_length,
        resolved_params=resolved_params,
        logger=logger,
    )
    duration_seconds = time.perf_counter() - warmup_start
    logger.info(
        "Prewarmed TFT fold core cache before Optuna: folds=%s cached_cores=%s duration_seconds=%.3f",
        len(folds),
        len(cached_cores),
        duration_seconds,
    )
    return {
        "cached_cores": len(cached_cores),
        "core_max_encoder_length": core_max_encoder_length,
        "duration_seconds": duration_seconds,
        "execution_policy": execution_plan,
    }


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
        cached_fold.train_row_count,
        len(cached_fold.fold_valid_frame),
    )
    model = fit_tft_model(
        cached_fold.fit_reference_frame,
        feature_cols,
        target_col=target_contract.learning_target_col,
        model_params=resolved_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=2000,
        valid_frame=cached_fold.fold_valid_frame,
        train_weights=None,
        valid_weights=cached_fold.fold_valid_weights,
        dataset_artifacts=cached_fold.dataset_artifacts,
    )
    logger.info("TFT tuning fold fit completed: fold=%s", fold_number)
    logger.info("TFT tuning fold point prediction started: fold=%s", fold_number)
    predictions = _reconstruct_absolute_predictions(
        raw_predictions=cast(
            Any,
            predict_with_tft_model(
                model,
                cached_fold.fold_valid_frame,
                feature_cols,
                fallback_policy="raise",
            ),
        ),
        valid_indices=cached_fold.valid_indices,
        target_contract=target_contract,
        shared_reconstruction_anchor=shared_reconstruction_anchor,
    )
    logger.info("TFT tuning fold quantile prediction started: fold=%s", fold_number)
    quantile_predictions = _reconstruct_absolute_quantiles(
        quantile_predictions=predict_quantiles_with_tft_model(
            model,
            cached_fold.fold_valid_frame,
            feature_cols,
            fallback_policy="raise",
        ),
        valid_indices=cached_fold.valid_indices,
        target_contract=target_contract,
        shared_reconstruction_anchor=shared_reconstruction_anchor,
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
        results_by_index = {
            future_to_index[future]: future.result()
            for future in as_completed(future_to_index)
        }
    return [results_by_index[index] for index in range(len(folds))]


def _dataset_macro_scores_from_fold_results(
    fold_results: list[dict[str, object]],
) -> dict[str, float]:
    dataset_mean_wape: dict[str, float] = {}
    for dataset_source in sorted(
        {str(result["dataset_source"]) for result in fold_results}
    ):
        dataset_scores = [
            float(cast(Any, result["wape"]))
            for result in fold_results
            if str(result["dataset_source"]) == dataset_source
        ]
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
    mean_abs_bias: float | None,
    mean_coverage_80: float | None,
    mean_coverage_95: float | None,
) -> None:
    macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
    logger.info(
        "TFT tuning fold completed: fold=%s/%s business_macro_wape=%.6f business_mean_abs_bias=%s business_mean_coverage_80=%s business_mean_coverage_95=%s dataset_business_mean_wape=%s",
        fold_number,
        total_folds,
        macro_mean_wape,
        "nan" if mean_abs_bias is None else f"{mean_abs_bias:.6f}",
        "nan" if mean_coverage_80 is None else f"{mean_coverage_80:.6f}",
        "nan" if mean_coverage_95 is None else f"{mean_coverage_95:.6f}",
        dataset_mean_wape,
    )


def _resolved_trial_learning_rate(
    *,
    folds: list[dict[str, object]],
    cached_folds: dict[int, CachedFoldArtifacts],
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> tuple[dict[str, object], float]:
    current_learning_rate = float(cast(Any, resolved_params["learning_rate"]))
    if not bool(cast(Any, resolved_params.get("use_learning_rate_finder", False))):
        return resolved_params, current_learning_rate
    first_fold = folds[0]
    selected_learning_rate = calibrate_tft_learning_rate(
        dataset_artifacts=cached_folds[id(first_fold)].dataset_artifacts,
        resolved_params=resolved_params,
    )
    logger.info(
        "Resolved TFT trial learning rate from first fold calibration: learning_rate=%s",
        selected_learning_rate,
    )
    return (
        {
            **resolved_params,
            "learning_rate": selected_learning_rate,
            "use_learning_rate_finder": False,
        },
        selected_learning_rate,
    )


def _should_use_memory_safe_sequential_fold_processing(
    *,
    folds: list[dict[str, object]],
    execution_plan: dict[str, object],
) -> bool:
    return len(folds) > 1 and int(cast(Any, execution_plan["fold_workers"])) == 1


def _build_single_fold_artifact(
    *,
    fold: dict[str, object],
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    train_dataset_source_counts: dict[str, int],
    tuning_dataset_sources: np.ndarray,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    max_encoder_length: int,
    core_max_encoder_length: int,
    logger: logging.Logger,
) -> CachedFoldArtifacts:
    cached_core = _build_cached_fold_core_artifacts(
        fold=fold,
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        train_dataset_source_counts=train_dataset_source_counts,
        feature_cols=feature_cols,
        target_contract=target_contract,
        core_max_encoder_length=core_max_encoder_length,
        resolved_params=resolved_params,
        logger=logger,
    )
    cached_fold = _dataset_artifacts_from_core(
        cached_core=cached_core,
        shared_dataset_sources=tuning_dataset_sources,
        max_encoder_length=max_encoder_length,
        feature_cols=feature_cols,
        logger=logger,
    )
    del cached_core
    gc.collect()
    return cached_fold


def _resolve_memory_safe_learning_rate(
    *,
    folds: list[dict[str, object]],
    first_cached_fold: CachedFoldArtifacts | None,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    train_dataset_source_counts: dict[str, int],
    tuning_dataset_sources: np.ndarray,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> tuple[dict[str, object], float, CachedFoldArtifacts | None]:
    current_learning_rate = float(cast(Any, resolved_params["learning_rate"]))
    if not bool(cast(Any, resolved_params.get("use_learning_rate_finder", False))):
        return resolved_params, current_learning_rate, first_cached_fold
    max_encoder_length = int(cast(Any, resolved_params["max_encoder_length"]))
    core_max_encoder_length = int(
        cast(
            Any,
            resolved_params.get("dataset_core_max_encoder_length", max_encoder_length),
        )
    )
    selected_first_fold = first_cached_fold or _build_single_fold_artifact(
        fold=folds[0],
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        train_dataset_source_counts=train_dataset_source_counts,
        tuning_dataset_sources=tuning_dataset_sources,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        max_encoder_length=max_encoder_length,
        core_max_encoder_length=core_max_encoder_length,
        logger=logger,
    )
    selected_learning_rate = calibrate_tft_learning_rate(
        dataset_artifacts=selected_first_fold.dataset_artifacts,
        resolved_params=resolved_params,
    )
    logger.info(
        "Resolved TFT trial learning rate from first fold calibration: learning_rate=%s",
        selected_learning_rate,
    )
    return (
        {
            **resolved_params,
            "learning_rate": selected_learning_rate,
            "use_learning_rate_finder": False,
        },
        selected_learning_rate,
        selected_first_fold,
    )


def _iterative_trial_scoring(
    *,
    logger: logging.Logger,
    trial: optuna.trial.Trial,
    folds: list[dict[str, object]],
    cached_folds: dict[int, CachedFoldArtifacts],
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
) -> tuple[list[dict[str, object]], int, float]:
    selected_learning_rate = float(cast(Any, resolved_params["learning_rate"]))
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
        fold_results.extend(
            cast(list[dict[str, object]], grouped_fold_result["dataset_results"])
        )
        dataset_mean_wape = _dataset_macro_scores_from_fold_results(fold_results)
        interim_macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
        mean_abs_bias = _mean_optional_metric(fold_results, metric_key="abs_bias")
        mean_coverage_80 = _mean_optional_metric(
            fold_results,
            metric_key="coverage_80",
        )
        mean_coverage_95 = _mean_optional_metric(
            fold_results,
            metric_key="coverage_95",
        )
        _log_fold_completion(
            logger=logger,
            fold_number=folds_completed,
            total_folds=len(folds),
            dataset_mean_wape=dataset_mean_wape,
            mean_abs_bias=mean_abs_bias,
            mean_coverage_80=mean_coverage_80,
            mean_coverage_95=mean_coverage_95,
        )
        trial.report(-interim_macro_mean_wape, step=folds_completed)
        if trial.should_prune():
            trial.set_user_attr("fold_results", fold_results)
            trial.set_user_attr("dataset_mean_wape", dataset_mean_wape)
            trial.set_user_attr("folds_completed", folds_completed)
            raise optuna.TrialPruned(
                f"Trial pruned after fold {folds_completed} with interim_wape={interim_macro_mean_wape:.6f}"
            )
    return fold_results, folds_completed, selected_learning_rate


def _iterative_trial_scoring_memory_safe(
    *,
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    folds: list[dict[str, object]],
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    train_dataset_source_counts: dict[str, int],
    tuning_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
) -> tuple[list[dict[str, object]], int, float]:
    max_encoder_length = int(cast(Any, resolved_params["max_encoder_length"]))
    core_max_encoder_length = int(
        cast(
            Any,
            resolved_params.get("dataset_core_max_encoder_length", max_encoder_length),
        )
    )
    prebuilt_first_fold: CachedFoldArtifacts | None = None
    effective_params, selected_learning_rate, prebuilt_first_fold = (
        _resolve_memory_safe_learning_rate(
            folds=folds,
            first_cached_fold=prebuilt_first_fold,
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            train_dataset_source_counts=train_dataset_source_counts,
            tuning_dataset_sources=tuning_dataset_sources,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
        )
    )
    fold_results: list[dict[str, object]] = []
    folds_completed = 0
    for folds_completed, fold in enumerate(folds, start=1):
        cached_fold = (
            prebuilt_first_fold
            if folds_completed == 1 and prebuilt_first_fold is not None
            else _build_single_fold_artifact(
                fold=fold,
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                train_dataset_source_counts=train_dataset_source_counts,
                tuning_dataset_sources=tuning_dataset_sources,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                max_encoder_length=max_encoder_length,
                core_max_encoder_length=core_max_encoder_length,
                logger=logger,
            )
        )
        grouped_fold_result = _fit_single_tuning_fold(
            cached_fold=cached_fold,
            shared_dataset_sources=tuning_dataset_sources,
            shared_absolute_target=shared_absolute_target,
            shared_reconstruction_anchor=shared_reconstruction_anchor,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=effective_params,
            logger=logger,
        )
        fold_results.extend(
            cast(list[dict[str, object]], grouped_fold_result["dataset_results"])
        )
        dataset_mean_wape = _dataset_macro_scores_from_fold_results(fold_results)
        interim_macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
        mean_abs_bias = _mean_optional_metric(fold_results, metric_key="abs_bias")
        mean_coverage_80 = _mean_optional_metric(
            fold_results,
            metric_key="coverage_80",
        )
        mean_coverage_95 = _mean_optional_metric(
            fold_results,
            metric_key="coverage_95",
        )
        _log_fold_completion(
            logger=logger,
            fold_number=folds_completed,
            total_folds=len(folds),
            dataset_mean_wape=dataset_mean_wape,
            mean_abs_bias=mean_abs_bias,
            mean_coverage_80=mean_coverage_80,
            mean_coverage_95=mean_coverage_95,
        )
        if trial is not None:
            trial.report(-interim_macro_mean_wape, step=folds_completed)
            if trial.should_prune():
                trial.set_user_attr("fold_results", fold_results)
                trial.set_user_attr("dataset_mean_wape", dataset_mean_wape)
                trial.set_user_attr("folds_completed", folds_completed)
                raise optuna.TrialPruned(
                    f"Trial pruned after fold {folds_completed} with interim_wape={interim_macro_mean_wape:.6f}"
                )
        del grouped_fold_result
        del cached_fold
        gc.collect()
    return fold_results, folds_completed, selected_learning_rate


def _score_all_folds(
    *,
    folds: list[dict[str, object]],
    cached_folds: dict[int, CachedFoldArtifacts],
    execution_plan: dict[str, object],
    logger: logging.Logger,
    shared_dataset_sources: np.ndarray,
    shared_absolute_target: np.ndarray,
    shared_reconstruction_anchor: np.ndarray | None,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    trial: optuna.trial.Trial | None,
) -> tuple[list[dict[str, object]], int, float]:
    if trial is not None and int(cast(Any, execution_plan["fold_workers"])) == 1:
        return _iterative_trial_scoring(
            logger=logger,
            trial=trial,
            folds=folds,
            cached_folds=cached_folds,
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
    fold_results = [
        result
        for grouped in grouped_fold_results
        for result in cast(list[dict[str, object]], grouped["dataset_results"])
    ]
    return fold_results, len(folds), float(cast(Any, resolved_params["learning_rate"]))


def fit_and_score_tft_model_on_tuning(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    *,
    logger: logging.Logger,
    model_params: dict[str, object] | None = None,
    total_threads: int | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    trial: optuna.trial.Trial | None = None,
) -> dict[str, object]:
    resolved_params, execution_plan, shared_context = _resolved_tft_tuning_inputs(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        logger=logger,
        model_params=model_params,
        total_threads=total_threads,
        target_transform=target_transform,
        hpo_mode=trial is not None,
    )
    use_memory_safe_sequential_path = (
        _should_use_memory_safe_sequential_fold_processing(
            folds=folds,
            execution_plan=execution_plan,
        )
    )
    if use_memory_safe_sequential_path:
        fold_results, folds_completed, selected_learning_rate = (
            _iterative_trial_scoring_memory_safe(
                logger=logger,
                trial=trial,
                folds=folds,
                train_frame=shared_context.train_frame,
                tuning_frame=shared_context.tuning_frame,
                train_dataset_source_counts=shared_context.train_dataset_source_counts,
                tuning_dataset_sources=shared_context.tuning_dataset_sources,
                shared_absolute_target=shared_context.tuning_absolute_target,
                shared_reconstruction_anchor=shared_context.tuning_reconstruction_anchor,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
            )
        )
    else:
        cached_folds = _cached_fold_artifacts(
            train_frame=shared_context.train_frame,
            tuning_frame=shared_context.tuning_frame,
            folds=folds,
            execution_plan=execution_plan,
            train_dataset_source_counts=shared_context.train_dataset_source_counts,
            tuning_dataset_sources=shared_context.tuning_dataset_sources,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
        )
        effective_params, selected_learning_rate = _resolved_trial_learning_rate(
            folds=folds,
            cached_folds=cached_folds,
            resolved_params=resolved_params,
            logger=logger,
        )
        fold_results, folds_completed, selected_learning_rate = _score_all_folds(
            folds=folds,
            cached_folds=cached_folds,
            execution_plan=execution_plan,
            logger=logger,
            shared_dataset_sources=shared_context.tuning_dataset_sources,
            shared_absolute_target=shared_context.tuning_absolute_target,
            shared_reconstruction_anchor=shared_context.tuning_reconstruction_anchor,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=effective_params,
            trial=trial,
        )
    dataset_mean_wape = _dataset_macro_scores_from_fold_results(fold_results)
    macro_mean_wape = float(np.mean(list(dataset_mean_wape.values())))
    mean_abs_bias = _mean_optional_metric(fold_results, metric_key="abs_bias")
    mean_coverage_80 = _mean_optional_metric(fold_results, metric_key="coverage_80")
    mean_coverage_95 = _mean_optional_metric(fold_results, metric_key="coverage_95")
    mean_pinball_loss = _mean_optional_metric(
        fold_results,
        metric_key="pinball_loss_p50",
    )
    if trial is not None and int(cast(Any, execution_plan["fold_workers"])) != 1:
        trial.report(-macro_mean_wape, step=max(1, folds_completed))
    return {
        "macro_mean_wape": macro_mean_wape,
        "dataset_mean_wape": dataset_mean_wape,
        "mean_abs_bias": mean_abs_bias,
        "mean_coverage_80": mean_coverage_80,
        "mean_coverage_95": mean_coverage_95,
        "mean_pinball_loss": mean_pinball_loss,
        "selected_learning_rate": selected_learning_rate,
        "fold_results": fold_results,
        "folds_completed": folds_completed,
        "execution_policy": execution_plan,
    }
