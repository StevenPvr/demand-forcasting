from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.frame_utils import (
    attach_group_and_time_columns,
    build_combined_frame,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    lazy_import_tft_dependencies,
    resolve_model_params,
)
from praedixa.demand_forecast.backends.tft.training_dataset import (
    TrainingDatasetArtifacts,
    build_training_dataset_artifacts,
)
from praedixa.demand_forecast.feature_screening.constants import (
    DEFAULT_MODEL_PARAMS,
)


@dataclass(frozen=True)
class CachedScreeningFoldArtifacts:
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    dataset_artifacts: TrainingDatasetArtifacts


_SCREENING_FOLD_CACHE: dict[tuple[object, ...], CachedScreeningFoldArtifacts] = {}
_SCREENING_FOLD_CACHE_LOCK = Lock()


def clear_screening_fold_cache() -> None:
    with _SCREENING_FOLD_CACHE_LOCK:
        _SCREENING_FOLD_CACHE.clear()


def _take_frame_rows(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
) -> pd.DataFrame:
    selected = frame.iloc[row_indices]
    return selected.copy()


def _fold_signature(fold: dict[str, object]) -> tuple[object, ...]:
    return (
        int(cast(Any, fold["fold"])),
        tuple(np.asarray(fold["train_idx"], dtype=np.int32).tolist()),
        tuple(np.asarray(fold["valid_idx"], dtype=np.int32).tolist()),
    )


def _cache_key(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_col: str,
    max_encoder_length: int,
) -> tuple[object, ...]:
    return (
        id(frame),
        _fold_signature(fold),
        tuple(feature_cols),
        target_col,
        max_encoder_length,
    )


def _fold_frames(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_indices = np.asarray(fold["train_idx"], dtype=np.int32)
    valid_indices = np.asarray(fold["valid_idx"], dtype=np.int32)
    train_frame = _take_frame_rows(frame, train_indices)
    valid_frame = _take_frame_rows(frame, valid_indices)
    return train_frame, valid_frame


def _prepared_fold_frame(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    combined_frame = build_combined_frame(
        train_frame,
        valid_frame,
        train_weights=None,
        valid_weights=None,
    )
    return attach_group_and_time_columns(combined_frame, feature_cols)


def _build_fold_artifacts(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_col: str,
    max_encoder_length: int,
) -> CachedScreeningFoldArtifacts:
    train_frame, valid_frame = _fold_frames(frame=frame, fold=fold)
    imports = lazy_import_tft_dependencies()
    prepared_frame = _prepared_fold_frame(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
    )
    dataset_artifacts = build_training_dataset_artifacts(
        imports,
        prepared_frame=prepared_frame,
        feature_cols=feature_cols,
        target_col=target_col,
        resolved_params={"max_encoder_length": max_encoder_length},
    )
    return CachedScreeningFoldArtifacts(
        train_frame=train_frame,
        valid_frame=valid_frame,
        dataset_artifacts=dataset_artifacts,
    )


def cached_screening_fold_artifacts(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, object] | None,
) -> CachedScreeningFoldArtifacts:
    resolved_params = resolve_model_params(
        model_params,
        default_params=DEFAULT_MODEL_PARAMS,
        default_max_iter=300,
    )
    max_encoder_length = int(cast(Any, resolved_params["max_encoder_length"]))
    cache_key = _cache_key(
        frame=frame,
        fold=fold,
        feature_cols=feature_cols,
        target_col=target_col,
        max_encoder_length=max_encoder_length,
    )
    with _SCREENING_FOLD_CACHE_LOCK:
        cached = _SCREENING_FOLD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = _build_fold_artifacts(
        frame=frame,
        fold=fold,
        feature_cols=feature_cols,
        target_col=target_col,
        max_encoder_length=max_encoder_length,
    )
    with _SCREENING_FOLD_CACHE_LOCK:
        _SCREENING_FOLD_CACHE[cache_key] = cached
    return cached
