from __future__ import annotations

import logging
import os
import platform
from typing import Mapping

import numpy as np
import pandas as pd

from praedixa.demand_forecast.feature_screening.baselines import (
    evaluate_statistical_baselines as _evaluate_statistical_baselines_internal,
)
from praedixa.demand_forecast.feature_screening.constants import (
    DEFAULT_CORRELATION_PROGRESS_LOG_EVERY,
    DEFAULT_DATE_COL,
    DEFAULT_LAG_PATTERNS,
    DEFAULT_N_FOLDS,
    DEFAULT_PEARSON_THRESHOLD,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPEARMAN_THRESHOLD,
    DEFAULT_TARGET_COL,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_TUNING_PROGRESS_LOG_EVERY,
    DEFAULT_TUNING_TRIALS,
)
from praedixa.demand_forecast.feature_screening.correlation import (
    filter_correlated_lag_features as _filter_correlated_lag_features_internal,
)
from praedixa.demand_forecast.feature_screening.correlation import (
    filter_correlated_lag_features_memmap as _filter_correlated_lag_features_memmap_internal,
)
from praedixa.demand_forecast.feature_screening.correlation import (
    get_lag_candidate_columns as _get_lag_candidate_columns_internal,
)
from praedixa.demand_forecast.feature_screening.correlation import (
    get_lag_candidate_columns_from_names as _get_lag_candidate_columns_from_names_internal,
)
from praedixa.demand_forecast.feature_screening.splits import (
    build_walk_forward_folds as _build_walk_forward_folds_internal,
)
from praedixa.demand_forecast.feature_screening.splits import (
    split_chronological_train_tuning as _split_chronological_train_tuning_internal,
)
from praedixa.demand_forecast.feature_screening.tuning import (
    optimize_non_lag_model_params as _optimize_non_lag_model_params_internal,
)
from praedixa.demand_forecast.feature_screening.orchestrator import (
    build_lag_selection_outputs,
)
from praedixa.demand_forecast.feature_screening.metrics import (
    compute_wape,
    compute_wape_improvement_pct,
)


logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_N_FOLDS",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TUNING_PROGRESS_LOG_EVERY",
    "DEFAULT_TUNING_TRIALS",
    "build_lag_selection_outputs",
    "build_walk_forward_folds",
    "compute_wape",
    "compute_wape_improvement_pct",
    "evaluate_statistical_baselines",
    "filter_correlated_lag_features",
    "get_lag_candidate_columns",
    "optimize_non_lag_model_params",
    "resolve_parallelism",
    "split_chronological_train_tuning",
]


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def get_lag_candidate_columns(frame: pd.DataFrame) -> list[str]:
    return _get_lag_candidate_columns_internal(frame, lag_patterns=DEFAULT_LAG_PATTERNS)


def get_lag_candidate_columns_from_names(column_names: list[str]) -> list[str]:
    return _get_lag_candidate_columns_from_names_internal(
        column_names, lag_patterns=DEFAULT_LAG_PATTERNS
    )


def split_chronological_train_tuning(
    frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    return _split_chronological_train_tuning_internal(
        frame, date_col=date_col, train_fraction=train_fraction, logger=logger
    )


def build_walk_forward_folds(
    frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    return _build_walk_forward_folds_internal(
        frame, date_col=date_col, n_folds=n_folds, logger=logger
    )


def filter_correlated_lag_features(
    frame: pd.DataFrame,
    candidate_cols: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    return _filter_correlated_lag_features_internal(
        frame,
        candidate_cols,
        target_col=target_col,
        pearson_threshold=pearson_threshold,
        spearman_threshold=spearman_threshold,
        logger=logger,
    )


def filter_correlated_lag_features_memmap(
    candidate_matrix: np.memmap,
    candidate_cols: list[str],
    target_values: np.ndarray,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    return _filter_correlated_lag_features_memmap_internal(
        candidate_matrix,
        candidate_cols,
        target_values,
        pearson_threshold=pearson_threshold,
        spearman_threshold=spearman_threshold,
        correlation_progress_log_every=DEFAULT_CORRELATION_PROGRESS_LOG_EVERY,
        logger=logger,
    )

def resolve_parallelism(
    requested_workers: int | None,
    candidate_count: int,
    total_threads: int | None = None,
) -> tuple[int, int]:
    available_threads = max(1, total_threads or (os.cpu_count() or 1))
    if _is_macos():
        logger.info(
            "macOS detected: forcing outer parallelism to 1 worker for la stabilite du backend TFT, using all cores inside each fit."
        )
        return 1, available_threads
    max_workers = max(1, requested_workers or available_threads)
    resolved_workers = max(1, min(max_workers, max(1, candidate_count)))
    threads_per_worker = max(1, available_threads // resolved_workers)
    return resolved_workers, threads_per_worker

def evaluate_statistical_baselines(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    target_col: str = DEFAULT_TARGET_COL,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return _evaluate_statistical_baselines_internal(
        frame, folds, logger=logger, target_col=target_col
    )


def optimize_non_lag_model_params(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    n_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    baseline_wape: float | None = None,
    base_model_params: dict[str, object] | None = None,
    num_workers: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    return _optimize_non_lag_model_params_internal(
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        logger=logger,
        target_col=target_col,
        n_trials=n_trials,
        random_seed=random_seed,
        baseline_wape=baseline_wape,
        base_model_params=base_model_params,
        num_workers=num_workers,
    )
