from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


OptimizeNonLagModelParamsFn = Callable[..., tuple[dict[str, object], pd.DataFrame]]
RunSingleAddonLagSelectionFn = Callable[..., tuple[list[str], pd.DataFrame, dict[str, object]]]


@dataclass(frozen=True)
class PreparedLagSelectionInputs:
    target_dir: Path
    source_path: Path
    all_columns: list[str]
    lag_candidate_cols: list[str]
    split_metadata: dict[str, object]
    selection_mask: np.ndarray
    holdout_mask: np.ndarray
    selection_train: pd.DataFrame
    base_frame: pd.DataFrame
    work_dir: Path
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class CorrelationOnlyContext:
    source_path: Path
    output_paths: dict[str, Path]
    split_metadata: dict[str, object]
    date_col: str
    target_col: str
    selection_mask: np.ndarray
    holdout_mask: np.ndarray
    non_lag_feature_cols: list[str]
    required_scoring_context_cols: list[str]
    filtered_lag_cols: list[str]
    lag_candidate_cols: list[str]
    lag_candidate_count_after_corr_filter: int
    n_folds: int
    pearson_threshold: float
    spearman_threshold: float
    tuning_trials: int
    tuning_random_seed: int
    model_params: dict[str, object] | None


@dataclass(frozen=True)
class FullLagSelectionContext:
    source_path: Path
    output_paths: dict[str, Path]
    split_metadata: dict[str, object]
    selection_train: pd.DataFrame
    base_frame: pd.DataFrame
    selection_mask: np.ndarray
    holdout_mask: np.ndarray
    date_col: str
    target_col: str
    lag_candidate_cols: list[str]
    lag_candidate_count_after_corr_filter: int
    non_lag_feature_cols: list[str]
    required_scoring_context_cols: list[str]
    filtered_lag_cols: list[str]
    candidate_matrix: np.memmap
    candidate_to_index: dict[str, int]
    baseline_feature_cols: list[str]
    n_folds: int
    pearson_threshold: float
    spearman_threshold: float
    tuning_trials: int
    tuning_random_seed: int
    model_params: dict[str, object] | None
    num_workers: int | None
    logger: logging.Logger


@dataclass(frozen=True)
class LagSelectionConfig:
    input_path: str | Path | None
    output_dir: str | Path
    duckdb_path: str | Path
    gold_table: str
    date_col: str
    target_col: str
    train_fraction: float
    n_folds: int
    pearson_threshold: float
    spearman_threshold: float
    tuning_trials: int
    tuning_random_seed: int
    model_params: dict[str, object] | None
    num_workers: int | None
    correlation_only: bool
    max_selected_lag_features: int | None


@dataclass(frozen=True)
class LagSelectionExecutionContext:
    prepared: PreparedLagSelectionInputs
    date_col: str
    target_col: str
    non_lag_feature_cols: list[str]
    n_folds: int
    pearson_threshold: float
    spearman_threshold: float
    tuning_trials: int
    tuning_random_seed: int
    model_params: dict[str, object] | None
    num_workers: int | None
    correlation_only: bool
    max_selected_lag_features: int | None
    logger: logging.Logger
