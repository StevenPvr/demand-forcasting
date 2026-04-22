from __future__ import annotations

import logging

import pandas as pd

from praedixa.demand_forecast.training.baselines import (
    evaluate_statistical_baselines_macro as _evaluate_statistical_baselines_macro_internal,
)
from praedixa.demand_forecast.training.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_N_FOLDS,
)
from praedixa.demand_forecast.training.folds import (
    build_grouped_tuning_walk_forward_folds_by_dataset as _build_grouped_tuning_walk_forward_folds_by_dataset_internal,
)
from praedixa.demand_forecast.training.folds import (
    build_tuning_walk_forward_folds as _build_tuning_walk_forward_folds_internal,
)
from praedixa.demand_forecast.training.folds import (
    build_tuning_walk_forward_folds_by_dataset as _build_tuning_walk_forward_folds_by_dataset_internal,
)
from praedixa.demand_forecast.training.orchestrator import (
    OptimisationBuildRequest,
    build_optimisation_outputs,
)
from praedixa.demand_forecast.training.sampling_loaders import (
    load_gold_train_tuning_frames,
    load_parquet_train_tuning_frames,
)
from praedixa.demand_forecast.contracts.targets import TargetContract


logger = logging.getLogger(__name__)

__all__ = [
    "build_grouped_tuning_walk_forward_folds_by_dataset",
    "OptimisationBuildRequest",
    "build_optimisation_outputs",
    "build_tuning_walk_forward_folds",
    "build_tuning_walk_forward_folds_by_dataset",
    "evaluate_statistical_baselines_macro",
    "load_gold_train_tuning_frames",
    "load_parquet_train_tuning_frames",
]

def build_tuning_walk_forward_folds(
    tuning_frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    return _build_tuning_walk_forward_folds_internal(
        tuning_frame,
        date_col=date_col,
        n_folds=n_folds,
        logger=logger,
    )


def build_tuning_walk_forward_folds_by_dataset(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    return _build_tuning_walk_forward_folds_by_dataset_internal(
        tuning_frame,
        date_col=date_col,
        dataset_source_col=dataset_source_col,
        n_folds=n_folds,
        logger=logger,
    )


def build_grouped_tuning_walk_forward_folds_by_dataset(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    dataset_source_col: str = DEFAULT_DATASET_SOURCE_COL,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[dict[str, object]]:
    return _build_grouped_tuning_walk_forward_folds_by_dataset_internal(
        tuning_frame,
        date_col=date_col,
        dataset_source_col=dataset_source_col,
        n_folds=n_folds,
        logger=logger,
    )

def evaluate_statistical_baselines_macro(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    *,
    absolute_target_col: str,
    target_contract: TargetContract | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return _evaluate_statistical_baselines_macro_internal(
        frame,
        folds,
        absolute_target_col=absolute_target_col,
        target_contract=target_contract,
        logger=logger,
    )
