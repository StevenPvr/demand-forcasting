from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from praedixa.demand_forecast.feature_screening.addon_selection import run_single_addon_lag_selection_memmap
from praedixa.demand_forecast.feature_screening.constants import (
    DEFAULT_CORRELATION_ONLY,
    DEFAULT_DATE_COL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_GOLD_TABLE,
    DEFAULT_INPUT_PATH,
    DEFAULT_MAX_SELECTED_LAG_FEATURES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PEARSON_THRESHOLD,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPEARMAN_THRESHOLD,
    DEFAULT_TARGET_COL,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_TUNING_TRIALS,
)
from praedixa.demand_forecast.feature_screening.orchestrator_flows import run_lag_selection_with_memmap
from praedixa.demand_forecast.feature_screening.orchestrator_io import prepare_lag_selection_inputs
from praedixa.demand_forecast.feature_screening.orchestrator_models import (
    LagSelectionConfig,
    LagSelectionExecutionContext,
)
from praedixa.demand_forecast.feature_screening.tuning import optimize_non_lag_model_params


def build_lag_selection_outputs(
    input_path: str | Path | None = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    logger: logging.Logger | None = None,
    **overrides: object,
) -> dict[str, Path]:
    resolved_logger = logger or logging.getLogger(__name__)
    config = resolve_lag_selection_config(input_path=input_path, output_dir=output_dir, overrides=overrides)
    prepared = prepare_lag_selection_inputs_from_config(config=config, logger=resolved_logger)
    non_lag_feature_cols = [column for column in prepared.base_frame.columns if column not in {config.target_col, config.date_col}]
    return run_lag_selection_with_memmap(
        context=LagSelectionExecutionContext(
            prepared=prepared,
            date_col=config.date_col,
            target_col=config.target_col,
            non_lag_feature_cols=non_lag_feature_cols,
            n_folds=config.n_folds,
            pearson_threshold=config.pearson_threshold,
            spearman_threshold=config.spearman_threshold,
            tuning_trials=config.tuning_trials,
            tuning_random_seed=config.tuning_random_seed,
            model_params=config.model_params,
            num_workers=config.num_workers,
            correlation_only=config.correlation_only,
            max_selected_lag_features=config.max_selected_lag_features,
            logger=resolved_logger,
        ),
        optimize_non_lag_model_params_fn=optimize_non_lag_model_params,
        run_single_addon_lag_selection_memmap_fn=run_single_addon_lag_selection_memmap,
    )


def resolve_lag_selection_config(
    *,
    input_path: str | Path | None,
    output_dir: str | Path,
    overrides: dict[str, object],
) -> LagSelectionConfig:
    allowed_keys = {
        "duckdb_path",
        "gold_table",
        "date_col",
        "target_col",
        "train_fraction",
        "n_folds",
        "pearson_threshold",
        "spearman_threshold",
        "tuning_trials",
        "tuning_random_seed",
        "model_params",
        "num_workers",
        "correlation_only",
        "max_selected_lag_features",
    }
    unknown_keys = sorted(set(overrides) - allowed_keys)
    if unknown_keys:
        raise TypeError(f"Unexpected lag selection options: {unknown_keys}")
    return LagSelectionConfig(
        input_path=input_path,
        output_dir=output_dir,
        duckdb_path=cast(str | Path, overrides.get("duckdb_path", DEFAULT_DUCKDB_PATH)),
        gold_table=cast(str, overrides.get("gold_table", DEFAULT_GOLD_TABLE)),
        date_col=cast(str, overrides.get("date_col", DEFAULT_DATE_COL)),
        target_col=cast(str, overrides.get("target_col", DEFAULT_TARGET_COL)),
        train_fraction=cast(float, overrides.get("train_fraction", DEFAULT_TRAIN_FRACTION)),
        n_folds=cast(int, overrides.get("n_folds", 5)),
        pearson_threshold=cast(float, overrides.get("pearson_threshold", DEFAULT_PEARSON_THRESHOLD)),
        spearman_threshold=cast(float, overrides.get("spearman_threshold", DEFAULT_SPEARMAN_THRESHOLD)),
        tuning_trials=cast(int, overrides.get("tuning_trials", DEFAULT_TUNING_TRIALS)),
        tuning_random_seed=cast(int, overrides.get("tuning_random_seed", DEFAULT_RANDOM_SEED)),
        model_params=cast(dict[str, object] | None, overrides.get("model_params")),
        num_workers=cast(int | None, overrides.get("num_workers")),
        correlation_only=cast(bool, overrides.get("correlation_only", DEFAULT_CORRELATION_ONLY)),
        max_selected_lag_features=cast(int | None, overrides.get("max_selected_lag_features", DEFAULT_MAX_SELECTED_LAG_FEATURES)),
    )


def prepare_lag_selection_inputs_from_config(
    *,
    config: LagSelectionConfig,
    logger: logging.Logger,
):
    return prepare_lag_selection_inputs(
        input_path=config.input_path,
        output_dir=config.output_dir,
        duckdb_path=config.duckdb_path,
        gold_table=config.gold_table,
        date_col=config.date_col,
        target_col=config.target_col,
        train_fraction=config.train_fraction,
        logger=logger,
    )
