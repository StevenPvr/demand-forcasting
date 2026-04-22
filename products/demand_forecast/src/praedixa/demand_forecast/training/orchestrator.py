from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from praedixa.demand_forecast.backends.tft.backend import raise_if_tft_backend_required
from praedixa.demand_forecast.training.constants import (
    DEFAULT_DATE_COL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_GOLD_TABLE,
    DEFAULT_N_FOLDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGET_COL,
    DEFAULT_TARGET_TRANSFORM,
    DEFAULT_TUNING_RANDOM_SEED,
    DEFAULT_TRAIN_INPUT_PATH,
    DEFAULT_TRAIN_SAMPLE_FRACTION,
    DEFAULT_TUNING_INPUT_PATH,
    DEFAULT_TUNING_SAMPLE_FRACTION,
    DEFAULT_TUNING_TRIALS,
)
from praedixa.demand_forecast.training.orchestrator_steps import (
    LoadedOptimisationFrames,
    OptimisationRunConfig,
    PreparedOptimisationContext,
    load_optimisation_frames,
    prepare_optimisation_context,
    run_optimisation_pipeline,
)
from praedixa.demand_forecast.training.tuning import optimize_tft_model_params


@dataclass(frozen=True)
class OptimisationBuildRequest:
    train_input_path: str | Path | None = DEFAULT_TRAIN_INPUT_PATH
    tuning_input_path: str | Path | None = DEFAULT_TUNING_INPUT_PATH
    bundle_dir: str | Path | None = None
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
    date_col: str = DEFAULT_DATE_COL
    target_col: str = DEFAULT_TARGET_COL
    n_folds: int = DEFAULT_N_FOLDS
    tuning_trials: int = DEFAULT_TUNING_TRIALS
    tuning_random_seed: int = DEFAULT_TUNING_RANDOM_SEED
    model_params: dict[str, object] | None = None
    duckdb_path: str | Path = DEFAULT_DUCKDB_PATH
    gold_table: str = DEFAULT_GOLD_TABLE
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION


def _prepare_output_dir(
    *,
    output_dir: str | Path,
    logger: logging.Logger,
    n_folds: int,
    tuning_trials: int,
    train_sample_fraction: float,
    tuning_sample_fraction: float,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting optimisation artifact build: output_dir=%s n_folds=%s tuning_trials=%s train_sample_fraction=%.4f tuning_sample_fraction=%.4f",
        target_dir,
        n_folds,
        tuning_trials,
        train_sample_fraction,
        tuning_sample_fraction,
    )
    return target_dir


def _loaded_context_and_run_config(
    *,
    request: OptimisationBuildRequest,
    logger: logging.Logger,
) -> tuple[LoadedOptimisationFrames, PreparedOptimisationContext, OptimisationRunConfig]:
    loaded = load_optimisation_frames(
        train_input_path=request.train_input_path,
        tuning_input_path=request.tuning_input_path,
        duckdb_path=request.duckdb_path,
        gold_table=request.gold_table,
        logger=logger,
        date_col=request.date_col,
        target_col=request.target_col,
        train_sample_fraction=request.train_sample_fraction,
        tuning_sample_fraction=request.tuning_sample_fraction,
    )
    context = prepare_optimisation_context(
        loaded=loaded,
        date_col=request.date_col,
        target_col=request.target_col,
        logger=logger,
    )
    return loaded, context, OptimisationRunConfig(
        bundle_dir=request.bundle_dir,
        date_col=request.date_col,
        n_folds=request.n_folds,
        tuning_trials=request.tuning_trials,
        tuning_random_seed=request.tuning_random_seed,
        resolved_model_params=request.model_params or {"n_jobs": os.cpu_count() or 1},
        resolved_target_transform=context.target_contract.target_mode or DEFAULT_TARGET_TRANSFORM,
        duckdb_path=request.duckdb_path,
        gold_table=request.gold_table,
    )


def build_optimisation_outputs(
    request: OptimisationBuildRequest | None = None,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    resolved_request = request or OptimisationBuildRequest()
    raise_if_tft_backend_required("optimisation.build_optimisation_outputs")
    resolved_logger = logger or logging.getLogger(__name__)
    target_dir = _prepare_output_dir(
        output_dir=resolved_request.output_dir,
        logger=resolved_logger,
        n_folds=resolved_request.n_folds,
        tuning_trials=resolved_request.tuning_trials,
        train_sample_fraction=resolved_request.train_sample_fraction,
        tuning_sample_fraction=resolved_request.tuning_sample_fraction,
    )
    loaded, context, run_config = _loaded_context_and_run_config(
        request=resolved_request,
        logger=resolved_logger,
    )
    return run_optimisation_pipeline(
        target_dir=target_dir,
        loaded=loaded,
        context=context,
        run_config=run_config,
        logger=resolved_logger,
        optimize_fn=optimize_tft_model_params,
    )
