from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATE_COL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_GOLD_TABLE,
    DEFAULT_MODEL_BACKEND,
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
    DEFAULT_XGBOOST_LOCAL_CPU_MAX_THREADS,
)
from praedixa.demand_forecast.training.shared.model_backends import (
    resolve_optimisation_model_backend,
)
from praedixa.demand_forecast.training.orchestration.steps import (
    LoadedOptimisationFrames,
    OptimisationRunConfig,
    PreparedOptimisationContext,
    load_optimisation_frames,
    prepare_optimisation_context,
    run_optimisation_pipeline,
)


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
    model_backend: str = DEFAULT_MODEL_BACKEND


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


def _resolve_requested_n_jobs(model_params: dict[str, object] | None) -> int:
    raw_n_jobs = (model_params or {}).get("n_jobs")
    default_n_jobs = os.cpu_count() or 1
    if isinstance(raw_n_jobs, bool):
        return default_n_jobs
    if isinstance(raw_n_jobs, int):
        return max(1, raw_n_jobs)
    if isinstance(raw_n_jobs, str):
        try:
            return max(1, int(raw_n_jobs.strip()))
        except ValueError:
            return default_n_jobs
    return default_n_jobs


def _loaded_context_and_run_config(
    *,
    request: OptimisationBuildRequest,
    logger: logging.Logger,
) -> tuple[LoadedOptimisationFrames, PreparedOptimisationContext, OptimisationRunConfig]:
    resolved_backend = request.model_backend.strip().lower()
    requested_n_jobs = _resolve_requested_n_jobs(request.model_params)
    resolved_n_jobs = (
        max(1, min(requested_n_jobs, DEFAULT_XGBOOST_LOCAL_CPU_MAX_THREADS))
        if resolved_backend == "xgboost"
        else max(1, requested_n_jobs)
    )
    resolved_model_params = {
        **(request.model_params or {}),
        "n_jobs": resolved_n_jobs,
        "model_backend": resolved_backend,
    }
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
        model_backend=resolved_backend,
        logger=logger,
    )
    return loaded, context, OptimisationRunConfig(
        bundle_dir=request.bundle_dir,
        model_backend=resolved_backend,
        date_col=request.date_col,
        n_folds=request.n_folds,
        tuning_trials=request.tuning_trials,
        tuning_random_seed=request.tuning_random_seed,
        resolved_model_params=resolved_model_params,
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
    model_backend = resolve_optimisation_model_backend(resolved_request.model_backend)
    model_backend.require_available("optimisation.build_optimisation_outputs")
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
        optimize_fn=model_backend.optimize_fn,
    )
