from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from praedixa.demand_forecast.training.config.main_config import (
    OFFICIAL_OPTIMISATION_MAIN_CONFIG,
    OptimisationMainConfig,
    build_optimisation_model_params,
    cuda_available,
    mps_available,
    validate_runtime_profile,
)
from praedixa.demand_forecast.backends.xgboost.runtime import (
    xgboost_cuda_preflight_available,
)


VERBOSE_TRAINING_LOGS_ENV = "PRAEDIXA_VERBOSE_TRAINING_LOGS"
QUIET_XGBOOST_LOGGER_NAMES = (
    "praedixa.demand_forecast.backends.xgboost",
    "praedixa.demand_forecast.training.baselines",
    "praedixa.demand_forecast.training.orchestration",
    "praedixa.demand_forecast.training.sampling",
    "praedixa.demand_forecast.training.validation",
    "praedixa.demand_forecast.training.xgboost",
)


def _validate_bundle_dir(bundle_dir: Path) -> tuple[Path, Path, Path]:
    optimisation_train_path = bundle_dir / "optimisation_train.parquet"
    optimisation_tuning_path = bundle_dir / "optimisation_tuning.parquet"
    if optimisation_train_path.exists() and optimisation_tuning_path.exists():
        return optimisation_train_path, optimisation_tuning_path, bundle_dir
    train_path = bundle_dir / "train.parquet"
    tuning_path = bundle_dir / "tuning.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training bundle is missing `{train_path}`.")
    if not tuning_path.exists():
        raise FileNotFoundError(f"Training bundle is missing `{tuning_path}`.")
    return train_path, tuning_path, bundle_dir


def _accelerator_availability_for_log(
    config: OptimisationMainConfig,
) -> tuple[bool, bool]:
    if config.model_backend == "xgboost":
        return xgboost_cuda_preflight_available(), False
    return cuda_available(), mps_available()


def _configure_backend_logging(config: OptimisationMainConfig) -> None:
    if os.getenv(VERBOSE_TRAINING_LOGS_ENV) == "1":
        return
    if config.model_backend != "xgboost":
        return
    for logger_name in QUIET_XGBOOST_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.INFO)


def run_optimisation_main(config: OptimisationMainConfig) -> None:
    from praedixa.demand_forecast.training.orchestration.pipeline import (
        OptimisationBuildRequest,
        build_optimisation_outputs,
    )

    _configure_backend_logging(config)
    logger = logging.getLogger(__name__)
    has_cuda, has_mps = _accelerator_availability_for_log(config)
    logger.info(
        "Selected official optimisation backend: model_backend=%s runtime_profile=%s cuda_available=%s mps_available=%s",
        config.model_backend,
        config.runtime_profile,
        has_cuda,
        has_mps,
    )
    train_input_path, tuning_input_path, resolved_bundle_dir = _validate_bundle_dir(
        config.bundle_dir
    )
    validate_runtime_profile(config.runtime_profile, model_backend=config.model_backend)
    outputs = build_optimisation_outputs(
        OptimisationBuildRequest(
            train_input_path=train_input_path,
            tuning_input_path=tuning_input_path,
            output_dir=config.output_dir,
            n_folds=config.n_folds,
            tuning_trials=config.max_trials,
            train_sample_fraction=config.train_sample_fraction,
            tuning_sample_fraction=config.tuning_sample_fraction,
            model_params=build_optimisation_model_params(config),
            bundle_dir=resolved_bundle_dir,
            model_backend=config.model_backend,
        )
    )
    logger.info(
        "Optimisation artifacts written: %s",
        json.dumps({name: str(path) for name, path in outputs.items()}, indent=2),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_optimisation_main(OFFICIAL_OPTIMISATION_MAIN_CONFIG)
