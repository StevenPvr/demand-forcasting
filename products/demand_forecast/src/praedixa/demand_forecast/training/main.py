from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

from praedixa.demand_forecast.training.pipeline import (
    OptimisationBuildRequest,
    build_optimisation_outputs,
)
from praedixa.demand_forecast.backends.tft.backend import TFTBackendNotReadyError
from praedixa.platform.runtime.paths import FEATURE_SELECTION_DIR
from praedixa.platform.runtime.paths import OPTIMISATION_DIR


DEFAULT_FEATURE_SELECTION_DIR = FEATURE_SELECTION_DIR
DEFAULT_OUTPUT_DIR = OPTIMISATION_DIR
DEFAULT_RUNTIME_PROFILE = "scaleway_l40s"
DEFAULT_N_FOLDS = 2
DEFAULT_MAX_TRIALS = 2
DEFAULT_STAGE_BUDGET = "standard"
DEFAULT_TRAIN_SAMPLE_FRACTION = 1.0
DEFAULT_TUNING_SAMPLE_FRACTION = 1.0
DEFAULT_TENSORBOARD_LOGDIR: Path | None = None


@dataclass(frozen=True)
class OptimisationMainConfig:
    bundle_dir: Path | None = None
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    n_folds: int = DEFAULT_N_FOLDS
    max_trials: int = DEFAULT_MAX_TRIALS
    stage_budget: str = DEFAULT_STAGE_BUDGET
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION
    tensorboard_logdir: Path | None = DEFAULT_TENSORBOARD_LOGDIR


def build_official_optimisation_main_config() -> OptimisationMainConfig:
    return OptimisationMainConfig()


def _resolve_bundle_inputs(bundle_dir: Path | None) -> tuple[Path, Path]:
    if bundle_dir is None:
        return (
            DEFAULT_FEATURE_SELECTION_DIR / "train_selection_70_selected.parquet",
            DEFAULT_FEATURE_SELECTION_DIR / "train_tuning_30_selected.parquet",
        )
    train_input_path = bundle_dir / "train.parquet"
    tuning_input_path = bundle_dir / "tuning.parquet"
    if not train_input_path.exists():
        raise FileNotFoundError(f"Training bundle is missing `{train_input_path}`.")
    if not tuning_input_path.exists():
        raise FileNotFoundError(f"Training bundle is missing `{tuning_input_path}`.")
    return train_input_path, tuning_input_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_official_optimisation_main_config()
    train_input_path, tuning_input_path = _resolve_bundle_inputs(config.bundle_dir)
    model_params: dict[str, object] = {
        "n_jobs": os.cpu_count() or 1,
        "runtime_profile": config.runtime_profile,
        "stage_budget": config.stage_budget,
        "precision": "32-true",
    }
    if config.tensorboard_logdir is not None:
        model_params["tensorboard_logdir"] = str(config.tensorboard_logdir)
    try:
        outputs = build_optimisation_outputs(
            OptimisationBuildRequest(
                train_input_path=train_input_path,
                tuning_input_path=tuning_input_path,
                output_dir=config.output_dir,
                n_folds=config.n_folds,
                tuning_trials=config.max_trials,
                train_sample_fraction=config.train_sample_fraction,
                tuning_sample_fraction=config.tuning_sample_fraction,
                model_params=model_params,
                bundle_dir=config.bundle_dir,
            ),
        )
    except TFTBackendNotReadyError:
        logging.getLogger(__name__).error(
            "Le backend TFT unique n'est pas encore branche : l'etape optimisation reste un placeholder structurel."
        )
        raise
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
