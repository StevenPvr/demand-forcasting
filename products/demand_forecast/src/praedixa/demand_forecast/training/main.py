from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Praedixa TFT optimisation from local parquets or a versioned bundle.")
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--runtime-profile", type=str, default="local_cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--stage-budget", type=str, default="standard")
    parser.add_argument("--train-sample-fraction", type=float, default=1.0)
    parser.add_argument("--tuning-sample-fraction", type=float, default=1.0)
    parser.add_argument("--tensorboard-logdir", type=Path, default=None)
    return parser.parse_args()


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
    args = _parse_args()
    train_input_path, tuning_input_path = _resolve_bundle_inputs(args.bundle_dir)
    model_params: dict[str, object] = {
        "n_jobs": os.cpu_count() or 1,
        "runtime_profile": args.runtime_profile,
        "stage_budget": args.stage_budget,
    }
    if args.tensorboard_logdir is not None:
        model_params["tensorboard_logdir"] = str(args.tensorboard_logdir)
    try:
        outputs = build_optimisation_outputs(
            OptimisationBuildRequest(
                train_input_path=train_input_path,
                tuning_input_path=tuning_input_path,
                output_dir=args.output_dir,
                n_folds=int(args.n_folds),
                tuning_trials=int(args.max_trials),
                train_sample_fraction=float(args.train_sample_fraction),
                tuning_sample_fraction=float(args.tuning_sample_fraction),
                model_params=model_params,
                bundle_dir=args.bundle_dir,
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
