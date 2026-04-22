from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys


def _bootstrap_import_paths() -> None:
    if __package__ not in {None, ""}:
        return
    current_file = Path(__file__).resolve()
    try:
        project_root = next(
            parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
        )
    except StopIteration as exc:  # pragma: no cover - repository invariant
        raise RuntimeError(
            f"Unable to resolve the Praedixa project root from {current_file}"
        ) from exc
    search_paths: tuple[Path, ...] = (
        project_root,
        project_root / "platform" / "python" / "src",
        project_root / "products" / "demand_forecast" / "src",
    )
    for path in reversed(search_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()


def _load_optimisation_main_defaults() -> tuple[str, str, int, int, str, float, float]:
    from praedixa.demand_forecast.training.constants import (
        DEFAULT_OPTIMISATION_MAIN_MAX_TRIALS,
        DEFAULT_OPTIMISATION_MAIN_N_FOLDS,
        DEFAULT_OPTIMISATION_MAIN_RUNTIME_PROFILE,
        DEFAULT_OPTIMISATION_MAIN_STAGE_BUDGET,
        DEFAULT_OPTIMISATION_MAIN_TRAIN_SAMPLE_FRACTION,
        DEFAULT_OPTIMISATION_MAIN_TUNING_SAMPLE_FRACTION,
    )
    from praedixa.platform.runtime.paths import TRAINING_BUNDLE_DIR

    return (
        str(TRAINING_BUNDLE_DIR),
        DEFAULT_OPTIMISATION_MAIN_RUNTIME_PROFILE,
        DEFAULT_OPTIMISATION_MAIN_N_FOLDS,
        DEFAULT_OPTIMISATION_MAIN_MAX_TRIALS,
        DEFAULT_OPTIMISATION_MAIN_STAGE_BUDGET,
        DEFAULT_OPTIMISATION_MAIN_TRAIN_SAMPLE_FRACTION,
        DEFAULT_OPTIMISATION_MAIN_TUNING_SAMPLE_FRACTION,
    )


(
    DEFAULT_BUNDLE_DIR,
    DEFAULT_RUNTIME_PROFILE,
    DEFAULT_N_FOLDS,
    DEFAULT_MAX_TRIALS,
    DEFAULT_STAGE_BUDGET,
    DEFAULT_TRAIN_SAMPLE_FRACTION,
    DEFAULT_TUNING_SAMPLE_FRACTION,
) = _load_optimisation_main_defaults()

DEFAULT_TENSORBOARD_LOGDIR: Path | None = None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _validate_runtime_profile(requested_profile: str) -> None:
    if requested_profile == "local_cpu":
        return
    if requested_profile == "mac_metal":
        if _mps_available():
            return
        raise RuntimeError(
            "The official optimisation run is configured with `mac_metal`, but MPS is unavailable on this machine. "
            "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
        )
    if requested_profile == "scaleway_l40s":
        if _cuda_available():
            return
        raise RuntimeError(
            "The official optimisation run is configured with `scaleway_l40s`, but CUDA is unavailable on this machine. "
            "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
        )
    raise RuntimeError(
        f"Unknown official optimisation runtime profile `{requested_profile}`. "
        "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
    )


def _default_output_dir() -> Path:
    from praedixa.platform.runtime.paths import OPTIMISATION_DIR

    return OPTIMISATION_DIR


def _default_runtime_profile() -> str:
    if _cuda_available():
        return "scaleway_l40s"
    if _mps_available():
        return "mac_metal"
    return DEFAULT_RUNTIME_PROFILE


@dataclass(frozen=True)
class OptimisationMainConfig:
    bundle_dir: Path = Path(DEFAULT_BUNDLE_DIR)
    runtime_profile: str = field(default_factory=_default_runtime_profile)
    output_dir: Path = field(default_factory=_default_output_dir)
    n_folds: int = DEFAULT_N_FOLDS
    max_trials: int = DEFAULT_MAX_TRIALS
    stage_budget: str = DEFAULT_STAGE_BUDGET
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION
    tensorboard_logdir: Path | None = DEFAULT_TENSORBOARD_LOGDIR


def build_default_optimisation_main_config() -> OptimisationMainConfig:
    return OptimisationMainConfig()


OFFICIAL_OPTIMISATION_MAIN_CONFIG = build_default_optimisation_main_config()


def _validate_bundle_dir(bundle_dir: Path) -> tuple[Path, Path, Path]:
    optimisation_train_input_path = bundle_dir / "optimisation_train.parquet"
    optimisation_tuning_input_path = bundle_dir / "optimisation_tuning.parquet"
    if optimisation_train_input_path.exists() and optimisation_tuning_input_path.exists():
        return optimisation_train_input_path, optimisation_tuning_input_path, bundle_dir
    train_input_path = bundle_dir / "train.parquet"
    tuning_input_path = bundle_dir / "tuning.parquet"
    if not train_input_path.exists():
        raise FileNotFoundError(f"Training bundle is missing `{train_input_path}`.")
    if not tuning_input_path.exists():
        raise FileNotFoundError(f"Training bundle is missing `{tuning_input_path}`.")
    return train_input_path, tuning_input_path, bundle_dir


def _resolve_official_bundle_inputs(bundle_dir: Path) -> tuple[Path, Path, Path]:
    return _validate_bundle_dir(bundle_dir)


def main() -> None:
    from praedixa.demand_forecast.backends.tft.backend import TFTBackendNotReadyError
    from praedixa.demand_forecast.training.pipeline import (
        OptimisationBuildRequest,
        build_optimisation_outputs,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = OFFICIAL_OPTIMISATION_MAIN_CONFIG
    logging.getLogger(__name__).info(
        "Selected official optimisation runtime profile: runtime_profile=%s cuda_available=%s mps_available=%s",
        config.runtime_profile,
        _cuda_available(),
        _mps_available(),
    )
    train_input_path, tuning_input_path, resolved_bundle_dir = _resolve_official_bundle_inputs(config.bundle_dir)
    _validate_runtime_profile(config.runtime_profile)
    model_params: dict[str, object] = {
        "n_jobs": os.cpu_count() or 1,
        "runtime_profile": config.runtime_profile,
        "stage_budget": config.stage_budget,
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
                bundle_dir=resolved_bundle_dir,
            ),
        )
    except TFTBackendNotReadyError as exc:
        logging.getLogger(__name__).error(
            "Optimisation aborted because the TFT backend is unavailable: %s",
            exc,
        )
        raise
    logging.getLogger(__name__).info(
        "Optimisation artifacts written: %s",
        json.dumps({name: str(path) for name, path in outputs.items()}, indent=2),
    )


if __name__ == "__main__":
    main()
