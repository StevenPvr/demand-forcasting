from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
import os
from pathlib import Path
from typing import Any


def _load_optimisation_main_defaults() -> tuple[
    str, str, str, int, int, str, float, float, int, int, str, float, float
]:
    from praedixa.demand_forecast.training.config.constants import (
        DEFAULT_H100_OPTIMISATION_MAIN_MAX_TRIALS,
        DEFAULT_H100_OPTIMISATION_MAIN_N_FOLDS,
        DEFAULT_H100_OPTIMISATION_MAIN_STAGE_BUDGET,
        DEFAULT_H100_OPTIMISATION_MAIN_TRAIN_SAMPLE_FRACTION,
        DEFAULT_H100_OPTIMISATION_MAIN_TUNING_SAMPLE_FRACTION,
        DEFAULT_MODEL_BACKEND,
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
        DEFAULT_MODEL_BACKEND,
        DEFAULT_OPTIMISATION_MAIN_RUNTIME_PROFILE,
        DEFAULT_OPTIMISATION_MAIN_N_FOLDS,
        DEFAULT_OPTIMISATION_MAIN_MAX_TRIALS,
        DEFAULT_OPTIMISATION_MAIN_STAGE_BUDGET,
        DEFAULT_OPTIMISATION_MAIN_TRAIN_SAMPLE_FRACTION,
        DEFAULT_OPTIMISATION_MAIN_TUNING_SAMPLE_FRACTION,
        DEFAULT_H100_OPTIMISATION_MAIN_N_FOLDS,
        DEFAULT_H100_OPTIMISATION_MAIN_MAX_TRIALS,
        DEFAULT_H100_OPTIMISATION_MAIN_STAGE_BUDGET,
        DEFAULT_H100_OPTIMISATION_MAIN_TRAIN_SAMPLE_FRACTION,
        DEFAULT_H100_OPTIMISATION_MAIN_TUNING_SAMPLE_FRACTION,
    )


(
    DEFAULT_BUNDLE_DIR,
    DEFAULT_MODEL_BACKEND,
    DEFAULT_RUNTIME_PROFILE,
    DEFAULT_N_FOLDS,
    DEFAULT_MAX_TRIALS,
    DEFAULT_STAGE_BUDGET,
    DEFAULT_TRAIN_SAMPLE_FRACTION,
    DEFAULT_TUNING_SAMPLE_FRACTION,
    DEFAULT_H100_N_FOLDS,
    DEFAULT_H100_MAX_TRIALS,
    DEFAULT_H100_STAGE_BUDGET,
    DEFAULT_H100_TRAIN_SAMPLE_FRACTION,
    DEFAULT_H100_TUNING_SAMPLE_FRACTION,
) = _load_optimisation_main_defaults()

DEFAULT_TENSORBOARD_LOGDIR: Path | None = None


def _torch_module() -> Any | None:
    if find_spec("torch") is None:
        return None
    return import_module("torch")


def _cuda_available() -> bool:
    torch_module = _torch_module()
    if torch_module is None:
        return False
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _cuda_device_name() -> str | None:
    torch_module = _torch_module()
    if torch_module is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        return None
    get_device_name = getattr(cuda, "get_device_name", None)
    if not callable(get_device_name):
        return None
    try:
        return str(get_device_name(0))
    except RuntimeError:
        return None


def _mps_available() -> bool:
    torch_module = _torch_module()
    if torch_module is None:
        return False
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def cuda_available() -> bool:
    return _cuda_available()


def mps_available() -> bool:
    return _mps_available()


def validate_runtime_profile(requested_profile: str, *, model_backend: str) -> None:
    if model_backend == "xgboost":
        if requested_profile == "local_cpu":
            return
        raise RuntimeError(
            "The XGBoost optimisation backend is currently wired as a CPU backend. "
            "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
        )
    if requested_profile == "local_cpu":
        return
    if requested_profile == "mac_metal":
        if _mps_available():
            return
        raise RuntimeError(
            "The official optimisation run is configured with `mac_metal`, but MPS is unavailable on this machine. "
            "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
        )
    if requested_profile in {"scaleway_l40s", "nvidia_h100"}:
        if _cuda_available():
            return
        raise RuntimeError(
            f"The official optimisation run is configured with `{requested_profile}`, but CUDA is unavailable on this machine. "
            "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
        )
    raise RuntimeError(
        f"Unknown official optimisation runtime profile `{requested_profile}`. "
        "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."
    )


def _default_output_dir() -> Path:
    from praedixa.platform.runtime.paths import OPTIMISATION_DIR

    return OPTIMISATION_DIR


def _default_xgboost_n_jobs(runtime_profile: str) -> int:
    from praedixa.demand_forecast.training.config.constants import (
        DEFAULT_XGBOOST_LOCAL_CPU_MAX_THREADS,
    )

    requested_threads = os.cpu_count() or 1
    if runtime_profile == "local_cpu":
        return max(1, min(requested_threads, DEFAULT_XGBOOST_LOCAL_CPU_MAX_THREADS))
    return max(1, requested_threads)


def _is_h100_cuda_device() -> bool:
    device_name = _cuda_device_name()
    return device_name is not None and "H100" in device_name.upper()


def _default_runtime_profile(*, model_backend: str = DEFAULT_MODEL_BACKEND) -> str:
    if model_backend == "xgboost":
        return DEFAULT_RUNTIME_PROFILE
    if _is_h100_cuda_device():
        return "nvidia_h100"
    if _cuda_available():
        return "scaleway_l40s"
    return DEFAULT_RUNTIME_PROFILE


def _default_budget_for_runtime(
    runtime_profile: str,
) -> tuple[int, int, str, float, float]:
    if runtime_profile == "nvidia_h100":
        return (
            DEFAULT_H100_N_FOLDS,
            DEFAULT_H100_MAX_TRIALS,
            DEFAULT_H100_STAGE_BUDGET,
            DEFAULT_H100_TRAIN_SAMPLE_FRACTION,
            DEFAULT_H100_TUNING_SAMPLE_FRACTION,
        )
    return (
        DEFAULT_N_FOLDS,
        DEFAULT_MAX_TRIALS,
        DEFAULT_STAGE_BUDGET,
        DEFAULT_TRAIN_SAMPLE_FRACTION,
        DEFAULT_TUNING_SAMPLE_FRACTION,
    )


@dataclass(frozen=True)
class OptimisationMainConfig:
    bundle_dir: Path = Path(DEFAULT_BUNDLE_DIR)
    model_backend: str = DEFAULT_MODEL_BACKEND
    runtime_profile: str = field(default_factory=_default_runtime_profile)
    output_dir: Path = field(default_factory=_default_output_dir)
    n_folds: int = DEFAULT_N_FOLDS
    max_trials: int = DEFAULT_MAX_TRIALS
    stage_budget: str = DEFAULT_STAGE_BUDGET
    train_sample_fraction: float = DEFAULT_TRAIN_SAMPLE_FRACTION
    tuning_sample_fraction: float = DEFAULT_TUNING_SAMPLE_FRACTION
    tensorboard_logdir: Path | None = DEFAULT_TENSORBOARD_LOGDIR


def build_default_optimisation_main_config() -> OptimisationMainConfig:
    return build_xgboost_optimisation_main_config()


def build_xgboost_optimisation_main_config() -> OptimisationMainConfig:
    return _build_backend_optimisation_main_config("xgboost")


def build_tft_optimisation_main_config() -> OptimisationMainConfig:
    return _build_backend_optimisation_main_config("tft")


def build_optimisation_model_params(
    config: OptimisationMainConfig,
) -> dict[str, object]:
    n_jobs = (
        _default_xgboost_n_jobs(config.runtime_profile)
        if config.model_backend == "xgboost"
        else os.cpu_count() or 1
    )
    model_params: dict[str, object] = {
        "n_jobs": n_jobs,
        "model_backend": config.model_backend,
        "runtime_profile": config.runtime_profile,
        "stage_budget": config.stage_budget,
    }
    if config.model_backend == "xgboost":
        from praedixa.demand_forecast.training.config.constants import (
            DEFAULT_XGBOOST_LOCAL_CPU_FOLD_WORKERS,
        )

        model_params["max_parallel_fold_workers"] = DEFAULT_XGBOOST_LOCAL_CPU_FOLD_WORKERS
    if config.tensorboard_logdir is not None:
        model_params["tensorboard_logdir"] = str(config.tensorboard_logdir)
    return model_params


def _build_backend_optimisation_main_config(
    model_backend: str,
) -> OptimisationMainConfig:
    runtime_profile = _default_runtime_profile(model_backend=model_backend)
    (
        n_folds,
        max_trials,
        stage_budget,
        train_sample_fraction,
        tuning_sample_fraction,
    ) = _default_budget_for_runtime(runtime_profile)
    return OptimisationMainConfig(
        model_backend=model_backend,
        runtime_profile=runtime_profile,
        n_folds=n_folds,
        max_trials=max_trials,
        stage_budget=stage_budget,
        train_sample_fraction=train_sample_fraction,
        tuning_sample_fraction=tuning_sample_fraction,
    )


OFFICIAL_OPTIMISATION_MAIN_CONFIG = build_default_optimisation_main_config()
