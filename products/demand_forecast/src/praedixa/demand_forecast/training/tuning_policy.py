from __future__ import annotations

import logging
import math
import os
import socket
from typing import Any, cast

import optuna

from praedixa.demand_forecast.backends.tft.runtime_profile import (
    DEFAULT_RUNTIME_PROFILE_NAME,
    resolve_runtime_profile,
)
from praedixa.demand_forecast.training.constants import DEFAULT_MAX_PARALLEL_FOLD_WORKERS

_HIDDEN_SIZE_CHOICES = [8, 16, 24, 32, 48, 64]
_HIDDEN_CONTINUOUS_SIZE_CHOICES = [4, 8, 12, 16, 24, 32]
_ATTENTION_HEAD_SIZE_CHOICES = [1, 2, 4]
_BATCH_SIZE_CHOICES = [32, 64, 128]
_ENCODER_LENGTH_CHOICES = [14, 28, 56]
_GRADIENT_CLIP_CHOICES = [0.05, 0.1, 0.5, 1.0]
_LSTM_LAYER_CHOICES = [1, 2, 3]
_HPO_STAGE_A_EPOCH_RANGE = (8, 12)
_HPO_STAGE_B_EPOCH_RANGE = (20, 30)
_HPO_PRUNER_CONFIG: dict[str, int | str] = {
    "type": "MedianPruner",
    "n_startup_trials": 2,
    "n_warmup_steps": 1,
    "interval_steps": 1,
}
_LEARNING_RATE_SEARCH_LOW = 1e-3
_LEARNING_RATE_SEARCH_HIGH = 1e-2


def _log_float_bounds(anchor: object, *, low: float, high: float) -> tuple[float, float]:
    if not isinstance(anchor, (int, float)):
        return low, high
    lower = max(low, float(anchor) * 0.5)
    upper = min(high, float(anchor) * 2.0)
    return (low, high) if lower >= upper else (lower, upper)


def _linear_float_bounds(anchor: object, *, low: float, high: float, margin: float) -> tuple[float, float]:
    if not isinstance(anchor, (int, float)):
        return low, high
    lower = max(low, float(anchor) - margin)
    upper = min(high, float(anchor) + margin)
    return (low, high) if lower >= upper else (lower, upper)


def resolve_stage_policy(
    tuning_trials: int,
    *,
    stage_budget: str = "standard",
) -> dict[str, dict[str, object]]:
    stage_a_ratio = 0.75
    stage_a_epoch_range = _HPO_STAGE_A_EPOCH_RANGE
    stage_b_epoch_range = _HPO_STAGE_B_EPOCH_RANGE
    if stage_budget == "quick":
        stage_a_ratio, stage_a_epoch_range, stage_b_epoch_range = 0.85, (6, 10), (12, 18)
    elif stage_budget == "full":
        stage_a_ratio, stage_a_epoch_range, stage_b_epoch_range = 0.65, (10, 14), (24, 36)
    stage_a_trials = max(1, min(tuning_trials, int(math.ceil(tuning_trials * stage_a_ratio))))
    return {
        "stage_a": {"trial_count": stage_a_trials, "epoch_range": list(stage_a_epoch_range), "search_mode": "broad"},
        "stage_b": {
            "trial_count": max(0, tuning_trials - stage_a_trials),
            "epoch_range": list(stage_b_epoch_range),
            "search_mode": "narrowed_top_k",
        },
        "stage_c": {"trial_count": 1, "search_mode": "final_selected_config", "execution": "outside_optuna", "stage_budget": stage_budget},
    }


def stage_name_for_trial(*, trial_number: int, tuning_trials: int, stage_budget: str) -> str:
    stage_a_trials = int(cast(Any, resolve_stage_policy(tuning_trials, stage_budget=stage_budget)["stage_a"]["trial_count"]))
    return "stage_a" if trial_number < stage_a_trials else "stage_b"


def best_completed_trial_params(study: optuna.study.Study) -> dict[str, object] | None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return None
    best_trial = max(completed_trials, key=required_trial_value)
    return dict(best_trial.params)


def sample_optuna_params(
    trial: optuna.trial.Trial,
    random_seed: int,
    *,
    stage_name: str = "stage_a",
    anchor_params: dict[str, object] | None = None,
    epoch_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    if stage_name == "stage_b":
        resolved_epoch_range = epoch_range or _HPO_STAGE_B_EPOCH_RANGE
        learning_rate_low, learning_rate_high = _log_float_bounds(
            anchor_params.get("learning_rate") if anchor_params else None,
            low=_LEARNING_RATE_SEARCH_LOW,
            high=_LEARNING_RATE_SEARCH_HIGH,
        )
        dropout_low, dropout_high = _linear_float_bounds(anchor_params.get("dropout") if anchor_params else None, low=0.05, high=0.30, margin=0.08)
        weight_decay_low, weight_decay_high = _log_float_bounds(anchor_params.get("weight_decay") if anchor_params else None, low=1e-6, high=1e-2)
        return {
            "max_epochs": trial.suggest_int("max_epochs", *resolved_epoch_range),
            "batch_size": trial.suggest_categorical("batch_size", _BATCH_SIZE_CHOICES),
            "max_encoder_length": trial.suggest_categorical("max_encoder_length", _ENCODER_LENGTH_CHOICES),
            "gradient_clip_val": trial.suggest_categorical("gradient_clip_val", _GRADIENT_CLIP_CHOICES),
            "learning_rate": trial.suggest_float("learning_rate", learning_rate_low, learning_rate_high, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", _HIDDEN_SIZE_CHOICES),
            "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", _HIDDEN_CONTINUOUS_SIZE_CHOICES),
            "attention_head_size": trial.suggest_categorical("attention_head_size", _ATTENTION_HEAD_SIZE_CHOICES),
            "lstm_layers": trial.suggest_categorical("lstm_layers", _LSTM_LAYER_CHOICES),
            "dropout": trial.suggest_float("dropout", dropout_low, dropout_high),
            "weight_decay": trial.suggest_float("weight_decay", weight_decay_low, weight_decay_high, log=True),
            "random_state": int(random_seed + trial.number + 1),
        }
    resolved_epoch_range = epoch_range or _HPO_STAGE_A_EPOCH_RANGE
    return {
        "max_epochs": trial.suggest_int("max_epochs", *resolved_epoch_range),
        "batch_size": trial.suggest_categorical("batch_size", _BATCH_SIZE_CHOICES),
        "max_encoder_length": trial.suggest_categorical("max_encoder_length", _ENCODER_LENGTH_CHOICES),
        "gradient_clip_val": trial.suggest_categorical("gradient_clip_val", _GRADIENT_CLIP_CHOICES),
        "learning_rate": trial.suggest_float("learning_rate", _LEARNING_RATE_SEARCH_LOW, _LEARNING_RATE_SEARCH_HIGH, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", _HIDDEN_SIZE_CHOICES),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", _HIDDEN_CONTINUOUS_SIZE_CHOICES),
        "attention_head_size": trial.suggest_categorical("attention_head_size", _ATTENTION_HEAD_SIZE_CHOICES),
        "lstm_layers": trial.suggest_categorical("lstm_layers", _LSTM_LAYER_CHOICES),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "random_state": int(random_seed + trial.number + 1),
    }


def required_trial_value(trial: optuna.trial.FrozenTrial) -> float:
    if trial.value is None:
        raise ValueError(f"Optuna trial {trial.number} has no objective value.")
    return float(trial.value)


def resolve_fold_execution_plan(
    *,
    folds: list[dict[str, object]],
    resolved_params: dict[str, object],
    total_threads: int | None,
    logger: logging.Logger,
) -> dict[str, object]:
    runtime_profile_name = str(resolved_params.get("runtime_profile", DEFAULT_RUNTIME_PROFILE_NAME))
    runtime_profile = resolve_runtime_profile(runtime_profile_name)
    accelerator = str(resolved_params.get("accelerator", runtime_profile["accelerator"]))
    devices = int(cast(Any, resolved_params.get("devices", runtime_profile["devices"])))
    max_parallel_fold_workers = int(cast(Any, resolved_params.get("max_parallel_fold_workers", DEFAULT_MAX_PARALLEL_FOLD_WORKERS)))
    num_threads = int(total_threads or cast(Any, resolved_params.get("n_jobs", os.cpu_count() or 1)))
    gpu_safe_mode = accelerator in {"gpu", "mps"} or runtime_profile_name == "mac_metal"
    fold_workers = 1 if gpu_safe_mode else max(1, min(len(folds), max_parallel_fold_workers))
    threads_per_fold = max(1, num_threads // fold_workers)
    logger.info(
        "Optuna fold execution plan: runtime_profile=%s accelerator=%s devices=%s gpu_safe_mode=%s fold_workers=%s threads_per_fold=%s total_threads=%s",
        runtime_profile_name, accelerator, devices, gpu_safe_mode, fold_workers, threads_per_fold, num_threads,
    )
    return {
        "runtime_profile": runtime_profile_name,
        "accelerator": accelerator,
        "devices": devices,
        "gpu_safe_mode": gpu_safe_mode,
        "fold_workers": fold_workers,
        "threads_per_fold": threads_per_fold,
        "total_threads": num_threads,
        "requested_max_parallel_fold_workers": max_parallel_fold_workers,
    }


def resolve_hpo_execution_policy(
    *,
    folds: list[dict[str, object]],
    resolved_params: dict[str, object],
    total_threads: int | None,
    logger: logging.Logger,
) -> dict[str, object]:
    return resolve_fold_execution_plan(
        folds=folds,
        resolved_params=resolved_params,
        total_threads=total_threads,
        logger=logger,
    )


def runtime_environment_metadata(*, execution_policy: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "host_name": socket.gethostname(),
        "instance_type": os.getenv("PRAEDIXA_INSTANCE_TYPE"),
        "gpu_name": None,
        "cuda_version": None,
        "device_count": None,
        "runtime_profile": execution_policy["runtime_profile"],
    }
    try:
        import torch
    except ImportError:
        return metadata
    if torch.cuda.is_available():
        metadata["device_count"] = int(torch.cuda.device_count())
        metadata["gpu_name"] = str(torch.cuda.get_device_name(0))
        metadata["cuda_version"] = getattr(torch.version, "cuda", None)
    return metadata


def build_hpo_runtime_metadata(
    *,
    study: optuna.study.Study,
    execution_policy: dict[str, object],
    tuning_trials: int,
    stage_budget: str,
) -> dict[str, object]:
    trial_states = [trial.state.name for trial in study.trials]
    status_counts = {state: trial_states.count(state) for state in sorted(set(trial_states))}
    best_trial = study.best_trial
    return {
        "execution_policy": execution_policy,
        "stage_policy": resolve_stage_policy(tuning_trials, stage_budget=stage_budget),
        "pruner": dict(_HPO_PRUNER_CONFIG),
        "system": runtime_environment_metadata(execution_policy=execution_policy),
        "trial_status_counts": status_counts,
        "best_trial_number": int(best_trial.number),
        "best_improvement_pct": required_trial_value(best_trial),
        "best_mean_wape": float(best_trial.user_attrs["mean_wape"]),
        "best_mean_abs_bias": best_trial.user_attrs.get("mean_abs_bias"),
        "best_coverage_80": best_trial.user_attrs.get("mean_coverage_80"),
        "best_coverage_95": best_trial.user_attrs.get("mean_coverage_95"),
    }


def create_optuna_study(*, random_seed: int) -> optuna.study.Study:
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(cast(Any, _HPO_PRUNER_CONFIG["n_startup_trials"])),
        n_warmup_steps=int(cast(Any, _HPO_PRUNER_CONFIG["n_warmup_steps"])),
        interval_steps=int(cast(Any, _HPO_PRUNER_CONFIG["interval_steps"])),
    )
    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
