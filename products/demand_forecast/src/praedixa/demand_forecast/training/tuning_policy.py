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
from praedixa.demand_forecast.backends.tft.model_common import (
    should_serialize_local_cpu_folds,
)
from praedixa.demand_forecast.training.constants import (
    DEFAULT_MAX_PARALLEL_FOLD_WORKERS,
    DEFAULT_STAGE_BUDGET,
    DEFAULT_TUNING_ATTENTION_HEAD_SIZE_CHOICES,
    DEFAULT_TUNING_BATCH_SIZE_CHOICES,
    DEFAULT_TUNING_DROPOUT_HIGH,
    DEFAULT_TUNING_DROPOUT_LOW,
    DEFAULT_TUNING_DROPOUT_STAGE_B_MARGIN,
    DEFAULT_TUNING_ENCODER_LENGTH_CHOICES,
    DEFAULT_TUNING_GRADIENT_CLIP_CHOICES,
    DEFAULT_TUNING_HIDDEN_CONTINUOUS_SIZE_CHOICES,
    DEFAULT_TUNING_HIDDEN_SIZE_CHOICES,
    DEFAULT_TUNING_LEARNING_RATE_HIGH,
    DEFAULT_TUNING_LEARNING_RATE_LOW,
    DEFAULT_TUNING_LEARNING_RATE_REFERENCE_BATCH_SIZE,
    DEFAULT_TUNING_LSTM_LAYER_CHOICES,
    DEFAULT_TUNING_MAX_BATCH_SCALED_LEARNING_RATE_SCALE,
    DEFAULT_TUNING_PRUNER_CONFIG,
    DEFAULT_TUNING_SAMPLER_STARTUP_TRIALS,
    DEFAULT_TUNING_STAGE_POLICIES,
    DEFAULT_TUNING_WEIGHT_DECAY_HIGH,
    DEFAULT_TUNING_WEIGHT_DECAY_LOW,
)


def _linear_float_bounds(
    anchor: object, *, low: float, high: float, margin: float
) -> tuple[float, float]:
    if not isinstance(anchor, (int, float)):
        return low, high
    lower = max(low, float(anchor) - margin)
    upper = min(high, float(anchor) + margin)
    return (low, high) if lower >= upper else (lower, upper)


def _runtime_profile_batch_size_choices(runtime_profile_name: str) -> tuple[int, ...]:
    _ = runtime_profile_name
    return DEFAULT_TUNING_BATCH_SIZE_CHOICES


def _batch_scaled_learning_rate_bounds(batch_size: int) -> tuple[float, float]:
    reference_batch_size = max(1, DEFAULT_TUNING_LEARNING_RATE_REFERENCE_BATCH_SIZE)
    batch_scale = min(
        DEFAULT_TUNING_MAX_BATCH_SCALED_LEARNING_RATE_SCALE,
        max(1.0, float(batch_size) / float(reference_batch_size)),
    )
    return (
        DEFAULT_TUNING_LEARNING_RATE_LOW * batch_scale,
        DEFAULT_TUNING_LEARNING_RATE_HIGH * batch_scale,
    )


def resolve_stage_policy(
    tuning_trials: int,
    *,
    stage_budget: str = DEFAULT_STAGE_BUDGET,
) -> dict[str, dict[str, object]]:
    policy = DEFAULT_TUNING_STAGE_POLICIES.get(
        stage_budget,
        DEFAULT_TUNING_STAGE_POLICIES[DEFAULT_STAGE_BUDGET],
    )
    stage_a_ratio = float(cast(Any, policy["stage_a_ratio"]))
    stage_a_epoch_range = cast(
        tuple[int, int], cast(Any, policy["stage_a_epoch_range"])
    )
    stage_b_epoch_range = cast(
        tuple[int, int], cast(Any, policy["stage_b_epoch_range"])
    )
    stage_a_trials = max(
        1, min(tuning_trials, int(math.ceil(tuning_trials * stage_a_ratio)))
    )
    return {
        "stage_a": {
            "trial_count": stage_a_trials,
            "epoch_range": list(stage_a_epoch_range),
            "search_mode": "broad",
        },
        "stage_b": {
            "trial_count": max(0, tuning_trials - stage_a_trials),
            "epoch_range": list(stage_b_epoch_range),
            "search_mode": "narrowed_top_k",
        },
        "stage_c": {
            "trial_count": 1,
            "search_mode": "final_selected_config",
            "execution": "outside_optuna",
            "stage_budget": stage_budget,
        },
    }


def stage_name_for_trial(
    *, trial_number: int, tuning_trials: int, stage_budget: str
) -> str:
    stage_a_trials = int(
        cast(
            Any,
            resolve_stage_policy(tuning_trials, stage_budget=stage_budget)["stage_a"][
                "trial_count"
            ],
        )
    )
    return "stage_a" if trial_number < stage_a_trials else "stage_b"


def best_completed_trial_params(study: optuna.study.Study) -> dict[str, object] | None:
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        return None
    best_trial = max(completed_trials, key=required_trial_value)
    return dict(best_trial.params)


def sample_optuna_params(
    trial: optuna.trial.Trial,
    random_seed: int,
    *,
    runtime_profile_name: str = DEFAULT_RUNTIME_PROFILE_NAME,
    stage_name: str = "stage_a",
    anchor_params: dict[str, object] | None = None,
    epoch_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    batch_size_choices = _runtime_profile_batch_size_choices(runtime_profile_name)
    if stage_name == "stage_b":
        standard_policy = DEFAULT_TUNING_STAGE_POLICIES[DEFAULT_STAGE_BUDGET]
        resolved_epoch_range = epoch_range or cast(
            tuple[int, int],
            cast(Any, standard_policy["stage_b_epoch_range"]),
        )
        batch_size = int(trial.suggest_categorical("batch_size", batch_size_choices))
        learning_rate_low, learning_rate_high = _batch_scaled_learning_rate_bounds(
            batch_size
        )
        dropout_low, dropout_high = _linear_float_bounds(
            anchor_params.get("dropout") if anchor_params else None,
            low=DEFAULT_TUNING_DROPOUT_LOW,
            high=DEFAULT_TUNING_DROPOUT_HIGH,
            margin=DEFAULT_TUNING_DROPOUT_STAGE_B_MARGIN,
        )
        return {
            "max_epochs": trial.suggest_int("max_epochs", *resolved_epoch_range),
            "batch_size": batch_size,
            "max_encoder_length": trial.suggest_categorical(
                "max_encoder_length", DEFAULT_TUNING_ENCODER_LENGTH_CHOICES
            ),
            "gradient_clip_val": trial.suggest_categorical(
                "gradient_clip_val", DEFAULT_TUNING_GRADIENT_CLIP_CHOICES
            ),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", DEFAULT_TUNING_HIDDEN_SIZE_CHOICES
            ),
            "hidden_continuous_size": trial.suggest_categorical(
                "hidden_continuous_size",
                DEFAULT_TUNING_HIDDEN_CONTINUOUS_SIZE_CHOICES,
            ),
            "attention_head_size": trial.suggest_categorical(
                "attention_head_size",
                DEFAULT_TUNING_ATTENTION_HEAD_SIZE_CHOICES,
            ),
            "lstm_layers": trial.suggest_categorical(
                "lstm_layers", DEFAULT_TUNING_LSTM_LAYER_CHOICES
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                learning_rate_low,
                learning_rate_high,
                log=True,
            ),
            "dropout": trial.suggest_float("dropout", dropout_low, dropout_high),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                DEFAULT_TUNING_WEIGHT_DECAY_LOW,
                DEFAULT_TUNING_WEIGHT_DECAY_HIGH,
                log=True,
            ),
            "use_learning_rate_finder": False,
            "random_state": int(random_seed + trial.number + 1),
        }
    standard_policy = DEFAULT_TUNING_STAGE_POLICIES[DEFAULT_STAGE_BUDGET]
    resolved_epoch_range = epoch_range or cast(
        tuple[int, int],
        cast(Any, standard_policy["stage_a_epoch_range"]),
    )
    batch_size = int(trial.suggest_categorical("batch_size", batch_size_choices))
    learning_rate_low, learning_rate_high = _batch_scaled_learning_rate_bounds(
        batch_size
    )
    return {
        "max_epochs": trial.suggest_int("max_epochs", *resolved_epoch_range),
        "batch_size": batch_size,
        "max_encoder_length": trial.suggest_categorical(
            "max_encoder_length", DEFAULT_TUNING_ENCODER_LENGTH_CHOICES
        ),
        "gradient_clip_val": trial.suggest_categorical(
            "gradient_clip_val", DEFAULT_TUNING_GRADIENT_CLIP_CHOICES
        ),
        "hidden_size": trial.suggest_categorical(
            "hidden_size", DEFAULT_TUNING_HIDDEN_SIZE_CHOICES
        ),
        "hidden_continuous_size": trial.suggest_categorical(
            "hidden_continuous_size",
            DEFAULT_TUNING_HIDDEN_CONTINUOUS_SIZE_CHOICES,
        ),
        "attention_head_size": trial.suggest_categorical(
            "attention_head_size",
            DEFAULT_TUNING_ATTENTION_HEAD_SIZE_CHOICES,
        ),
        "lstm_layers": trial.suggest_categorical(
            "lstm_layers", DEFAULT_TUNING_LSTM_LAYER_CHOICES
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            learning_rate_low,
            learning_rate_high,
            log=True,
        ),
        "dropout": trial.suggest_float(
            "dropout", DEFAULT_TUNING_DROPOUT_LOW, DEFAULT_TUNING_DROPOUT_HIGH
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            DEFAULT_TUNING_WEIGHT_DECAY_LOW,
            DEFAULT_TUNING_WEIGHT_DECAY_HIGH,
            log=True,
        ),
        "use_learning_rate_finder": False,
        "random_state": int(random_seed + trial.number + 1),
    }


def required_trial_value(trial: optuna.trial.FrozenTrial) -> float:
    if trial.value is None:
        raise ValueError(f"Optuna trial {trial.number} has no objective value.")
    return float(trial.value)


def resolved_trial_status_name(
    trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
    *,
    default_status: str | None = None,
) -> str:
    terminal_status = trial.user_attrs.get("terminal_status")
    if isinstance(terminal_status, str) and terminal_status:
        return terminal_status
    trial_state = getattr(trial, "state", None)
    if isinstance(trial_state, optuna.trial.TrialState):
        return trial_state.name
    if isinstance(default_status, str) and default_status:
        return default_status
    return "RUNNING"


def resolve_fold_execution_plan(
    *,
    folds: list[dict[str, object]],
    resolved_params: dict[str, object],
    total_threads: int | None,
    logger: logging.Logger,
) -> dict[str, object]:
    runtime_profile_name = str(
        resolved_params.get("runtime_profile", DEFAULT_RUNTIME_PROFILE_NAME)
    )
    runtime_profile = resolve_runtime_profile(runtime_profile_name)
    accelerator = str(
        resolved_params.get("accelerator", runtime_profile["accelerator"])
    )
    devices = int(cast(Any, resolved_params.get("devices", runtime_profile["devices"])))
    max_parallel_fold_workers = int(
        cast(
            Any,
            resolved_params.get(
                "max_parallel_fold_workers", DEFAULT_MAX_PARALLEL_FOLD_WORKERS
            ),
        )
    )
    num_threads = int(
        total_threads or cast(Any, resolved_params.get("n_jobs", os.cpu_count() or 1))
    )
    gpu_safe_mode = accelerator in {"gpu", "mps"} or runtime_profile_name == "mac_metal"
    if runtime_profile_name == "local_cpu" and should_serialize_local_cpu_folds():
        gpu_safe_mode = True
    fold_workers = (
        1 if gpu_safe_mode else max(1, min(len(folds), max_parallel_fold_workers))
    )
    threads_per_fold = max(1, num_threads // fold_workers)
    logger.info(
        "Optuna fold execution plan: runtime_profile=%s accelerator=%s devices=%s gpu_safe_mode=%s fold_workers=%s threads_per_fold=%s total_threads=%s",
        runtime_profile_name,
        accelerator,
        devices,
        gpu_safe_mode,
        fold_workers,
        threads_per_fold,
        num_threads,
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


def runtime_environment_metadata(
    *, execution_policy: dict[str, object]
) -> dict[str, object]:
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
    best_trial: optuna.trial.FrozenTrial | None = None,
) -> dict[str, object]:
    trial_states = [resolved_trial_status_name(trial) for trial in study.trials]
    status_counts = {
        state: trial_states.count(state) for state in sorted(set(trial_states))
    }
    resolved_best_trial = study.best_trial if best_trial is None else best_trial
    return {
        "execution_policy": execution_policy,
        "stage_policy": resolve_stage_policy(tuning_trials, stage_budget=stage_budget),
        "pruner": dict(DEFAULT_TUNING_PRUNER_CONFIG),
        "system": runtime_environment_metadata(execution_policy=execution_policy),
        "trial_status_counts": status_counts,
        "best_trial_number": int(resolved_best_trial.number),
        "best_improvement_pct": (
            required_trial_value(resolved_best_trial)
            if resolved_best_trial.value is not None
            else None
        ),
        "best_mean_wape": float(resolved_best_trial.user_attrs["mean_wape"]),
        "best_mean_abs_bias": resolved_best_trial.user_attrs.get("mean_abs_bias"),
        "best_coverage_80": resolved_best_trial.user_attrs.get("mean_coverage_80"),
        "best_coverage_95": resolved_best_trial.user_attrs.get("mean_coverage_95"),
    }


def create_optuna_study(*, random_seed: int) -> optuna.study.Study:
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(
        seed=random_seed,
        multivariate=True,
        n_startup_trials=DEFAULT_TUNING_SAMPLER_STARTUP_TRIALS,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(
            cast(Any, DEFAULT_TUNING_PRUNER_CONFIG["n_startup_trials"])
        ),
        n_warmup_steps=int(cast(Any, DEFAULT_TUNING_PRUNER_CONFIG["n_warmup_steps"])),
        interval_steps=int(cast(Any, DEFAULT_TUNING_PRUNER_CONFIG["interval_steps"])),
    )
    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
