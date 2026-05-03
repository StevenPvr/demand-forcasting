from __future__ import annotations

import math
import os
import socket
from typing import Any, cast
import warnings

import optuna

from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_STAGE_BUDGET,
    DEFAULT_TUNING_PRUNER_CONFIG,
    DEFAULT_TUNING_SAMPLER_STARTUP_TRIALS,
    DEFAULT_TUNING_STAGE_POLICIES,
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


def required_trial_value(trial: optuna.trial.FrozenTrial) -> float:
    if trial.value is None:
        raise ValueError(f"Optuna trial {trial.number} has no objective value.")
    return float(trial.value)


def create_optuna_study(*, random_seed: int) -> optuna.study.Study:
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=optuna.exceptions.ExperimentalWarning,
            message="Argument ``multivariate`` is an experimental feature.*",
        )
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


def runtime_environment_metadata(
    *, execution_policy: dict[str, object]
) -> dict[str, object]:
    return {
        "host_name": socket.gethostname(),
        "instance_type": os.getenv("PRAEDIXA_INSTANCE_TYPE"),
        "gpu_name": None,
        "cuda_version": None,
        "device_count": None,
        "runtime_profile": execution_policy["runtime_profile"],
    }


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
        "best_improvement_pct": resolved_best_trial.user_attrs.get(
            "baseline_wape_improvement_pct"
        ),
        "best_objective_score": required_trial_value(resolved_best_trial)
        if resolved_best_trial.value is not None
        else None,
        "best_mean_wape": float(resolved_best_trial.user_attrs["mean_wape"]),
        "best_mean_abs_bias": resolved_best_trial.user_attrs.get("mean_abs_bias"),
        "best_decision_loss": resolved_best_trial.user_attrs.get(
            "mean_decision_loss"
        ),
        "best_normalized_economic_loss": resolved_best_trial.user_attrs.get(
            "mean_normalized_economic_loss"
        ),
        "best_normalized_bias": resolved_best_trial.user_attrs.get(
            "mean_normalized_bias"
        ),
        "best_positive_bias_penalty": resolved_best_trial.user_attrs.get(
            "mean_positive_bias_penalty"
        ),
        "best_severe_negative_bias_penalty": resolved_best_trial.user_attrs.get(
            "mean_severe_negative_bias_penalty"
        ),
        "validation_monitor_metric": resolved_best_trial.user_attrs.get(
            "validation_monitor_metric"
        ),
        "economic_objective_config": resolved_best_trial.user_attrs.get(
            "economic_objective_config"
        ),
        "segmented_economic_objective_config": resolved_best_trial.user_attrs.get(
            "segmented_economic_objective_config",
            resolved_best_trial.user_attrs.get("economic_objective_config"),
        ),
        "segment_decision_loss": resolved_best_trial.user_attrs.get(
            "segment_decision_loss"
        ),
        "segment_bias": resolved_best_trial.user_attrs.get("segment_bias"),
        "product_decision_loss": resolved_best_trial.user_attrs.get(
            "product_decision_loss"
        ),
        "product_bias": resolved_best_trial.user_attrs.get("product_bias"),
        "best_coverage_80": resolved_best_trial.user_attrs.get("mean_coverage_80"),
        "best_coverage_95": resolved_best_trial.user_attrs.get("mean_coverage_95"),
    }
