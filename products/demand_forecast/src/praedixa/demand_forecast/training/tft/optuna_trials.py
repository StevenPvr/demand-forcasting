from __future__ import annotations

import logging
from typing import Any, cast

import optuna
import pandas as pd

from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_TUNING_GUARDRAIL_COVERAGE_80_BOUNDS,
    DEFAULT_TUNING_GUARDRAIL_COVERAGE_95_BOUNDS,
    DEFAULT_TUNING_GUARDRAIL_DATASET_WAPE_COLLAPSE_MULTIPLIER,
    DEFAULT_TUNING_GUARDRAIL_MIN_ABS_BIAS_LIMIT,
)
from praedixa.demand_forecast.training.tft.policy import resolved_trial_status_name


def log_trial_start(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    trial_params: dict[str, object],
) -> None:
    logger.info(
        "[HPO trial %s/%s | %s] started with params=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        _serializable_trial_params(trial_params),
    )


def log_trial_completion(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    mean_wape: float,
    improvement_pct: float,
    folds_completed: int,
    dataset_mean_wape: dict[str, float],
    mean_abs_bias: float | None,
    mean_coverage_80: float | None,
    mean_coverage_95: float | None,
) -> None:
    logger.info(
        "[HPO trial %s/%s | %s] completed business_mean_wape=%.6f improvement_pct=%.6f business_mean_abs_bias=%s business_mean_coverage_80=%s business_mean_coverage_95=%s folds_completed=%s dataset_business_mean_wape=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        mean_wape,
        improvement_pct,
        "nan" if mean_abs_bias is None else f"{mean_abs_bias:.6f}",
        "nan" if mean_coverage_80 is None else f"{mean_coverage_80:.6f}",
        "nan" if mean_coverage_95 is None else f"{mean_coverage_95:.6f}",
        folds_completed,
        dataset_mean_wape,
    )


def log_trial_failure(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    error: Exception,
) -> None:
    logger.warning(
        "[HPO trial %s/%s | %s] failed error=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        error,
    )


def log_trial_pruned(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    error: Exception,
) -> None:
    logger.info(
        "[HPO trial %s/%s | %s] pruned reason=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        error,
    )


def log_trial_rejected(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    error: Exception,
) -> None:
    logger.info(
        "[HPO trial %s/%s | %s] rejected reason=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        error,
    )


def tuning_result_summary(
    tuning_result: dict[str, object],
) -> tuple[float, dict[str, float], int, float | None, float | None, float | None]:
    return (
        float(cast(Any, tuning_result["macro_mean_wape"])),
        cast(dict[str, float], tuning_result["dataset_mean_wape"]),
        int(cast(Any, tuning_result["folds_completed"])),
        float(cast(Any, tuning_result["mean_abs_bias"]))
        if tuning_result.get("mean_abs_bias") is not None
        else None,
        float(cast(Any, tuning_result["mean_coverage_80"]))
        if tuning_result.get("mean_coverage_80") is not None
        else None,
        float(cast(Any, tuning_result["mean_coverage_95"]))
        if tuning_result.get("mean_coverage_95") is not None
        else None,
    )


def objective_score(tuning_result: dict[str, object]) -> float:
    macro_mean_wape = float(cast(Any, tuning_result["macro_mean_wape"]))
    coverage_penalty = _coverage_penalty(
        tuning_result.get("mean_coverage_80"),
        target=0.80,
    ) + _coverage_penalty(
        tuning_result.get("mean_coverage_95"),
        target=0.95,
    )
    pinball_penalty = (
        float(cast(Any, tuning_result["mean_pinball_loss"]))
        if tuning_result.get("mean_pinball_loss") is not None
        else 0.0
    )
    return float(-macro_mean_wape - 0.10 * coverage_penalty - 0.05 * pinball_penalty)


def guardrail_failure_reason(
    *,
    tuning_result: dict[str, object],
    baseline_wape: float,
    baseline_dataset_wape: dict[str, float],
) -> str | None:
    dataset_mean_wape = cast(dict[str, float], tuning_result["dataset_mean_wape"])
    collapsed = [
        dataset_source
        for dataset_source, score in dataset_mean_wape.items()
        if float(score)
        > (
            baseline_dataset_wape.get(dataset_source, baseline_wape)
            * DEFAULT_TUNING_GUARDRAIL_DATASET_WAPE_COLLAPSE_MULTIPLIER
        )
    ]
    if collapsed:
        return f"dataset_wape_collapse:{','.join(collapsed)}"
    mean_abs_bias = tuning_result.get("mean_abs_bias")
    if isinstance(mean_abs_bias, (int, float)) and float(mean_abs_bias) > max(
        DEFAULT_TUNING_GUARDRAIL_MIN_ABS_BIAS_LIMIT,
        baseline_wape,
    ):
        return "absolute_bias_too_high"
    coverage_80 = tuning_result.get("mean_coverage_80")
    if isinstance(coverage_80, (int, float)) and not (
        DEFAULT_TUNING_GUARDRAIL_COVERAGE_80_BOUNDS[0]
        <= float(coverage_80)
        <= DEFAULT_TUNING_GUARDRAIL_COVERAGE_80_BOUNDS[1]
    ):
        return "coverage_80_out_of_bounds"
    coverage_95 = tuning_result.get("mean_coverage_95")
    if isinstance(coverage_95, (int, float)) and not (
        DEFAULT_TUNING_GUARDRAIL_COVERAGE_95_BOUNDS[0]
        <= float(coverage_95)
        <= DEFAULT_TUNING_GUARDRAIL_COVERAGE_95_BOUNDS[1]
    ):
        return "coverage_95_out_of_bounds"
    return None


def build_tuning_report(study: optuna.study.Study) -> pd.DataFrame:
    rows = [_tuning_report_row(trial) for trial in study.trials]
    return (
        pd.DataFrame(rows)
        .sort_values(by="objective_score", ascending=False)
        .reset_index(drop=True)
    )


def _serializable_trial_params(trial_params: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in trial_params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
    return serialized


def _coverage_penalty(value: object, *, target: float) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return abs(float(value) - target)


def _trial_duration_seconds(trial: optuna.trial.FrozenTrial) -> float:
    if trial.duration is None:
        return float("nan")
    return float(trial.duration.total_seconds())


def _trial_user_attr_float(trial: optuna.trial.FrozenTrial, key: str) -> float:
    value = trial.user_attrs.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _trial_epochs_completed(trial: optuna.trial.FrozenTrial) -> int:
    return int(
        trial.user_attrs.get("epochs_completed", trial.params.get("max_epochs", 0))
    )


def _tuning_report_row(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
    return {
        "trial": trial.number,
        "stage_name": str(trial.user_attrs.get("stage_name", "unknown")),
        "trial_status": resolved_trial_status_name(trial),
        "trial_duration_seconds": _trial_duration_seconds(trial),
        "epochs_completed": _trial_epochs_completed(trial),
        "folds_completed": int(trial.user_attrs.get("folds_completed", 0)),
        "mean_wape": _trial_user_attr_float(trial, "mean_wape"),
        "mean_abs_bias": _trial_user_attr_float(trial, "mean_abs_bias"),
        "coverage_80": _trial_user_attr_float(trial, "mean_coverage_80"),
        "coverage_95": _trial_user_attr_float(trial, "mean_coverage_95"),
        "objective_score": _trial_user_attr_float(trial, "objective_score"),
        "selected_learning_rate": _trial_user_attr_float(
            trial, "selected_learning_rate"
        ),
        "failure_reason": str(trial.user_attrs.get("failure_reason", "")),
        "baseline_wape_improvement_pct": _trial_user_attr_float(
            trial,
            "baseline_wape_improvement_pct",
        ),
        "fold_wape_scores": trial.user_attrs.get("fold_wape_scores", []),
        "dataset_mean_wape": trial.user_attrs.get("dataset_mean_wape", {}),
        **trial.params,
    }


__all__ = [
    "build_tuning_report",
    "guardrail_failure_reason",
    "log_trial_completion",
    "log_trial_failure",
    "log_trial_pruned",
    "log_trial_rejected",
    "log_trial_start",
    "objective_score",
    "tuning_result_summary",
]

