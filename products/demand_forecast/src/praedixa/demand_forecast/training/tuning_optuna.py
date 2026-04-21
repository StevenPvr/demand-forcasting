from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, cast

import optuna
import pandas as pd

from praedixa.demand_forecast.backends.tft.model_utils import DEFAULT_TFT_MODEL_PARAMS
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.feature_screening.pipeline import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_TUNING_PROGRESS_LOG_EVERY,
    DEFAULT_TUNING_TRIALS,
    compute_wape_improvement_pct,
)
from praedixa.demand_forecast.training.tuning_policy import (
    best_completed_trial_params,
    build_hpo_runtime_metadata,
    create_optuna_study,
    required_trial_value,
    resolve_fold_execution_plan,
    resolve_stage_policy,
    sample_optuna_params,
    stage_name_for_trial,
)

ScoreFn = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class ObjectiveContext:
    study: optuna.study.Study
    score_fn: ScoreFn
    train_frame: pd.DataFrame
    tuning_frame: pd.DataFrame
    folds: list[dict[str, object]]
    feature_cols: list[str]
    baseline_wape: float
    target_contract: TargetContract
    logger: logging.Logger
    tuning_trials: int
    random_seed: int
    model_params: dict[str, object] | None
    total_threads: int
    target_transform: str
    stage_budget: str
    stage_policy: dict[str, dict[str, object]]


def _serializable_trial_params(trial_params: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in trial_params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
    return serialized


def _log_trial_start(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    trial_params: dict[str, object],
) -> None:
    logger.info(
        "Optuna trial started: trial=%s/%s stage=%s params=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        _serializable_trial_params(trial_params),
    )


def _log_trial_completion(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    mean_wape: float,
    improvement_pct: float,
    folds_completed: int,
    dataset_mean_wape: dict[str, float],
) -> None:
    logger.info(
        "Optuna trial completed: trial=%s/%s stage=%s mean_wape=%.6f improvement_pct=%.6f folds_completed=%s dataset_mean_wape=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        mean_wape,
        improvement_pct,
        folds_completed,
        dataset_mean_wape,
    )


def _log_trial_failure(
    *,
    logger: logging.Logger,
    trial_number: int,
    tuning_trials: int,
    stage_name: str,
    error: Exception,
) -> None:
    logger.warning(
        "Optuna trial failed: trial=%s/%s stage=%s error=%s",
        trial_number + 1,
        tuning_trials,
        stage_name,
        error,
    )


def _tuning_result_summary(tuning_result: dict[str, object]) -> tuple[float, dict[str, float], int]:
    return (
        float(cast(Any, tuning_result["macro_mean_wape"])),
        cast(dict[str, float], tuning_result["dataset_mean_wape"]),
        int(cast(Any, tuning_result["folds_completed"])),
    )


def _build_trial_params(
    *,
    trial: optuna.trial.Trial,
    study: optuna.study.Study,
    stage_name: str,
    random_seed: int,
    model_params: dict[str, object] | None,
    stage_policy: dict[str, dict[str, object]],
) -> dict[str, object]:
    anchor_params = best_completed_trial_params(study) if stage_name == "stage_b" else None
    stage_config = stage_policy[stage_name]
    return {
        **(model_params or {}),
        **sample_optuna_params(
            trial=trial,
            random_seed=random_seed,
            stage_name=stage_name,
            anchor_params=anchor_params,
            epoch_range=cast(tuple[int, int], tuple(cast(list[int], stage_config["epoch_range"]))),
        ),
    }


def _log_optuna_progress(
    *,
    study: optuna.study.Study,
    logger: logging.Logger,
    completed: int,
    tuning_trials: int,
    stage_name: str,
    improvement_pct: float,
) -> None:
    if completed == 1 or completed % DEFAULT_TUNING_PROGRESS_LOG_EVERY == 0 or completed == tuning_trials:
        logger.info(
            "Final optimisation progress: trial=%s/%s stage=%s current_best_improvement_pct=%.6f",
            completed,
            tuning_trials,
            stage_name,
            max((trial.value for trial in study.trials if trial.value is not None), default=improvement_pct),
        )


def build_objective(*, context: ObjectiveContext) -> Any:
    trial_counter = {"completed": 0}

    def objective(trial: optuna.trial.Trial) -> float:
        stage_name = stage_name_for_trial(
            trial_number=trial.number,
            tuning_trials=context.tuning_trials,
            stage_budget=context.stage_budget,
        )
        trial.set_user_attr("stage_name", stage_name)
        trial_params = _build_trial_params(
            trial=trial,
            study=context.study,
            stage_name=stage_name,
            random_seed=context.random_seed,
            model_params=context.model_params,
            stage_policy=context.stage_policy,
        )
        trial.set_user_attr("epochs_completed", int(cast(Any, trial_params["max_epochs"])))
        _log_trial_start(
            logger=context.logger,
            trial_number=trial.number,
            tuning_trials=context.tuning_trials,
            stage_name=stage_name,
            trial_params=trial_params,
        )
        try:
            tuning_result = context.score_fn(
                train_frame=context.train_frame,
                tuning_frame=context.tuning_frame,
                folds=context.folds,
                feature_cols=context.feature_cols,
                target_contract=context.target_contract,
                logger=context.logger,
                model_params=trial_params,
                total_threads=context.total_threads,
                target_transform=context.target_transform,
                trial=trial,
            )
            mean_wape, dataset_mean_wape, folds_completed = _tuning_result_summary(tuning_result)
            improvement_pct = compute_wape_improvement_pct(context.baseline_wape, mean_wape)
            trial.set_user_attr("mean_wape", mean_wape)
            trial.set_user_attr("dataset_mean_wape", dataset_mean_wape)
            trial.set_user_attr("fold_wape_scores", cast(Any, tuning_result["fold_results"]))
            trial.set_user_attr("folds_completed", folds_completed)
            trial_counter["completed"] += 1
            _log_trial_completion(
                logger=context.logger,
                trial_number=trial.number,
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                mean_wape=mean_wape,
                improvement_pct=improvement_pct,
                folds_completed=folds_completed,
                dataset_mean_wape=dataset_mean_wape,
            )
            _log_optuna_progress(
                study=context.study,
                logger=context.logger,
                completed=trial_counter["completed"],
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                improvement_pct=improvement_pct,
            )
            return improvement_pct
        except optuna.TrialPruned as exc:
            _log_trial_failure(
                logger=context.logger,
                trial_number=trial.number,
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                error=exc,
            )
            raise
        except Exception as exc:
            _log_trial_failure(
                logger=context.logger,
                trial_number=trial.number,
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                error=exc,
            )
            raise

    return objective


def _build_tuning_report(study: optuna.study.Study) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        rows.append(
            {
                "trial": trial.number,
                "stage_name": str(trial.user_attrs.get("stage_name", "unknown")),
                "trial_status": trial.state.name,
                "trial_duration_seconds": float(trial.duration.total_seconds()) if trial.duration else float("nan"),
                "epochs_completed": int(trial.user_attrs.get("epochs_completed", trial.params.get("max_epochs", 0))),
                "folds_completed": int(trial.user_attrs.get("folds_completed", 0)),
                "mean_wape": float(trial.user_attrs["mean_wape"]) if "mean_wape" in trial.user_attrs else float("nan"),
                "baseline_wape_improvement_pct": float(trial.value) if trial.value is not None else float("nan"),
                "fold_wape_scores": trial.user_attrs.get("fold_wape_scores", []),
                "dataset_mean_wape": trial.user_attrs.get("dataset_mean_wape", {}),
                **trial.params,
            }
        )
    return pd.DataFrame(rows).sort_values(by="baseline_wape_improvement_pct", ascending=False).reset_index(drop=True)


def _finalize_best_params(
    *,
    study: optuna.study.Study,
    model_params: dict[str, object] | None,
    random_seed: int,
    total_threads: int,
) -> dict[str, object]:
    best_params = {**(model_params or {}), **study.best_params}
    best_params["random_state"] = random_seed + study.best_trial.number + 1
    best_params["n_jobs"] = total_threads
    best_params.pop("max_parallel_fold_workers", None)
    return best_params


def _optuna_search_context(
    *,
    folds: list[dict[str, object]],
    logger: logging.Logger,
    tuning_trials: int,
    feature_cols: list[str],
    baseline_wape: float,
    model_params: dict[str, object] | None,
) -> tuple[int, str, dict[str, dict[str, object]], dict[str, object]]:
    total_threads = int(cast(Any, (model_params or {}).get("n_jobs", os.cpu_count() or 1)))
    stage_budget = str((model_params or {}).get("stage_budget", "standard"))
    execution_policy = resolve_fold_execution_plan(
        folds=folds,
        resolved_params={**DEFAULT_TFT_MODEL_PARAMS, **(model_params or {})},
        total_threads=total_threads,
        logger=logger,
    )
    logger.info(
        "Starting final TFT optimisation with Optuna: trials=%s feature_count=%s baseline_wape=%.6f",
        tuning_trials,
        len(feature_cols),
        baseline_wape,
    )
    return total_threads, stage_budget, resolve_stage_policy(tuning_trials, stage_budget=stage_budget), execution_policy


def _objective_for_search(*, context: ObjectiveContext) -> Any:
    return build_objective(context=context)


def _finalize_search_outputs(
    *,
    study: optuna.study.Study,
    execution_policy: dict[str, object],
    tuning_trials: int,
    stage_budget: str,
    model_params: dict[str, object] | None,
    random_seed: int,
    total_threads: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    return (
        _finalize_best_params(study=study, model_params=model_params, random_seed=random_seed, total_threads=total_threads),
        _build_tuning_report(study),
        build_hpo_runtime_metadata(
            study=study,
            execution_policy=execution_policy,
            tuning_trials=tuning_trials,
            stage_budget=stage_budget,
        ),
    )


def run_optuna_search(
    *, score_fn: ScoreFn, train_frame: pd.DataFrame, tuning_frame: pd.DataFrame, folds: list[dict[str, object]],
    feature_cols: list[str], baseline_wape: float, target_contract: TargetContract, logger: logging.Logger,
    tuning_trials: int = DEFAULT_TUNING_TRIALS, random_seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, object] | None = None, target_transform: str,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    total_threads, stage_budget, stage_policy, execution_policy = _optuna_search_context(
        folds=folds,
        logger=logger,
        tuning_trials=tuning_trials,
        feature_cols=feature_cols,
        baseline_wape=baseline_wape,
        model_params=model_params,
    )
    study = create_optuna_study(random_seed=random_seed)
    objective_context = ObjectiveContext(
        study=study,
        score_fn=score_fn,
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        baseline_wape=baseline_wape,
        target_contract=target_contract,
        logger=logger,
        tuning_trials=tuning_trials,
        random_seed=random_seed,
        model_params=model_params,
        total_threads=total_threads,
        target_transform=target_transform,
        stage_budget=stage_budget,
        stage_policy=stage_policy,
    )
    study.optimize(_objective_for_search(context=objective_context), n_trials=tuning_trials, n_jobs=1)
    logger.info(
        "Final TFT optimisation complete: best_trial=%s best_wape=%.6f best_improvement_pct=%.6f",
        study.best_trial.number,
        float(study.best_trial.user_attrs["mean_wape"]),
        required_trial_value(study.best_trial),
    )
    return _finalize_search_outputs(
        study=study,
        execution_policy=execution_policy,
        tuning_trials=tuning_trials,
        stage_budget=stage_budget,
        model_params=model_params,
        random_seed=random_seed,
        total_threads=total_threads,
    )
