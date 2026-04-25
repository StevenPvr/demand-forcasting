from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, cast

import optuna
import pandas as pd

from praedixa.demand_forecast.backends.tft.model_utils import DEFAULT_TFT_MODEL_PARAMS
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.training.shared.metrics import (
    compute_wape_improvement_pct,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_TUNING_PROGRESS_LOG_EVERY,
    DEFAULT_TUNING_RANDOM_SEED,
    DEFAULT_TUNING_TRIALS,
)
from praedixa.demand_forecast.training.tft.policy import (
    best_completed_trial_params,
    build_hpo_runtime_metadata,
    create_optuna_study,
    resolved_trial_status_name,
    resolve_fold_execution_plan,
    resolve_stage_policy,
    sample_optuna_params,
    stage_name_for_trial,
)
from praedixa.demand_forecast.training.tft.optuna_trials import (
    build_tuning_report,
    guardrail_failure_reason,
    log_trial_completion,
    log_trial_failure,
    log_trial_pruned,
    log_trial_rejected,
    log_trial_start,
    objective_score as compute_objective_score,
    tuning_result_summary,
)

ScoreFn = Callable[..., dict[str, object]]
PrewarmFn = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class ObjectiveContext:
    study: optuna.study.Study
    score_fn: ScoreFn
    train_frame: pd.DataFrame
    tuning_frame: pd.DataFrame
    folds: list[dict[str, object]]
    feature_cols: list[str]
    baseline_wape: float
    baseline_dataset_wape: dict[str, float]
    target_contract: TargetContract
    logger: logging.Logger
    tuning_trials: int
    random_seed: int
    model_params: dict[str, object] | None
    total_threads: int
    target_transform: str
    stage_budget: str
    stage_policy: dict[str, dict[str, object]]


def _build_trial_params(
    *,
    trial: optuna.trial.Trial,
    study: optuna.study.Study,
    stage_name: str,
    random_seed: int,
    model_params: dict[str, object] | None,
    stage_policy: dict[str, dict[str, object]],
) -> dict[str, object]:
    anchor_params = (
        best_completed_trial_params(study) if stage_name == "stage_b" else None
    )
    stage_config = stage_policy[stage_name]
    runtime_profile_name = str(
        (model_params or {}).get(
            "runtime_profile", DEFAULT_TFT_MODEL_PARAMS["runtime_profile"]
        )
    )
    return {
        **(model_params or {}),
        **sample_optuna_params(
            trial=trial,
            random_seed=random_seed,
            runtime_profile_name=runtime_profile_name,
            stage_name=stage_name,
            anchor_params=anchor_params,
            epoch_range=cast(
                tuple[int, int], tuple(cast(list[int], stage_config["epoch_range"]))
            ),
        ),
    }


def _log_optuna_progress(
    *,
    study: optuna.study.Study,
    logger: logging.Logger,
    completed: int,
    tuning_trials: int,
    stage_name: str,
    objective_score: float,
) -> None:
    if (
        completed == 1
        or completed % DEFAULT_TUNING_PROGRESS_LOG_EVERY == 0
        or completed == tuning_trials
    ):
        logger.info(
            "Final optimisation progress: trial=%s/%s stage=%s current_bestobjective_score=%.6f",
            completed,
            tuning_trials,
            stage_name,
            max(
                (trial.value for trial in study.trials if trial.value is not None),
                default=objective_score,
            ),
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
        trial.set_user_attr(
            "epochs_completed", int(cast(Any, trial_params["max_epochs"]))
        )
        log_trial_start(
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
            (
                mean_wape,
                dataset_mean_wape,
                folds_completed,
                mean_abs_bias,
                mean_coverage_80,
                mean_coverage_95,
            ) = tuning_result_summary(tuning_result)
            improvement_pct = compute_wape_improvement_pct(
                context.baseline_wape, mean_wape
            )
            trial.set_user_attr("mean_wape", mean_wape)
            trial.set_user_attr("dataset_mean_wape", dataset_mean_wape)
            trial.set_user_attr("mean_abs_bias", tuning_result.get("mean_abs_bias"))
            trial.set_user_attr(
                "mean_coverage_80", tuning_result.get("mean_coverage_80")
            )
            trial.set_user_attr(
                "mean_coverage_95", tuning_result.get("mean_coverage_95")
            )
            trial.set_user_attr(
                "selected_learning_rate",
                tuning_result.get("selected_learning_rate"),
            )
            score = compute_objective_score(tuning_result)
            trial.set_user_attr("objective_score", score)
            trial.set_user_attr("baseline_wape_improvement_pct", improvement_pct)
            trial.set_user_attr(
                "fold_wape_scores", cast(Any, tuning_result["fold_results"])
            )
            trial.set_user_attr("folds_completed", folds_completed)
            failure_reason = guardrail_failure_reason(
                tuning_result=tuning_result,
                baseline_wape=context.baseline_wape,
                baseline_dataset_wape=context.baseline_dataset_wape,
            )
            if failure_reason is not None:
                trial.set_user_attr("failure_reason", failure_reason)
                trial.set_user_attr("terminal_status", "REJECTED")
                raise optuna.TrialPruned(f"Guardrail rejected trial: {failure_reason}")
            trial_counter["completed"] += 1
            log_trial_completion(
                logger=context.logger,
                trial_number=trial.number,
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                mean_wape=mean_wape,
                improvement_pct=improvement_pct,
                folds_completed=folds_completed,
                dataset_mean_wape=dataset_mean_wape,
                mean_abs_bias=mean_abs_bias,
                mean_coverage_80=mean_coverage_80,
                mean_coverage_95=mean_coverage_95,
            )
            _log_optuna_progress(
                study=context.study,
                logger=context.logger,
                completed=trial_counter["completed"],
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                objective_score=score,
            )
            return score
        except optuna.TrialPruned as exc:
            if "failure_reason" not in trial.user_attrs:
                trial.set_user_attr("failure_reason", str(exc))
            trial_status = resolved_trial_status_name(trial, default_status="PRUNED")
            if trial_status == "REJECTED":
                log_trial_rejected(
                    logger=context.logger,
                    trial_number=trial.number,
                    tuning_trials=context.tuning_trials,
                    stage_name=stage_name,
                    error=exc,
                )
            else:
                log_trial_pruned(
                    logger=context.logger,
                    trial_number=trial.number,
                    tuning_trials=context.tuning_trials,
                    stage_name=stage_name,
                    error=exc,
                )
            raise
        except Exception as exc:
            trial.set_user_attr("failure_reason", str(exc))
            log_trial_failure(
                logger=context.logger,
                trial_number=trial.number,
                tuning_trials=context.tuning_trials,
                stage_name=stage_name,
                error=exc,
            )
            raise

    return objective

def _finalize_best_params(
    *,
    best_trial: optuna.trial.FrozenTrial,
    model_params: dict[str, object] | None,
    random_seed: int,
    total_threads: int,
) -> dict[str, object]:
    best_params = {**(model_params or {}), **best_trial.params}
    selected_learning_rate = best_trial.user_attrs.get("selected_learning_rate")
    if isinstance(selected_learning_rate, (int, float)):
        best_params["learning_rate"] = float(selected_learning_rate)
        best_params["use_learning_rate_finder"] = False
    best_params["random_state"] = random_seed + best_trial.number + 1
    best_params["n_jobs"] = total_threads
    best_params.pop("max_parallel_fold_workers", None)
    return best_params


def _optuna_search_context(
    *,
    folds: list[dict[str, object]],
    logger: logging.Logger,
    tuning_trials: int,
    model_params: dict[str, object] | None,
) -> tuple[int, str, dict[str, dict[str, object]], dict[str, object]]:
    total_threads = int(
        cast(Any, (model_params or {}).get("n_jobs", os.cpu_count() or 1))
    )
    stage_budget = str((model_params or {}).get("stage_budget", "standard"))
    execution_policy = resolve_fold_execution_plan(
        folds=folds,
        resolved_params={**DEFAULT_TFT_MODEL_PARAMS, **(model_params or {})},
        total_threads=total_threads,
        logger=logger,
    )
    return (
        total_threads,
        stage_budget,
        resolve_stage_policy(tuning_trials, stage_budget=stage_budget),
        execution_policy,
    )


def _objective_for_search(*, context: ObjectiveContext) -> Any:
    return build_objective(context=context)


def _finalize_search_outputs(
    *,
    study: optuna.study.Study,
    best_trial: optuna.trial.FrozenTrial,
    execution_policy: dict[str, object],
    tuning_trials: int,
    stage_budget: str,
    model_params: dict[str, object] | None,
    random_seed: int,
    total_threads: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    return (
        _finalize_best_params(
            best_trial=best_trial,
            model_params=model_params,
            random_seed=random_seed,
            total_threads=total_threads,
        ),
        build_tuning_report(study),
        build_hpo_runtime_metadata(
            study=study,
            execution_policy=execution_policy,
            tuning_trials=tuning_trials,
            stage_budget=stage_budget,
            best_trial=best_trial,
        ),
    )


def _resolve_best_trial(
    *,
    study: optuna.study.Study,
    logger: logging.Logger,
) -> optuna.trial.FrozenTrial:
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if completed_trials:
        return study.best_trial
    failure_reasons = [
        str(trial.user_attrs.get("failure_reason", trial.state.name))
        for trial in study.trials
    ]
    logger.error(
        "Optuna search completed without guardrail-approved trials; refusing to promote rejected/pruned trials. reasons=%s",
        failure_reasons,
    )
    raise RuntimeError(
        "Optuna search completed without guardrail-approved trials. "
        f"Failure reasons: {failure_reasons}"
    )


def run_optuna_search(
    *,
    score_fn: ScoreFn,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    baseline_wape: float,
    baseline_dataset_wape: dict[str, float] | None = None,
    target_contract: TargetContract,
    logger: logging.Logger,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_TUNING_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    target_transform: str,
    prewarm_fn: PrewarmFn | None = None,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    total_threads, stage_budget, stage_policy, execution_policy = (
        _optuna_search_context(
            folds=folds,
            logger=logger,
            tuning_trials=tuning_trials,
            model_params=model_params,
        )
    )
    if prewarm_fn is not None:
        prewarm_fn(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=feature_cols,
            target_contract=target_contract,
            logger=logger,
            model_params=model_params,
            total_threads=total_threads,
            target_transform=target_transform,
        )
    logger.info(
        "Starting final TFT optimisation with Optuna: trials=%s feature_count=%s baseline_wape=%.6f",
        tuning_trials,
        len(feature_cols),
        baseline_wape,
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
        baseline_dataset_wape=baseline_dataset_wape or {},
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
    study.optimize(
        _objective_for_search(context=objective_context),
        n_trials=tuning_trials,
        n_jobs=1,
        catch=(RuntimeError,),
    )
    best_trial = _resolve_best_trial(study=study, logger=logger)
    logger.info(
        "Final TFT optimisation complete: best_trial=%s best_wape=%.6f best_improvement_pct=%.6f",
        best_trial.number,
        float(best_trial.user_attrs["mean_wape"]),
        compute_wape_improvement_pct(
            baseline_wape,
            float(best_trial.user_attrs["mean_wape"]),
        ),
    )
    return _finalize_search_outputs(
        study=study,
        best_trial=best_trial,
        execution_policy=execution_policy,
        tuning_trials=tuning_trials,
        stage_budget=stage_budget,
        model_params=model_params,
        random_seed=random_seed,
        total_threads=total_threads,
    )
