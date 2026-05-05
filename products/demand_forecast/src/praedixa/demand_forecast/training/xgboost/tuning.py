from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, cast

import optuna
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from praedixa.demand_forecast.backends.xgboost.model_common import (
    resolve_xgboost_model_params,
)
from praedixa.demand_forecast.backends.xgboost.model_fit import (
    fit_xgboost_model_and_predict_validation,
)
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    reconstruct_absolute_predictions,
)
from praedixa.demand_forecast.training.shared.metrics import (
    compute_wape_improvement_pct,
)
from praedixa.demand_forecast.training.shared.economic_objective import (
    economic_objective_config_payload,
    fit_economic_decision_calibration,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_TARGET_TRANSFORM,
    DEFAULT_TUNING_GUARDRAIL_DATASET_WAPE_COLLAPSE_MULTIPLIER,
    DEFAULT_TUNING_GUARDRAIL_MIN_ABS_BIAS_LIMIT,
    DEFAULT_TUNING_PROGRESS_LOG_EVERY,
    DEFAULT_TUNING_RANDOM_SEED,
    DEFAULT_TUNING_TRIALS,
    DEFAULT_XGBOOST_TUNING_COLSAMPLE_BYTREE_RANGE,
    DEFAULT_XGBOOST_TUNING_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGBOOST_TUNING_LEARNING_RATE_RANGE,
    DEFAULT_XGBOOST_TUNING_MAX_BIN_CHOICES,
    DEFAULT_XGBOOST_TUNING_MAX_CAT_THRESHOLD_CHOICES,
    DEFAULT_XGBOOST_TUNING_MAX_CAT_TO_ONEHOT_CHOICES,
    DEFAULT_XGBOOST_TUNING_MAX_DEPTH_CHOICES,
    DEFAULT_XGBOOST_TUNING_MIN_CHILD_WEIGHT_RANGE,
    DEFAULT_XGBOOST_TUNING_REG_ALPHA_RANGE,
    DEFAULT_XGBOOST_TUNING_REG_LAMBDA_RANGE,
    DEFAULT_XGBOOST_TUNING_SUBSAMPLE_RANGE,
)
from praedixa.demand_forecast.training.shared.hpo import (
    build_hpo_runtime_metadata,
    create_optuna_study,
    resolved_trial_status_name,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    filter_training_eligible_rows,
)
from praedixa.demand_forecast.training.xgboost.scoring import (
    XGBoostFoldMatrixCache,
    build_xgboost_fold_matrix_cache,
    fit_and_score_xgboost_model_on_tuning,
    xgboost_execution_policy,
)


def sample_xgboost_optuna_params(
    trial: optuna.trial.Trial,
    random_seed: int = DEFAULT_TUNING_RANDOM_SEED,
) -> dict[str, object]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", *DEFAULT_XGBOOST_TUNING_LEARNING_RATE_RANGE, log=True
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", DEFAULT_XGBOOST_TUNING_MAX_DEPTH_CHOICES
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            *DEFAULT_XGBOOST_TUNING_MIN_CHILD_WEIGHT_RANGE,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", *DEFAULT_XGBOOST_TUNING_SUBSAMPLE_RANGE
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *DEFAULT_XGBOOST_TUNING_COLSAMPLE_BYTREE_RANGE
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *DEFAULT_XGBOOST_TUNING_REG_ALPHA_RANGE, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *DEFAULT_XGBOOST_TUNING_REG_LAMBDA_RANGE, log=True
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin", DEFAULT_XGBOOST_TUNING_MAX_BIN_CHOICES
        ),
        "max_cat_to_onehot": trial.suggest_categorical(
            "max_cat_to_onehot",
            DEFAULT_XGBOOST_TUNING_MAX_CAT_TO_ONEHOT_CHOICES,
        ),
        "max_cat_threshold": trial.suggest_categorical(
            "max_cat_threshold",
            DEFAULT_XGBOOST_TUNING_MAX_CAT_THRESHOLD_CHOICES,
        ),
        "enable_early_stopping": True,
        "enable_categorical": True,
        "early_stopping_rounds": DEFAULT_XGBOOST_TUNING_EARLY_STOPPING_ROUNDS,
        "random_state": int(random_seed + trial.number + 1),
    }


def _objective_score(tuning_result: dict[str, object]) -> float:
    decision_loss = tuning_result.get("mean_decision_loss")
    if isinstance(decision_loss, bool) or not isinstance(decision_loss, (int, float)):
        raise ValueError(
            "XGBoost optimisation requires a numeric `mean_decision_loss`; "
            "refusing to fall back to a predictive-error objective."
        )
    return -float(decision_loss)


def guardrail_failure_reason(
    *,
    tuning_result: dict[str, object],
    baseline_wape: float,
    baseline_dataset_wape: dict[str, float],
) -> str | None:
    _ = baseline_dataset_wape
    mean_abs_normalized_bias = tuning_result.get("mean_abs_normalized_bias")
    if isinstance(mean_abs_normalized_bias, (int, float)) and float(
        mean_abs_normalized_bias
    ) > max(
        DEFAULT_TUNING_GUARDRAIL_MIN_ABS_BIAS_LIMIT,
        baseline_wape,
    ):
        return "absolute_bias_too_high"
    return None


def dataset_wape_collapse_warning(
    *,
    tuning_result: dict[str, object],
    baseline_wape: float,
    baseline_dataset_wape: dict[str, float],
) -> str | None:
    dataset_mean_wape = cast(dict[str, float], tuning_result["dataset_mean_wape"])
    collapsed = [
        dataset_source
        for dataset_source, score in dataset_mean_wape.items()
        if score
        > baseline_dataset_wape.get(dataset_source, baseline_wape)
        * DEFAULT_TUNING_GUARDRAIL_DATASET_WAPE_COLLAPSE_MULTIPLIER
    ]
    if collapsed:
        return f"dataset_wape_collapse:{','.join(collapsed)}"
    return None


def _tuning_report_row(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
    return {
        "trial": int(trial.number),
        "stage_name": "xgboost",
        "trial_status": resolved_trial_status_name(trial),
        "trial_duration_seconds": float(trial.duration.total_seconds())
        if trial.duration is not None
        else float("nan"),
        "epochs_completed": int(trial.user_attrs.get("selected_n_estimators", 0)),
        "folds_completed": int(trial.user_attrs.get("folds_completed", 0)),
        "mean_wape": _trial_user_attr_float(trial, "mean_wape"),
        "mean_abs_bias": _trial_user_attr_float(trial, "mean_abs_bias"),
        "mean_abs_normalized_bias": _trial_user_attr_float(
            trial,
            "mean_abs_normalized_bias",
        ),
        "mean_decision_loss": _trial_user_attr_float(trial, "mean_decision_loss"),
        "mean_normalized_economic_loss": _trial_user_attr_float(
            trial,
            "mean_normalized_economic_loss",
        ),
        "mean_normalized_bias": _trial_user_attr_float(
            trial,
            "mean_normalized_bias",
        ),
        "mean_positive_bias_penalty": _trial_user_attr_float(
            trial,
            "mean_positive_bias_penalty",
        ),
        "mean_severe_negative_bias_penalty": _trial_user_attr_float(
            trial,
            "mean_severe_negative_bias_penalty",
        ),
        "validation_economic_loss": _trial_user_attr_float(
            trial,
            "validation_economic_loss",
        ),
        "validation_bias": _trial_user_attr_float(trial, "validation_bias"),
        "validation_positive_bias_penalty": _trial_user_attr_float(
            trial,
            "validation_positive_bias_penalty",
        ),
        "validation_severe_negative_bias_penalty": _trial_user_attr_float(
            trial,
            "validation_severe_negative_bias_penalty",
        ),
        "validation_wape": _trial_user_attr_float(trial, "validation_wape"),
        "validation_monitor_metric": str(
            trial.user_attrs.get(
                "validation_monitor_metric",
                "segmented_asymmetric_decision_loss",
            )
        ),
        "segment_decision_loss": trial.user_attrs.get("segment_decision_loss", {}),
        "segment_bias": trial.user_attrs.get("segment_bias", {}),
        "product_decision_loss": trial.user_attrs.get("product_decision_loss", {}),
        "product_bias": trial.user_attrs.get("product_bias", {}),
        "coverage_80": float("nan"),
        "coverage_95": float("nan"),
        "objective_score": _trial_user_attr_float(trial, "objective_score"),
        "selected_learning_rate": _trial_user_attr_float(
            trial, "selected_learning_rate"
        ),
        "selected_n_estimators": _trial_user_attr_float(
            trial,
            "selected_n_estimators",
        ),
        "native_best_score": _trial_user_attr_float(trial, "native_best_score"),
        "failure_reason": str(trial.user_attrs.get("failure_reason", "")),
        "guardrail_warning": str(trial.user_attrs.get("guardrail_warning", "")),
        "baseline_wape_improvement_pct": _trial_user_attr_float(
            trial, "baseline_wape_improvement_pct"
        ),
        "fold_wape_scores": trial.user_attrs.get("fold_wape_scores", []),
        "dataset_mean_wape": trial.user_attrs.get("dataset_mean_wape", {}),
        **trial.params,
    }


def _trial_user_attr_float(trial: optuna.trial.FrozenTrial, key: str) -> float:
    value = trial.user_attrs.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _build_tuning_report(study: optuna.study.Study) -> pd.DataFrame:
    return (
        pd.DataFrame([_tuning_report_row(trial) for trial in study.trials])
        .sort_values(by="objective_score", ascending=False)
        .reset_index(drop=True)
    )


def _is_categorical_feature(series: pd.Series) -> bool:
    return bool(
        isinstance(series.dtype, pd.CategoricalDtype)
        or is_object_dtype(series)
        or is_string_dtype(series)
    )


def _is_numeric_feature(series: pd.Series) -> bool:
    return bool(is_numeric_dtype(series) or is_bool_dtype(series))


def _top_categorical_cardinalities(
    frame: pd.DataFrame,
    categorical_cols: list[str],
) -> dict[str, int]:
    cardinalities = {
        column: int(frame[column].astype("string").nunique(dropna=True))
        for column in categorical_cols
    }
    return dict(
        sorted(cardinalities.items(), key=lambda item: item[1], reverse=True)[:10]
    )


def _log_xgboost_feature_space(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
) -> None:
    categorical_cols = [
        column
        for column in feature_cols
        if _is_categorical_feature(train_frame[column])
    ]
    numeric_cols = [
        column for column in feature_cols if _is_numeric_feature(train_frame[column])
    ]
    logger.info(
        "XGBoost feature space resolved: feature_count=%s numeric_features=%s "
        "categorical_features=%s top_categorical_cardinalities=%s",
        len(feature_cols),
        len(numeric_cols),
        len(categorical_cols),
        _top_categorical_cardinalities(train_frame, categorical_cols),
    )


def _best_completed_trial(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        reasons = [
            str(trial.user_attrs.get("failure_reason", trial.state.name))
            for trial in study.trials
        ]
        raise RuntimeError(
            "XGBoost Optuna search completed without guardrail-approved trials. "
            f"Failure reasons: {reasons}"
        )
    return study.best_trial


def _final_best_params(
    *,
    best_trial: optuna.trial.FrozenTrial,
    model_params: dict[str, object] | None,
    random_seed: int,
    total_threads: int,
) -> dict[str, object]:
    best_params = _without_hpo_sample_weight_params(
        {**(model_params or {}), **best_trial.params}
    )
    best_params["model_backend"] = "xgboost"
    best_params["random_state"] = random_seed + best_trial.number + 1
    best_params["n_jobs"] = total_threads
    best_params["enable_categorical"] = True
    best_params["enable_early_stopping"] = True
    best_params["early_stopping_rounds"] = DEFAULT_XGBOOST_TUNING_EARLY_STOPPING_ROUNDS
    best_params["validation_monitor_metric"] = "segmented_asymmetric_decision_loss"
    best_params["economic_objective_config"] = best_trial.user_attrs.get(
        "economic_objective_config",
        economic_objective_config_payload(),
    )
    best_params["segmented_economic_objective_config"] = best_trial.user_attrs.get(
        "segmented_economic_objective_config",
        best_params["economic_objective_config"],
    )
    for key in _BEST_TRIAL_DECISION_ATTRS:
        if key in best_trial.user_attrs:
            best_params[key] = best_trial.user_attrs[key]
    return resolve_xgboost_model_params(best_params)


def _without_hpo_sample_weight_params(params: dict[str, object]) -> dict[str, object]:
    cleaned = dict(params)
    cleaned.pop("bakery_sample_weight_multiplier", None)
    cleaned.pop("enable_dataset_sample_weight", None)
    return cleaned


def _build_objective(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    baseline_wape: float,
    baseline_dataset_wape: dict[str, float],
    target_contract: TargetContract,
    logger: logging.Logger,
    random_seed: int,
    model_params: dict[str, object] | None,
    total_threads: int,
    tuning_trials: int,
    study: optuna.study.Study,
) -> Callable[[optuna.trial.Trial], float]:
    completed = {"count": 0}
    base_train_frame = filter_training_eligible_rows(
        train_frame,
        label="xgboost_optuna_shared_train",
        logger=logger,
    )
    logger.debug(
        "XGBoost Optuna shared train frame filtered once: rows_before=%s rows_after=%s",
        len(train_frame),
        len(base_train_frame),
    )
    fold_matrix_cache = _build_fold_matrix_cache_for_objective(
        train_frame=base_train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        logger=logger,
        model_params=model_params,
        total_threads=total_threads,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        objective_start = time.perf_counter()
        logger.info(
            "XGBoost Optuna trial entering objective: trial=%s/%s train_rows=%s tuning_rows=%s folds=%s features=%s threads_per_fold=%s",
            trial.number + 1,
            tuning_trials,
            len(train_frame),
            len(tuning_frame),
            len(folds),
            len(feature_cols),
            total_threads,
        )
        trial_params = {
            **(model_params or {}),
            **sample_xgboost_optuna_params(trial, random_seed=random_seed),
        }
        trial_params = _without_hpo_sample_weight_params(trial_params)
        logger.info(
            "XGBoost Optuna trial sampled params: trial=%s params=%s",
            trial.number,
            {
                key: trial_params[key]
                for key in sorted(trial_params)
                if key
                in {
                    "colsample_bytree",
                    "early_stopping_rounds",
                    "enable_early_stopping",
                    "learning_rate",
                    "max_bin",
                    "max_cat_threshold",
                    "max_cat_to_onehot",
                    "max_parallel_fold_workers",
                    "max_depth",
                    "min_child_weight",
                    "n_estimators",
                    "n_jobs",
                    "reg_alpha",
                    "reg_lambda",
                    "subsample",
                    "tree_method",
                }
            },
        )
        try:
            result = fit_and_score_xgboost_model_on_tuning(
                train_frame=base_train_frame,
                tuning_frame=tuning_frame,
                folds=folds,
                feature_cols=feature_cols,
                target_contract=target_contract,
                logger=logger,
                model_params=trial_params,
                total_threads=total_threads,
                trial=trial,
                train_frame_is_eligible=True,
                fold_matrix_cache=fold_matrix_cache,
            )
            objective_score = _record_trial_result(
                trial=trial,
                tuning_result=result,
                baseline_wape=baseline_wape,
                baseline_dataset_wape=baseline_dataset_wape,
            )
            completed["count"] += 1
            _log_progress(
                completed=completed["count"],
                tuning_trials=tuning_trials,
                study=study,
                objective_score=objective_score,
                logger=logger,
            )
            logger.info(
                "XGBoost Optuna trial completed: trial=%s objective_score=%.6f mean_wape=%.6f mean_decision_loss=%s selected_n_estimators=%s duration_seconds=%.3f",
                trial.number,
                objective_score,
                float(cast(Any, result["macro_mean_wape"])),
                result.get("mean_decision_loss"),
                result.get("selected_n_estimators"),
                time.perf_counter() - objective_start,
            )
            return objective_score
        except optuna.TrialPruned:
            logger.debug(
                "XGBoost Optuna trial pruned: trial=%s duration_seconds=%.3f",
                trial.number,
                time.perf_counter() - objective_start,
            )
            raise
        except Exception as exc:
            trial.set_user_attr("failure_reason", str(exc))
            logger.exception(
                "XGBoost Optuna trial failed with Python exception: trial=%s duration_seconds=%.3f",
                trial.number,
                time.perf_counter() - objective_start,
            )
            raise

    return objective


def _build_fold_matrix_cache_for_objective(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    logger: logging.Logger,
    model_params: dict[str, object] | None,
    total_threads: int,
) -> XGBoostFoldMatrixCache | None:
    return build_xgboost_fold_matrix_cache(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        logger=logger,
        model_params=_without_hpo_sample_weight_params(model_params or {}),
        total_threads=total_threads,
    )


def _record_trial_result(
    *,
    trial: optuna.trial.Trial,
    tuning_result: dict[str, object],
    baseline_wape: float,
    baseline_dataset_wape: dict[str, float],
) -> float:
    mean_wape = float(cast(Any, tuning_result["macro_mean_wape"]))
    improvement_pct = compute_wape_improvement_pct(baseline_wape, mean_wape)
    objective_score = _objective_score(tuning_result)
    trial.set_user_attr("mean_wape", mean_wape)
    trial.set_user_attr("dataset_mean_wape", tuning_result["dataset_mean_wape"])
    trial.set_user_attr("mean_abs_bias", tuning_result.get("mean_abs_bias"))
    trial.set_user_attr(
        "mean_abs_normalized_bias",
        tuning_result.get("mean_abs_normalized_bias"),
    )
    _record_decision_trial_attrs(trial=trial, tuning_result=tuning_result)
    trial.set_user_attr(
        "selected_learning_rate", tuning_result["selected_learning_rate"]
    )
    trial.set_user_attr("selected_n_estimators", tuning_result["selected_n_estimators"])
    trial.set_user_attr("native_best_score", tuning_result["native_best_score"])
    trial.set_user_attr("objective_score", objective_score)
    trial.set_user_attr("baseline_wape_improvement_pct", improvement_pct)
    trial.set_user_attr("fold_wape_scores", tuning_result["fold_results"])
    trial.set_user_attr("folds_completed", tuning_result["folds_completed"])
    warning = dataset_wape_collapse_warning(
        tuning_result=tuning_result,
        baseline_wape=baseline_wape,
        baseline_dataset_wape=baseline_dataset_wape,
    )
    if warning is not None:
        trial.set_user_attr("guardrail_warning", warning)
    failure_reason = guardrail_failure_reason(
        tuning_result=tuning_result,
        baseline_wape=baseline_wape,
        baseline_dataset_wape=baseline_dataset_wape,
    )
    if failure_reason is not None:
        trial.set_user_attr("failure_reason", failure_reason)
        trial.set_user_attr("terminal_status", "REJECTED")
        raise optuna.TrialPruned(f"Guardrail rejected trial: {failure_reason}")
    return objective_score


_BEST_TRIAL_DECISION_ATTRS: tuple[str, ...] = (
    "mean_decision_loss",
    "mean_normalized_economic_loss",
    "mean_economic_loss",
    "mean_normalized_bias",
    "mean_positive_bias_penalty",
    "mean_severe_negative_bias_penalty",
    "mean_wape_guardrail_penalty",
    "validation_economic_loss",
    "validation_bias",
    "validation_positive_bias_penalty",
    "validation_severe_negative_bias_penalty",
    "validation_wape",
    "segment_decision_loss",
    "segment_bias",
    "product_decision_loss",
    "product_bias",
)


def _record_decision_trial_attrs(
    *,
    trial: optuna.trial.Trial,
    tuning_result: dict[str, object],
) -> None:
    for key in _BEST_TRIAL_DECISION_ATTRS:
        if key in tuning_result:
            trial.set_user_attr(key, tuning_result[key])
    trial.set_user_attr(
        "validation_economic_loss",
        tuning_result.get("mean_normalized_economic_loss"),
    )
    trial.set_user_attr("validation_bias", tuning_result.get("mean_normalized_bias"))
    trial.set_user_attr(
        "validation_positive_bias_penalty",
        tuning_result.get("mean_positive_bias_penalty"),
    )
    trial.set_user_attr(
        "validation_severe_negative_bias_penalty",
        tuning_result.get("mean_severe_negative_bias_penalty"),
    )
    trial.set_user_attr("validation_wape", tuning_result.get("macro_mean_wape"))
    trial.set_user_attr(
        "validation_monitor_metric",
        tuning_result.get(
            "validation_monitor_metric",
            "segmented_asymmetric_decision_loss",
        ),
    )
    trial.set_user_attr(
        "economic_objective_config",
        tuning_result.get(
            "economic_objective_config",
            economic_objective_config_payload(),
        ),
    )
    trial.set_user_attr(
        "segmented_economic_objective_config",
        tuning_result.get(
            "segmented_economic_objective_config",
            tuning_result.get(
                "economic_objective_config", economic_objective_config_payload()
            ),
        ),
    )


def _log_progress(
    *,
    completed: int,
    tuning_trials: int,
    study: optuna.study.Study,
    objective_score: float,
    logger: logging.Logger,
) -> None:
    if (
        completed == 1
        or completed == tuning_trials
        or completed % DEFAULT_TUNING_PROGRESS_LOG_EVERY == 0
    ):
        logger.info(
            "XGBoost optimisation progress: trial=%s/%s current_best_objective_score=%.6f",
            completed,
            tuning_trials,
            max(
                (trial.value for trial in study.trials if trial.value is not None),
                default=objective_score,
            ),
        )


def optimize_xgboost_model_params(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    baseline_wape: float,
    target_contract: TargetContract,
    *,
    baseline_dataset_wape: dict[str, float] | None = None,
    logger: logging.Logger,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_TUNING_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    _ = target_transform
    total_threads = int(
        cast(Any, (model_params or {}).get("n_jobs", os.cpu_count() or 1))
    )
    resolved_base_params = resolve_xgboost_model_params(
        {**(model_params or {}), "n_jobs": total_threads}
    )
    total_threads = int(cast(Any, resolved_base_params["n_jobs"]))
    execution_policy = xgboost_execution_policy(
        resolved_base_params,
        fold_count=len(folds),
    )
    _log_xgboost_feature_space(train_frame, feature_cols, logger)
    logger.info(
        "Starting final XGBoost optimisation with Optuna: trials=%s feature_count=%s baseline_wape=%.6f train_rows=%s tuning_rows=%s folds=%s execution_policy=%s",
        tuning_trials,
        len(feature_cols),
        baseline_wape,
        len(train_frame),
        len(tuning_frame),
        len(folds),
        execution_policy,
    )
    study = create_optuna_study(random_seed=random_seed)
    objective = _build_objective(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        baseline_wape=baseline_wape,
        baseline_dataset_wape=baseline_dataset_wape or {},
        target_contract=target_contract,
        logger=logger,
        random_seed=random_seed,
        model_params=resolved_base_params,
        total_threads=total_threads,
        tuning_trials=tuning_trials,
        study=study,
    )
    logger.debug(
        "XGBoost Optuna study created; about to call study.optimize: trials=%s sampler=%s pruner=%s",
        tuning_trials,
        study.sampler.__class__.__name__,
        study.pruner.__class__.__name__,
    )
    study.optimize(
        objective,
        n_trials=tuning_trials,
        n_jobs=1,
        catch=(RuntimeError,),
    )
    logger.debug(
        "XGBoost Optuna study.optimize returned: trials_finished=%s",
        len(study.trials),
    )
    best_trial = _best_completed_trial(study)
    best_params = _final_best_params(
        best_trial=best_trial,
        model_params=resolved_base_params,
        random_seed=random_seed,
        total_threads=total_threads,
    )
    return (
        _best_params_with_economic_calibration(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
            best_params=best_params,
            logger=logger,
        ),
        _build_tuning_report(study),
        build_hpo_runtime_metadata(
            study=study,
            execution_policy=execution_policy,
            tuning_trials=tuning_trials,
            stage_budget=str(resolved_base_params.get("stage_budget", "standard")),
            best_trial=best_trial,
        ),
    )


def _best_params_with_economic_calibration(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    best_params: dict[str, object],
    logger: logging.Logger,
) -> dict[str, object]:
    calibrated_params = dict(best_params)
    try:
        eligible_train_frame = filter_training_eligible_rows(
            train_frame,
            label="xgboost_economic_calibration_train",
            logger=logger,
            log_level=logging.DEBUG,
        )
        _, raw_predictions = fit_xgboost_model_and_predict_validation(
            eligible_train_frame,
            tuning_frame,
            feature_cols,
            target_col=target_contract.learning_target_col,
            model_params=calibrated_params,
        )
        absolute_predictions = reconstruct_absolute_predictions(
            raw_predictions,
            tuning_frame,
            target_contract,
        )
        calibration = fit_economic_decision_calibration(
            tuning_frame,
            tuning_frame[target_contract.absolute_target_col],
            absolute_predictions,
        )
        calibrated_params["economic_calibration"] = calibration
        logger.info(
            "XGBoost economic calibration fitted: applied=%s global=%s families=%s products=%s",
            calibration.get("applied"),
            cast(dict[str, object], calibration.get("global", {})),
            len(cast(dict[str, object], calibration.get("families", {}))),
            len(cast(dict[str, object], calibration.get("products", {}))),
        )
    except Exception as exc:
        calibrated_params["economic_calibration"] = {
            "version": "economic_decision_calibration_final",
            "enabled": False,
            "applied": False,
            "failure_reason": str(exc),
        }
        logger.warning("XGBoost economic calibration skipped: %s", exc)
    return calibrated_params
