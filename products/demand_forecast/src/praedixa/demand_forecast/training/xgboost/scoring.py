from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import time
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd

from praedixa.demand_forecast.backends.xgboost.model_common import (
    resolve_xgboost_model_params,
)
from praedixa.demand_forecast.backends.xgboost.model_fit import (
    XGBoostPreparedTrainingMatrices,
    fit_xgboost_model_and_predict_validation,
    fit_prepared_xgboost_matrices_and_predict_validation,
    prepare_xgboost_training_matrices,
    warm_up_xgboost_runtime,
)
from praedixa.demand_forecast.backends.xgboost.runtime import xgboost_uses_cuda
from praedixa.demand_forecast.contracts.targets import (
    TargetContract,
    reconstruct_absolute_predictions,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_TARGET_TRANSFORM,
    DEFAULT_XGBOOST_LOCAL_CPU_FOLD_WORKERS,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    filter_training_eligible_rows,
)
from praedixa.demand_forecast.training.shared.metrics import compute_wape


def _param_snapshot(params: dict[str, object]) -> dict[str, object]:
    keys = (
        "colsample_bytree",
        "early_stopping_rounds",
        "enable_categorical",
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
        "device",
        "xgboost_matrix_type",
        "xgboost_gpu_input_backend",
    )
    return {key: params[key] for key in keys if key in params}


def _resolve_fold_workers(
    *,
    fold_count: int,
    resolved_params: dict[str, object],
) -> int:
    requested = int(
        cast(
            Any,
            resolved_params.get(
                "max_parallel_fold_workers",
                DEFAULT_XGBOOST_LOCAL_CPU_FOLD_WORKERS,
            ),
        )
    )
    return max(1, min(fold_count, requested))


def xgboost_execution_policy(
    resolved_params: dict[str, object],
    *,
    fold_count: int | None = None,
) -> dict[str, object]:
    folds = int(fold_count or 1)
    uses_cuda = xgboost_uses_cuda(resolved_params)
    if uses_cuda:
        fold_workers = 1
        threads_per_fold = 1
    else:
        fold_workers = _resolve_fold_workers(
            fold_count=folds,
            resolved_params=resolved_params,
        )
        threads_per_fold = int(cast(Any, resolved_params["n_jobs"]))
    return {
        "backend": "xgboost",
        "runtime_profile": str(resolved_params.get("runtime_profile", "local_cpu")),
        "accelerator": "gpu" if uses_cuda else "cpu",
        "devices": 1,
        "gpu_safe_mode": uses_cuda,
        "gpu_input_backend": str(
            resolved_params.get("xgboost_gpu_input_backend", "cpu")
        ),
        "fold_workers": fold_workers,
        "threads_per_fold": threads_per_fold,
        "total_threads": fold_workers * threads_per_fold,
    }


@dataclass(frozen=True)
class XGBoostFoldMatrixCache:
    resolved_params: dict[str, object]
    folds: list["_CachedXGBoostFold"]


@dataclass(frozen=True)
class _CachedXGBoostFold:
    fold: dict[str, object]
    fold_position: int
    fold_valid: pd.DataFrame
    prepared_matrices: XGBoostPreparedTrainingMatrices


def build_xgboost_fold_matrix_cache(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    logger: logging.Logger,
    model_params: dict[str, object],
    total_threads: int | None = None,
    train_frame_is_eligible: bool = False,
) -> XGBoostFoldMatrixCache | None:
    resolved_params = _resolved_scoring_params(model_params, total_threads)
    if not _should_cache_xgboost_fold_matrices(resolved_params):
        return None
    cache_start = time.perf_counter()
    base_train_frame = _eligible_base_train_frame(
        train_frame,
        logger=logger,
        train_frame_is_eligible=train_frame_is_eligible,
    )
    try:
        cached_folds = _build_cached_xgboost_folds(
            train_frame=base_train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
        )
    except Exception as exc:
        logger.warning(
            "XGBoost fold matrix cache disabled after build failure: %s",
            exc,
        )
        return None
    logger.info(
        "XGBoost fold matrix cache ready: folds=%s matrix_type=%s gpu_input_backend=%s max_bin=%s duration_seconds=%.3f",
        len(cached_folds),
        resolved_params.get("xgboost_matrix_type"),
        resolved_params.get("xgboost_gpu_input_backend"),
        resolved_params.get("max_bin"),
        time.perf_counter() - cache_start,
    )
    return XGBoostFoldMatrixCache(
        resolved_params=resolved_params,
        folds=cached_folds,
    )


def _resolved_scoring_params(
    model_params: dict[str, object] | None,
    total_threads: int | None,
) -> dict[str, object]:
    resolved_params = resolve_xgboost_model_params(model_params)
    resolved_params["n_jobs"] = int(
        total_threads or cast(Any, resolved_params["n_jobs"])
    )
    if xgboost_uses_cuda(resolved_params):
        resolved_params["n_jobs"] = 1
        resolved_params["max_parallel_fold_workers"] = 1
    return resolved_params


def _eligible_base_train_frame(
    train_frame: pd.DataFrame,
    *,
    logger: logging.Logger,
    train_frame_is_eligible: bool,
) -> pd.DataFrame:
    if train_frame_is_eligible:
        return train_frame
    return filter_training_eligible_rows(
        train_frame,
        label="xgboost_optuna_shared_train",
        logger=logger,
    )


def _should_cache_xgboost_fold_matrices(params: dict[str, object]) -> bool:
    matrix_type = str(params.get("xgboost_matrix_type", "dmatrix"))
    if matrix_type == "quantile":
        return params.get("max_bin") is not None
    return matrix_type == "dmatrix"


def _build_cached_xgboost_folds(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> list[_CachedXGBoostFold]:
    cached_folds: list[_CachedXGBoostFold] = []
    for position, fold in enumerate(folds, start=1):
        cached_folds.append(
            _build_cached_xgboost_fold(
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                fold=fold,
                fold_position=position,
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
            )
        )
    return cached_folds


def _build_cached_xgboost_fold(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    fold_position: int,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
) -> _CachedXGBoostFold:
    fold_start = time.perf_counter()
    fold_train, fold_valid = _fold_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        fold=fold,
        logger=logger,
    )
    prepared_matrices = prepare_xgboost_training_matrices(
        fold_train,
        fold_valid,
        feature_cols,
        target_col=target_contract.learning_target_col,
        model_params=resolved_params,
        train_sample_weight=_sample_weights(fold_train)
        if bool(resolved_params.get("enable_dataset_sample_weight", False))
        else None,
    )
    logger.debug(
        "XGBoost cached fold matrix built: fold=%s position=%s train_rows=%s valid_rows=%s duration_seconds=%.3f",
        int(cast(Any, fold["fold"])),
        fold_position,
        len(fold_train),
        len(fold_valid),
        time.perf_counter() - fold_start,
    )
    return _CachedXGBoostFold(
        fold=fold,
        fold_position=fold_position,
        fold_valid=fold_valid,
        prepared_matrices=prepared_matrices,
    )


def fit_and_score_xgboost_model_on_tuning(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    *,
    logger: logging.Logger,
    model_params: dict[str, object] | None = None,
    total_threads: int | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    trial: optuna.trial.Trial | None = None,
    train_frame_is_eligible: bool = False,
    fold_matrix_cache: XGBoostFoldMatrixCache | None = None,
) -> dict[str, object]:
    _ = target_transform
    trial_number = trial.number if trial is not None else None
    scoring_start = time.perf_counter()
    resolved_params = _resolved_scoring_params(model_params, total_threads)
    execution_policy = xgboost_execution_policy(
        resolved_params,
        fold_count=len(folds),
    )
    if fold_matrix_cache is not None:
        execution_policy = {
            **execution_policy,
            "fold_matrix_cache": "prebuilt",
            "fold_matrix_cache_folds": len(fold_matrix_cache.folds),
        }
    logger.debug(
        "XGBoost scoring started: trial=%s train_rows=%s tuning_rows=%s folds=%s features=%s execution_policy=%s params=%s",
        trial_number,
        len(train_frame),
        len(tuning_frame),
        len(folds),
        len(feature_cols),
        execution_policy,
        _param_snapshot(resolved_params),
    )
    if train_frame_is_eligible:
        base_train_frame = train_frame
    else:
        base_train_frame = filter_training_eligible_rows(
            train_frame,
            label="xgboost_optuna_shared_train",
            logger=logger,
        )
    logger.debug(
        "XGBoost scoring eligible train frame ready: trial=%s rows_before=%s rows_after=%s",
        trial_number,
        len(train_frame),
        len(base_train_frame),
    )
    fold_results, folds_completed = _score_xgboost_folds_with_optional_cache(
        train_frame=base_train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
        trial=trial,
        trial_number=trial_number,
        fold_workers=int(cast(Any, execution_policy["fold_workers"])),
        fold_matrix_cache=fold_matrix_cache,
    )
    dataset_mean_wape = _dataset_macro_scores(fold_results)
    selected_n_estimators = _mean_metric(
        fold_results,
        metric_key="selected_n_estimators",
    )
    result = {
        "macro_mean_wape": float(np.mean(list(dataset_mean_wape.values()))),
        "dataset_mean_wape": dataset_mean_wape,
        "mean_abs_bias": _mean_metric(fold_results, metric_key="abs_bias"),
        "mean_coverage_80": None,
        "mean_coverage_95": None,
        "mean_pinball_loss": None,
        "selected_learning_rate": resolved_params["learning_rate"],
        "selected_n_estimators": int(round(selected_n_estimators))
        if selected_n_estimators is not None
        else None,
        "native_best_score": _mean_metric(fold_results, metric_key="native_best_score"),
        "fold_results": fold_results,
        "folds_completed": folds_completed,
        "execution_policy": execution_policy,
    }
    logger.debug(
        "XGBoost scoring completed: trial=%s folds_completed=%s macro_mean_wape=%.6f selected_n_estimators=%s native_best_score=%s duration_seconds=%.3f",
        trial_number,
        folds_completed,
        result["macro_mean_wape"],
        result["selected_n_estimators"],
        result["native_best_score"],
        time.perf_counter() - scoring_start,
    )
    return result


def _score_xgboost_folds_with_optional_cache(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
    fold_workers: int,
    fold_matrix_cache: XGBoostFoldMatrixCache | None,
) -> tuple[list[dict[str, object]], int]:
    if fold_matrix_cache is not None:
        return _score_cached_xgboost_folds(
            fold_matrix_cache=fold_matrix_cache,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
            trial=trial,
            trial_number=trial_number,
            fold_workers=fold_workers,
        )
    return _score_xgboost_folds(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
        trial=trial,
        trial_number=trial_number,
        fold_workers=fold_workers,
    )


def _score_cached_xgboost_folds(
    *,
    fold_matrix_cache: XGBoostFoldMatrixCache,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
    fold_workers: int,
) -> tuple[list[dict[str, object]], int]:
    if fold_workers <= 1:
        return _score_cached_xgboost_folds_sequential(
            fold_matrix_cache=fold_matrix_cache,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
            trial=trial,
            trial_number=trial_number,
        )
    return _score_cached_xgboost_folds_parallel(
        fold_matrix_cache=fold_matrix_cache,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
        trial=trial,
        trial_number=trial_number,
        fold_workers=fold_workers,
    )


def _score_xgboost_folds(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
    fold_workers: int,
) -> tuple[list[dict[str, object]], int]:
    if fold_workers <= 1:
        return _score_xgboost_folds_sequential(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            folds=folds,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
            trial=trial,
            trial_number=trial_number,
        )
    return _score_xgboost_folds_parallel(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
        trial=trial,
        trial_number=trial_number,
        fold_workers=fold_workers,
    )


def _score_cached_xgboost_folds_sequential(
    *,
    fold_matrix_cache: XGBoostFoldMatrixCache,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
) -> tuple[list[dict[str, object]], int]:
    fold_results: list[dict[str, object]] = []
    folds_completed = 0
    fold_count = len(fold_matrix_cache.folds)
    for folds_completed, cached_fold in enumerate(
        fold_matrix_cache.folds,
        start=1,
    ):
        single_fold_results, duration_seconds = _score_cached_fold_with_logging(
            cached_fold=cached_fold,
            fold_count=fold_count,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
            trial_number=trial_number,
        )
        fold_results.extend(single_fold_results)
        _log_fold_completed(
            logger=logger,
            trial_number=trial_number,
            fold=cached_fold.fold,
            fold_position=cached_fold.fold_position,
            fold_count=fold_count,
            single_fold_results=single_fold_results,
            duration_seconds=duration_seconds,
        )
        if trial is not None:
            _report_trial_progress(trial, fold_results, step=folds_completed)
            if trial.should_prune():
                trial.set_user_attr("fold_results", fold_results)
                trial.set_user_attr("folds_completed", folds_completed)
                raise optuna.TrialPruned(
                    f"XGBoost trial pruned after fold {folds_completed}."
                )
    return fold_results, folds_completed


def _score_cached_xgboost_folds_parallel(
    *,
    fold_matrix_cache: XGBoostFoldMatrixCache,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
    fold_workers: int,
) -> tuple[list[dict[str, object]], int]:
    logger.debug(
        "XGBoost cached folds parallel execution starting: trial=%s folds=%s workers=%s threads_per_worker=%s",
        trial_number,
        len(fold_matrix_cache.folds),
        fold_workers,
        int(cast(Any, resolved_params["n_jobs"])),
    )
    if int(cast(Any, resolved_params["n_jobs"])) > 1:
        warm_up_xgboost_runtime(int(cast(Any, resolved_params["n_jobs"])))
    completed: list[_FoldResult] = []
    with ThreadPoolExecutor(max_workers=fold_workers) as executor:
        future_to_position = {
            executor.submit(
                _score_cached_fold_with_logging,
                cached_fold=cached_fold,
                fold_count=len(fold_matrix_cache.folds),
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
                trial_number=trial_number,
            ): position
            for position, cached_fold in enumerate(fold_matrix_cache.folds, start=1)
        }
        for future in as_completed(future_to_position):
            position = future_to_position[future]
            results, duration_seconds = future.result()
            cached_fold = fold_matrix_cache.folds[position - 1]
            completed.append(
                _FoldResult(
                    position=position,
                    results=results,
                    duration_seconds=duration_seconds,
                )
            )
            _log_fold_completed(
                logger=logger,
                trial_number=trial_number,
                fold=cached_fold.fold,
                fold_position=position,
                fold_count=len(fold_matrix_cache.folds),
                single_fold_results=results,
                duration_seconds=duration_seconds,
            )
    fold_results = [
        row for fold_result in sorted(completed) for row in fold_result.results
    ]
    if trial is not None:
        _report_trial_progress(trial, fold_results, step=len(fold_matrix_cache.folds))
    return fold_results, len(completed)


def _score_cached_fold_with_logging(
    *,
    cached_fold: _CachedXGBoostFold,
    fold_count: int,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial_number: int | None,
) -> tuple[list[dict[str, object]], float]:
    fold_start = time.perf_counter()
    _log_fold_starting(
        logger=logger,
        trial_number=trial_number,
        fold=cached_fold.fold,
        fold_position=cached_fold.fold_position,
        fold_count=fold_count,
    )
    return (
        _score_cached_fold(
            cached_fold=cached_fold,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
        ),
        time.perf_counter() - fold_start,
    )


def _score_cached_fold(
    *,
    cached_fold: _CachedXGBoostFold,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
) -> list[dict[str, object]]:
    model, raw_predictions = fit_prepared_xgboost_matrices_and_predict_validation(
        cached_fold.prepared_matrices,
        model_params=resolved_params,
    )
    predictions = reconstruct_absolute_predictions(
        raw_predictions,
        cached_fold.fold_valid,
        target_contract,
    )
    return _score_predictions_by_dataset(
        frame=cached_fold.fold_valid,
        predictions=predictions,
        absolute_target_col=target_contract.absolute_target_col,
        fold_number=int(cast(Any, cached_fold.fold["fold"])),
        feature_cols=feature_cols,
        selected_n_estimators=_selected_n_estimators(model.model),
        native_best_score=_native_best_score(model.model),
    )


def _score_xgboost_folds_sequential(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
) -> tuple[list[dict[str, object]], int]:
    fold_results: list[dict[str, object]] = []
    folds_completed = 0
    for folds_completed, fold in enumerate(folds, start=1):
        single_fold_results, duration_seconds = _score_single_fold_with_logging(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            fold=fold,
            fold_position=folds_completed,
            fold_count=len(folds),
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
            trial_number=trial_number,
        )
        fold_results.extend(single_fold_results)
        logger.debug(
            "XGBoost fold completed: trial=%s fold=%s/%s fold_id=%s dataset_results=%s duration_seconds=%.3f",
            trial_number,
            folds_completed,
            len(folds),
            int(cast(Any, fold["fold"])),
            single_fold_results,
            duration_seconds,
        )
        if trial is not None:
            _report_trial_progress(trial, fold_results, step=folds_completed)
            if trial.should_prune():
                trial.set_user_attr("fold_results", fold_results)
                trial.set_user_attr("folds_completed", folds_completed)
                raise optuna.TrialPruned(
                    f"XGBoost trial pruned after fold {folds_completed}."
                )
    return fold_results, folds_completed


def _score_xgboost_folds_parallel(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial: optuna.trial.Trial | None,
    trial_number: int | None,
    fold_workers: int,
) -> tuple[list[dict[str, object]], int]:
    logger.debug(
        "XGBoost folds parallel execution starting: trial=%s folds=%s workers=%s threads_per_worker=%s",
        trial_number,
        len(folds),
        fold_workers,
        int(cast(Any, resolved_params["n_jobs"])),
    )
    if int(cast(Any, resolved_params["n_jobs"])) > 1:
        warm_up_xgboost_runtime(int(cast(Any, resolved_params["n_jobs"])))
    completed = _parallel_fold_results(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        resolved_params=resolved_params,
        logger=logger,
        trial_number=trial_number,
        fold_workers=fold_workers,
    )
    fold_results = [
        row for fold_result in sorted(completed) for row in fold_result.results
    ]
    if trial is not None:
        _report_trial_progress(trial, fold_results, step=len(folds))
    return fold_results, len(completed)


def _report_trial_progress(
    trial: optuna.trial.Trial,
    fold_results: list[dict[str, object]],
    *,
    step: int,
) -> None:
    dataset_mean_wape = _dataset_macro_scores(fold_results)
    trial.report(
        -float(np.mean(list(dataset_mean_wape.values()))),
        step=step,
    )


@dataclass(frozen=True, order=True)
class _FoldResult:
    position: int
    results: list[dict[str, object]] = field(compare=False)
    duration_seconds: float = field(compare=False)


def _parallel_fold_results(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial_number: int | None,
    fold_workers: int,
) -> list[_FoldResult]:
    completed: list[_FoldResult] = []
    with ThreadPoolExecutor(max_workers=fold_workers) as executor:
        future_to_position = {
            executor.submit(
                _score_single_fold_with_logging,
                train_frame=train_frame,
                tuning_frame=tuning_frame,
                fold=fold,
                fold_position=position,
                fold_count=len(folds),
                feature_cols=feature_cols,
                target_contract=target_contract,
                resolved_params=resolved_params,
                logger=logger,
                trial_number=trial_number,
            ): position
            for position, fold in enumerate(folds, start=1)
        }
        for future in as_completed(future_to_position):
            position = future_to_position[future]
            results, duration_seconds = future.result()
            completed.append(
                _FoldResult(
                    position=position,
                    results=results,
                    duration_seconds=duration_seconds,
                )
            )
            _log_fold_completed(
                logger=logger,
                trial_number=trial_number,
                fold=folds[position - 1],
                fold_position=position,
                fold_count=len(folds),
                single_fold_results=results,
                duration_seconds=duration_seconds,
            )
    return completed


def _score_single_fold_with_logging(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    fold_position: int,
    fold_count: int,
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial_number: int | None,
) -> tuple[list[dict[str, object]], float]:
    fold_start = time.perf_counter()
    _log_fold_starting(
        logger=logger,
        trial_number=trial_number,
        fold=fold,
        fold_position=fold_position,
        fold_count=fold_count,
    )
    return (
        _score_single_fold(
            train_frame=train_frame,
            tuning_frame=tuning_frame,
            fold=fold,
            feature_cols=feature_cols,
            target_contract=target_contract,
            resolved_params=resolved_params,
            logger=logger,
            trial_number=trial_number,
        ),
        time.perf_counter() - fold_start,
    )


def _log_fold_starting(
    *,
    logger: logging.Logger,
    trial_number: int | None,
    fold: dict[str, object],
    fold_position: int,
    fold_count: int,
) -> None:
    logger.debug(
        "XGBoost fold starting: trial=%s fold=%s/%s fold_id=%s train_extension_rows=%s valid_rows=%s",
        trial_number,
        fold_position,
        fold_count,
        int(cast(Any, fold["fold"])),
        len(cast(Any, fold["train_idx"])),
        len(cast(Any, fold["valid_idx"])),
    )


def _log_fold_completed(
    *,
    logger: logging.Logger,
    trial_number: int | None,
    fold: dict[str, object],
    fold_position: int,
    fold_count: int,
    single_fold_results: list[dict[str, object]],
    duration_seconds: float,
) -> None:
    logger.debug(
        "XGBoost fold completed: trial=%s fold=%s/%s fold_id=%s dataset_results=%s duration_seconds=%.3f",
        trial_number,
        fold_position,
        fold_count,
        int(cast(Any, fold["fold"])),
        single_fold_results,
        duration_seconds,
    )


def _score_single_fold(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    feature_cols: list[str],
    target_contract: TargetContract,
    resolved_params: dict[str, object],
    logger: logging.Logger,
    trial_number: int | None,
) -> list[dict[str, object]]:
    fold_train, fold_valid = _fold_frames(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        fold=fold,
        logger=logger,
    )
    logger.debug(
        "XGBoost fold frames ready: trial=%s fold=%s train_rows=%s valid_rows=%s feature_count=%s",
        trial_number,
        int(cast(Any, fold["fold"])),
        len(fold_train),
        len(fold_valid),
        len(feature_cols),
    )
    model, raw_predictions = fit_xgboost_model_and_predict_validation(
        fold_train,
        fold_valid,
        feature_cols,
        target_col=target_contract.learning_target_col,
        model_params=resolved_params,
        train_sample_weight=_sample_weights(fold_train)
        if bool(resolved_params.get("enable_dataset_sample_weight", False))
        else None,
    )
    predictions = reconstruct_absolute_predictions(
        raw_predictions,
        fold_valid,
        target_contract,
    )
    return _score_predictions_by_dataset(
        frame=fold_valid,
        predictions=predictions,
        absolute_target_col=target_contract.absolute_target_col,
        fold_number=int(cast(Any, fold["fold"])),
        feature_cols=feature_cols,
        selected_n_estimators=_selected_n_estimators(model.model),
        native_best_score=_native_best_score(model.model),
    )


def _selected_n_estimators(model: object) -> int | None:
    try:
        best_iteration = getattr(model, "best_iteration", None)
    except AttributeError:
        return None
    if isinstance(best_iteration, bool) or not isinstance(best_iteration, int):
        return None
    return best_iteration + 1


def _native_best_score(model: object) -> float | None:
    try:
        best_score = getattr(model, "best_score", None)
    except AttributeError:
        return None
    if isinstance(best_score, bool) or not isinstance(best_score, (int, float)):
        return None
    return float(best_score)


def _fold_frames(
    *,
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    fold: dict[str, object],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_number = int(cast(Any, fold["fold"]))
    train_idx = np.sort(np.asarray(fold["train_idx"], dtype=np.int32))
    valid_idx = np.sort(np.asarray(fold["valid_idx"], dtype=np.int32))
    train_extension = filter_training_eligible_rows(
        tuning_frame.iloc[train_idx].copy(),
        label=f"xgboost_optuna_fold_{fold_number}:tuning_train_extension",
        logger=logger,
    )
    fold_train = pd.concat([train_frame, train_extension], ignore_index=True)
    fold_valid = tuning_frame.iloc[valid_idx].copy().reset_index(drop=True)
    return fold_train.reset_index(drop=True), fold_valid


def _score_predictions_by_dataset(
    *,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    absolute_target_col: str,
    fold_number: int,
    feature_cols: list[str],
    selected_n_estimators: int | None,
    native_best_score: float | None,
) -> list[dict[str, object]]:
    scored = frame[[DEFAULT_DATASET_SOURCE_COL, absolute_target_col]].copy()
    scored["prediction"] = predictions
    rows: list[dict[str, object]] = []
    for dataset_source, dataset_frame in scored.groupby(
        DEFAULT_DATASET_SOURCE_COL, sort=False
    ):
        target = dataset_frame[absolute_target_col].astype(float)
        prediction = dataset_frame["prediction"].astype(float)
        bias = float((prediction - target).mean())
        rows.append(
            {
                "dataset_source": str(dataset_source),
                "fold": fold_number,
                "wape": float(compute_wape(target, prediction)),
                "bias": bias,
                "abs_bias": abs(bias),
                "coverage_80": None,
                "coverage_95": None,
                "pinball_loss_p50": None,
                "feature_count": int(len(feature_cols)),
                "rows_scored": int(len(dataset_frame)),
                "selected_n_estimators": selected_n_estimators,
                "native_best_score": native_best_score,
            }
        )
    return rows


def _dataset_weight_by_source(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame[DEFAULT_DATASET_SOURCE_COL].value_counts(sort=False)
    if counts.empty:
        raise ValueError("Cannot train XGBoost without dataset source weights.")
    dataset_total_weight = 1.0 / float(len(counts))
    return {
        str(dataset_source): dataset_total_weight / float(row_count)
        for dataset_source, row_count in counts.items()
    }


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    weight_by_source = _dataset_weight_by_source(frame)
    return (
        pd.Series(frame[DEFAULT_DATASET_SOURCE_COL], copy=False)
        .astype("string")
        .map(weight_by_source)
        .fillna(0.0)
        .to_numpy(dtype=np.float64, copy=True)
    )


def _dataset_macro_scores(fold_results: list[dict[str, object]]) -> dict[str, float]:
    dataset_scores: dict[str, float] = {}
    for dataset_source in sorted({str(row["dataset_source"]) for row in fold_results}):
        scores = [
            float(cast(Any, row["wape"]))
            for row in fold_results
            if str(row["dataset_source"]) == dataset_source
        ]
        dataset_scores[dataset_source] = float(np.mean(scores))
    return dataset_scores


def _mean_metric(
    fold_results: list[dict[str, object]],
    *,
    metric_key: str,
) -> float | None:
    values = [
        float(cast(Any, row[metric_key]))
        for row in fold_results
        if row.get(metric_key) is not None
    ]
    return float(np.mean(values)) if values else None


__all__ = [
    "XGBoostFoldMatrixCache",
    "build_xgboost_fold_matrix_cache",
    "fit_and_score_xgboost_model_on_tuning",
    "xgboost_execution_policy",
]
