from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_metrics import (
    REFERENCE_DATE_COL,
    REFERENCE_PRODUCT_COL,
    build_predictions_frame,
    rmse_score,
)
from praedixa.demand_forecast.evaluation.folds import build_daily_walk_forward_folds
from praedixa.demand_forecast.evaluation.modeling import (
    fit_evaluation_model,
    predict_absolute,
    predict_absolute_quantiles,
)
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.training.validation.eligibility import filter_training_eligible_rows

_PREDICTION_METADATA_COLS: tuple[str, ...] = (
    "dataset_source",
    "target_semantics",
    "censor_flag",
    "label_quality_score",
    "usable_for_training_flag",
)
_DIAGNOSTIC_FEATURE_NULL_LIMIT = 12


@dataclass(frozen=True)
class RefitExecutionContext:
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_cols: list[str]
    target_contract: TargetContract
    model_params: dict[str, object]
    logger: logging.Logger
    fit_model_fn: Callable[..., tuple[object, int]]
    predict_absolute_fn: Callable[..., np.ndarray]
    predict_quantiles_absolute_fn: Callable[..., pd.DataFrame]
    base_train_rows: int
    base_valid_rows: int


def evaluate_daily_refit_predictions(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    logger: logging.Logger,
    fit_model_fn: Callable[..., tuple[object, int]] = fit_evaluation_model,
    predict_absolute_fn: Callable[..., np.ndarray] = predict_absolute,
    predict_quantiles_absolute_fn: Callable[..., pd.DataFrame] = predict_absolute_quantiles,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    folds = build_daily_walk_forward_folds(test_frame, date_col=REFERENCE_DATE_COL)
    logger.info(
        "Daily evaluation refit starting: test_days=%s base_train_rows=%s valid_rows=%s "
        "test_rows=%s feature_count=%s target_col=%s",
        len(folds),
        len(train_frame),
        len(valid_frame),
        len(test_frame),
        len(feature_cols),
        target_contract.learning_target_col,
    )
    logger.debug(
        "Daily evaluation refit starting: folds=%s base_train_rows=%s valid_rows=%s "
        "test_rows=%s feature_count=%s target_col=%s feature_cols=%s",
        len(folds),
        len(train_frame),
        len(valid_frame),
        len(test_frame),
        len(feature_cols),
        target_contract.learning_target_col,
        feature_cols,
    )
    daily_prediction_parts: list[pd.DataFrame] = []
    daily_reports: list[dict[str, object]] = []
    best_iterations: list[int] = []
    eligible_train_frame = _filter_base_refit_train_frame(
        train_frame,
        logger=logger,
    )
    context = RefitExecutionContext(
        train_frame=eligible_train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=model_params,
        logger=logger,
        fit_model_fn=fit_model_fn,
        predict_absolute_fn=predict_absolute_fn,
        predict_quantiles_absolute_fn=predict_quantiles_absolute_fn,
        base_train_rows=int(len(train_frame)),
        base_valid_rows=int(len(valid_frame)),
    )

    for fold in folds:
        fold_predictions, fold_report, best_iteration = _run_refit_fold(
            fold=fold,
            total_fold_count=len(folds),
            context=context,
        )
        daily_prediction_parts.append(fold_predictions)
        daily_reports.append(fold_report)
        best_iterations.append(best_iteration)
    return _refit_outputs(daily_prediction_parts, daily_reports, best_iterations)


def _run_refit_fold(
    *,
    fold: dict[str, object],
    total_fold_count: int,
    context: RefitExecutionContext,
) -> tuple[pd.DataFrame, dict[str, object], int]:
    eval_date = str(fold["eval_date"])
    bakery_history_frame, fold_test_frame, known_history_rows = _refit_fold_frames(
        fold=fold,
        test_frame=context.test_frame,
        base_train_rows=context.base_train_rows,
        base_valid_rows=context.base_valid_rows,
    )
    context.logger.debug(
        "Daily evaluation refit fold entering model fit: fold=%s/%s eval_date=%s "
        "bakery_history_rows=%s fold_test_rows=%s known_history_rows=%s",
        int(cast(Any, fold["fold"])),
        total_fold_count,
        eval_date,
        len(bakery_history_frame),
        len(fold_test_frame),
        known_history_rows,
    )
    context.logger.info(
        "Daily evaluation test-day progress: fold=%s/%s eval_date=%s status=starting "
        "bakery_history_rows=%s test_rows=%s known_history_rows=%s",
        int(cast(Any, fold["fold"])),
        total_fold_count,
        eval_date,
        len(bakery_history_frame),
        len(fold_test_frame),
        known_history_rows,
    )
    model, best_iteration = _refit_model_and_iteration(
        fold=fold,
        total_fold_count=total_fold_count,
        bakery_history_frame=bakery_history_frame,
        context=context,
    )
    context.logger.debug(
        "Daily evaluation refit fold model fit returned: fold=%s/%s eval_date=%s best_iteration=%s",
        int(cast(Any, fold["fold"])),
        total_fold_count,
        eval_date,
        best_iteration,
    )
    fold_predictions = _build_fold_predictions(
        model=model,
        fold_test_frame=fold_test_frame,
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        known_history_rows=known_history_rows,
        predict_absolute_fn=context.predict_absolute_fn,
        predict_quantiles_absolute_fn=context.predict_quantiles_absolute_fn,
    )
    return _finalize_refit_fold(
        fold=fold,
        total_fold_count=total_fold_count,
        context=context,
        eval_date=eval_date,
        bakery_history_frame=bakery_history_frame,
        fold_test_frame=fold_test_frame,
        fold_predictions=fold_predictions,
        best_iteration=best_iteration,
        known_history_rows=known_history_rows,
    )


def _refit_model_and_iteration(
    *,
    fold: dict[str, object],
    total_fold_count: int,
    bakery_history_frame: pd.DataFrame,
    context: RefitExecutionContext,
) -> tuple[object, int]:
    return _fit_refit_fold_model(
        fold=fold,
        total_fold_count=total_fold_count,
        logger=context.logger,
        fit_model_fn=context.fit_model_fn,
        train_frame=context.train_frame,
        valid_frame=context.valid_frame,
        bakery_history_frame=bakery_history_frame,
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        model_params=context.model_params,
    )


def _filter_base_refit_train_frame(
    train_frame: pd.DataFrame,
    *,
    logger: logging.Logger,
) -> pd.DataFrame:
    return filter_training_eligible_rows(
        train_frame,
        label="evaluation_base_refit_train",
        logger=logger,
        log_level=logging.DEBUG,
    )


def _finalize_refit_fold(
    *,
    fold: dict[str, object],
    total_fold_count: int,
    context: RefitExecutionContext,
    eval_date: str,
    bakery_history_frame: pd.DataFrame,
    fold_test_frame: pd.DataFrame,
    fold_predictions: pd.DataFrame,
    best_iteration: int,
    known_history_rows: int,
) -> tuple[pd.DataFrame, dict[str, object], int]:
    fold_report = _daily_refit_report_row(
        fold=fold,
        eval_date=eval_date,
        bakery_history_frame=bakery_history_frame,
        fold_predictions=fold_predictions,
        fold_test_frame=fold_test_frame,
        absolute_target_col=context.target_contract.absolute_target_col,
        best_iteration=best_iteration,
        known_history_rows=known_history_rows,
    )
    _log_refit_fold(
        logger=context.logger,
        fold_number=int(cast(Any, fold["fold"])),
        total_fold_count=total_fold_count,
        eval_date=eval_date,
        known_history_rows=known_history_rows,
        valid_rows=len(fold_predictions),
        best_iteration=best_iteration,
        mae=cast(float, fold_report["mae"]),
        rmse=cast(float, fold_report["rmse"]),
    )
    return fold_predictions, fold_report, best_iteration


def _build_fold_predictions(
    *,
    model: object,
    fold_test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    known_history_rows: int,
    predict_absolute_fn: Callable[..., np.ndarray],
    predict_quantiles_absolute_fn: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    reference_cols = [
        REFERENCE_DATE_COL,
        REFERENCE_PRODUCT_COL,
        *[
            column
            for column in _PREDICTION_METADATA_COLS
            if column in fold_test_frame.columns
        ],
    ]
    fold_reference = fold_test_frame.loc[:, reference_cols].copy()
    absolute_quantiles, absolute_predictions = _fold_absolute_predictions(
        model=model,
        fold_test_frame=fold_test_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        predict_absolute_fn=predict_absolute_fn,
        predict_quantiles_absolute_fn=predict_quantiles_absolute_fn,
    )
    return build_predictions_frame(
        fold_reference,
        actual=fold_test_frame[target_contract.absolute_target_col],
        prediction_raw=absolute_predictions,
        train_rows_used=known_history_rows,
        quantile_predictions=absolute_quantiles,
    )


def _log_refit_fold(
    *,
    logger: logging.Logger,
    fold_number: int,
    total_fold_count: int,
    eval_date: str,
    known_history_rows: int,
    valid_rows: int,
    best_iteration: int,
    mae: float,
    rmse: float,
) -> None:
    logger.info(
        "Daily evaluation test-day progress: fold=%s/%s eval_date=%s status=completed "
        "history_rows=%s test_rows=%s best_iteration=%s mae=%.6f rmse=%.6f",
        fold_number,
        total_fold_count,
        eval_date,
        known_history_rows,
        valid_rows,
        best_iteration,
        mae,
        rmse,
    )


def _refit_outputs(
    daily_prediction_parts: list[pd.DataFrame],
    daily_reports: list[dict[str, object]],
    best_iterations: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    predictions_df = pd.concat(daily_prediction_parts, ignore_index=True)
    predictions_df = predictions_df.sort_values(["product", "target_date"]).reset_index(drop=True)
    daily_report_df = pd.DataFrame(daily_reports).sort_values("fold").reset_index(drop=True)
    resolved_best_iteration = int(round(float(np.mean(best_iterations)))) if best_iterations else 2000
    return predictions_df, daily_report_df, resolved_best_iteration


def _refit_fold_frames(
    *,
    fold: dict[str, object],
    test_frame: pd.DataFrame,
    base_train_rows: int,
    base_valid_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    history_idx = np.asarray(fold["history_idx"], dtype=np.int32)
    valid_idx = np.asarray(fold["valid_idx"], dtype=np.int32)
    bakery_history_frame = test_frame.iloc[history_idx].copy()
    fold_test_frame = test_frame.iloc[valid_idx].copy()
    known_history_rows = int(base_train_rows + base_valid_rows + len(bakery_history_frame))
    return bakery_history_frame, fold_test_frame, known_history_rows


def _fit_refit_fold_model(
    *,
    fold: dict[str, object],
    total_fold_count: int,
    logger: logging.Logger,
    fit_model_fn: Callable[..., tuple[object, int]],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    bakery_history_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[object, int]:
    fold_number = int(cast(Any, fold["fold"]))
    eval_date = str(fold["eval_date"])
    logger.debug(
        "Daily evaluation refit assembling training frame: fold=%s/%s eval_date=%s "
        "base_train_rows=%s bakery_history_rows=%s valid_rows=%s",
        fold_number,
        total_fold_count,
        eval_date,
        len(train_frame),
        len(bakery_history_frame),
        len(valid_frame),
    )
    aligned_history_frame = bakery_history_frame.reindex(columns=train_frame.columns)
    eligible_history_frame = filter_training_eligible_rows(
        aligned_history_frame,
        label="evaluation_daily_refit_history",
        logger=logger,
        log_level=logging.DEBUG,
    )
    if eligible_history_frame.empty:
        refit_train_frame = train_frame
    else:
        refit_train_frame = pd.concat(
            [train_frame, eligible_history_frame],
            ignore_index=True,
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Daily evaluation refit frame after incremental eligibility: fold=%s/%s eval_date=%s %s",
            fold_number,
            total_fold_count,
            eval_date,
            _frame_diagnostic_summary(
                refit_train_frame,
                feature_cols=feature_cols,
                target_col=target_contract.learning_target_col,
            ),
        )
    logger.debug(
        "Daily evaluation refit calling backend fit: fold=%s/%s eval_date=%s "
        "train_rows=%s valid_rows=%s feature_count=%s",
        fold_number,
        total_fold_count,
        eval_date,
        len(refit_train_frame),
        len(valid_frame),
        len(feature_cols),
    )
    return fit_model_fn(
        train_frame=refit_train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=model_params,
    )


def _frame_diagnostic_summary(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, object]:
    present_features = [column for column in feature_cols if column in frame.columns]
    missing_features = [column for column in feature_cols if column not in frame.columns]
    null_counts = frame[present_features].isna().sum()
    nonzero_null_counts = {
        str(column): int(count)
        for column, count in null_counts.sort_values(ascending=False).items()
        if int(count) > 0
    }
    dtype_counts = frame[present_features].dtypes.astype(str).value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "cols": int(len(frame.columns)),
        "feature_count": int(len(present_features)),
        "missing_features": missing_features,
        "target_nulls": int(frame[target_col].isna().sum()) if target_col in frame.columns else None,
        "feature_dtype_counts": {str(key): int(value) for key, value in dtype_counts.items()},
        "feature_null_counts": dict(
            list(nonzero_null_counts.items())[:_DIAGNOSTIC_FEATURE_NULL_LIMIT]
        ),
    }


def _daily_refit_report_row(
    *,
    fold: dict[str, object],
    eval_date: str,
    bakery_history_frame: pd.DataFrame,
    fold_predictions: pd.DataFrame,
    fold_test_frame: pd.DataFrame,
    absolute_target_col: str,
    best_iteration: int,
    known_history_rows: int,
) -> dict[str, object]:
    absolute_target = fold_test_frame[absolute_target_col].astype(float).to_numpy()
    prediction_values = fold_predictions["prediction_raw"].astype(float).to_numpy()
    return {
        "fold": int(cast(Any, fold["fold"])),
        "eval_date": eval_date,
        "history_rows": known_history_rows,
        "bakery_history_rows": int(len(bakery_history_frame)),
        "valid_rows": int(len(fold_predictions)),
        "best_iteration": int(best_iteration),
        "mae": float(np.mean(np.abs(absolute_target - prediction_values))),
        "rmse": rmse_score(absolute_target, prediction_values),
    }


def _fold_absolute_predictions(
    *,
    model: object,
    fold_test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    predict_absolute_fn: Callable[..., np.ndarray],
    predict_quantiles_absolute_fn: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame | None, np.ndarray]:
    supports_quantile_predictions = hasattr(model, "quantiles") and hasattr(model, "target_col")
    absolute_quantiles = (
        predict_quantiles_absolute_fn(
            model,
            fold_test_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
        )
        if supports_quantile_predictions
        else None
    )
    if absolute_quantiles is not None and "prediction_p50" in absolute_quantiles.columns:
        return absolute_quantiles, absolute_quantiles["prediction_p50"].to_numpy(dtype=float)
    return (
        absolute_quantiles,
        predict_absolute_fn(
            model,
            fold_test_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
        ),
    )
