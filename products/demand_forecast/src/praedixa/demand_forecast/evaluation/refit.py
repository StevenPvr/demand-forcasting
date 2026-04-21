from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.bakery_metrics import REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL, build_predictions_frame
from praedixa.demand_forecast.evaluation.folds import build_daily_walk_forward_folds
from praedixa.demand_forecast.evaluation.modeling import fit_evaluation_model, predict_absolute, predict_absolute_quantiles
from praedixa.demand_forecast.contracts.targets import TargetContract


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
    daily_prediction_parts: list[pd.DataFrame] = []
    daily_reports: list[dict[str, object]] = []
    best_iterations: list[int] = []
    context = RefitExecutionContext(
        train_frame=train_frame,
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
    model, best_iteration = _refit_model_and_iteration(bakery_history_frame, context)
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
    bakery_history_frame: pd.DataFrame,
    context: RefitExecutionContext,
) -> tuple[object, int]:
    return _fit_refit_fold_model(
        fit_model_fn=context.fit_model_fn,
        train_frame=context.train_frame,
        valid_frame=context.valid_frame,
        bakery_history_frame=bakery_history_frame,
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        model_params=context.model_params,
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
    fold_reference = fold_test_frame.loc[:, [REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL]].copy()
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
) -> None:
    logger.info(
        "Daily evaluation refit complete: fold=%s/%s eval_date=%s history_rows=%s valid_rows=%s best_iteration=%s",
        fold_number,
        total_fold_count,
        eval_date,
        known_history_rows,
        valid_rows,
        best_iteration,
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
    fit_model_fn: Callable[..., tuple[object, int]],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    bakery_history_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[object, int]:
    refit_train_frame = pd.concat([train_frame, bakery_history_frame], ignore_index=True)
    return fit_model_fn(
        train_frame=refit_train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=model_params,
    )


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
        "rmse": float(np.sqrt(np.mean(np.square(absolute_target - prediction_values)))),
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
