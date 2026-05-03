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
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATASET_SOURCE_COL,
)
from praedixa.demand_forecast.training.shared.economic_objective import (
    ECONOMIC_SEGMENT_METADATA_COLS,
    apply_economic_decision_calibration_to_predictions,
)
from praedixa.demand_forecast.training.validation.eligibility import (
    filter_training_eligible_rows,
)

_PREDICTION_METADATA_COLS: tuple[str, ...] = (
    "dataset_source",
    "target_semantics",
    "censor_flag",
    "label_quality_score",
    "usable_for_training_flag",
    *ECONOMIC_SEGMENT_METADATA_COLS,
)
_DIAGNOSTIC_FEATURE_NULL_LIMIT = 12
_CALIBRATED_QUANTILE_COLS: tuple[str, ...] = (
    "prediction_p2_5",
    "prediction_p10",
    "prediction_p50",
    "prediction_p90",
    "prediction_p97_5",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
)
_CALIBRATION_ROW_ORDER_COL = "__praedixa_calibration_row_order"
_DEFAULT_CALIBRATION_PROGRESS_SERIES_BATCH_SIZE = 512
_BAKERY_DATASET_SOURCE = "bakery"
_DEFAULT_DAILY_REFIT_BAKERY_WEIGHT_MULTIPLIER = 1.0
_DEFAULT_DAILY_REFIT_BAKERY_MIN_WEIGHT_MULTIPLIER = 5.0
_DEFAULT_DAILY_REFIT_BAKERY_WEIGHT_WARMUP_DAYS = 28
_DEFAULT_DAILY_REFIT_BAKERY_MIN_EFFECTIVE_SHARE = 0.50
_DEFAULT_DAILY_REFIT_BAKERY_MAX_EFFECTIVE_SHARE = 0.99
_BUSINESS_SAMPLE_WEIGHT_COL = "sample_weight_business"
_BAKERY_REFIT_SEED_FLAG_COL = "bakery_refit_seed_flag"


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
    base_bakery_history_rows: int
    base_bakery_history_days: int
    base_bakery_pretest_seed_rows: int
    base_bakery_pretest_seed_days: int


@dataclass
class PredictionCalibrationState:
    enabled: bool
    initialize_from_valid: bool
    min_rows: int
    slope_min: float
    slope_max: float
    progress_series_batch_size: int
    raw_predictions: list[float]
    actuals: list[float]


@dataclass(frozen=True)
class PredictionCalibrationParams:
    slope: float
    intercept: float
    row_count: int
    applied: bool


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
    predict_quantiles_absolute_fn: Callable[
        ..., pd.DataFrame
    ] = predict_absolute_quantiles,
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
    base_bakery_history_rows = _bakery_row_count(eligible_train_frame)
    base_bakery_history_days = _bakery_day_count(eligible_train_frame)
    base_bakery_mask = _bakery_mask(eligible_train_frame)
    base_bakery_pretest_seed_rows = _bakery_pretest_seed_row_count(
        eligible_train_frame,
        base_bakery_mask,
    )
    base_bakery_pretest_seed_days = _bakery_pretest_seed_day_count(
        eligible_train_frame,
        base_bakery_mask,
    )
    logger.info(
        "Daily evaluation refit base bakery history available before first prediction: "
        "rows=%s days=%s eligible_bakery_pretest_seed_rows=%s "
        "eligible_bakery_pretest_seed_days=%s",
        base_bakery_history_rows,
        base_bakery_history_days,
        base_bakery_pretest_seed_rows,
        base_bakery_pretest_seed_days,
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
        base_bakery_history_rows=base_bakery_history_rows,
        base_bakery_history_days=base_bakery_history_days,
        base_bakery_pretest_seed_rows=base_bakery_pretest_seed_rows,
        base_bakery_pretest_seed_days=base_bakery_pretest_seed_days,
    )
    calibration_state = _prediction_calibration_state(model_params)
    _initialize_prediction_calibration_history(
        context,
        calibration_state=calibration_state,
    )

    for fold in folds:
        fold_predictions, fold_report, best_iteration = _run_refit_fold(
            fold=fold,
            total_fold_count=len(folds),
            context=context,
            calibration_state=calibration_state,
        )
        daily_prediction_parts.append(fold_predictions)
        daily_reports.append(fold_report)
        best_iterations.append(best_iteration)
        _update_prediction_calibration_history(
            calibration_state,
            fold_predictions=fold_predictions,
        )
    return _refit_outputs(daily_prediction_parts, daily_reports, best_iterations)


def _run_refit_fold(
    *,
    fold: dict[str, object],
    total_fold_count: int,
    context: RefitExecutionContext,
    calibration_state: PredictionCalibrationState,
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
        "bakery_history_rows=%s bakery_pretest_history_rows=%s "
        "bakery_total_history_rows=%s fold_test_rows=%s known_history_rows=%s",
        int(cast(Any, fold["fold"])),
        total_fold_count,
        eval_date,
        len(bakery_history_frame),
        context.base_bakery_history_rows,
        context.base_bakery_history_rows + len(bakery_history_frame),
        len(fold_test_frame),
        known_history_rows,
    )
    context.logger.info(
        "Daily evaluation test-day progress: fold=%s/%s eval_date=%s status=starting "
        "bakery_history_rows=%s bakery_pretest_history_rows=%s "
        "bakery_total_history_rows=%s test_rows=%s known_history_rows=%s",
        int(cast(Any, fold["fold"])),
        total_fold_count,
        eval_date,
        len(bakery_history_frame),
        context.base_bakery_history_rows,
        context.base_bakery_history_rows + len(bakery_history_frame),
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
    fold_predictions = _apply_prediction_calibration(
        fold_predictions,
        calibration_state=calibration_state,
    )
    fold_predictions = _apply_economic_decision_calibration(
        fold_predictions,
        model_params=context.model_params,
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


def _bakery_row_count(frame: pd.DataFrame) -> int:
    return int(_bakery_mask(frame).sum())


def _bakery_day_count(frame: pd.DataFrame) -> int:
    bakery_mask = _bakery_mask(frame)
    if not bool(bakery_mask.any()):
        return 0
    date_col = REFERENCE_DATE_COL if REFERENCE_DATE_COL in frame.columns else "dt"
    if date_col not in frame.columns:
        return 1
    bakery_dates = pd.to_datetime(frame.loc[bakery_mask, date_col], errors="coerce").dropna()
    return int(bakery_dates.dt.normalize().nunique())


def _bakery_mask(frame: pd.DataFrame) -> pd.Series:
    if DEFAULT_DATASET_SOURCE_COL not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[DEFAULT_DATASET_SOURCE_COL].astype(str) == _BAKERY_DATASET_SOURCE


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
        bakery_pretest_seed_rows=context.base_bakery_pretest_seed_rows,
        bakery_pretest_seed_days=context.base_bakery_pretest_seed_days,
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
        calibration_applied=cast(bool, fold_report["prediction_calibration_applied"]),
        calibration_rows=cast(int, fold_report["prediction_calibration_rows"]),
        calibration_slope=cast(float, fold_report["prediction_calibration_slope"]),
        calibration_intercept=cast(
            float, fold_report["prediction_calibration_intercept"]
        ),
        economic_calibration_scope=cast(
            str,
            fold_report.get("economic_calibration_scope", "none"),
        ),
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
    fold_test_frame = _with_reference_prediction_columns(fold_test_frame)
    reference_cols = [
        REFERENCE_DATE_COL,
        REFERENCE_PRODUCT_COL,
        *[
            column
            for column in _PREDICTION_METADATA_COLS
            if column in fold_test_frame.columns
            and column not in {REFERENCE_DATE_COL, REFERENCE_PRODUCT_COL}
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


def _with_reference_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if REFERENCE_DATE_COL in frame.columns and REFERENCE_PRODUCT_COL in frame.columns:
        return frame
    output = frame.copy()
    if REFERENCE_DATE_COL not in output.columns:
        if "dt" not in output.columns:
            raise KeyError(REFERENCE_DATE_COL)
        output[REFERENCE_DATE_COL] = output["dt"].to_numpy()
    if REFERENCE_PRODUCT_COL not in output.columns:
        if "product_id" not in output.columns:
            raise KeyError(REFERENCE_PRODUCT_COL)
        output[REFERENCE_PRODUCT_COL] = output["product_id"].astype(str).to_numpy()
    return output


def _prediction_calibration_state(
    model_params: dict[str, object],
) -> PredictionCalibrationState:
    return PredictionCalibrationState(
        enabled=_bool_param(
            model_params.get("enable_daily_prediction_calibration"), default=False
        ),
        initialize_from_valid=_bool_param(
            model_params.get("initialize_daily_prediction_calibration_from_valid"),
            default=True,
        ),
        min_rows=max(
            _int_param(
                model_params.get("daily_prediction_calibration_min_rows"), default=35
            ),
            2,
        ),
        slope_min=max(
            _float_param(
                model_params.get("daily_prediction_calibration_slope_min"), default=0.25
            ),
            0.0,
        ),
        slope_max=max(
            _float_param(
                model_params.get("daily_prediction_calibration_slope_max"), default=2.0
            ),
            0.0,
        ),
        progress_series_batch_size=max(
            _int_param(
                model_params.get(
                    "daily_prediction_calibration_progress_series_batch_size"
                ),
                default=_DEFAULT_CALIBRATION_PROGRESS_SERIES_BATCH_SIZE,
            ),
            1,
        ),
        raw_predictions=[],
        actuals=[],
    )


def _initialize_prediction_calibration_history(
    context: RefitExecutionContext,
    *,
    calibration_state: PredictionCalibrationState,
) -> None:
    if (
        not calibration_state.enabled
        or not calibration_state.initialize_from_valid
        or context.valid_frame.empty
    ):
        return
    context.logger.info(
        "Daily prediction calibration initialization starting: base_train_rows=%s "
        "valid_rows=%s min_rows=%s slope_bounds=[%.3f, %.3f]",
        len(context.train_frame),
        len(context.valid_frame),
        calibration_state.min_rows,
        calibration_state.slope_min,
        calibration_state.slope_max,
    )
    calibration_model, _ = context.fit_model_fn(
        train_frame=context.train_frame,
        valid_frame=context.valid_frame.head(0),
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        model_params=context.model_params,
    )
    calibration_frame = _frame_with_known_model_series(
        context.valid_frame,
        model=calibration_model,
    )
    if calibration_frame.empty:
        context.logger.warning(
            "Daily prediction calibration skipped: validation history has no series present in model context."
        )
        return
    context.logger.info(
        "Daily prediction calibration validation frame resolved: valid_rows=%s "
        "eligible_rows=%s eligible_series=%s",
        len(context.valid_frame),
        len(calibration_frame),
        int(_prediction_series_ids(calibration_frame).nunique()),
    )
    calibration_predictions = _build_calibration_predictions_with_progress(
        context=context,
        model=calibration_model,
        calibration_frame=calibration_frame,
        calibration_state=calibration_state,
    )
    _update_prediction_calibration_history(
        calibration_state,
        fold_predictions=calibration_predictions,
    )
    calibration = _fit_prediction_calibration(calibration_state)
    context.logger.info(
        "Daily prediction calibration initialized from validation history: rows=%s "
        "min_rows=%s applied=%s slope=%.6f intercept=%.6f",
        len(calibration_state.actuals),
        calibration_state.min_rows,
        calibration.applied,
        calibration.slope,
        calibration.intercept,
    )


def _build_calibration_predictions_with_progress(
    *,
    context: RefitExecutionContext,
    model: object,
    calibration_frame: pd.DataFrame,
    calibration_state: PredictionCalibrationState,
) -> pd.DataFrame:
    frame = calibration_frame.copy()
    frame[_CALIBRATION_ROW_ORDER_COL] = np.arange(len(frame), dtype=np.int64)
    chunks = _calibration_series_chunks(
        frame,
        series_batch_size=calibration_state.progress_series_batch_size,
    )
    if len(chunks) <= 1:
        predictions = _build_calibration_chunk_predictions(
            context=context,
            model=model,
            calibration_chunk=frame,
        )
        return predictions.drop(columns=[_CALIBRATION_ROW_ORDER_COL])
    context.logger.info(
        "Daily prediction calibration validation prediction starting: rows=%s "
        "series=%s batches=%s series_per_batch=%s",
        len(frame),
        int(_prediction_series_ids(frame).nunique()),
        len(chunks),
        calibration_state.progress_series_batch_size,
    )
    prediction_parts: list[pd.DataFrame] = []
    for chunk in _progress_calibration_chunks(
        chunks,
        description="prediction calibration valid",
    ):
        prediction_parts.append(
            _build_calibration_chunk_predictions(
                context=context,
                model=model,
                calibration_chunk=chunk,
            )
        )
    return (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(_CALIBRATION_ROW_ORDER_COL, kind="mergesort")
        .drop(columns=[_CALIBRATION_ROW_ORDER_COL])
        .reset_index(drop=True)
    )


def _build_calibration_chunk_predictions(
    *,
    context: RefitExecutionContext,
    model: object,
    calibration_chunk: pd.DataFrame,
) -> pd.DataFrame:
    row_order = calibration_chunk[_CALIBRATION_ROW_ORDER_COL].to_numpy(dtype=np.int64)
    fold_test_frame = calibration_chunk.drop(columns=[_CALIBRATION_ROW_ORDER_COL])
    predictions = _build_fold_predictions(
        model=model,
        fold_test_frame=fold_test_frame,
        feature_cols=context.feature_cols,
        target_contract=context.target_contract,
        known_history_rows=context.base_train_rows,
        predict_absolute_fn=context.predict_absolute_fn,
        predict_quantiles_absolute_fn=context.predict_quantiles_absolute_fn,
    )
    predictions[_CALIBRATION_ROW_ORDER_COL] = row_order
    return predictions


def _calibration_series_chunks(
    frame: pd.DataFrame,
    *,
    series_batch_size: int,
) -> list[pd.DataFrame]:
    series_ids = _prediction_series_ids(frame).astype(str)
    ordered_series = series_ids.drop_duplicates().tolist()
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(ordered_series), series_batch_size):
        batch_series = set(ordered_series[start : start + series_batch_size])
        chunks.append(frame.loc[series_ids.isin(batch_series)].copy())
    return chunks


def _progress_calibration_chunks(
    chunks: list[pd.DataFrame],
    *,
    description: str,
) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return chunks
    return tqdm(
        chunks,
        desc=description,
        unit="batch",
        total=len(chunks),
        mininterval=1.0,
    )


def _frame_with_known_model_series(
    frame: pd.DataFrame, *, model: object
) -> pd.DataFrame:
    series_order = getattr(model, "series_order", None)
    if not isinstance(series_order, list) or not series_order:
        return frame
    known_series = {str(series_id) for series_id in cast(list[object], series_order)}
    series_ids = _prediction_series_ids(frame)
    return frame.loc[series_ids.astype(str).isin(known_series)].copy()


def _prediction_series_ids(frame: pd.DataFrame) -> pd.Series:
    product_ids = _prediction_product_ids(frame)
    if "dataset_source" in frame.columns:
        bakery_mask = frame["dataset_source"].astype(str) == "bakery"
    else:
        bakery_mask = pd.Series(False, index=frame.index)
    if "client_id" not in frame.columns:
        return product_ids
    client_ids = frame["client_id"].astype("string")
    return client_ids.where(~bakery_mask & client_ids.notna(), product_ids).astype(str)


def _prediction_product_ids(frame: pd.DataFrame) -> pd.Series:
    if REFERENCE_PRODUCT_COL in frame.columns:
        return frame[REFERENCE_PRODUCT_COL].astype(str)
    if "product_id" in frame.columns:
        return frame["product_id"].astype(str)
    raise KeyError(REFERENCE_PRODUCT_COL)


def _apply_prediction_calibration(
    predictions_df: pd.DataFrame,
    *,
    calibration_state: PredictionCalibrationState,
) -> pd.DataFrame:
    if not calibration_state.enabled:
        return predictions_df
    calibration = _fit_prediction_calibration(calibration_state)
    output = predictions_df.copy()
    raw_predictions = output["prediction_raw"].astype(float).to_numpy()
    output["prediction_raw_uncalibrated"] = raw_predictions
    output["prediction_calibration_slope"] = calibration.slope
    output["prediction_calibration_intercept"] = calibration.intercept
    output["prediction_calibration_rows"] = calibration.row_count
    output["prediction_calibration_applied"] = calibration.applied
    if not calibration.applied:
        return output
    _transform_prediction_columns(output, calibration=calibration)
    return output


def _apply_economic_decision_calibration(
    predictions_df: pd.DataFrame,
    *,
    model_params: dict[str, object],
) -> pd.DataFrame:
    calibration = model_params.get("economic_calibration")
    if not isinstance(calibration, dict):
        return predictions_df
    return apply_economic_decision_calibration_to_predictions(
        predictions_df,
        calibration=cast(dict[str, object], calibration),
    )


def _fit_prediction_calibration(
    calibration_state: PredictionCalibrationState,
) -> PredictionCalibrationParams:
    row_count = len(calibration_state.actuals)
    if row_count < calibration_state.min_rows:
        return PredictionCalibrationParams(
            slope=1.0,
            intercept=0.0,
            row_count=row_count,
            applied=False,
        )
    raw_predictions = np.asarray(calibration_state.raw_predictions, dtype=np.float64)
    actuals = np.asarray(calibration_state.actuals, dtype=np.float64)
    finite_mask = np.isfinite(raw_predictions) & np.isfinite(actuals)
    if int(finite_mask.sum()) < calibration_state.min_rows:
        return PredictionCalibrationParams(
            slope=1.0,
            intercept=0.0,
            row_count=int(finite_mask.sum()),
            applied=False,
        )
    x_values = raw_predictions[finite_mask]
    y_values = actuals[finite_mask]
    slope, intercept = _least_squares_calibration(
        x_values,
        y_values,
        slope_min=calibration_state.slope_min,
        slope_max=calibration_state.slope_max,
    )
    return PredictionCalibrationParams(
        slope=slope,
        intercept=intercept,
        row_count=int(finite_mask.sum()),
        applied=True,
    )


def _least_squares_calibration(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    slope_min: float,
    slope_max: float,
) -> tuple[float, float]:
    if float(np.var(x_values)) <= 1e-12:
        return 1.0, float(np.mean(y_values - x_values))
    design = np.column_stack([x_values, np.ones_like(x_values)])
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    if not np.isfinite(slope) or not np.isfinite(intercept) or float(slope) <= 0.0:
        return 1.0, float(np.mean(y_values - x_values))
    clipped_slope = float(np.clip(float(slope), slope_min, slope_max))
    if clipped_slope != float(slope):
        intercept = float(np.mean(y_values - clipped_slope * x_values))
    return clipped_slope, float(intercept)


def _transform_prediction_columns(
    predictions_df: pd.DataFrame,
    *,
    calibration: PredictionCalibrationParams,
) -> None:
    for column in ("prediction_raw", *_CALIBRATED_QUANTILE_COLS):
        if column not in predictions_df.columns:
            continue
        transformed = _calibrated_values(
            predictions_df[column].astype(float).to_numpy(),
            calibration=calibration,
        )
        predictions_df[column] = transformed
    predictions_df["prediction_rounded"] = np.round(
        predictions_df["prediction_raw"].astype(float).to_numpy(),
        0,
    )


def _calibrated_values(
    values: np.ndarray,
    *,
    calibration: PredictionCalibrationParams,
) -> np.ndarray:
    return np.clip(
        values.astype(float) * calibration.slope + calibration.intercept,
        a_min=0.0,
        a_max=None,
    )


def _update_prediction_calibration_history(
    calibration_state: PredictionCalibrationState,
    *,
    fold_predictions: pd.DataFrame,
) -> None:
    if not calibration_state.enabled:
        return
    raw_col = (
        "prediction_raw_uncalibrated"
        if "prediction_raw_uncalibrated" in fold_predictions.columns
        else "prediction_raw"
    )
    calibration_state.raw_predictions.extend(
        fold_predictions[raw_col].astype(float).to_numpy().tolist()
    )
    calibration_state.actuals.extend(
        fold_predictions["actual"].astype(float).to_numpy().tolist()
    )


def _bool_param(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int_param(value: object, *, default: int) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _float_param(value: object, *, default: float) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


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
    calibration_applied: bool,
    calibration_rows: int,
    calibration_slope: float,
    calibration_intercept: float,
    economic_calibration_scope: str,
) -> None:
    logger.info(
        "Daily evaluation test-day progress: fold=%s/%s eval_date=%s status=completed "
        "history_rows=%s test_rows=%s best_iteration=%s mae=%.6f rmse=%.6f "
        "calibration_applied=%s calibration_rows=%s calibration_slope=%.6f "
        "calibration_intercept=%.6f calibration_scope=%s",
        fold_number,
        total_fold_count,
        eval_date,
        known_history_rows,
        valid_rows,
        best_iteration,
        mae,
        rmse,
        calibration_applied,
        calibration_rows,
        calibration_slope,
        calibration_intercept,
        economic_calibration_scope,
    )


def _refit_outputs(
    daily_prediction_parts: list[pd.DataFrame],
    daily_reports: list[dict[str, object]],
    best_iterations: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    predictions_df = pd.concat(daily_prediction_parts, ignore_index=True)
    predictions_df = predictions_df.sort_values(["product", "target_date"]).reset_index(
        drop=True
    )
    daily_report_df = (
        pd.DataFrame(daily_reports).sort_values("fold").reset_index(drop=True)
    )
    resolved_best_iteration = (
        int(round(float(np.mean(best_iterations)))) if best_iterations else 2000
    )
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
    known_history_rows = int(
        base_train_rows + base_valid_rows + len(bakery_history_frame)
    )
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
        refit_train_frame = _concat_refit_train_with_history(
            train_frame=train_frame,
            eligible_history_frame=eligible_history_frame,
        )
    refit_train_frame = _with_bakery_refit_business_weights(
        refit_train_frame,
        model_params=model_params,
        logger=logger,
        fold_number=fold_number,
        total_fold_count=total_fold_count,
        eval_date=eval_date,
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


def _concat_refit_train_with_history(
    *,
    train_frame: pd.DataFrame,
    eligible_history_frame: pd.DataFrame,
) -> pd.DataFrame:
    non_empty_history_cols = [
        column
        for column in eligible_history_frame.columns
        if not bool(eligible_history_frame[column].isna().all())
    ]
    history_for_concat = eligible_history_frame.loc[:, non_empty_history_cols].copy()
    concatenated_frame = pd.concat(
        [train_frame, history_for_concat],
        ignore_index=True,
    )
    return concatenated_frame.reindex(columns=train_frame.columns)


def _with_bakery_refit_business_weights(
    frame: pd.DataFrame,
    *,
    model_params: dict[str, object],
    logger: logging.Logger,
    fold_number: int,
    total_fold_count: int,
    eval_date: str,
) -> pd.DataFrame:
    if DEFAULT_DATASET_SOURCE_COL not in frame.columns:
        return frame
    multiplier = _float_param(
        model_params.get("daily_refit_bakery_sample_weight_multiplier"),
        default=_DEFAULT_DAILY_REFIT_BAKERY_WEIGHT_MULTIPLIER,
    )
    if multiplier <= 1.0:
        return frame
    bakery_mask = frame[DEFAULT_DATASET_SOURCE_COL].astype(str) == _BAKERY_DATASET_SOURCE
    bakery_rows = int(bakery_mask.sum())
    if bakery_rows == 0:
        return frame
    bakery_weight_history_days = _bakery_refit_weight_history_day_count(
        frame, bakery_mask
    )
    bakery_pretest_seed_days = _bakery_pretest_seed_day_count(frame, bakery_mask)
    weighted = frame.copy()
    base_weights = _base_business_weights(weighted)
    bakery_index = bakery_mask.to_numpy(dtype=bool)
    effective_multiplier, non_bakery_multiplier, target_bakery_share = (
        _bakery_refit_weight_multipliers(
            model_params,
            base_weights=base_weights,
            bakery_index=bakery_index,
            max_multiplier=multiplier,
            bakery_history_days=bakery_weight_history_days,
        )
    )
    base_weights[~bakery_index] *= non_bakery_multiplier
    base_weights[bakery_index] *= effective_multiplier
    effective_bakery_share = _effective_weight_share(base_weights, bakery_index)
    weighted[_BUSINESS_SAMPLE_WEIGHT_COL] = base_weights
    logger.info(
        "Daily evaluation refit bakery business weights applied: fold=%s/%s "
        "eval_date=%s bakery_rows=%s total_rows=%s bakery_multiplier=%.3f "
        "bakery_max_multiplier=%.3f bakery_weight_history_days=%s "
        "bakery_pretest_seed_days=%s target_bakery_share=%.6f "
        "effective_bakery_share=%.6f non_bakery_multiplier=%.6f "
        "weight_min=%.6f weight_mean=%.6f weight_max=%.6f",
        fold_number,
        total_fold_count,
        eval_date,
        bakery_rows,
        len(weighted),
        effective_multiplier,
        multiplier,
        bakery_weight_history_days,
        bakery_pretest_seed_days,
        target_bakery_share,
        effective_bakery_share,
        non_bakery_multiplier,
        float(base_weights.min()),
        float(base_weights.mean()),
        float(base_weights.max()),
    )
    return weighted


def _bakery_refit_weight_history_day_count(
    frame: pd.DataFrame, bakery_mask: pd.Series
) -> int:
    date_col = REFERENCE_DATE_COL if REFERENCE_DATE_COL in frame.columns else "dt"
    if date_col not in frame.columns:
        return 1
    bakery_dates = pd.to_datetime(
        frame.loc[bakery_mask, date_col],
        errors="coerce",
    ).dropna()
    return max(int(bakery_dates.dt.normalize().nunique()), 1)


def _bakery_pretest_seed_day_count(frame: pd.DataFrame, bakery_mask: pd.Series) -> int:
    if _BAKERY_REFIT_SEED_FLAG_COL not in frame.columns:
        return 0
    seed_mask = bakery_mask & frame[_BAKERY_REFIT_SEED_FLAG_COL].fillna(False).astype(bool)
    if not bool(seed_mask.any()):
        return 0
    date_col = REFERENCE_DATE_COL if REFERENCE_DATE_COL in frame.columns else "dt"
    if date_col not in frame.columns:
        return 1
    seed_dates = pd.to_datetime(frame.loc[seed_mask, date_col], errors="coerce").dropna()
    return int(seed_dates.dt.normalize().nunique())


def _bakery_pretest_seed_row_count(frame: pd.DataFrame, bakery_mask: pd.Series) -> int:
    if _BAKERY_REFIT_SEED_FLAG_COL not in frame.columns:
        return 0
    seed_mask = (
        bakery_mask
        & frame[_BAKERY_REFIT_SEED_FLAG_COL].fillna(False).astype(bool)
    )
    return int(seed_mask.sum())


def _bakery_refit_weight_multipliers(
    model_params: dict[str, object],
    *,
    base_weights: np.ndarray,
    bakery_index: np.ndarray,
    max_multiplier: float,
    bakery_history_days: int,
) -> tuple[float, float, float]:
    strategy = str(
        model_params.get(
            "daily_refit_bakery_weight_strategy",
            "progressive_multiplier",
        )
    )
    if strategy == "target_effective_share":
        target_share = _target_bakery_effective_share(
            model_params,
            bakery_history_days=bakery_history_days,
        )
        return (
            _bakery_multiplier_for_target_share(
                base_weights,
                bakery_index=bakery_index,
                target_share=target_share,
            ),
            1.0,
            target_share,
        )
    effective_multiplier = _effective_bakery_refit_weight_multiplier(
        model_params,
        max_multiplier=max_multiplier,
        bakery_history_days=bakery_history_days,
    )
    non_bakery_multiplier = _non_bakery_refit_weight_multiplier(
        model_params,
        effective_bakery_multiplier=effective_multiplier,
    )
    adjusted_weights = base_weights.copy()
    adjusted_weights[~bakery_index] *= non_bakery_multiplier
    adjusted_weights[bakery_index] *= effective_multiplier
    return (
        effective_multiplier,
        non_bakery_multiplier,
        _effective_weight_share(adjusted_weights, bakery_index),
    )


def _target_bakery_effective_share(
    model_params: dict[str, object],
    *,
    bakery_history_days: int,
) -> float:
    min_share = _bounded_share_param(
        model_params.get("daily_refit_bakery_min_effective_share"),
        default=_DEFAULT_DAILY_REFIT_BAKERY_MIN_EFFECTIVE_SHARE,
    )
    max_share = _bounded_share_param(
        model_params.get("daily_refit_bakery_max_effective_share"),
        default=_DEFAULT_DAILY_REFIT_BAKERY_MAX_EFFECTIVE_SHARE,
    )
    upper = max(min_share, max_share)
    return _piecewise_bakery_effective_share(
        bakery_history_days=bakery_history_days,
        min_share=min_share,
        max_share=upper,
    )


def _piecewise_bakery_effective_share(
    *,
    bakery_history_days: int,
    min_share: float,
    max_share: float,
) -> float:
    milestones: tuple[tuple[int, float], ...] = (
        (30, max(min_share, min(max_share, 0.70))),
        (60, max(min_share, min(max_share, 0.85))),
        (90, max(min_share, min(max_share, 0.95))),
        (180, max_share),
    )
    previous_day = 0
    previous_share = min_share
    days = max(int(bakery_history_days), 0)
    for milestone_day, milestone_share in milestones:
        if days <= milestone_day:
            progress = (days - previous_day) / max(milestone_day - previous_day, 1)
            return previous_share + progress * (milestone_share - previous_share)
        previous_day = milestone_day
        previous_share = milestone_share
    return max_share


def _bakery_multiplier_for_target_share(
    base_weights: np.ndarray,
    *,
    bakery_index: np.ndarray,
    target_share: float,
) -> float:
    bakery_weight_sum = float(base_weights[bakery_index].sum())
    non_bakery_weight_sum = float(base_weights[~bakery_index].sum())
    if bakery_weight_sum <= 0.0 or non_bakery_weight_sum <= 0.0:
        return 1.0
    clipped_share = min(max(float(target_share), 1e-6), 0.999999)
    return max(
        1.0,
        clipped_share
        * non_bakery_weight_sum
        / ((1.0 - clipped_share) * bakery_weight_sum),
    )


def _effective_weight_share(weights: np.ndarray, bakery_index: np.ndarray) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return 0.0
    return float(weights[bakery_index].sum() / total_weight)


def _effective_bakery_refit_weight_multiplier(
    model_params: dict[str, object],
    *,
    max_multiplier: float,
    bakery_history_days: int,
) -> float:
    min_multiplier = _float_param(
        model_params.get("daily_refit_bakery_min_sample_weight_multiplier"),
        default=_DEFAULT_DAILY_REFIT_BAKERY_MIN_WEIGHT_MULTIPLIER,
    )
    warmup_days = max(
        _int_param(
            model_params.get("daily_refit_bakery_weight_warmup_days"),
            default=_DEFAULT_DAILY_REFIT_BAKERY_WEIGHT_WARMUP_DAYS,
        ),
        1,
    )
    upper = max(float(max_multiplier), 1.0)
    lower = min(max(float(min_multiplier), 1.0), upper)
    progress = min(max(float(bakery_history_days) / float(warmup_days), 0.0), 1.0)
    return lower + progress * (upper - lower)


def _non_bakery_refit_weight_multiplier(
    model_params: dict[str, object],
    *,
    effective_bakery_multiplier: float,
) -> float:
    mode = str(
        model_params.get(
            "daily_refit_non_bakery_sample_weight_mode",
            "explicit_or_identity",
        )
    )
    if mode == "inverse_effective_bakery":
        return 1.0 / max(float(effective_bakery_multiplier), 1.0)
    multiplier = _float_param(
        model_params.get("daily_refit_non_bakery_sample_weight_multiplier"),
        default=1.0,
    )
    if not np.isfinite(multiplier):
        return 1.0
    return max(float(multiplier), 0.0)


def _base_business_weights(frame: pd.DataFrame) -> np.ndarray:
    if _BUSINESS_SAMPLE_WEIGHT_COL not in frame.columns:
        return np.ones(len(frame), dtype=float)
    weights = pd.to_numeric(
        frame[_BUSINESS_SAMPLE_WEIGHT_COL],
        errors="coerce",
    ).to_numpy(dtype=float)
    return np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)


def _bounded_share_param(value: object, *, default: float) -> float:
    resolved = _float_param(value, default=default)
    if not np.isfinite(resolved):
        return default
    return min(max(float(resolved), 1e-6), 0.999999)


def _frame_diagnostic_summary(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, object]:
    present_features = [column for column in feature_cols if column in frame.columns]
    missing_features = [
        column for column in feature_cols if column not in frame.columns
    ]
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
        "target_nulls": int(frame[target_col].isna().sum())
        if target_col in frame.columns
        else None,
        "feature_dtype_counts": {
            str(key): int(value) for key, value in dtype_counts.items()
        },
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
    bakery_pretest_seed_rows: int,
    bakery_pretest_seed_days: int,
) -> dict[str, object]:
    absolute_target = fold_test_frame[absolute_target_col].astype(float).to_numpy()
    prediction_values = fold_predictions["prediction_raw"].astype(float).to_numpy()
    calibration_applied = (
        bool(fold_predictions["prediction_calibration_applied"].iloc[0])
        if "prediction_calibration_applied" in fold_predictions.columns
        else False
    )
    calibration_rows = (
        int(fold_predictions["prediction_calibration_rows"].iloc[0])
        if "prediction_calibration_rows" in fold_predictions.columns
        else 0
    )
    calibration_slope = (
        float(fold_predictions["prediction_calibration_slope"].iloc[0])
        if "prediction_calibration_slope" in fold_predictions.columns
        else 1.0
    )
    calibration_intercept = (
        float(fold_predictions["prediction_calibration_intercept"].iloc[0])
        if "prediction_calibration_intercept" in fold_predictions.columns
        else 0.0
    )
    economic_calibration_applied = (
        bool(fold_predictions["economic_calibration_applied"].any())
        if "economic_calibration_applied" in fold_predictions.columns
        else False
    )
    economic_calibration_scope = (
        ",".join(sorted(fold_predictions["economic_calibration_scope"].astype(str).unique()))
        if "economic_calibration_scope" in fold_predictions.columns
        else "none"
    )
    return {
        "fold": int(cast(Any, fold["fold"])),
        "eval_date": eval_date,
        "history_rows": known_history_rows,
        "bakery_history_rows": int(len(bakery_history_frame)),
        "bakery_pretest_seed_rows": int(bakery_pretest_seed_rows),
        "bakery_pretest_seed_days": int(bakery_pretest_seed_days),
        "valid_rows": int(len(fold_predictions)),
        "best_iteration": int(best_iteration),
        "mae": float(np.mean(np.abs(absolute_target - prediction_values))),
        "rmse": rmse_score(absolute_target, prediction_values),
        "prediction_calibration_applied": calibration_applied,
        "prediction_calibration_rows": calibration_rows,
        "prediction_calibration_slope": calibration_slope,
        "prediction_calibration_intercept": calibration_intercept,
        "economic_calibration_applied": economic_calibration_applied,
        "economic_calibration_scope": economic_calibration_scope,
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
    supports_quantile_predictions = hasattr(model, "quantiles") and hasattr(
        model, "target_col"
    )
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
    if (
        absolute_quantiles is not None
        and "prediction_p50" in absolute_quantiles.columns
    ):
        return absolute_quantiles, absolute_quantiles["prediction_p50"].to_numpy(
            dtype=float
        )
    return (
        absolute_quantiles,
        predict_absolute_fn(
            model,
            fold_test_frame,
            feature_cols=feature_cols,
            target_contract=target_contract,
        ),
    )
