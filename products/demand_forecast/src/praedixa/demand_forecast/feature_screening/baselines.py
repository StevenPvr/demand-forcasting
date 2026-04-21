from __future__ import annotations

import logging
from typing import Callable, TypeAlias, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.feature_screening.constants import DEFAULT_TARGET_COL
from praedixa.demand_forecast.feature_screening.metrics import compute_wape


def _same_weekday_mean_4(frame: pd.DataFrame) -> pd.Series:
    if "target_same_dow_mean_4w" in frame.columns:
        return frame["target_same_dow_mean_4w"]
    candidate_cols = ["sale_amount_lag_7", "sale_amount_lag_14", "sale_amount_lag_21", "sale_amount_lag_28"]
    available_cols = [column for column in candidate_cols if column in frame.columns]
    if not available_cols:
        return pd.Series(np.nan, index=frame.index)
    return frame[available_cols].mean(axis=1)


def _resolve_baseline_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return None


BaselinePredictor: TypeAlias = Callable[[pd.DataFrame], pd.Series | None]


def _naive_lag_1(frame: pd.DataFrame) -> pd.Series | None:
    return _resolve_baseline_column(frame, ("sale_amount_lag_1", "lag_1"))


def _seasonal_naive_lag_7(frame: pd.DataFrame) -> pd.Series | None:
    return _resolve_baseline_column(
        frame,
        ("target_lag_7", "target_seasonal_naive_d7", "sale_amount_lag_7", "lag_7"),
    )


def _trailing_mean_7(frame: pd.DataFrame) -> pd.Series | None:
    return _resolve_baseline_column(frame, ("sale_amount_rolling_mean_7", "rolling_mean_7"))


def _trailing_mean_28(frame: pd.DataFrame) -> pd.Series | None:
    return _resolve_baseline_column(frame, ("sale_amount_rolling_mean_28", "rolling_mean_28"))


def _mean_wape(row: dict[str, object]) -> float:
    return float(cast(float, row["mean_wape"]))


def _score_baseline_fold(
    *,
    frame: pd.DataFrame,
    fold: dict[str, object],
    predictor: BaselinePredictor,
    target_col: str,
) -> float | None:
    valid_idx = np.asarray(fold["valid_idx"], dtype=np.int32)
    valid_frame = frame.iloc[valid_idx]
    y_true = valid_frame[target_col]
    y_pred = predictor(valid_frame)
    if y_pred is None or bool(pd.isna(y_pred).all()):
        return None
    return compute_wape(y_true, y_pred)


def _evaluate_baseline_predictor(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    baseline_name: str,
    predictor: BaselinePredictor,
    target_col: str,
) -> dict[str, object] | None:
    fold_scores: list[float] = []
    for fold in folds:
        fold_score = _score_baseline_fold(
            frame=frame,
            fold=fold,
            predictor=predictor,
            target_col=target_col,
        )
        if fold_score is None:
            return None
        fold_scores.append(fold_score)
    return {
        "baseline_name": baseline_name,
        "mean_wape": float(np.mean(fold_scores)),
        "fold_wape_scores": [float(score) for score in fold_scores],
    }


def evaluate_statistical_baselines(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    *,
    logger: logging.Logger,
    target_col: str = DEFAULT_TARGET_COL,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_predictors: dict[str, BaselinePredictor] = {
        "naive_lag_1": _naive_lag_1,
        "seasonal_naive_lag_7": _seasonal_naive_lag_7,
        "trailing_mean_7": _trailing_mean_7,
        "trailing_mean_28": _trailing_mean_28,
        "same_weekday_mean_4": _same_weekday_mean_4,
    }

    report_rows: list[dict[str, object]] = []
    for baseline_name, predictor in baseline_predictors.items():
        baseline_row = _evaluate_baseline_predictor(
            frame=frame,
            folds=folds,
            baseline_name=baseline_name,
            predictor=predictor,
            target_col=target_col,
        )
        if baseline_row is None:
            continue
        report_rows.append(baseline_row)

    if not report_rows:
        raise ValueError("No statistical baseline could be evaluated with the available columns.")

    best_row = min(report_rows, key=_mean_wape)
    logger.info(
        "Best statistical baseline: name=%s mean_wape=%.6f",
        best_row["baseline_name"],
        best_row["mean_wape"],
    )
    return report_rows, best_row
