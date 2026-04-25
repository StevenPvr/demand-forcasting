from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.folds import (
    build_daily_walk_forward_folds as _build_daily_walk_forward_folds_internal,
)
from praedixa.demand_forecast.evaluation.modeling import (
    fit_evaluation_model as _fit_evaluation_model_internal,
)
from praedixa.demand_forecast.evaluation.modeling import (
    predict_absolute as _predict_absolute_internal,
)
from praedixa.demand_forecast.evaluation.modeling import (
    predict_absolute_quantiles as _predict_absolute_quantiles_internal,
)
from praedixa.demand_forecast.evaluation.orchestrator import (
    EvaluationBuildRequest,
    build_evaluation_outputs,
)
from praedixa.demand_forecast.evaluation.reference import (
    build_bakery_reference_feature_frame as _build_bakery_reference_feature_frame_internal,
)
from praedixa.demand_forecast.evaluation.refit import (
    evaluate_daily_refit_predictions as _evaluate_daily_refit_predictions_internal,
)
from praedixa.demand_forecast.training.config.constants import (
    DEFAULT_DATE_COL,
)
from praedixa.demand_forecast.contracts.targets import TargetContract


logger = logging.getLogger(__name__)

__all__ = [
    "build_bakery_reference_feature_frame",
    "evaluate_daily_refit_predictions",
    "build_daily_walk_forward_folds",
    "EvaluationBuildRequest",
    "build_evaluation_outputs",
]


def _fit_evaluation_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[Any, int]:
    return _fit_evaluation_model_internal(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=model_params,
    )


def _predict_absolute(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> np.ndarray:
    return _predict_absolute_internal(
        model,
        frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
    )


def _predict_absolute_quantiles(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> pd.DataFrame:
    return _predict_absolute_quantiles_internal(
        model,
        frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
    )


def build_daily_walk_forward_folds(
    frame: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
) -> list[dict[str, object]]:
    return _build_daily_walk_forward_folds_internal(frame, date_col=date_col)


def evaluate_daily_refit_predictions(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    return _evaluate_daily_refit_predictions_internal(
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=model_params,
        logger=logger,
        fit_model_fn=_fit_evaluation_model,
        predict_absolute_fn=_predict_absolute,
        predict_quantiles_absolute_fn=_predict_absolute_quantiles,
    )


def build_bakery_reference_feature_frame(
    reference_full_df: pd.DataFrame,
    reference_test_df: pd.DataFrame,
    gold_base_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    return _build_bakery_reference_feature_frame_internal(
        reference_full_df, reference_test_df, gold_base_panel_df
    )
