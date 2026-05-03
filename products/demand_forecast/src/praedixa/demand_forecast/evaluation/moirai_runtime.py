from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.moirai.model import (
    MoiraiZeroShotModel,
    fit_moirai_zero_shot_model,
    predict_moirai_median,
    predict_moirai_quantiles,
    save_moirai_model_marker,
)
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.evaluation.refit import evaluate_daily_refit_predictions


def evaluate_moirai_daily_refit_predictions(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    resolved_params = _moirai_model_params(model_params)
    logger.info(
        "Moirai zero-shot evaluation starting: model_path=%s context_length=%s "
        "batch_size=%s covariates_used=%s",
        resolved_params["model_path"],
        resolved_params["context_length"],
        resolved_params["batch_size"],
        _available_covariate_count(train_frame, feature_cols),
    )
    return evaluate_daily_refit_predictions(
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=resolved_params,
        logger=logger,
        fit_model_fn=fit_moirai_evaluation_model,
        predict_absolute_fn=predict_absolute_moirai,
        predict_quantiles_absolute_fn=predict_absolute_moirai_quantiles,
    )


def fit_moirai_evaluation_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[MoiraiZeroShotModel, int]:
    return fit_moirai_zero_shot_model(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        target_col=target_contract.absolute_target_col,
        model_params=model_params,
    )


def predict_absolute_moirai(
    model: MoiraiZeroShotModel,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> np.ndarray:
    _ = feature_cols, target_contract
    return predict_moirai_median(model, frame)


def predict_absolute_moirai_quantiles(
    model: MoiraiZeroShotModel,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> pd.DataFrame:
    _ = feature_cols, target_contract
    return predict_moirai_quantiles(model, frame)


def fit_final_moirai_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    num_boost_round: int,
) -> MoiraiZeroShotModel:
    _ = num_boost_round
    model, _ = fit_moirai_zero_shot_model(
        train_frame=fit_frame,
        valid_frame=fit_frame.head(0),
        feature_cols=feature_cols,
        target_col=target_contract.absolute_target_col,
        model_params=_moirai_model_params(model_params),
    )
    return model


def save_moirai_model(model: MoiraiZeroShotModel, output_path: Path) -> Path:
    return save_moirai_model_marker(model, output_path)


def _moirai_model_params(model_params: dict[str, object]) -> dict[str, object]:
    resolved = dict(model_params)
    resolved["model_backend"] = "moirai"
    resolved.setdefault("model_path", "Salesforce/moirai-2.0-R-small")
    resolved.setdefault("context_length", 1680)
    resolved.setdefault("batch_size", 32)
    resolved.setdefault("num_samples", 100)
    resolved.setdefault("quantile_levels", [0.1, 0.5, 0.9])
    resolved.setdefault("enable_daily_prediction_calibration", False)
    resolved.setdefault("initialize_daily_prediction_calibration_from_valid", True)
    resolved.setdefault("daily_prediction_calibration_min_rows", 35)
    resolved.setdefault("daily_prediction_calibration_slope_min", 0.25)
    resolved.setdefault("daily_prediction_calibration_slope_max", 2.0)
    resolved.setdefault("daily_prediction_calibration_progress_series_batch_size", 512)
    return resolved


def _available_covariate_count(frame: pd.DataFrame, feature_cols: list[str]) -> int:
    return sum(1 for column in feature_cols if column in frame.columns)
