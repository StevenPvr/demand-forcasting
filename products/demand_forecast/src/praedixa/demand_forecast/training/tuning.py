from __future__ import annotations

import logging

import optuna
import pandas as pd

from praedixa.demand_forecast.backends.tft.model_utils import DEFAULT_TFT_MODEL_PARAMS
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.feature_screening.pipeline import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_TUNING_TRIALS,
)
from praedixa.demand_forecast.training.constants import DEFAULT_TARGET_TRANSFORM
from praedixa.demand_forecast.training.tuning_policy import (
    resolve_hpo_execution_policy as _resolve_hpo_execution_policy,
    sample_optuna_params,
)
from praedixa.demand_forecast.training.tuning_optuna import run_optuna_search
from praedixa.demand_forecast.training.tuning_scoring import (
    fit_and_score_tft_model_on_tuning as _fit_and_score_tft_model_on_tuning,
)


def resolve_hpo_execution_policy(
    *,
    folds: list[dict[str, object]],
    model_params: dict[str, object] | None,
    total_threads: int | None,
    logger: logging.Logger,
) -> dict[str, object]:
    return _resolve_hpo_execution_policy(
        folds=folds,
        resolved_params={**DEFAULT_TFT_MODEL_PARAMS, **(model_params or {})},
        total_threads=total_threads,
        logger=logger,
    )


def fit_and_score_tft_model_on_tuning(
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
) -> dict[str, object]:
    return _fit_and_score_tft_model_on_tuning(
        train_frame=train_frame,
        tuning_frame=tuning_frame,
        folds=folds,
        feature_cols=feature_cols,
        target_contract=target_contract,
        logger=logger,
        model_params=model_params,
        total_threads=total_threads,
        target_transform=target_transform,
        trial=trial,
    )


def optimize_tft_model_params(
    train_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    folds: list[dict[str, object]],
    feature_cols: list[str],
    baseline_wape: float,
    target_contract: TargetContract,
    *,
    logger: logging.Logger,
    tuning_trials: int = DEFAULT_TUNING_TRIALS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, object] | None = None,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    return run_optuna_search(
        score_fn=fit_and_score_tft_model_on_tuning,
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
        target_transform=target_transform,
    )


__all__ = [
    "fit_and_score_tft_model_on_tuning",
    "optimize_tft_model_params",
    "resolve_hpo_execution_policy",
    "sample_optuna_params",
]
