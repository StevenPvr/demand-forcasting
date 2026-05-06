from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from praedixa.demand_forecast.backends.xgboost.model_fit import FittedXGBoostModel
from praedixa.demand_forecast.contracts.targets import TargetContract
from praedixa.demand_forecast.evaluation.modeling import (
    fit_xgboost_evaluation_model,
    predict_absolute_xgboost,
)
from praedixa.demand_forecast.evaluation.refit import evaluate_daily_refit_predictions


def evaluate_xgboost_daily_refit_predictions(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    refit_model_params = _xgboost_daily_refit_model_params(model_params)
    logger.info(
        "XGBoost daily evaluation refit resources resolved: n_jobs=%s workers=%s runtime_profile=%s",
        refit_model_params.get("n_jobs"),
        refit_model_params.get("evaluation_daily_refit_workers"),
        refit_model_params.get("runtime_profile"),
    )
    return evaluate_daily_refit_predictions(
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        feature_cols=feature_cols,
        target_contract=target_contract,
        model_params=refit_model_params,
        logger=logger,
        fit_model_fn=fit_xgboost_evaluation_model,
        predict_absolute_fn=predict_absolute_xgboost,
        predict_quantiles_absolute_fn=_empty_quantile_predictions,
    )


def _xgboost_daily_refit_model_params(
    model_params: dict[str, object],
) -> dict[str, object]:
    resolved = dict(model_params)
    resolved["n_jobs"] = _available_cpu_cores()
    resolved["evaluation_daily_refit_workers"] = 1
    return resolved


def _available_cpu_cores() -> int:
    return max(1, int(os.cpu_count() or 1))


def _empty_quantile_predictions(*_: object, **__: object) -> pd.DataFrame:
    return pd.DataFrame()


def save_xgboost_model(model: FittedXGBoostModel, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.model.save_model(str(output_path))
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "feature_cols": model.feature_spec.feature_cols,
                "categorical_feature_cols": model.feature_spec.categorical_feature_cols,
                "numeric_feature_cols": model.feature_spec.numeric_feature_cols,
                "category_values": model.feature_spec.category_values,
                "native_categorical": model.feature_spec.native_categorical,
                "params": model.params,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return output_path
