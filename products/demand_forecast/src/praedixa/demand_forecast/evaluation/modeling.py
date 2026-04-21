from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from praedixa.demand_forecast.evaluation.constants import DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS
from praedixa.demand_forecast.training.constants import (
    DEFAULT_DATASET_SOURCE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
    DEFAULT_IDENTIFIER_FEATURE_COLS,
)
from praedixa.demand_forecast.contracts.targets import TargetContract, reconstruct_absolute_predictions
from praedixa.demand_forecast.backends.tft.model_utils import (
    DEFAULT_TFT_MODEL_PARAMS,
    fit_tft_model,
    predict_quantiles_with_tft_model,
    predict_with_tft_model,
    select_tft_feature_columns,
)


def compute_equal_dataset_row_weights(
    frame: pd.DataFrame,
    *,
    dataset_source_col: str,
) -> np.ndarray | None:
    if dataset_source_col not in frame.columns:
        return None
    dataset_sources = frame[dataset_source_col].astype("string").fillna("<NA>").astype(str)
    dataset_counts = dataset_sources.value_counts(dropna=False)
    dataset_count = max(1, len(dataset_counts))
    relative_weights = np.asarray(
        [1.0 / (dataset_count * float(dataset_counts.loc[dataset])) for dataset in dataset_sources.tolist()],
        dtype=float,
    )
    return relative_weights * float(len(frame))


def drop_constant_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    # Keep constant metadata so evaluation matches the TFT training feature space.
    _ = frame
    return list(feature_cols), []


def select_feature_columns(
    train_frame: pd.DataFrame,
    *,
    learning_target_col: str,
    absolute_target_col: str,
) -> tuple[list[str], list[str], list[str]]:
    numeric_feature_cols = select_tft_feature_columns(
        train_frame,
        excluded_cols={
            DEFAULT_DATE_COL,
            learning_target_col,
            absolute_target_col,
            *DEFAULT_EVALUATION_EXCLUDED_HELPER_FEATURE_COLS,
            *DEFAULT_EXCLUDED_RISKY_FEATURE_COLS,
        },
    )
    identifier_feature_cols = [
        column
        for column in DEFAULT_IDENTIFIER_FEATURE_COLS
        if column in train_frame.columns and column not in numeric_feature_cols
    ]
    filtered_feature_cols, constant_feature_cols = drop_constant_feature_columns(train_frame, numeric_feature_cols)
    return filtered_feature_cols, constant_feature_cols, identifier_feature_cols


def fit_evaluation_model(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
) -> tuple[Any, int]:
    learning_target_col = target_contract.learning_target_col
    train_weights = compute_equal_dataset_row_weights(train_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL)
    valid_weights = compute_equal_dataset_row_weights(valid_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL)
    train_for_model = train_frame.copy()
    valid_for_model = valid_frame.copy()
    if target_contract.target_mode == "log1p":
        train_for_model[learning_target_col] = np.log1p(train_for_model[learning_target_col].astype(float))
        valid_for_model[learning_target_col] = np.log1p(valid_for_model[learning_target_col].astype(float))
    model = fit_tft_model(
        train_for_model,
        feature_cols,
        target_col=learning_target_col,
        model_params=model_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=2000,
        valid_frame=valid_for_model,
        train_weights=train_weights,
        valid_weights=valid_weights,
    )
    best_iteration = getattr(model, "best_iteration", None)
    resolved_rounds = int(best_iteration + 1) if best_iteration is not None and best_iteration >= 0 else 2000
    return model, resolved_rounds


def predict_absolute(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> np.ndarray:
    predictions = predict_with_tft_model(model, frame, feature_cols)
    return reconstruct_absolute_predictions(predictions, frame, target_contract)


def predict_absolute_quantiles(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_contract: TargetContract,
) -> pd.DataFrame:
    quantile_predictions = predict_quantiles_with_tft_model(model, frame, feature_cols)
    absolute_quantiles = {
        column: reconstruct_absolute_predictions(
            quantile_predictions[column].to_numpy(dtype=float),
            frame,
            target_contract,
        )
        for column in quantile_predictions.columns
    }
    return pd.DataFrame(absolute_quantiles)


def fit_final_model(
    *,
    fit_frame: pd.DataFrame,
    feature_cols: list[str],
    target_contract: TargetContract,
    model_params: dict[str, object],
    num_boost_round: int,
) -> Any:
    learning_target_col = target_contract.learning_target_col
    fit_weights = compute_equal_dataset_row_weights(fit_frame, dataset_source_col=DEFAULT_DATASET_SOURCE_COL)
    fit_for_model = fit_frame.copy()
    if target_contract.target_mode == "log1p":
        fit_for_model[learning_target_col] = np.log1p(fit_for_model[learning_target_col].astype(float))
    return fit_tft_model(
        fit_for_model,
        feature_cols,
        target_col=learning_target_col,
        model_params=model_params,
        default_params=DEFAULT_TFT_MODEL_PARAMS,
        default_max_iter=num_boost_round,
        train_weights=fit_weights,
    )
