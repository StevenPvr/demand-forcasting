from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from sklearn.preprocessing import StandardScaler

from praedixa.demand_forecast.backends.tft.feature_contract import (
    validate_feature_contract,
)
from praedixa.demand_forecast.backends.tft.feature_mapping import TFTLayout
from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    SPLIT_COL,
    TIME_IDX_COL,
    WEIGHT_COL,
    defragment_frame,
    resolve_layout,
)


@dataclass(frozen=True)
class TrainingDatasetArtifacts:
    training_slice: pd.DataFrame
    training_dataset: Any
    validation_dataset: Any
    layout: TFTLayout
    feature_scalers: dict[str, StandardScaler]
    normalization_strategy: dict[str, Any]


def _build_timeseries_dataset(
    imports: dict[str, Any],
    frame: pd.DataFrame,
    *,
    target_col: str,
    max_encoder_length: int,
    weight_col: str | None,
    layout: TFTLayout,
    categorical_encoders: dict[str, Any] | None,
    real_feature_scalers: dict[str, StandardScaler] | None,
) -> Any:
    target_normalizer = imports["GroupNormalizer"](groups=[GROUP_COL], method="standard")
    return imports["TimeSeriesDataSet"](
        defragment_frame(frame),
        time_idx=TIME_IDX_COL,
        target=target_col,
        group_ids=[GROUP_COL],
        weight=weight_col,
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=layout["static_categoricals"],
        static_reals=layout["static_reals"],
        time_varying_known_categoricals=layout["time_varying_known_categoricals"],
        time_varying_known_reals=layout["time_varying_known_reals"],
        time_varying_unknown_categoricals=layout["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=[*layout["time_varying_unknown_reals"], target_col],
        allow_missing_timesteps=False,
        add_target_scales=True,
        target_normalizer=target_normalizer,
        categorical_encoders=categorical_encoders,
        scalers=real_feature_scalers,
    )


def _build_validation_dataset(
    imports: dict[str, Any],
    training_dataset: Any,
    prepared_frame: pd.DataFrame,
) -> Any:
    valid_frame = defragment_frame(prepared_frame.loc[prepared_frame[SPLIT_COL] == "valid"].copy())
    if valid_frame.empty:
        return None
    return imports["TimeSeriesDataSet"].from_dataset(
        training_dataset,
        defragment_frame(prepared_frame),
        min_prediction_idx=int(valid_frame[TIME_IDX_COL].min()),
        stop_randomization=True,
    )


def _real_feature_scaler_columns(layout: TFTLayout) -> list[str]:
    columns: list[str] = []
    for column in [*layout["static_reals"], *layout["time_varying_known_reals"], *layout["time_varying_unknown_reals"]]:
        if column != TIME_IDX_COL and column not in columns:
            columns.append(column)
    return columns


def _categorical_encoder_columns(layout: TFTLayout) -> list[str]:
    columns = [GROUP_COL]
    for column in [*layout["static_categoricals"], *layout["time_varying_known_categoricals"], *layout["time_varying_unknown_categoricals"]]:
        if column not in columns:
            columns.append(column)
    return columns


def _build_categorical_encoders(
    imports: dict[str, Any],
    *,
    layout: TFTLayout,
) -> dict[str, Any]:
    return {
        column: imports["NaNLabelEncoder"](add_nan=True, warn=False)
        for column in _categorical_encoder_columns(layout)
    }


def _fit_real_feature_scalers(
    training_slice: pd.DataFrame,
    *,
    layout: TFTLayout,
) -> dict[str, StandardScaler]:
    scalers: dict[str, StandardScaler] = {}
    for column in _real_feature_scaler_columns(layout):
        numeric_values = pd.to_numeric(training_slice[column], errors="coerce").dropna()
        if numeric_values.empty:
            continue
        scaler = StandardScaler()
        cast(Any, scaler).fit(numeric_values.to_frame(name=column).astype(float))
        scalers[column] = scaler
    return scalers


def build_training_dataset_artifacts(
    imports: dict[str, Any],
    *,
    prepared_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
) -> TrainingDatasetArtifacts:
    validate_feature_contract(feature_cols)
    layout = resolve_layout(prepared_frame, feature_cols)
    training_slice = defragment_frame(prepared_frame.loc[prepared_frame[SPLIT_COL] == "train"].copy())
    categorical_encoders = _build_categorical_encoders(imports, layout=layout)
    feature_scalers = _fit_real_feature_scalers(training_slice, layout=layout)
    training_dataset = _build_timeseries_dataset(
        imports,
        training_slice,
        target_col=target_col,
        max_encoder_length=int(cast(Any, resolved_params["max_encoder_length"])),
        weight_col=WEIGHT_COL if WEIGHT_COL in training_slice.columns else None,
        layout=layout,
        categorical_encoders=categorical_encoders,
        real_feature_scalers=feature_scalers or None,
    )
    validation_dataset = _build_validation_dataset(imports, training_dataset, prepared_frame)
    return TrainingDatasetArtifacts(
        training_slice=training_slice,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        layout=layout,
        feature_scalers=feature_scalers,
        normalization_strategy={
            "kind": "group_normalizer",
            "method": "standard",
            "groups": [GROUP_COL],
            "legacy_external_scaler": False,
        },
    )
