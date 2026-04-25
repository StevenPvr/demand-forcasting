from __future__ import annotations

from dataclasses import dataclass
import logging
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
from praedixa.demand_forecast.backends.tft.model_common import (
    suppress_tft_dataframe_fragmentation_warnings,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingDatasetArtifacts:
    training_slice: pd.DataFrame
    training_dataset: Any
    validation_dataset: Any
    layout: TFTLayout
    feature_scalers: dict[str, StandardScaler]
    normalization_strategy: dict[str, Any]


def _supported_group_ids_with_sufficient_history(
    frame: pd.DataFrame,
    *,
    max_encoder_length: int,
) -> set[str]:
    if frame.empty:
        return set()
    if SPLIT_COL in frame.columns:
        train_rows = frame.loc[frame[SPLIT_COL] == "train"]
    else:
        train_rows = frame
    if train_rows.empty:
        return set()
    group_sizes = train_rows.groupby(GROUP_COL, sort=False).size()
    return {
        str(group_id)
        for group_id, size in group_sizes.items()
        if int(size) >= max_encoder_length
    }


def filter_groups_with_sufficient_history(
    frame: pd.DataFrame,
    *,
    max_encoder_length: int,
) -> pd.DataFrame:
    if frame.empty:
        return defragment_frame(frame)
    supported_group_ids = _supported_group_ids_with_sufficient_history(
        frame,
        max_encoder_length=max_encoder_length,
    )
    if not supported_group_ids:
        LOGGER.warning(
            "No TFT groups remain after enforcing min encoder history: max_encoder_length=%s rows=%s",
            max_encoder_length,
            len(frame),
        )
        return pd.DataFrame(columns=frame.columns)
    filtered = frame.loc[frame[GROUP_COL].astype("string").isin(supported_group_ids)].copy()
    return defragment_frame(filtered)


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
    with suppress_tft_dataframe_fragmentation_warnings():
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
    *,
    max_encoder_length: int,
) -> Any:
    validation_frame = _build_validation_frame(
        prepared_frame=prepared_frame,
        max_encoder_length=max_encoder_length,
    )
    if validation_frame.empty:
        return None
    with suppress_tft_dataframe_fragmentation_warnings():
        return imports["TimeSeriesDataSet"].from_dataset(
            training_dataset,
            validation_frame,
            min_prediction_idx=max_encoder_length,
            stop_randomization=True,
        )


def _validation_group_frames(
    prepared_frame: pd.DataFrame,
    *,
    max_encoder_length: int,
) -> list[pd.DataFrame]:
    grouped_frames: list[pd.DataFrame] = []
    for _, group_frame in prepared_frame.groupby(GROUP_COL, sort=False):
        train_rows = group_frame.loc[group_frame[SPLIT_COL] == "train"].copy()
        valid_rows = group_frame.loc[group_frame[SPLIT_COL] == "valid"].copy()
        if valid_rows.empty or len(train_rows) < max_encoder_length:
            continue
        history_tail = train_rows.tail(max_encoder_length).copy()
        validation_window = pd.concat([history_tail, valid_rows], ignore_index=True)
        validation_window[TIME_IDX_COL] = pd.RangeIndex(
            start=0,
            stop=len(validation_window),
            step=1,
            name=TIME_IDX_COL,
        ).astype("int32")
        grouped_frames.append(validation_window)
    return grouped_frames


def _build_validation_frame(
    *,
    prepared_frame: pd.DataFrame,
    max_encoder_length: int,
) -> pd.DataFrame:
    grouped_frames = _validation_group_frames(
        prepared_frame,
        max_encoder_length=max_encoder_length,
    )
    if not grouped_frames:
        return pd.DataFrame(columns=prepared_frame.columns)
    return defragment_frame(pd.concat(grouped_frames, ignore_index=True))


def _rebase_training_slice_time_idx(
    training_slice: pd.DataFrame,
) -> pd.DataFrame:
    if training_slice.empty:
        return defragment_frame(training_slice)
    rebased = training_slice.sort_values([GROUP_COL, TIME_IDX_COL, "dt"]).reset_index(drop=True).copy()
    rebased[TIME_IDX_COL] = rebased.groupby(GROUP_COL, sort=False).cumcount().astype("int32")
    return defragment_frame(rebased)


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


def build_categorical_encoders(
    imports: dict[str, Any],
    *,
    layout: TFTLayout,
) -> dict[str, Any]:
    return _build_categorical_encoders(imports, layout=layout)


def fit_real_feature_scalers(
    training_slice: pd.DataFrame,
    *,
    layout: TFTLayout,
) -> dict[str, StandardScaler]:
    return _fit_real_feature_scalers(training_slice, layout=layout)


def build_training_dataset_artifacts(
    imports: dict[str, Any],
    *,
    prepared_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    resolved_params: dict[str, object],
) -> TrainingDatasetArtifacts:
    validate_feature_contract(
        feature_cols,
        decision_profile=cast(
            Any,
            resolved_params.get("decision_profile", "post_close_d_plus_1"),
        ),
    )
    max_encoder_length = int(cast(Any, resolved_params["max_encoder_length"]))
    prepared_frame = filter_groups_with_sufficient_history(
        prepared_frame,
        max_encoder_length=max_encoder_length,
    )
    layout = resolve_layout(prepared_frame, feature_cols)
    training_slice = _rebase_training_slice_time_idx(
        prepared_frame.loc[prepared_frame[SPLIT_COL] == "train"].copy()
    )
    categorical_encoders = build_categorical_encoders(imports, layout=layout)
    feature_scalers = fit_real_feature_scalers(training_slice, layout=layout)
    training_dataset = _build_timeseries_dataset(
        imports,
        training_slice,
        target_col=target_col,
        max_encoder_length=max_encoder_length,
        weight_col=WEIGHT_COL if WEIGHT_COL in training_slice.columns else None,
        layout=layout,
        categorical_encoders=categorical_encoders,
        real_feature_scalers=feature_scalers or None,
    )
    validation_dataset = _build_validation_dataset(
        imports,
        training_dataset,
        prepared_frame,
        max_encoder_length=max_encoder_length,
    )
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
        },
    )
