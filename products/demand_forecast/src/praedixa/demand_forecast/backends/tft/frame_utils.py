from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import (
    TFTLayout,
    resolve_explicit_tft_layout,
    select_explicit_tft_categorical_columns,
    select_explicit_tft_group_id_columns,
)


GROUP_COL = "__tft_group_id"
SPLIT_COL = "__tft_split"
TIME_IDX_COL = "__tft_time_idx"
WEIGHT_COL = "__tft_sample_weight"
PREDICTION_ROW_ID_COL = "__tft_prediction_row_id"


def defragment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy()


def build_group_identifier(frame: pd.DataFrame, group_source_cols: list[str]) -> pd.Series:
    if not group_source_cols:
        return pd.Series(["global_series"] * len(frame), index=frame.index, dtype="string")
    parts = [frame[column].astype("string").fillna("<NA>") for column in group_source_cols]
    group_key = parts[0]
    for next_part in parts[1:]:
        group_key = group_key.str.cat(next_part, sep="__")
    return group_key.astype("string")


def with_optional_weights(frame: pd.DataFrame, weights: np.ndarray | None) -> pd.DataFrame:
    weighted = frame.copy()
    if weights is None:
        return weighted
    if len(weights) != len(frame):
        raise ValueError("Sample weights length must match frame length.")
    weighted[WEIGHT_COL] = np.asarray(weights, dtype=float)
    return weighted


def attach_prediction_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    with_ids = frame.copy()
    with_ids[PREDICTION_ROW_ID_COL] = np.arange(len(with_ids), dtype=np.int32)
    return with_ids


def prepare_split_frame(
    frame: pd.DataFrame,
    *,
    split_name: str,
    weights: np.ndarray | None,
) -> pd.DataFrame:
    prepared = with_optional_weights(frame, weights)
    prepared[SPLIT_COL] = split_name
    return prepared


def build_combined_frame(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    *,
    train_weights: np.ndarray | None,
    valid_weights: np.ndarray | None,
) -> pd.DataFrame:
    train_part = prepare_split_frame(train_frame, split_name="train", weights=train_weights)
    if valid_frame is None:
        return defragment_frame(train_part)
    valid_part = prepare_split_frame(valid_frame, split_name="valid", weights=valid_weights)
    return defragment_frame(pd.concat([train_part, valid_part], ignore_index=True))


def cast_categorical_columns(frame: pd.DataFrame, categorical_cols: Iterable[str]) -> pd.DataFrame:
    casted = frame.copy()
    for column in categorical_cols:
        casted[column] = casted[column].astype("string").fillna("<NA>")
    return defragment_frame(casted)


def attach_group_and_time_columns(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["dt"] = pd.to_datetime(prepared["dt"])
    group_source_cols = select_explicit_tft_group_id_columns(prepared)
    prepared[GROUP_COL] = build_group_identifier(prepared, group_source_cols)
    prepared = prepared.sort_values([GROUP_COL, "dt"]).reset_index(drop=True)
    prepared[TIME_IDX_COL] = prepared.groupby(GROUP_COL, sort=False).cumcount().astype(np.int32)
    feature_categoricals = select_explicit_tft_categorical_columns(feature_cols)
    return defragment_frame(cast_categorical_columns(prepared, [*feature_categoricals, GROUP_COL]))


def resolve_layout(frame: pd.DataFrame, feature_cols: list[str]) -> TFTLayout:
    _ = frame
    layout = resolve_explicit_tft_layout(feature_cols)
    layout["time_varying_known_reals"] = [TIME_IDX_COL, *layout["time_varying_known_reals"]]
    return layout
