from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import pandas as pd

from praedixa.demand_forecast.backends.tft.feature_mapping import TFTLayout
from praedixa.demand_forecast.backends.tft.frame_utils import (
    GROUP_COL,
    PREDICTION_ROW_ID_COL,
    SPLIT_COL,
    TIME_IDX_COL,
    WEIGHT_COL,
    defragment_frame,
)
from praedixa.demand_forecast.backends.tft.model_common import (
    suppress_tft_dataframe_fragmentation_warnings,
)


@dataclass(frozen=True)
class TimeSeriesDatasetCore:
    dataset: Any
    preprocessed_frame: pd.DataFrame


_DATA_TO_TENSORS_CAPTURE_LOCK = Lock()


def _capturing_data_to_tensors(
    original_method: Any,
) -> Any:
    def capturing_method(self: Any, data: pd.DataFrame) -> dict[str, Any]:
        self._praedixa_preprocessed_frame = data
        return original_method(self, data)

    return capturing_method


def _build_dataset_with_captured_preprocessed_frame(
    dataset_class: type[Any],
    build_fn: Any,
) -> TimeSeriesDatasetCore:
    with _DATA_TO_TENSORS_CAPTURE_LOCK:
        original_data_to_tensors = dataset_class._data_to_tensors
        dataset_class._data_to_tensors = _capturing_data_to_tensors(
            original_data_to_tensors
        )
        try:
            with suppress_tft_dataframe_fragmentation_warnings():
                dataset = build_fn()
        finally:
            dataset_class._data_to_tensors = original_data_to_tensors
    return TimeSeriesDatasetCore(
        dataset=dataset,
        preprocessed_frame=cast(pd.DataFrame, dataset._praedixa_preprocessed_frame),
    )


def build_training_dataset_core(
    imports: dict[str, Any],
    *,
    training_frame: pd.DataFrame,
    target_col: str,
    max_encoder_length: int,
    weight_col: str | None,
    layout: TFTLayout,
    categorical_encoders: dict[str, Any] | None,
    real_feature_scalers: dict[str, Any] | None,
) -> TimeSeriesDatasetCore:
    dataset_class = cast(type[Any], imports["TimeSeriesDataSet"])
    return _build_dataset_with_captured_preprocessed_frame(
        dataset_class,
        lambda: dataset_class(
            defragment_frame(training_frame),
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
            target_normalizer=imports["GroupNormalizer"](groups=[GROUP_COL], method="standard"),
            categorical_encoders=categorical_encoders,
            scalers=real_feature_scalers,
        ),
    )


def build_validation_dataset_core(
    *,
    training_core: TimeSeriesDatasetCore,
    validation_frame: pd.DataFrame,
    min_prediction_idx: int,
) -> TimeSeriesDatasetCore:
    dataset_class = training_core.dataset.__class__
    return _build_dataset_with_captured_preprocessed_frame(
        dataset_class,
        lambda: dataset_class.from_dataset(
            training_core.dataset,
            validation_frame,
            min_prediction_idx=min_prediction_idx,
            stop_randomization=True,
        ),
    )


def clone_training_dataset_from_core(
    *,
    core: TimeSeriesDatasetCore,
    max_encoder_length: int,
) -> Any:
    return _clone_dataset_from_core(
        core=core,
        preprocessed_frame=core.preprocessed_frame,
        max_encoder_length=max_encoder_length,
        min_prediction_idx=None,
        reuse_data=True,
    )


def clone_validation_dataset_from_core(
    *,
    core: TimeSeriesDatasetCore,
    max_encoder_length: int,
    kept_prediction_row_ids: set[int],
    valid_weight_by_row_id: dict[int, float],
) -> Any:
    trimmed_frame = _trim_validation_preprocessed_frame(
        core.preprocessed_frame,
        max_encoder_length=max_encoder_length,
        kept_prediction_row_ids=kept_prediction_row_ids,
        valid_weight_by_row_id=valid_weight_by_row_id,
    )
    return _clone_dataset_from_core(
        core=core,
        preprocessed_frame=trimmed_frame,
        max_encoder_length=max_encoder_length,
        min_prediction_idx=max_encoder_length,
        reuse_data=False,
    )


def _clone_dataset_from_core(
    *,
    core: TimeSeriesDatasetCore,
    preprocessed_frame: pd.DataFrame,
    max_encoder_length: int,
    min_prediction_idx: int | None,
    reuse_data: bool,
) -> Any:
    cloned = object.__new__(core.dataset.__class__)
    cloned.__dict__ = core.dataset.__dict__.copy()
    cloned.max_encoder_length = max_encoder_length
    cloned.min_encoder_length = max_encoder_length
    if min_prediction_idx is not None:
        cloned.min_prediction_idx = min_prediction_idx
    cloned._praedixa_preprocessed_frame = preprocessed_frame
    with suppress_tft_dataframe_fragmentation_warnings():
        if reuse_data:
            cloned.data = core.dataset.data
        else:
            cloned.data = cloned._data_to_tensors(preprocessed_frame)
        cloned.index = cloned._construct_index(preprocessed_frame, predict_mode=cloned.predict_mode)
    return cloned


def _trim_validation_preprocessed_frame(
    preprocessed_frame: pd.DataFrame,
    *,
    max_encoder_length: int,
    kept_prediction_row_ids: set[int],
    valid_weight_by_row_id: dict[int, float],
) -> pd.DataFrame:
    grouped_parts: list[pd.DataFrame] = []
    for _, group_frame in preprocessed_frame.groupby(GROUP_COL, sort=False):
        valid_rows = group_frame.loc[group_frame[SPLIT_COL] == "valid"].copy()
        valid_rows = valid_rows.loc[valid_rows[PREDICTION_ROW_ID_COL].isin(kept_prediction_row_ids)].copy()
        if valid_rows.empty:
            continue
        history_rows = group_frame.loc[group_frame[SPLIT_COL] == "train"].tail(max_encoder_length).copy()
        valid_rows[WEIGHT_COL] = valid_rows[PREDICTION_ROW_ID_COL].map(valid_weight_by_row_id).astype(float)
        grouped_parts.append(pd.concat([history_rows, valid_rows], ignore_index=True))
    if not grouped_parts:
        return pd.DataFrame(columns=preprocessed_frame.columns)
    return defragment_frame(pd.concat(grouped_parts, ignore_index=True))
