from __future__ import annotations

from dataclasses import dataclass
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
    prepared_frame = defragment_frame(training_frame)
    with suppress_tft_dataframe_fragmentation_warnings():
        dataset = dataset_class(
            prepared_frame,
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
            time_varying_unknown_categoricals=layout[
                "time_varying_unknown_categoricals"
            ],
            time_varying_unknown_reals=[
                *layout["time_varying_unknown_reals"],
                target_col,
            ],
            allow_missing_timesteps=False,
            add_target_scales=True,
            target_normalizer=imports["GroupNormalizer"](
                groups=[GROUP_COL], method="standard"
            ),
            categorical_encoders=categorical_encoders,
            scalers=real_feature_scalers,
        )
    return TimeSeriesDatasetCore(dataset=dataset, preprocessed_frame=prepared_frame)


def build_validation_dataset_core(
    *,
    training_core: TimeSeriesDatasetCore,
    validation_frame: pd.DataFrame,
    min_prediction_idx: int,
) -> TimeSeriesDatasetCore:
    prepared_frame = defragment_frame(validation_frame)
    with suppress_tft_dataframe_fragmentation_warnings():
        dataset = training_core.dataset.__class__.from_dataset(
            training_core.dataset,
            prepared_frame,
            min_prediction_idx=min_prediction_idx,
            stop_randomization=True,
        )
    return TimeSeriesDatasetCore(dataset=dataset, preprocessed_frame=prepared_frame)


def clone_training_dataset_from_core(
    *,
    core: TimeSeriesDatasetCore,
    max_encoder_length: int,
) -> Any:
    return _dataset_from_core_parameters(
        core=core,
        frame=core.preprocessed_frame,
        max_encoder_length=max_encoder_length,
        min_prediction_idx=None,
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
    return _dataset_from_core_parameters(
        core=core,
        frame=trimmed_frame,
        max_encoder_length=max_encoder_length,
        min_prediction_idx=max_encoder_length,
    )


def _dataset_from_core_parameters(
    *,
    core: TimeSeriesDatasetCore,
    frame: pd.DataFrame,
    max_encoder_length: int,
    min_prediction_idx: int | None,
) -> Any:
    parameters = dict(cast(dict[str, Any], core.dataset.get_parameters()))
    parameters["max_encoder_length"] = max_encoder_length
    parameters["min_encoder_length"] = max_encoder_length
    kwargs: dict[str, object] = {"stop_randomization": True}
    if min_prediction_idx is not None:
        kwargs["min_prediction_idx"] = min_prediction_idx
    with suppress_tft_dataframe_fragmentation_warnings():
        return core.dataset.__class__.from_parameters(
            parameters,
            defragment_frame(frame),
            **kwargs,
        )


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
        valid_rows = valid_rows.loc[
            valid_rows[PREDICTION_ROW_ID_COL].isin(kept_prediction_row_ids)
        ].copy()
        if valid_rows.empty:
            continue
        history_rows = (
            group_frame.loc[group_frame[SPLIT_COL] == "train"]
            .tail(max_encoder_length)
            .copy()
        )
        valid_rows[WEIGHT_COL] = (
            valid_rows[PREDICTION_ROW_ID_COL].map(valid_weight_by_row_id).astype(float)
        )
        grouped_parts.append(pd.concat([history_rows, valid_rows], ignore_index=True))
    if not grouped_parts:
        return pd.DataFrame(columns=preprocessed_frame.columns)
    return defragment_frame(pd.concat(grouped_parts, ignore_index=True))
