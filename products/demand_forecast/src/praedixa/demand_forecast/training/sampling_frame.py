from __future__ import annotations

import numpy as np
import pandas as pd

from praedixa.demand_forecast.training.sampling_common import resolve_sampling_order_cols
from praedixa.demand_forecast.training.sampling_models import FrameSamplingSpec


def sample_frame_stratified_by_date_store(
    frame: pd.DataFrame,
    *,
    spec: FrameSamplingSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty:
        return frame.copy(), _empty_sampling_metadata(spec=spec)
    _validate_sampling_inputs(spec=spec)
    ordered, sort_columns = _ordered_sampling_frame(frame=frame, spec=spec)
    sampled_parts, dataset_metadata = _sampled_datasets(ordered, spec=spec)
    sampled_frame = pd.concat(sampled_parts, axis=0, ignore_index=False)
    sampled_frame = sampled_frame.sort_values(sort_columns).reset_index(drop=True)
    return sampled_frame, {
        "sample_fraction": float(spec.sample_fraction),
        "min_samples_per_dataset": int(spec.min_samples_per_dataset),
        "sample_strategy": "date_store_stratified",
        "sample_store_col": spec.sample_store_col,
        "original_rows": int(frame.shape[0]),
        "sampled_rows": int(sampled_frame.shape[0]),
        "datasets": dataset_metadata,
    }


def _empty_sampling_metadata(*, spec: FrameSamplingSpec) -> dict[str, object]:
    return {
        "sample_fraction": float(spec.sample_fraction),
        "min_samples_per_dataset": int(spec.min_samples_per_dataset),
        "sample_strategy": "date_store_stratified",
        "sample_store_col": spec.sample_store_col,
        "original_rows": 0,
        "sampled_rows": 0,
        "datasets": {},
    }


def _validate_sampling_inputs(*, spec: FrameSamplingSpec) -> None:
    if not 0.0 < spec.sample_fraction <= 1.0:
        raise ValueError("Sampling fraction must be in (0, 1].")
    if spec.min_samples_per_dataset <= 0:
        raise ValueError("Minimum samples per dataset must be positive.")


def _ordered_sampling_frame(
    *,
    frame: pd.DataFrame,
    spec: FrameSamplingSpec,
) -> tuple[pd.DataFrame, list[str]]:
    ordered = frame.copy()
    ordered[spec.date_col] = pd.to_datetime(ordered[spec.date_col])
    sort_columns = [
        spec.dataset_source_col,
        spec.date_col,
        spec.sample_store_col,
        *resolve_sampling_order_cols(ordered),
    ]
    ordered = ordered.sort_values(sort_columns).reset_index(drop=True)
    return ordered, sort_columns


def _sampled_datasets(
    ordered: pd.DataFrame,
    *,
    spec: FrameSamplingSpec,
) -> tuple[list[pd.DataFrame], dict[str, dict[str, int | float]]]:
    sampled_parts: list[pd.DataFrame] = []
    dataset_metadata: dict[str, dict[str, int | float]] = {}
    for dataset_source, dataset_frame in ordered.groupby(spec.dataset_source_col, sort=False):
        sampled_dataset_frame = _sampled_dataset_frame(dataset_frame=dataset_frame, spec=spec)
        sampled_parts.append(sampled_dataset_frame)
        dataset_metadata[str(dataset_source)] = _sampled_dataset_metadata(
            dataset_frame=dataset_frame,
            sampled_dataset_frame=sampled_dataset_frame,
            spec=spec,
        )
    return sampled_parts, dataset_metadata


def _sampled_dataset_frame(
    *,
    dataset_frame: pd.DataFrame,
    spec: FrameSamplingSpec,
) -> pd.DataFrame:
    sampled_strata_parts = _sampled_strata_parts(dataset_frame, spec=spec)
    sampled_dataset_frame = (
        pd.concat(sampled_strata_parts, axis=0, ignore_index=False)
        if sampled_strata_parts
        else dataset_frame.iloc[0:0].copy()
    )
    return _top_up_sampled_dataset_frame(
        dataset_frame=dataset_frame,
        sampled_dataset_frame=sampled_dataset_frame,
        target_sample_size=_target_dataset_sample_size(dataset_frame=dataset_frame, spec=spec),
    )


def _sampled_strata_parts(
    dataset_frame: pd.DataFrame,
    *,
    spec: FrameSamplingSpec,
) -> list[pd.DataFrame]:
    per_dataset_parts: list[pd.DataFrame] = []
    for _, stratum_frame in dataset_frame.groupby([spec.date_col, spec.sample_store_col], sort=False):
        sample_size = int(np.floor(stratum_frame.shape[0] * spec.sample_fraction))
        if sample_size <= 0:
            continue
        per_dataset_parts.append(stratum_frame.head(sample_size))
    return per_dataset_parts


def _target_dataset_sample_size(
    *,
    dataset_frame: pd.DataFrame,
    spec: FrameSamplingSpec,
) -> int:
    return int(
        min(
            dataset_frame.shape[0],
            max(
                int(np.ceil(dataset_frame.shape[0] * spec.sample_fraction)),
                spec.min_samples_per_dataset,
            ),
        )
    )


def _top_up_sampled_dataset_frame(
    *,
    dataset_frame: pd.DataFrame,
    sampled_dataset_frame: pd.DataFrame,
    target_sample_size: int,
) -> pd.DataFrame:
    if sampled_dataset_frame.shape[0] >= target_sample_size:
        return sampled_dataset_frame
    remaining_dataset_frame = dataset_frame.loc[~dataset_frame.index.isin(sampled_dataset_frame.index)]
    top_up_count = target_sample_size - sampled_dataset_frame.shape[0]
    if top_up_count <= 0 or remaining_dataset_frame.empty:
        return sampled_dataset_frame
    return pd.concat(
        [sampled_dataset_frame, remaining_dataset_frame.head(top_up_count)],
        axis=0,
        ignore_index=False,
    )


def _sampled_dataset_metadata(
    *,
    dataset_frame: pd.DataFrame,
    sampled_dataset_frame: pd.DataFrame,
    spec: FrameSamplingSpec,
) -> dict[str, int | float]:
    return {
        "original_rows": int(dataset_frame.shape[0]),
        "sampled_rows": int(sampled_dataset_frame.shape[0]),
        "sample_fraction": float(spec.sample_fraction),
        "min_samples_per_dataset": int(spec.min_samples_per_dataset),
        "store_count": int(dataset_frame[spec.sample_store_col].nunique()),
        "unique_dates": int(dataset_frame[spec.date_col].nunique()),
        "strata_count": int(dataset_frame.groupby([spec.date_col, spec.sample_store_col], sort=False).ngroups),
    }


__all__ = ["sample_frame_stratified_by_date_store"]
