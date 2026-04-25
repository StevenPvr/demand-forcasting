from __future__ import annotations

import logging
from typing import cast

import pandas as pd


def resolve_sampling_store_col(frame: pd.DataFrame) -> str:
    candidate_columns = ("location_id", "store_id", "client_id", "product_id")
    for candidate in candidate_columns:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Unable to resolve a sampling store column. Tried: {', '.join(candidate_columns)}."
    )


def log_sampling_summary(
    label: str,
    metadata: dict[str, object],
    *,
    logger: logging.Logger,
) -> None:
    logger.info(
        "%s sampling summary: sampled_rows=%s original_rows=%s requested_sample_fraction=%.4f effective_sample_fraction=%.4f strategy=%s sample_store_col=%s",
        label,
        metadata["sampled_rows"],
        metadata["original_rows"],
        metadata["sample_fraction"],
        metadata.get("effective_sample_fraction", metadata["sample_fraction"]),
        metadata["sample_strategy"],
        metadata["sample_store_col"],
    )
    datasets = cast(dict[str, dict[str, object]], metadata["datasets"])
    for dataset_source, dataset_metadata in datasets.items():
        logger.info(
            "%s retained rows: dataset=%s sampled_rows=%s original_rows=%s requested_sample_fraction=%.4f effective_sample_fraction=%.4f stores=%s unique_dates=%s strata=%s",
            label,
            dataset_source,
            dataset_metadata["sampled_rows"],
            dataset_metadata["original_rows"],
            dataset_metadata.get("sample_fraction", metadata["sample_fraction"]),
            dataset_metadata.get(
                "effective_sample_fraction",
                dataset_metadata.get("sample_fraction", metadata["sample_fraction"]),
            ),
            dataset_metadata["store_count"],
            dataset_metadata["unique_dates"],
            dataset_metadata["strata_count"],
        )


def resolve_sampling_order_columns(
    selected_columns: list[str],
    *,
    date_col: str,
    sample_store_col: str,
) -> list[str]:
    ordering_columns = [column for column in (date_col, sample_store_col) if column in selected_columns]
    return _append_sampling_order_candidates(ordering_columns, selected_columns)


def _append_sampling_order_candidates(
    ordering_columns: list[str],
    selected_columns: list[str],
) -> list[str]:
    resolved = list(ordering_columns)
    for candidate in ("location_id", "product_id", "client_id"):
        if candidate in selected_columns and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def resolve_sampling_order_cols(frame: pd.DataFrame) -> list[str]:
    order_cols: list[str] = []
    for candidate in ("product_id", "client_id"):
        if candidate in frame.columns:
            order_cols.append(candidate)
    return order_cols


__all__ = [
    "log_sampling_summary",
    "resolve_sampling_order_cols",
    "resolve_sampling_order_columns",
    "resolve_sampling_store_col",
]
