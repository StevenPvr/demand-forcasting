from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl


LOGGER = logging.getLogger(__name__)


def _fold_ids(frame_length: int, folds: list[dict[str, object]]) -> np.ndarray:
    fold_ids = np.full(frame_length, -1, dtype=np.int16)
    for fold in folds:
        valid_idx = np.asarray(fold["valid_idx"], dtype=np.int32)
        fold_ids[valid_idx] = int(cast(Any, fold["fold"]))
    return fold_ids


def evaluate_polars_baseline_macro(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    baseline_name: str,
    baseline_column_name: str,
    absolute_target_col: str,
    dataset_source_col: str,
    logger: logging.Logger = LOGGER,
) -> dict[str, object] | None:
    started_at = time.perf_counter()
    logger.info(
        "Starting statistical baseline evaluation: baseline=%s fold_evaluations=%s engine=polars threads=%s",
        baseline_name,
        len(folds),
        pl.thread_pool_size(),
    )
    fold_ids = _fold_ids(len(frame), folds)
    baseline_frame = pl.DataFrame(
        {
            "dataset_source": frame[dataset_source_col].astype("string").fillna("<NA>").astype(str),
            "fold": fold_ids,
            "target": pd.to_numeric(frame[absolute_target_col], errors="coerce").to_numpy(
                dtype=np.float64,
                copy=False,
            ),
            "prediction": pd.to_numeric(frame[baseline_column_name], errors="coerce").to_numpy(
                dtype=np.float64,
                copy=False,
            ),
        }
    )
    fold_metrics = (
        baseline_frame.lazy()
        .filter(
            (pl.col("fold") > 0)
            & pl.col("target").is_finite()
            & pl.col("prediction").is_finite()
        )
        .group_by(["dataset_source", "fold"])
        .agg(
            [
                pl.len().alias("rows_scored"),
                pl.col("target").abs().sum().alias("target_abs_sum"),
                (pl.col("target") - pl.col("prediction")).abs().sum().alias("abs_error_sum"),
            ]
        )
        .with_columns(
            pl.when(pl.col("target_abs_sum") > 0.0)
            .then(pl.col("abs_error_sum") / pl.col("target_abs_sum"))
            .otherwise(None)
            .alias("wape")
        )
        .drop(["target_abs_sum", "abs_error_sum"])
        .drop_nulls(["wape"])
        .collect()
    )
    if fold_metrics.is_empty():
        return None
    dataset_metrics = fold_metrics.group_by("dataset_source").agg(
        pl.col("wape").mean().alias("dataset_mean_wape")
    )
    mean_wape = float(cast(Any, dataset_metrics["dataset_mean_wape"].mean()))
    fold_results = [
        {
            "dataset_source": str(row["dataset_source"]),
            "fold": int(row["fold"]),
            "wape": float(row["wape"]),
            "rows_scored": int(row["rows_scored"]),
        }
        for row in fold_metrics.sort(["dataset_source", "fold"]).to_dicts()
    ]
    dataset_mean_wape = {
        str(row["dataset_source"]): float(row["dataset_mean_wape"])
        for row in dataset_metrics.sort("dataset_source").to_dicts()
    }
    result: dict[str, object] = {
        "baseline_name": baseline_name,
        "mean_wape": mean_wape,
        "dataset_mean_wape": dataset_mean_wape,
        "fold_wape_scores": fold_results,
    }
    logger.info(
        "Finished statistical baseline evaluation: baseline=%s mean_wape=%.6f duration_seconds=%.3f engine=polars",
        baseline_name,
        mean_wape,
        time.perf_counter() - started_at,
    )
    return result


__all__ = ["evaluate_polars_baseline_macro"]
