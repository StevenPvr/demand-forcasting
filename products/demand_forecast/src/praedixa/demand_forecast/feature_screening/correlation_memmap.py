from __future__ import annotations

import logging
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from praedixa.demand_forecast.feature_screening.correlation_common import corr_arrays
from praedixa.platform.utils.memory import load_numeric_parquet_column


LOGGER = logging.getLogger(__name__)


def build_candidate_memmap(
    source_path: Path,
    candidate_cols: list[str],
    selection_mask: np.ndarray,
    work_dir: Path,
    *,
    memmap_progress_log_every: int,
    logger: logging.Logger = LOGGER,
) -> tuple[Path, np.memmap]:
    with tempfile.NamedTemporaryFile(prefix="lag_candidates_", suffix=".mmap", dir=work_dir, delete=False) as handle:
        mmap_path = Path(handle.name)
    row_count = int(selection_mask.sum())
    candidate_matrix = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(row_count, len(candidate_cols)))
    for index, column in enumerate(candidate_cols, start=1):
        column_values = load_numeric_parquet_column(source_path, column)
        candidate_matrix[:, index - 1] = column_values[selection_mask]
        log_correlation_progress(
            processed=index,
            total=len(candidate_cols),
            progress_log_every=memmap_progress_log_every,
            logger=logger,
            stage="memmap materialization",
        )
    candidate_matrix.flush()
    return mmap_path, candidate_matrix


def log_correlation_progress(
    *,
    processed: int,
    total: int,
    progress_log_every: int,
    logger: logging.Logger,
    stage: str,
) -> None:
    if processed == 1 or processed == total or processed % progress_log_every == 0:
        logger.info("Correlation %s progress: processed=%s/%s", stage, processed, total)


def candidate_column_values(
    candidate_matrix: np.memmap,
    candidate_to_index: dict[str, int],
    candidate: str,
) -> np.ndarray:
    return np.asarray(candidate_matrix[:, candidate_to_index[candidate]], dtype=np.float32)


def scan_target_correlations_memmap(
    *,
    candidate_matrix: np.memmap,
    candidate_cols: list[str],
    candidate_to_index: dict[str, int],
    target_values: np.ndarray,
    correlation_progress_log_every: int,
    logger: logging.Logger,
) -> tuple[list[str], dict[str, str], dict[str, float]]:
    survivors: list[str] = []
    zero_var_dropped: dict[str, str] = {}
    target_abs_corr: dict[str, float] = {}
    for index, candidate in enumerate(candidate_cols, start=1):
        candidate_values = candidate_column_values(candidate_matrix, candidate_to_index, candidate)
        if pd.Series(candidate_values).nunique(dropna=False) <= 1:
            zero_var_dropped[candidate] = "zero_variance"
        else:
            survivors.append(candidate)
            target_abs_corr[candidate] = corr_arrays(candidate_values, target_values, "pearson")
        log_correlation_progress(
            processed=index,
            total=len(candidate_cols),
            progress_log_every=correlation_progress_log_every,
            logger=logger,
            stage="target scan",
        )
    return survivors, zero_var_dropped, target_abs_corr


def empty_memmap_correlation_report(
    *,
    candidate_cols: list[str],
    target_abs_corr: dict[str, float],
    zero_var_dropped: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": column,
                "target_abs_corr": float(target_abs_corr.get(column, 0.0)),
                "pearson_selected": False,
                "spearman_selected": False,
                "dropped_by": zero_var_dropped.get(column, "unknown"),
                "dropped_with": "",
                "corr_value": np.nan,
            }
            for column in candidate_cols
        ]
    )


def prune_ranked_candidates_memmap(
    *,
    candidate_matrix: np.memmap,
    ranked_candidates: list[str],
    candidate_to_index: dict[str, int],
    method: str,
    threshold: float,
    correlation_progress_log_every: int,
    logger: logging.Logger,
) -> tuple[list[str], dict[str, tuple[str, float, str]]]:
    survivors: list[str] = []
    dropped: dict[str, tuple[str, float, str]] = {}
    for index, candidate in enumerate(ranked_candidates, start=1):
        drop_reason = memmap_drop_reason(
            candidate_matrix=candidate_matrix,
            candidate=candidate,
            survivors=survivors,
            candidate_to_index=candidate_to_index,
            method=method,
            threshold=threshold,
        )
        if drop_reason is None:
            survivors.append(candidate)
        else:
            dropped[candidate] = drop_reason
        log_correlation_progress(
            processed=index,
            total=len(ranked_candidates),
            progress_log_every=correlation_progress_log_every,
            logger=logger,
            stage=f"{method} prune",
        )
    return survivors, dropped


def memmap_drop_reason(
    *,
    candidate_matrix: np.memmap,
    candidate: str,
    survivors: list[str],
    candidate_to_index: dict[str, int],
    method: str,
    threshold: float,
) -> tuple[str, float, str] | None:
    candidate_values = candidate_column_values(candidate_matrix, candidate_to_index, candidate)
    for survivor in survivors:
        survivor_values = candidate_column_values(candidate_matrix, candidate_to_index, survivor)
        corr_value = corr_arrays(candidate_values, survivor_values, method)
        if corr_value >= threshold:
            return survivor, corr_value, method
    return None


def build_memmap_correlation_report(
    *,
    candidate_cols: list[str],
    target_abs_corr: dict[str, float],
    zero_var_dropped: dict[str, str],
    pearson_survivors: list[str],
    pearson_dropped: dict[str, tuple[str, float, str]],
    spearman_survivors: list[str],
    spearman_dropped: dict[str, tuple[str, float, str]],
) -> pd.DataFrame:
    report_rows = [
        memmap_correlation_report_row(
            column=column,
            target_abs_corr=target_abs_corr,
            zero_var_dropped=zero_var_dropped,
            pearson_dropped=pearson_dropped,
            spearman_dropped=spearman_dropped,
            pearson_survivors=pearson_survivors,
            spearman_survivors=spearman_survivors,
        )
        for column in candidate_cols
    ]
    return pd.DataFrame(report_rows).sort_values(
        by=["spearman_selected", "pearson_selected", "target_abs_corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def memmap_correlation_report_row(
    *,
    column: str,
    target_abs_corr: dict[str, float],
    zero_var_dropped: dict[str, str],
    pearson_dropped: dict[str, tuple[str, float, str]],
    spearman_dropped: dict[str, tuple[str, float, str]],
    pearson_survivors: list[str],
    spearman_survivors: list[str],
) -> dict[str, object]:
    dropped_by, dropped_with, corr_value = memmap_drop_details(
        column=column,
        zero_var_dropped=zero_var_dropped,
        pearson_dropped=pearson_dropped,
        spearman_dropped=spearman_dropped,
    )
    return {
        "feature": column,
        "target_abs_corr": float(target_abs_corr.get(column, 0.0)),
        "pearson_selected": column in set(pearson_survivors),
        "spearman_selected": column in set(spearman_survivors),
        "dropped_by": dropped_by,
        "dropped_with": dropped_with,
        "corr_value": corr_value,
    }


def memmap_drop_details(
    *,
    column: str,
    zero_var_dropped: dict[str, str],
    pearson_dropped: dict[str, tuple[str, float, str]],
    spearman_dropped: dict[str, tuple[str, float, str]],
) -> tuple[str, str, float]:
    if column in zero_var_dropped:
        return zero_var_dropped[column], "", float("nan")
    if column in pearson_dropped:
        dropped_with, corr_value, dropped_by = pearson_dropped[column]
        return dropped_by, dropped_with, corr_value
    if column in spearman_dropped:
        dropped_with, corr_value, dropped_by = spearman_dropped[column]
        return dropped_by, dropped_with, corr_value
    return "", "", float("nan")


def filter_correlated_lag_features_memmap(
    candidate_matrix: np.memmap,
    candidate_cols: list[str],
    target_values: np.ndarray,
    *,
    pearson_threshold: float,
    spearman_threshold: float,
    correlation_progress_log_every: int,
    logger: logging.Logger = LOGGER,
) -> tuple[list[str], pd.DataFrame]:
    log_memmap_filter_start(candidate_cols, pearson_threshold, spearman_threshold, logger)
    candidate_to_index, survivors, zero_var_dropped, target_abs_corr = memmap_scan_state(
        candidate_matrix=candidate_matrix,
        candidate_cols=candidate_cols,
        target_values=target_values,
        correlation_progress_log_every=correlation_progress_log_every,
        logger=logger,
    )
    if not survivors:
        return _empty_memmap_filter_result(candidate_cols, target_abs_corr, zero_var_dropped)
    pearson_survivors, pearson_dropped, spearman_survivors, spearman_dropped = memmap_pruning_stages(
        candidate_matrix=candidate_matrix,
        survivors=survivors,
        target_abs_corr=target_abs_corr,
        candidate_to_index=candidate_to_index,
        pearson_threshold=pearson_threshold,
        spearman_threshold=spearman_threshold,
        correlation_progress_log_every=correlation_progress_log_every,
        logger=logger,
    )
    report = build_memmap_correlation_report(
        candidate_cols=candidate_cols,
        target_abs_corr=target_abs_corr,
        zero_var_dropped=zero_var_dropped,
        pearson_survivors=pearson_survivors,
        pearson_dropped=pearson_dropped,
        spearman_survivors=spearman_survivors,
        spearman_dropped=spearman_dropped,
    )
    _log_memmap_filter_complete(candidate_cols, survivors, pearson_survivors, spearman_survivors, logger)
    return spearman_survivors, report


def memmap_scan_state(
    *,
    candidate_matrix: np.memmap,
    candidate_cols: list[str],
    target_values: np.ndarray,
    correlation_progress_log_every: int,
    logger: logging.Logger,
) -> tuple[dict[str, int], list[str], dict[str, str], dict[str, float]]:
    candidate_to_index = {column: index for index, column in enumerate(candidate_cols)}
    survivors, zero_var_dropped, target_abs_corr = scan_target_correlations_memmap(
        candidate_matrix=candidate_matrix,
        candidate_cols=candidate_cols,
        candidate_to_index=candidate_to_index,
        target_values=target_values,
        correlation_progress_log_every=correlation_progress_log_every,
        logger=logger,
    )
    return candidate_to_index, survivors, zero_var_dropped, target_abs_corr


def _empty_memmap_filter_result(
    candidate_cols: list[str],
    target_abs_corr: dict[str, float],
    zero_var_dropped: dict[str, str],
) -> tuple[list[str], pd.DataFrame]:
    return [], empty_memmap_correlation_report(
        candidate_cols=candidate_cols,
        target_abs_corr=target_abs_corr,
        zero_var_dropped=zero_var_dropped,
    )


def _log_memmap_filter_complete(
    candidate_cols: list[str],
    survivors: list[str],
    pearson_survivors: list[str],
    spearman_survivors: list[str],
    logger: logging.Logger,
) -> None:
    logger.info(
        "Correlation filtering complete: initial=%s after_zero_variance=%s after_pearson=%s after_spearman=%s",
        len(candidate_cols),
        len(survivors),
        len(pearson_survivors),
        len(spearman_survivors),
    )


def log_memmap_filter_start(
    candidate_cols: list[str],
    pearson_threshold: float,
    spearman_threshold: float,
    logger: logging.Logger,
) -> None:
    logger.info(
        "Starting correlation filtering on memmap candidates: candidates=%s pearson_threshold=%.2f spearman_threshold=%.2f",
        len(candidate_cols),
        pearson_threshold,
        spearman_threshold,
    )


def memmap_pruning_stages(
    *,
    candidate_matrix: np.memmap,
    survivors: list[str],
    target_abs_corr: dict[str, float],
    candidate_to_index: dict[str, int],
    pearson_threshold: float,
    spearman_threshold: float,
    correlation_progress_log_every: int,
    logger: logging.Logger,
) -> tuple[list[str], dict[str, tuple[str, float, str]], list[str], dict[str, tuple[str, float, str]]]:
    ranked_candidates = sorted(survivors, key=lambda column: target_abs_corr[column], reverse=True)
    pearson_survivors, pearson_dropped = prune_ranked_candidates_memmap(
        candidate_matrix=candidate_matrix,
        ranked_candidates=ranked_candidates,
        candidate_to_index=candidate_to_index,
        method="pearson",
        threshold=pearson_threshold,
        correlation_progress_log_every=correlation_progress_log_every,
        logger=logger,
    )
    spearman_survivors, spearman_dropped = prune_ranked_candidates_memmap(
        candidate_matrix=candidate_matrix,
        ranked_candidates=pearson_survivors,
        candidate_to_index=candidate_to_index,
        method="spearman",
        threshold=spearman_threshold,
        correlation_progress_log_every=correlation_progress_log_every,
        logger=logger,
    )
    return pearson_survivors, pearson_dropped, spearman_survivors, spearman_dropped
