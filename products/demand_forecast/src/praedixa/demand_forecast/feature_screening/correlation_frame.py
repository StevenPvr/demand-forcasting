from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def drop_zero_variance_columns(frame: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[str], dict[str, str]]:
    survivors: list[str] = []
    dropped: dict[str, str] = {}
    for column in candidate_cols:
        if frame[column].nunique(dropna=False) <= 1:
            dropped[column] = "zero_variance"
        else:
            survivors.append(column)
    return survivors, dropped


def empty_correlation_report(candidate_cols: list[str], zero_var_dropped: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": candidate_cols,
            "target_abs_corr": 0.0,
            "pearson_selected": False,
            "spearman_selected": False,
            "dropped_by": [zero_var_dropped.get(column, "unknown") for column in candidate_cols],
            "dropped_with": "",
            "corr_value": np.nan,
        }
    )


def frame_target_correlations(
    *,
    frame: pd.DataFrame,
    survivors: list[str],
    target_col: str,
) -> pd.Series:
    return frame[survivors].corrwith(frame[target_col], method="pearson").abs().fillna(0.0)


def greedy_corr_filter(
    corr_matrix: pd.DataFrame,
    ranked_candidates: list[str],
    threshold: float,
    stage: str,
) -> tuple[list[str], dict[str, tuple[str, float, str]]]:
    survivors: list[str] = []
    dropped: dict[str, tuple[str, float, str]] = {}
    for column in ranked_candidates:
        drop_reason = candidate_drop_reason(corr_matrix, column, survivors, threshold, stage)
        if drop_reason is None:
            survivors.append(column)
        else:
            dropped[column] = drop_reason
    return survivors, dropped


def candidate_drop_reason(
    corr_matrix: pd.DataFrame,
    column: str,
    survivors: list[str],
    threshold: float,
    stage: str,
) -> tuple[str, float, str] | None:
    for survivor in survivors:
        corr_value = float(np.asarray([corr_matrix.loc[column, survivor]], dtype=float)[0])
        if corr_value >= threshold:
            return survivor, corr_value, stage
    return None


def frame_correlation_report(
    *,
    candidate_cols: list[str],
    target_abs_corr: pd.Series,
    zero_var_dropped: dict[str, str],
    pearson_survivors: list[str],
    pearson_dropped: dict[str, tuple[str, float, str]],
    spearman_survivors: list[str],
    spearman_dropped: dict[str, tuple[str, float, str]],
) -> pd.DataFrame:
    pearson_selected = set(pearson_survivors)
    spearman_selected = set(spearman_survivors)
    report_rows = [
        correlation_report_row(
            column=column,
            target_abs_corr=target_abs_corr,
            zero_var_dropped=zero_var_dropped,
            pearson_dropped=pearson_dropped,
            spearman_dropped=spearman_dropped,
            pearson_selected=pearson_selected,
            spearman_selected=spearman_selected,
        )
        for column in candidate_cols
    ]
    return pd.DataFrame(report_rows).sort_values(
        by=["spearman_selected", "pearson_selected", "target_abs_corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def correlation_report_row(
    *,
    column: str,
    target_abs_corr: pd.Series,
    zero_var_dropped: dict[str, str],
    pearson_dropped: dict[str, tuple[str, float, str]],
    spearman_dropped: dict[str, tuple[str, float, str]],
    pearson_selected: set[str],
    spearman_selected: set[str],
) -> dict[str, object]:
    dropped_by, dropped_with, corr_value = drop_details(
        column=column,
        zero_var_dropped=zero_var_dropped,
        pearson_dropped=pearson_dropped,
        spearman_dropped=spearman_dropped,
    )
    return {
        "feature": column,
        "target_abs_corr": float(target_abs_corr.get(column, 0.0)),
        "pearson_selected": column in pearson_selected,
        "spearman_selected": column in spearman_selected,
        "dropped_by": dropped_by,
        "dropped_with": dropped_with,
        "corr_value": corr_value,
    }


def drop_details(
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


def corr_stage(
    *,
    frame: pd.DataFrame,
    ranked_candidates: list[str],
    threshold: float,
    method: Literal["pearson", "spearman"],
    stage: str,
) -> tuple[list[str], dict[str, tuple[str, float, str]]]:
    candidate_frame = frame.loc[:, ranked_candidates]
    corr = candidate_frame.corr(method=method)
    matrix = corr.apply(np.abs).fillna(0.0)
    return greedy_corr_filter(
        corr_matrix=matrix,
        ranked_candidates=ranked_candidates,
        threshold=threshold,
        stage=stage,
    )


def filter_correlated_lag_features(
    frame: pd.DataFrame,
    candidate_cols: list[str],
    *,
    target_col: str,
    pearson_threshold: float,
    spearman_threshold: float,
    logger: logging.Logger = LOGGER,
) -> tuple[list[str], pd.DataFrame]:
    survivors, zero_var_dropped = drop_zero_variance_columns(frame, candidate_cols)
    if not survivors:
        return [], empty_correlation_report(candidate_cols, zero_var_dropped)
    target_abs_corr = frame_target_correlations(frame=frame, survivors=survivors, target_col=target_col)
    ranked_candidates = list(target_abs_corr.sort_values(ascending=False).index.tolist())
    pearson_survivors, pearson_dropped = corr_stage(
        frame=frame,
        ranked_candidates=ranked_candidates,
        threshold=pearson_threshold,
        method="pearson",
        stage="pearson",
    )
    spearman_survivors, spearman_dropped = corr_stage(
        frame=frame,
        ranked_candidates=pearson_survivors,
        threshold=spearman_threshold,
        method="spearman",
        stage="spearman",
    )
    report = frame_correlation_report(
        candidate_cols=candidate_cols,
        target_abs_corr=target_abs_corr,
        zero_var_dropped=zero_var_dropped,
        pearson_survivors=pearson_survivors,
        pearson_dropped=pearson_dropped,
        spearman_survivors=spearman_survivors,
        spearman_dropped=spearman_dropped,
    )
    logger.info(
        "Correlation filtering complete: initial=%s after_zero_variance=%s after_pearson=%s after_spearman=%s",
        len(candidate_cols),
        len(survivors),
        len(pearson_survivors),
        len(spearman_survivors),
    )
    return spearman_survivors, report
