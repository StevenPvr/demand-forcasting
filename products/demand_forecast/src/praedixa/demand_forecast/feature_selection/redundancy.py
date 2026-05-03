"""Redundancy filters used before selector-model optimisation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from numbers import Real
import os
from typing import cast

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def drop_linear_correlated_candidate_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str,
    correlation_threshold: float,
) -> tuple[list[str], list[str]]:
    """Drop numeric candidates with Pearson redundancy above the threshold."""

    return _drop_redundant_features(
        train_df,
        candidate_columns,
        target_column,
        correlation_threshold,
        association_fn=_pearson_association,
        stage_name="linear",
    )


def drop_nonlinear_correlated_candidate_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str,
    correlation_threshold: float,
) -> tuple[list[str], list[str]]:
    """Drop numeric candidates with nonlinear redundancy above the threshold."""

    return _drop_redundant_features(
        train_df,
        candidate_columns,
        target_column,
        correlation_threshold,
        association_fn=_nonlinear_association,
        stage_name="nonlinear",
    )


def _drop_redundant_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str,
    correlation_threshold: float,
    *,
    association_fn: "_AssociationFn",
    stage_name: str,
) -> tuple[list[str], list[str]]:
    if len(candidate_columns) <= 1:
        LOGGER.info(
            "Skipping %s correlation redundancy filter: candidate_count=%d",
            stage_name,
            len(candidate_columns),
        )
        return candidate_columns, []
    LOGGER.info(
        "Starting %s correlation redundancy filter: candidate_count=%d threshold=%.3f",
        stage_name,
        len(candidate_columns),
        correlation_threshold,
    )
    ranked_columns = _rank_by_target_correlation(
        train_df,
        candidate_columns,
        target_column,
    )
    pair_associations = _parallel_pair_associations(
        train_df,
        ranked_columns,
        association_fn=association_fn,
        stage_name=stage_name,
    )
    kept_columns: list[str] = []
    dropped_columns: list[str] = []
    for column in ranked_columns:
        if any(
            _pair_association(pair_associations, column, kept_column)
            > correlation_threshold
            for kept_column in kept_columns
        ):
            dropped_columns.append(column)
            continue
        kept_columns.append(column)
    LOGGER.info(
        "Finished %s correlation redundancy filter: kept=%d dropped=%d",
        stage_name,
        len(kept_columns),
        len(dropped_columns),
    )
    return kept_columns, dropped_columns


def _parallel_pair_associations(
    train_df: pd.DataFrame,
    ranked_columns: list[str],
    *,
    association_fn: "_AssociationFn",
    stage_name: str,
) -> dict[tuple[str, str], float]:
    pairs = _candidate_pairs(ranked_columns)
    pair_count = len(pairs)
    worker_count = _correlation_worker_count(pair_count)
    if pair_count == 0:
        return {}
    LOGGER.info(
        "Computing %s feature-pair correlations: pairs=%d workers=%d",
        stage_name,
        pair_count,
        worker_count,
    )
    progress_step = max(1, pair_count // 20)
    results: dict[tuple[str, str], float] = {}

    def compute(pair: tuple[str, str]) -> tuple[tuple[str, str], float]:
        left, right = pair
        return pair, association_fn(train_df[left], train_df[right])

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for completed_count, (pair, association) in enumerate(
            executor.map(compute, pairs),
            start=1,
        ):
            results[pair] = association
            if completed_count % progress_step == 0 or completed_count == pair_count:
                LOGGER.info(
                    "%s correlation progress: %d/%d pairs computed",
                    stage_name,
                    completed_count,
                    pair_count,
                )
    return results


def _candidate_pairs(columns: list[str]) -> list[tuple[str, str]]:
    return [
        (left, right)
        for left_index, left in enumerate(columns)
        for right in columns[left_index + 1 :]
    ]


def _correlation_worker_count(pair_count: int) -> int:
    return max(1, min(pair_count, os.cpu_count() or 1))


def _pair_association(
    pair_associations: dict[tuple[str, str], float],
    left: str,
    right: str,
) -> float:
    return pair_associations.get((left, right), pair_associations[(right, left)])


def _rank_by_target_correlation(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str,
) -> list[str]:
    corr_frame = train_df.loc[:, [*candidate_columns, target_column]].corr().abs()
    corr_to_target = corr_frame[target_column]
    return sorted(
        candidate_columns,
        key=lambda column: (
            -_coerce_float(cast(object, corr_to_target.at[column])),
            column,
        ),
    )


def _pearson_association(left: pd.Series, right: pd.Series) -> float:
    pair = _numeric_pair_frame(left, right)
    if pair.empty:
        return 0.0
    value = pair["left"].corr(pair["right"])
    if pd.isna(value):
        return 0.0
    return abs(float(value))


def _nonlinear_association(left: pd.Series, right: pd.Series) -> float:
    pair = _numeric_pair_frame(left, right)
    if pair.empty:
        return 0.0
    return max(
        _correlation_ratio(pair["left"], pair["right"]),
        _correlation_ratio(pair["right"], pair["left"]),
    )


def _numeric_pair_frame(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if pair["left"].nunique(dropna=False) <= 1:
        return pd.DataFrame(columns=["left", "right"])
    if pair["right"].nunique(dropna=False) <= 1:
        return pd.DataFrame(columns=["left", "right"])
    return pair


def _correlation_ratio(categories_source: pd.Series, values_source: pd.Series) -> float:
    categories = _quantile_bins(categories_source)
    if categories.nunique(dropna=True) <= 1:
        return 0.0
    values = pd.to_numeric(values_source, errors="coerce")
    grand_mean = float(values.mean())
    total_var = float(((values - grand_mean) ** 2).sum())
    if total_var <= 0.0:
        return 0.0
    explained = 0.0
    grouped = values.groupby(categories, observed=True)
    for _, group in grouped:
        if group.empty:
            continue
        group_mean = float(group.mean())
        explained += float(len(group)) * ((group_mean - grand_mean) ** 2)
    return float(np.sqrt(max(0.0, min(1.0, explained / total_var))))


def _quantile_bins(values: pd.Series) -> pd.Series:
    unique_count = int(values.nunique(dropna=True))
    bin_count = max(2, min(20, unique_count))
    try:
        return pd.qcut(values, q=bin_count, duplicates="drop")
    except ValueError:
        return pd.cut(values, bins=bin_count, duplicates="drop")


def _coerce_float(value: object) -> float:
    if isinstance(value, Real):
        if np.isnan(float(value)):
            return 0.0
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {type(value).__name__}.")


type _AssociationFn = Callable[[pd.Series, pd.Series], float]
