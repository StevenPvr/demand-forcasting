from __future__ import annotations

import logging
from typing import Any, Mapping, cast

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _sorted_unique_dates(frame: pd.DataFrame, date_col: str) -> pd.DatetimeIndex:
    date_values = pd.Series(pd.to_datetime(frame[date_col], errors="raise"), copy=False)
    return pd.DatetimeIndex(date_values.drop_duplicates().sort_values())


def _format_date_bound(value: object) -> str:
    return pd.Timestamp(cast(Any, value)).strftime("%Y-%m-%d")


def split_chronological_train_tuning(
    frame: pd.DataFrame,
    *,
    date_col: str = "dt",
    train_fraction: float = 0.7,
    logger: logging.Logger = LOGGER,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    """Decoupe un panel en train de selection et holdout tuning par dates uniques."""

    dated = frame.copy()
    dated[date_col] = pd.to_datetime(dated[date_col])
    dated = dated.sort_values([date_col]).reset_index(drop=True)

    unique_dates = _sorted_unique_dates(dated, date_col)
    split_idx = max(1, int(len(unique_dates) * train_fraction))
    split_idx = min(split_idx, len(unique_dates) - 1)

    train_dates = unique_dates[:split_idx]
    holdout_dates = unique_dates[split_idx:]
    dated_dates = dated[date_col]

    selection_train = dated[dated_dates.isin(train_dates.tolist())].copy().reset_index(drop=True)
    tuning_holdout = dated[dated_dates.isin(holdout_dates.tolist())].copy().reset_index(drop=True)

    metadata: dict[str, object] = {
        "train_fraction": train_fraction,
        "total_rows": int(len(frame)),
        "train_rows": int(len(selection_train)),
        "holdout_rows": int(len(tuning_holdout)),
        "total_unique_dates": int(len(unique_dates)),
        "train_unique_dates": int(len(train_dates)),
        "holdout_unique_dates": int(len(holdout_dates)),
        "train_start_date": _format_date_bound(train_dates.min()),
        "train_end_date": _format_date_bound(train_dates.max()),
        "holdout_start_date": _format_date_bound(holdout_dates.min()),
        "holdout_end_date": _format_date_bound(holdout_dates.max()),
    }
    logger.info(
        "Chronological split complete: train_rows=%s holdout_rows=%s train_dates=%s holdout_dates=%s",
        metadata["train_rows"],
        metadata["holdout_rows"],
        metadata["train_unique_dates"],
        metadata["holdout_unique_dates"],
    )
    return selection_train, tuning_holdout, metadata


def build_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    date_col: str = "dt",
    n_folds: int = 5,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, object]]:
    """Construit des folds expanding-window sur des dates uniques ordonnees."""
    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col]).reset_index(drop=True)
    unique_dates = _sorted_unique_dates(ordered, date_col)
    ordered_dates = ordered[date_col]
    valid_dates_per_fold = _validate_fold_request(unique_dates, n_folds)
    folds = [
        _walk_forward_fold(ordered, ordered_dates, unique_dates, fold_idx, valid_dates_per_fold)
        for fold_idx in range(n_folds)
    ]
    if any(fold is None for fold in folds):
        raise ValueError(f"Expected {n_folds} folds, got {sum(fold is not None for fold in folds)}.")
    logger.info(
        "Walk-forward folds built: n_folds=%s valid_dates_per_fold=%s",
        n_folds,
        valid_dates_per_fold,
    )
    return [cast(dict[str, object], fold) for fold in folds]


def _validate_fold_request(unique_dates: pd.DatetimeIndex, n_folds: int) -> int:
    if len(unique_dates) < (n_folds + 1):
        raise ValueError(f"Need at least {n_folds + 1} unique dates to build {n_folds} walk-forward folds.")
    return max(1, len(unique_dates) // (n_folds + 1))


def _walk_forward_fold(
    ordered: pd.DataFrame,
    ordered_dates: pd.Series,
    unique_dates: pd.DatetimeIndex,
    fold_idx: int,
    valid_dates_per_fold: int,
) -> dict[str, object] | None:
    train_end = valid_dates_per_fold * (fold_idx + 1)
    valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
    train_dates = unique_dates[:train_end]
    valid_dates = unique_dates[train_end:valid_end]
    if len(valid_dates) == 0:
        return None
    train_mask = ordered_dates.isin(train_dates.tolist())
    valid_mask = ordered_dates.isin(valid_dates.tolist())
    return {
        "fold": fold_idx + 1,
        "train_idx": ordered.index[train_mask].to_numpy(),
        "valid_idx": ordered.index[valid_mask].to_numpy(),
        "train_dates": int(len(train_dates)),
        "valid_dates": int(len(valid_dates)),
        "train_end_date": _format_date_bound(train_dates.max()),
        "valid_start_date": _format_date_bound(valid_dates.min()),
        "valid_end_date": _format_date_bound(valid_dates.max()),
    }
