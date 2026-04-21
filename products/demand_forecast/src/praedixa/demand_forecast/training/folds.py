from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _sorted_unique_dates(frame: pd.DataFrame, date_col: str) -> pd.DatetimeIndex:
    date_values = pd.Series(pd.to_datetime(frame[date_col], errors="raise"), copy=False)
    return pd.DatetimeIndex(date_values.drop_duplicates().sort_values())


def _format_date_bound(value: object) -> str:
    return pd.Timestamp(cast(Any, value)).strftime("%Y-%m-%d")


def _dataset_supports_requested_folds(
    *,
    dataset_source: str,
    unique_dates: pd.Index,
    requested_n_folds: int,
    logger: logging.Logger = LOGGER,
) -> bool:
    supports_requested_folds = len(unique_dates) >= (requested_n_folds + 1)
    if not supports_requested_folds:
        logger.warning(
            "Skipping dataset `%s` from walk-forward validation and treating it as prediction-only: requested_folds=%s unique_dates=%s",
            dataset_source,
            requested_n_folds,
            len(unique_dates),
        )
    return supports_requested_folds


def _dataset_fold_metadata(
    *,
    dataset_source: str | None,
    fold_idx: int,
    dataset_positions: np.ndarray,
    ordered_dates: pd.Series,
    train_dates: pd.Index,
    valid_dates: pd.Index,
) -> dict[str, object]:
    train_mask = ordered_dates.isin(train_dates.tolist()).to_numpy()
    valid_mask = ordered_dates.isin(valid_dates.tolist()).to_numpy()
    payload: dict[str, object] = {
        "fold": fold_idx + 1,
        "train_idx": dataset_positions[train_mask],
        "valid_idx": dataset_positions[valid_mask],
        "train_dates": int(len(train_dates)),
        "valid_dates": int(len(valid_dates)),
        "train_end_date": _format_date_bound(train_dates.max()),
        "valid_start_date": _format_date_bound(valid_dates.min()),
        "valid_end_date": _format_date_bound(valid_dates.max()),
    }
    if dataset_source is not None:
        payload["dataset_source"] = dataset_source
    return payload


def _build_dataset_fold_sequence(
    *,
    dataset_source: str,
    dataset_frame: pd.DataFrame,
    date_col: str,
    n_folds: int,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    ordered = dataset_frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col])
    dataset_positions = ordered.index.to_numpy(dtype=np.int32)
    unique_dates = _sorted_unique_dates(ordered, date_col)
    ordered_dates = ordered[date_col]
    if not _dataset_supports_requested_folds(
        dataset_source=dataset_source,
        unique_dates=unique_dates,
        requested_n_folds=n_folds,
        logger=logger,
    ):
        return []
    valid_dates_per_fold = max(1, len(unique_dates) // (n_folds + 1))
    dataset_fold_metadata: list[dict[str, object]] = []
    for fold_idx in range(n_folds):
        train_end = valid_dates_per_fold * (fold_idx + 1)
        valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[train_end:valid_end]
        if len(valid_dates) == 0:
            break
        dataset_fold_metadata.append(
            _dataset_fold_metadata(
                dataset_source=dataset_source,
                fold_idx=fold_idx,
                dataset_positions=dataset_positions,
                ordered_dates=ordered_dates,
                train_dates=train_dates,
                valid_dates=valid_dates,
            )
        )
    return dataset_fold_metadata


def _combined_fold_indices(
    per_dataset_fold: list[dict[str, object]],
    index_key: str,
) -> np.ndarray:
    return np.concatenate(
        [np.asarray(item[index_key], dtype=np.int32) for item in per_dataset_fold]
    )


def _per_dataset_fold_payload(
    per_dataset_fold: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            key: value
            for key, value in item.items()
            if key not in {"train_idx", "valid_idx"}
        }
        for item in per_dataset_fold
    ]


def _grouped_fold_payload(
    *,
    fold_idx: int,
    per_dataset_fold: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "fold": fold_idx + 1,
        "train_idx": _combined_fold_indices(per_dataset_fold, "train_idx"),
        "valid_idx": _combined_fold_indices(per_dataset_fold, "valid_idx"),
        "datasets": [str(item["dataset_source"]) for item in per_dataset_fold],
        "per_dataset": _per_dataset_fold_payload(per_dataset_fold),
    }


def _dataset_fold_metadata_map(
    *,
    tuning_frame: pd.DataFrame,
    dataset_source_col: str,
    date_col: str,
    n_folds: int,
    logger: logging.Logger,
) -> dict[str, list[dict[str, object]]]:
    per_dataset_metadata: dict[str, list[dict[str, object]]] = {}
    for dataset_source, dataset_frame in tuning_frame.groupby(dataset_source_col, sort=False):
        dataset_fold_metadata = _build_dataset_fold_sequence(
            dataset_source=str(dataset_source),
            dataset_frame=dataset_frame,
            date_col=date_col,
            n_folds=n_folds,
            logger=logger,
        )
        if dataset_fold_metadata:
            per_dataset_metadata[str(dataset_source)] = dataset_fold_metadata
    return per_dataset_metadata


def build_tuning_walk_forward_folds(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = "dt",
    n_folds: int = 5,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, object]]:
    ordered = tuning_frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col]).reset_index(drop=True)
    unique_dates = _sorted_unique_dates(ordered, date_col)
    ordered_dates = ordered[date_col]

    if len(unique_dates) < (n_folds + 1):
        raise ValueError(f"Need at least {n_folds + 1} unique dates to build {n_folds} walk-forward folds.")

    valid_dates_per_fold = max(1, len(unique_dates) // (n_folds + 1))
    folds: list[dict[str, object]] = []
    for fold_idx in range(n_folds):
        train_end = valid_dates_per_fold * (fold_idx + 1)
        valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[train_end:valid_end]
        if len(valid_dates) == 0:
            break

        train_mask = ordered_dates.isin(train_dates.tolist())
        valid_mask = ordered_dates.isin(valid_dates.tolist())
        folds.append(
            {
                "fold": fold_idx + 1,
                "train_idx": ordered.index[train_mask].to_numpy(),
                "valid_idx": ordered.index[valid_mask].to_numpy(),
                "train_dates": int(len(train_dates)),
                "valid_dates": int(len(valid_dates)),
                "train_end_date": _format_date_bound(train_dates.max()),
                "valid_start_date": _format_date_bound(valid_dates.min()),
                "valid_end_date": _format_date_bound(valid_dates.max()),
            }
        )

    if len(folds) != n_folds:
        raise ValueError(f"Expected {n_folds} folds, got {len(folds)}.")
    logger.info(
        "Optimisation walk-forward folds built: n_folds=%s valid_dates_per_fold=%s",
        len(folds),
        valid_dates_per_fold,
    )
    return folds


def build_tuning_walk_forward_folds_by_dataset(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = "dt",
    dataset_source_col: str = "dataset_source",
    n_folds: int = 5,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, object]]:
    dataset_folds: list[dict[str, object]] = []
    for dataset_source, dataset_frame in tuning_frame.groupby(dataset_source_col, sort=False):
        ordered, unique_dates = _ordered_dataset_dates(dataset_frame, date_col=date_col)
        if not _dataset_supports_requested_folds(dataset_source=str(dataset_source), unique_dates=unique_dates, requested_n_folds=n_folds, logger=logger):
            continue
        dataset_folds.extend(
            _dataset_walk_forward_folds(
                dataset_source=str(dataset_source),
                ordered=ordered,
                date_col=date_col,
                unique_dates=unique_dates,
                n_folds=n_folds,
            )
        )
    logger.info(
        "Optimisation walk-forward folds built by dataset: datasets=%s total_fold_evaluations=%s",
        sorted(tuning_frame[dataset_source_col].dropna().unique().tolist()),
        len(dataset_folds),
    )
    return dataset_folds


def _ordered_dataset_dates(
    dataset_frame: pd.DataFrame,
    *,
    date_col: str,
) -> tuple[pd.DataFrame, pd.Index]:
    ordered = dataset_frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col]).reset_index(drop=True)
    return ordered, _sorted_unique_dates(ordered, date_col)


def _dataset_walk_forward_folds(
    *,
    dataset_source: str,
    ordered: pd.DataFrame,
    date_col: str,
    unique_dates: pd.Index,
    n_folds: int,
) -> list[dict[str, object]]:
    ordered_dates = ordered[date_col]
    valid_dates_per_fold = max(1, len(unique_dates) // (n_folds + 1))
    dataset_folds: list[dict[str, object]] = []
    for fold_idx in range(n_folds):
        train_end = valid_dates_per_fold * (fold_idx + 1)
        valid_end = min(train_end + valid_dates_per_fold, len(unique_dates))
        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[train_end:valid_end]
        if len(valid_dates) == 0:
            break
        train_mask = ordered_dates.isin(train_dates.tolist())
        valid_mask = ordered_dates.isin(valid_dates.tolist())
        dataset_folds.append(
            {
                "dataset_source": dataset_source,
                "fold": fold_idx + 1,
                "train_idx": ordered.index[train_mask].to_numpy(),
                "valid_idx": ordered.index[valid_mask].to_numpy(),
                "train_dates": int(len(train_dates)),
                "valid_dates": int(len(valid_dates)),
                "train_end_date": _format_date_bound(train_dates.max()),
                "valid_start_date": _format_date_bound(valid_dates.min()),
                "valid_end_date": _format_date_bound(valid_dates.max()),
            }
        )
    return dataset_folds


def build_grouped_tuning_walk_forward_folds_by_dataset(
    tuning_frame: pd.DataFrame,
    *,
    date_col: str = "dt",
    dataset_source_col: str = "dataset_source",
    n_folds: int = 5,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, object]]:
    grouped_folds: list[dict[str, object]] = []
    per_dataset_metadata = _dataset_fold_metadata_map(
        tuning_frame=tuning_frame,
        dataset_source_col=dataset_source_col,
        date_col=date_col,
        n_folds=n_folds,
        logger=logger,
    )

    if not per_dataset_metadata:
        raise ValueError("No dataset has enough unique dates to build walk-forward validation folds.")
    for fold_idx in range(n_folds):
        per_dataset_fold = [metadata[fold_idx] for metadata in per_dataset_metadata.values() if len(metadata) > fold_idx]
        grouped_folds.append(_grouped_fold_payload(fold_idx=fold_idx, per_dataset_fold=per_dataset_fold))
    logger.info(
        "Grouped optimisation folds built by dataset: n_folds=%s datasets=%s",
        len(grouped_folds),
        sorted(per_dataset_metadata.keys()),
    )
    return grouped_folds
