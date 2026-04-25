from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd


def _normalized_date_series(frame: pd.DataFrame, *, date_col: str) -> pd.Series:
    if date_col not in frame.columns:
        raise KeyError(f"Missing evaluation date column: {date_col}")
    dates = pd.to_datetime(frame[date_col], errors="raise")
    return pd.Series(dates, copy=False).reset_index(drop=True)


def _row_indices_by_date(dates: pd.Series) -> dict[pd.Timestamp, np.ndarray]:
    row_index = pd.DataFrame(
        {
            "__dt": dates,
            "__row_nr": np.arange(len(dates), dtype=np.int32),
        }
    )
    return {
        pd.Timestamp(cast(Any, date)): group["__row_nr"].to_numpy(
            dtype=np.int32,
            copy=True,
        )
        for date, group in row_index.groupby("__dt", sort=True)
    }


def build_daily_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    date_col: str = "dt",
) -> list[dict[str, object]]:
    dates = _normalized_date_series(frame, date_col=date_col)
    indices_by_date = _row_indices_by_date(dates)
    unique_dates = sorted(indices_by_date)
    if not unique_dates:
        raise ValueError("Need at least one evaluation date to build daily folds.")

    folds: list[dict[str, object]] = []
    history_idx = np.empty(0, dtype=np.int32)
    for fold_idx, eval_date in enumerate(unique_dates, start=1):
        valid_idx = indices_by_date[eval_date]
        folds.append(
            {
                "fold": fold_idx,
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "history_idx": history_idx.copy(),
                "valid_idx": valid_idx,
                "history_rows": int(len(history_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
        history_idx = np.sort(np.concatenate((history_idx, valid_idx))).astype(
            np.int32,
            copy=False,
        )
    return folds
