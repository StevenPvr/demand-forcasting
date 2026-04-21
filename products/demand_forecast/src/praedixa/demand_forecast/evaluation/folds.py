from __future__ import annotations

import pandas as pd


def build_daily_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    date_col: str = "dt",
) -> list[dict[str, object]]:
    ordered = frame.copy().reset_index(drop=True)
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered_dates = pd.Series(pd.to_datetime(ordered[date_col], errors="raise"), copy=False)
    unique_dates = pd.DatetimeIndex(ordered_dates.drop_duplicates().sort_values())
    if len(unique_dates) < 1:
        raise ValueError("Need at least one evaluation date to build daily folds.")
    folds: list[dict[str, object]] = []
    for fold_idx, eval_date in enumerate(unique_dates, start=1):
        history_mask = ordered[date_col] < eval_date
        valid_mask = ordered[date_col] == eval_date
        folds.append(
            {
                "fold": fold_idx,
                "eval_date": pd.Timestamp(eval_date).strftime("%Y-%m-%d"),
                "history_idx": ordered.index[history_mask].to_numpy(),
                "valid_idx": ordered.index[valid_mask].to_numpy(),
                "history_rows": int(history_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        )
    return folds
