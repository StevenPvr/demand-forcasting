from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def _ordered_fold_frame(frame: pd.DataFrame, *, date_col: str) -> pl.DataFrame:
    ordered = pl.from_pandas(frame.reset_index(drop=True), include_index=False).with_row_index("__row_nr")
    date_expr = pl.col(date_col)
    if ordered.schema.get(date_col) == pl.String:
        date_expr = date_expr.str.to_datetime(strict=True)
    else:
        date_expr = date_expr.cast(pl.Datetime, strict=True)
    return ordered.with_columns(date_expr.alias(date_col))


def build_daily_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    date_col: str = "dt",
) -> list[dict[str, object]]:
    ordered = _ordered_fold_frame(frame, date_col=date_col)
    unique_dates = ordered.get_column(date_col).unique().sort().to_list()
    if len(unique_dates) < 1:
        raise ValueError("Need at least one evaluation date to build daily folds.")
    folds: list[dict[str, object]] = []
    for fold_idx, eval_date in enumerate(unique_dates, start=1):
        history_idx = ordered.filter(pl.col(date_col) < eval_date).get_column("__row_nr").to_numpy().astype(np.int32)
        valid_idx = ordered.filter(pl.col(date_col) == eval_date).get_column("__row_nr").to_numpy().astype(np.int32)
        folds.append(
            {
                "fold": fold_idx,
                "eval_date": pd.Timestamp(eval_date).strftime("%Y-%m-%d"),
                "history_idx": history_idx,
                "valid_idx": valid_idx,
                "history_rows": int(len(history_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
    return folds
