from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


def get_parquet_columns(path: str | Path) -> list[str]:
    return list(pl.scan_parquet(str(path)).collect_schema().names())


def read_parquet_projected(
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    projected = list(columns) if columns is not None else None
    frame = pl.read_parquet(str(path), columns=projected)
    return downcast_pandas_frame(frame.to_pandas())


def downcast_pandas_frame(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy()
    for column in compact.columns:
        series = compact[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            compact[column] = series.astype("boolean")
            continue
        if pd.api.types.is_float_dtype(series):
            compact[column] = pd.to_numeric(series, downcast="float")
            continue
        if pd.api.types.is_integer_dtype(series):
            compact[column] = pd.to_numeric(series, downcast="integer")
            continue
    return compact


def load_numeric_parquet_column(
    path: str | Path,
    column: str,
    mask: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    table = pq.read_table(str(path), columns=[column])
    values = table[column].to_numpy(zero_copy_only=False)
    array = np.asarray(values, dtype=dtype)
    if mask is not None:
        array = array[mask]
    return array
