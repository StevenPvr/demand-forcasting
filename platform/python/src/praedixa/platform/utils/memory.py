from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


PARQUET_API: Any = pq


def get_parquet_columns(path: str | Path) -> list[str]:
    parquet_file: Any = PARQUET_API.ParquetFile(str(path))
    return list(parquet_file.schema.names)


def read_parquet_projected(
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    projected = list(columns) if columns is not None else None
    frame = pl.read_parquet(str(path), columns=projected)
    return downcast_pandas_frame(frame.to_pandas())


def _downcast_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    if pd.api.types.is_float_dtype(series):
        return pd.to_numeric(series, downcast="float")
    if pd.api.types.is_integer_dtype(series):
        return pd.to_numeric(series, downcast="integer")
    return series


def downcast_pandas_frame(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy(deep=False)
    for column in compact.columns:
        compact[column] = _downcast_series(compact[column])
    return compact


def load_numeric_parquet_column(
    path: str | Path,
    column: str,
    mask: np.ndarray | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    table: Any = PARQUET_API.read_table(str(path), columns=[column])
    values: Any = table[column].to_numpy(zero_copy_only=False)
    array = np.asarray(values, dtype=dtype)
    if mask is not None:
        array = array[mask]
    return array
