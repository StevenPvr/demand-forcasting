from __future__ import annotations

import logging

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def get_lag_candidate_columns(
    frame: pd.DataFrame,
    *,
    lag_patterns: tuple[str, ...],
) -> list[str]:
    return [column for column in frame.columns if any(pattern in column for pattern in lag_patterns)]


def get_lag_candidate_columns_from_names(
    column_names: list[str],
    *,
    lag_patterns: tuple[str, ...],
) -> list[str]:
    return [column for column in column_names if any(pattern in column for pattern in lag_patterns)]


def arrays_have_enough_variation(left_valid: np.ndarray, right_valid: np.ndarray) -> bool:
    if left_valid.size <= 1 or right_valid.size <= 1:
        return False
    if np.nanstd(left_valid) == 0.0 or np.nanstd(right_valid) == 0.0:
        return False
    return True


def valid_correlation_arrays(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid_mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(valid_mask):
        return None
    left_valid = left[valid_mask]
    right_valid = right[valid_mask]
    if not arrays_have_enough_variation(left_valid, right_valid):
        return None
    return left_valid, right_valid


def correlation_value(left_valid: np.ndarray, right_valid: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(np.corrcoef(left_valid, right_valid)[0, 1])
    if method == "spearman":
        return float(pd.Series(left_valid).corr(pd.Series(right_valid), method="spearman"))
    raise ValueError(f"Unsupported correlation method: {method}")


def corr_arrays(left: np.ndarray, right: np.ndarray, method: str) -> float:
    valid_arrays = valid_correlation_arrays(left, right)
    if valid_arrays is None:
        return 0.0
    left_valid, right_valid = valid_arrays
    corr_value = correlation_value(left_valid, right_valid, method)
    if pd.isna(corr_value):
        return 0.0
    return float(abs(corr_value))
