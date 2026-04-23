from __future__ import annotations

import numpy as np
import pandas as pd

_FLOAT_COMPARISON_EPSILON = 1e-8


def compute_wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = float(np.abs(y_true_arr).sum())
    if abs(denominator) <= _FLOAT_COMPARISON_EPSILON:
        return float("inf")
    return float(np.abs(y_true_arr - y_pred_arr).sum() / denominator)


def compute_wape_improvement_pct(base_wape: float, candidate_wape: float) -> float:
    if abs(base_wape) <= _FLOAT_COMPARISON_EPSILON:
        return 0.0
    return float(100.0 * (base_wape - candidate_wape) / base_wape)
