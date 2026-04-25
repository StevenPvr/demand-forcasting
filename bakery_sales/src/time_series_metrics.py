from __future__ import annotations

"""Metriques partagees pour les modeles temporels du projet."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def mae_score(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    """Calcule la MAE sur l'echelle originale."""

    return float(mean_absolute_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))


def rmse_score(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    """Calcule la RMSE sur l'echelle originale."""

    return float(
        root_mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    )


def smape(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    """Calcule la sMAPE de maniere numeriquement stable."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    if len(actual) == 0:
        return 0.0
    scaled_errors = np.zeros_like(actual, dtype=float)
    valid_mask = denominator > 0.0
    scaled_errors[valid_mask] = (2.0 * np.abs(actual[valid_mask] - predicted[valid_mask])) / denominator[valid_mask]
    return float(np.mean(scaled_errors))


def mase_score(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    insample_series: pd.Series,
    seasonal_period: int = 1,
) -> float:
    """Calcule la MASE avec un denominateur base sur la serie in-sample."""

    resolved_period = max(int(seasonal_period), 1)
    insample_values = np.asarray(insample_series, dtype=float)
    if len(insample_values) <= resolved_period:
        return math.inf
    naive_errors = np.abs(insample_values[resolved_period:] - insample_values[:-resolved_period])
    denominator = float(np.mean(naive_errors))
    if denominator == 0.0:
        return math.inf
    return mae_score(y_true, y_pred) / denominator


def interval_coverage(
    y_true: pd.Series,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> float:
    """Calcule la couverture empirique d'un intervalle de prediction."""

    actual = np.asarray(y_true, dtype=float)
    lower_bound = np.asarray(lower, dtype=float)
    upper_bound = np.asarray(upper, dtype=float)
    covered = (actual >= lower_bound) & (actual <= upper_bound)
    return float(np.mean(covered))


def lag_baseline_predictions(
    history_values: pd.Series,
    future_values: pd.Series,
    lag: int,
) -> pd.Series:
    """Construit des predictions baseline en utilisant les vraies observations passees."""

    resolved_lag = max(int(lag), 1)
    observed_values = list(np.asarray(history_values, dtype=float))
    predictions: list[float] = []
    for actual_value in np.asarray(future_values, dtype=float):
        if len(observed_values) < resolved_lag:
            predictions.append(float(observed_values[-1]))
        else:
            predictions.append(float(observed_values[-resolved_lag]))
        observed_values.append(float(actual_value))
    return pd.Series(predictions, dtype=float)
