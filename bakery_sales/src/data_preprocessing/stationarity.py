from __future__ import annotations

"""Diagnostics de stationnarite descriptifs pour la cible journaliere."""

import logging
import warnings
from typing import Any, cast

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.data_preprocessing.constants import ADF_PVALUE_THRESHOLD, SEASONAL_PERIOD_WEEKLY

LOGGER: logging.Logger = logging.getLogger(__name__)


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Normalise une serie en float sans NaN pour le test ADF."""

    return series.astype(float).dropna()


def _adf_test_result(series: pd.Series, pvalue_threshold: float) -> dict[str, Any]:
    """Execute un test ADF robuste sur une serie numerique."""

    clean_series = _clean_numeric_series(series)
    if len(clean_series) < 8:
        return {
            "pvalue": None,
            "stationary": True,
            "status": "too_short",
            "row_count": len(clean_series),
        }
    if clean_series.nunique(dropna=False) <= 1:
        return {
            "pvalue": 0.0,
            "stationary": True,
            "status": "constant",
            "row_count": len(clean_series),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pvalue = float(adfuller(clean_series, autolag="AIC")[1])
    return {
        "pvalue": pvalue,
        "stationary": bool(pvalue < pvalue_threshold),
        "status": "ok",
        "row_count": len(clean_series),
    }


def _candidate_d_values(level_result: dict[str, Any]) -> list[int]:
    """Retourne les ordres d'integration a tester sans imposer la differenciation en amont."""

    if bool(level_result["stationary"]):
        return [0]
    return [0, 1]


def build_stationarity_report(
    train_df: pd.DataFrame,
    target_column: str,
    pvalue_threshold: float = ADF_PVALUE_THRESHOLD,
    seasonal_period: int = SEASONAL_PERIOD_WEEKLY,
) -> dict[str, Any]:
    """Construit un rapport descriptif de stationnarite sur la cible originale."""

    target_series = cast(pd.Series, train_df[target_column].astype(float))
    level_result = _adf_test_result(target_series, pvalue_threshold)
    diff_1_result = _adf_test_result(target_series.diff(), pvalue_threshold)
    seasonal_diff_result = _adf_test_result(
        target_series.diff(seasonal_period),
        pvalue_threshold,
    )
    candidate_d_values = _candidate_d_values(level_result)
    candidate_D_values = [0] if seasonal_diff_result["status"] == "constant" else [0, 1]
    LOGGER.info(
        "Built stationarity report for %s with candidate d=%s and D=%s",
        target_column,
        candidate_d_values,
        candidate_D_values,
    )
    return {
        "target_column": target_column,
        "row_count": int(len(train_df)),
        "seasonal_period": seasonal_period,
        "pvalue_threshold": pvalue_threshold,
        "candidate_d_values": candidate_d_values,
        "candidate_D_values": candidate_D_values,
        "target": {
            "level": level_result,
            "difference_1": diff_1_result,
            f"seasonal_difference_{seasonal_period}": seasonal_diff_result,
        },
    }
