from __future__ import annotations

"""Feature engineering temporel causal sur la date cible de prevision."""

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.data_preprocessing.constants import (
    DAY_OF_MONTH_PERIOD,
    DAY_OF_WEEK_PERIOD,
    DAY_OF_YEAR_PERIOD,
    MONTH_PERIOD,
    ORIGIN_DATE_COLUMN,
    PAYDAY_WINDOW_DAYS,
    TARGET_ROLLING_WINDOWS,
    QUARTER_PERIOD,
    TARGET_DATE_COLUMN,
    TARGET_AUTOREGRESSIVE_LAGS,
    WEEK_OF_MONTH_PERIOD,
    WEEK_OF_YEAR_PERIOD,
    YEARLY_FOURIER_HARMONICS,
)
from src.data_preprocessing.open_data_features import build_open_data_features

LOGGER: logging.Logger = logging.getLogger(__name__)


def _sort_daily_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Trie le dataset journalier par origin_date croissante."""

    ordered_df: pd.DataFrame = dataset_df.copy()
    ordered_df[ORIGIN_DATE_COLUMN] = pd.to_datetime(
        ordered_df[ORIGIN_DATE_COLUMN],
        format="%Y-%m-%d",
    )
    ordered_df[TARGET_DATE_COLUMN] = pd.to_datetime(
        ordered_df[TARGET_DATE_COLUMN],
        format="%Y-%m-%d",
    )
    return ordered_df.sort_values([ORIGIN_DATE_COLUMN, TARGET_DATE_COLUMN]).reset_index(drop=True)


def _cyclical_pair(values: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode une variable calendaire sur le cercle unite."""

    angle = (2.0 * np.pi * values.astype(float)) / float(period)
    return np.sin(angle), np.cos(angle)


def _fourier_harmonics_frame(
    values: pd.Series,
    period: int,
    harmonics: tuple[int, ...],
    prefix: str,
    index: pd.Index,
) -> pd.DataFrame:
    """Construit des harmoniques de Fourier additionnelles pour une saisonnalite longue."""

    harmonic_columns: dict[str, np.ndarray] = {}
    for harmonic in harmonics:
        angle = (2.0 * np.pi * harmonic * values.astype(float)) / float(period)
        harmonic_columns[f"{prefix}_sin_{harmonic}"] = np.sin(angle)
        harmonic_columns[f"{prefix}_cos_{harmonic}"] = np.cos(angle)
    return pd.DataFrame(harmonic_columns, index=index)


def _add_target_calendar_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les exogenes calendrier calculees sur la date cible."""

    date_series = dataset_df[TARGET_DATE_COLUMN]
    iso_calendar = date_series.dt.isocalendar()
    week_of_year = iso_calendar.week.astype(int)
    week_of_month = ((date_series.dt.day - 1) // 7) + 1
    day_of_week_sin, day_of_week_cos = _cyclical_pair(date_series.dt.dayofweek, DAY_OF_WEEK_PERIOD)
    day_of_month_sin, day_of_month_cos = _cyclical_pair(date_series.dt.day, DAY_OF_MONTH_PERIOD)
    day_of_year_sin, day_of_year_cos = _cyclical_pair(date_series.dt.dayofyear, DAY_OF_YEAR_PERIOD)
    week_of_year_sin, week_of_year_cos = _cyclical_pair(week_of_year, WEEK_OF_YEAR_PERIOD)
    week_of_month_sin, week_of_month_cos = _cyclical_pair(week_of_month, WEEK_OF_MONTH_PERIOD)
    month_sin, month_cos = _cyclical_pair(date_series.dt.month, MONTH_PERIOD)
    quarter_sin, quarter_cos = _cyclical_pair(date_series.dt.quarter, QUARTER_PERIOD)
    yearly_harmonics = _fourier_harmonics_frame(
        values=date_series.dt.dayofyear,
        period=DAY_OF_YEAR_PERIOD,
        harmonics=YEARLY_FOURIER_HARMONICS,
        prefix="exog_calendar_target_day_of_year",
        index=dataset_df.index,
    )
    calendar_frame = pd.DataFrame(
        {
            "exog_calendar_target_day_of_week": date_series.dt.dayofweek.astype(int),
            "exog_calendar_target_day_of_month": date_series.dt.day.astype(int),
            "exog_calendar_target_day_of_year": date_series.dt.dayofyear.astype(int),
            "exog_calendar_target_week_of_year": week_of_year,
            "exog_calendar_target_week_of_month": week_of_month.astype(int),
            "exog_calendar_target_month": date_series.dt.month.astype(int),
            "exog_calendar_target_quarter": date_series.dt.quarter.astype(int),
            "exog_calendar_target_year": date_series.dt.year.astype(int),
            "exog_calendar_target_days_in_month": date_series.dt.days_in_month.astype(int),
            "exog_calendar_target_days_from_month_start": (date_series.dt.day - 1).astype(int),
            "exog_calendar_target_days_to_month_end": (
                date_series.dt.days_in_month - date_series.dt.day
            ).astype(int),
            "exog_calendar_target_is_weekend": (date_series.dt.dayofweek >= 5).astype(int),
            "exog_calendar_target_is_month_start": date_series.dt.is_month_start.astype(int),
            "exog_calendar_target_is_month_end": date_series.dt.is_month_end.astype(int),
            "exog_calendar_target_is_quarter_start": date_series.dt.is_quarter_start.astype(int),
            "exog_calendar_target_is_quarter_end": date_series.dt.is_quarter_end.astype(int),
            "exog_calendar_target_is_year_start": date_series.dt.is_year_start.astype(int),
            "exog_calendar_target_is_year_end": date_series.dt.is_year_end.astype(int),
            "exog_calendar_target_is_payday_window": date_series.dt.day.isin(PAYDAY_WINDOW_DAYS).astype(int),
            "exog_calendar_target_is_early_month": (date_series.dt.day <= 5).astype(int),
            "exog_calendar_target_is_late_month": (date_series.dt.day >= 25).astype(int),
            "exog_calendar_target_is_summer_season": date_series.dt.month.isin((6, 7, 8)).astype(int),
            "exog_calendar_target_is_winter_season": date_series.dt.month.isin((12, 1, 2)).astype(int),
            "exog_calendar_target_day_of_week_sin": day_of_week_sin,
            "exog_calendar_target_day_of_week_cos": day_of_week_cos,
            "exog_calendar_target_day_of_month_sin": day_of_month_sin,
            "exog_calendar_target_day_of_month_cos": day_of_month_cos,
            "exog_calendar_target_day_of_year_sin": day_of_year_sin,
            "exog_calendar_target_day_of_year_cos": day_of_year_cos,
            "exog_calendar_target_week_of_year_sin": week_of_year_sin,
            "exog_calendar_target_week_of_year_cos": week_of_year_cos,
            "exog_calendar_target_week_of_month_sin": week_of_month_sin,
            "exog_calendar_target_week_of_month_cos": week_of_month_cos,
            "exog_calendar_target_month_sin": month_sin,
            "exog_calendar_target_month_cos": month_cos,
            "exog_calendar_target_quarter_sin": quarter_sin,
            "exog_calendar_target_quarter_cos": quarter_cos,
        },
        index=dataset_df.index,
    )
    return pd.concat([dataset_df.copy(), calendar_frame, yearly_harmonics], axis=1)


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calcule un ratio robuste en annulant les divisions par zero."""

    safe_denominator = denominator.replace(0.0, np.nan)
    divided_series = cast(pd.Series, numerator.divide(safe_denominator))
    sanitized_series = cast(pd.Series, divided_series.replace([np.inf, -np.inf], np.nan))
    return cast(pd.Series, sanitized_series.fillna(0.0))


def _rolling_target_feature_frame(target_series: pd.Series, index: pd.Index) -> pd.DataFrame:
    """Construit des statistiques causales sur l'historique recent de la cible."""

    observed_target = target_series.shift(1)
    rolling_columns: dict[str, pd.Series] = {}
    for window in TARGET_ROLLING_WINDOWS:
        rolling_window = observed_target.rolling(window=window, min_periods=window)
        rolling_columns[f"exog_autoreg_target_roll_mean_{window}"] = cast(pd.Series, rolling_window.mean())
        rolling_columns[f"exog_autoreg_target_roll_std_{window}"] = cast(
            pd.Series,
            rolling_window.std().fillna(0.0),
        )
    same_weekday_mean = cast(
        pd.Series,
        pd.concat(
            [target_series.shift(7 * step) for step in range(1, 5)],
            axis=1,
        ).mean(axis=1),
    )
    lag_1 = cast(pd.Series, observed_target)
    lag_7 = cast(pd.Series, target_series.shift(7))
    roll_mean_7 = cast(pd.Series, rolling_columns["exog_autoreg_target_roll_mean_7"])
    rolling_columns["exog_autoreg_target_same_weekday_roll_mean_4"] = same_weekday_mean
    rolling_columns["exog_autoreg_target_gap_lag_1_vs_roll_mean_7"] = lag_1 - roll_mean_7
    rolling_columns["exog_autoreg_target_ratio_lag_1_vs_roll_mean_7"] = _safe_ratio_series(
        lag_1,
        roll_mean_7,
    )
    rolling_columns["exog_autoreg_target_gap_lag_7_vs_same_weekday_mean_4"] = lag_7 - same_weekday_mean
    return pd.DataFrame(rolling_columns, index=index)


def _add_target_autoregressive_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les lags et statistiques autoregressives causales de la cible."""

    target_series = cast(pd.Series, dataset_df["target_baguette_t_plus_1"])
    lag_columns = {
        f"exog_autoreg_target_lag_{lag}": target_series.shift(lag)
        for lag in TARGET_AUTOREGRESSIVE_LAGS
    }
    rolling_df = _rolling_target_feature_frame(target_series=target_series, index=dataset_df.index)
    autoreg_df = pd.concat([pd.DataFrame(lag_columns, index=dataset_df.index), rolling_df], axis=1)
    featured_df = pd.concat([dataset_df.copy(), autoreg_df], axis=1)
    return featured_df.dropna(subset=list(autoreg_df.columns)).reset_index(drop=True)


def _add_context_interaction_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute quelques interactions causales entre contexte date cible et meteo origine."""

    interaction_df = dataset_df.copy()
    if {
        "exog_weather_origin_is_rainy_day",
        "exog_calendar_target_is_weekend",
    }.issubset(interaction_df.columns):
        interaction_df["exog_weather_origin_is_rainy_day_x_target_is_weekend"] = (
            interaction_df["exog_weather_origin_is_rainy_day"]
            * interaction_df["exog_calendar_target_is_weekend"]
        )
    if {
        "exog_weather_origin_is_hot_day",
        "exog_calendar_target_is_weekend",
    }.issubset(interaction_df.columns):
        interaction_df["exog_weather_origin_is_hot_day_x_target_is_weekend"] = (
            interaction_df["exog_weather_origin_is_hot_day"]
            * interaction_df["exog_calendar_target_is_weekend"]
        )
    if {
        "exog_holiday_target_is_school_holiday",
        "exog_calendar_target_is_weekend",
    }.issubset(interaction_df.columns):
        interaction_df["exog_holiday_target_is_school_holiday_x_target_is_weekend"] = (
            interaction_df["exog_holiday_target_is_school_holiday"]
            * interaction_df["exog_calendar_target_is_weekend"]
        )
    return interaction_df


def build_modeling_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables calendrier a partir de la date cible."""

    ordered_df: pd.DataFrame = _sort_daily_dataset(dataset_df)
    featured_df: pd.DataFrame = _add_target_calendar_features(ordered_df)
    featured_df = _add_target_autoregressive_features(featured_df)
    open_data_df: pd.DataFrame = build_open_data_features(featured_df)
    featured_df = pd.concat([featured_df, open_data_df], axis=1)
    featured_df = _add_context_interaction_features(featured_df)
    featured_df[ORIGIN_DATE_COLUMN] = featured_df[ORIGIN_DATE_COLUMN].dt.strftime("%Y-%m-%d")
    featured_df[TARGET_DATE_COLUMN] = featured_df[TARGET_DATE_COLUMN].dt.strftime("%Y-%m-%d")
    LOGGER.info(
        "Built %d target-date calendar columns",
        len(
            [
                column
                for column in featured_df.columns
                if column.startswith("exog_calendar_target_")
            ]
        ),
    )
    LOGGER.info(
        "Built %d autoregressive target lag columns",
        len([column for column in featured_df.columns if column.startswith("exog_autoreg_target_lag_")]),
    )
    return featured_df.reset_index(drop=True)
