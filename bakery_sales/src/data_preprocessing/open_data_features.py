from __future__ import annotations

"""Features open data causales: vacances scolaires sur date cible et meteo sur date d'origine."""

import json
import logging
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from src.data_preprocessing.constants import (
    CSV_ENCODING,
    COLD_DAY_TEMPERATURE_THRESHOLD,
    DEFAULT_SCHOOL_HOLIDAYS_URL,
    DEFAULT_PUBLIC_HOLIDAYS_URL,
    DEFAULT_SCHOOL_ZONE,
    DEFAULT_WEATHER_API_BASE_URL,
    DEFAULT_WEATHER_LATITUDE,
    DEFAULT_WEATHER_LONGITUDE,
    DEFAULT_WEATHER_TIMEZONE,
    HEAVY_RAIN_MM_THRESHOLD,
    HOLIDAY_DISTANCE_FILL_VALUE,
    HOT_DAY_TEMPERATURE_THRESHOLD,
    ORIGIN_DATE_COLUMN,
    SUNNY_DAY_HOURS_THRESHOLD,
    TARGET_DATE_COLUMN,
    WEATHER_DAILY_VARIABLES,
    WINDY_DAY_KMH_THRESHOLD,
)
from src.data_preprocessing.paths import (
    PUBLIC_HOLIDAYS_CACHE_CSV,
    SCHOOL_HOLIDAYS_CACHE_CSV,
    WEATHER_CACHE_CSV,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
TextDownloader = Callable[[str], str]
JsonDownloader = Callable[[str], dict[str, Any]]


def _download_text(url: str) -> str:
    """Telecharge une ressource texte UTF-8."""

    with urlopen(url, timeout=30.0) as response:
        payload = response.read()
    return payload.decode(CSV_ENCODING)


def _download_json(url: str) -> dict[str, Any]:
    """Telecharge une ressource JSON."""

    return json.loads(_download_text(url))


def _read_csv_cache(cache_path: Path) -> pd.DataFrame | None:
    """Charge un cache CSV s'il existe deja."""

    if not cache_path.exists():
        return None
    return pd.read_csv(cache_path)


def _write_csv_cache(dataset_df: pd.DataFrame, cache_path: Path) -> None:
    """Persiste un cache CSV local."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(cache_path, index=False, encoding=CSV_ENCODING)


def _load_school_holidays(
    cache_path: Path,
    source_url: str,
    download_text: TextDownloader,
) -> pd.DataFrame:
    """Charge le calendrier scolaire depuis le cache ou la source officielle."""

    cached_df = _read_csv_cache(cache_path)
    if cached_df is not None:
        return cached_df
    holidays_df = pd.read_csv(StringIO(download_text(source_url)))
    _write_csv_cache(holidays_df, cache_path)
    return holidays_df


def _load_public_holidays(
    cache_path: Path,
    source_url: str,
    download_json: JsonDownloader,
) -> pd.DataFrame:
    """Charge les jours feries officiels depuis le cache ou la source ouverte."""

    cached_df = _read_csv_cache(cache_path)
    if cached_df is not None:
        return cached_df
    payload = download_json(source_url)
    public_holidays_df = pd.DataFrame(
        {
            "date": list(payload.keys()),
            "holiday_name": list(payload.values()),
        }
    )
    _write_csv_cache(public_holidays_df, cache_path)
    return public_holidays_df


def _normalize_boolean_series(value_series: pd.Series) -> pd.Series:
    """Convertit une serie booleenne heterogene en entier 0/1."""

    return (
        value_series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    )


def _ascii_lower(value_series: pd.Series) -> pd.Series:
    """Normalise un texte en ASCII minuscule pour des comparaisons stables."""

    return (
        value_series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
    )


def _days_to_previous_and_next_event(
    reference_dates: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Calcule les distances en jours vers l'evenement precedent et suivant."""

    if len(event_dates) == 0:
        fill_values = [HOLIDAY_DISTANCE_FILL_VALUE] * len(reference_dates)
        return (
            pd.Series(fill_values, index=reference_dates, dtype=float),
            pd.Series(fill_values, index=reference_dates, dtype=float),
        )
    event_days = event_dates.sort_values().values.astype("datetime64[D]")
    reference_days = reference_dates.values.astype("datetime64[D]")
    next_positions = event_days.searchsorted(reference_days, side="left")
    previous_positions = next_positions - 1
    previous_days = []
    next_days = []
    for index, reference_day in enumerate(reference_days):
        previous_day = (
            event_days[previous_positions[index]]
            if previous_positions[index] >= 0
            else None
        )
        next_day = (
            event_days[next_positions[index]]
            if next_positions[index] < len(event_days)
            else None
        )
        previous_days.append(
            HOLIDAY_DISTANCE_FILL_VALUE
            if previous_day is None
            else float((reference_day - previous_day).astype("timedelta64[D]").astype(int))
        )
        next_days.append(
            HOLIDAY_DISTANCE_FILL_VALUE
            if next_day is None
            else float((next_day - reference_day).astype("timedelta64[D]").astype(int))
        )
    return (
        pd.Series(previous_days, index=reference_dates, dtype=float),
        pd.Series(next_days, index=reference_dates, dtype=float),
    )


def _shifted_binary_feature(event_series: pd.Series, offset_days: int) -> pd.Series:
    """Decale une serie binaire journaliere sur des jours calendaires."""

    shifted_index = event_series.index + pd.Timedelta(days=offset_days)
    shifted_series = pd.Series(event_series.to_numpy(), index=shifted_index)
    return shifted_series


def _build_holiday_feature_frame(
    target_dates: pd.Series,
    holidays_df: pd.DataFrame,
    school_zone: str,
) -> pd.DataFrame:
    """Construit les flags de vacances scolaires sur la date cible."""

    zone_column = f"vacances_zone_{school_zone.lower()}"
    normalized_df = holidays_df.copy()
    normalized_df["date"] = pd.to_datetime(normalized_df["date"], format="%Y-%m-%d")
    zone_series = cast(pd.Series, normalized_df[zone_column])
    normalized_df[zone_column] = _normalize_boolean_series(zone_series)
    holiday_name_series = cast(pd.Series, normalized_df["nom_vacances"])
    normalized_df["nom_vacances_ascii"] = _ascii_lower(holiday_name_series)
    holiday_index = normalized_df.set_index("date")
    target_index = pd.DatetimeIndex(pd.to_datetime(target_dates, format="%Y-%m-%d"))
    holiday_flag_series = cast(pd.Series, holiday_index[zone_column].astype(int))
    holiday_name_series = holiday_index["nom_vacances_ascii"].reindex(target_index).fillna("")
    previous_days, next_days = _days_to_previous_and_next_event(
        reference_dates=target_index,
        event_dates=cast(
            pd.DatetimeIndex,
            cast(pd.Series, holiday_flag_series[holiday_flag_series == 1]).index,
        ),
    )
    day_before_series = _shifted_binary_feature(holiday_flag_series, offset_days=-1)
    day_after_series = _shifted_binary_feature(holiday_flag_series, offset_days=1)
    return pd.DataFrame(
        {
            "exog_holiday_target_is_school_holiday": holiday_flag_series.reindex(target_index).fillna(0).astype(int).to_numpy(),
            "exog_holiday_target_is_summer_school_holiday": holiday_name_series.str.contains("ete")
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_autumn_school_holiday": holiday_name_series.str.contains("toussaint")
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_christmas_school_holiday": holiday_name_series.str.contains("noel")
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_winter_school_holiday": holiday_name_series.str.contains("hiver")
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_spring_school_holiday": holiday_name_series.str.contains("printemps")
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_day_before_school_holiday": day_before_series.reindex(target_index)
            .fillna(0)
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_day_after_school_holiday": day_after_series.reindex(target_index)
            .fillna(0)
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_days_since_previous_school_holiday": previous_days.to_numpy(),
            "exog_holiday_target_days_until_next_school_holiday": next_days.to_numpy(),
        },
        index=target_dates.index,
    )


def _build_public_holiday_feature_frame(
    target_dates: pd.Series,
    public_holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construit les features de jours feries officiels sur la date cible."""

    normalized_df = public_holidays_df.copy()
    normalized_df["date"] = pd.to_datetime(normalized_df["date"], format="%Y-%m-%d")
    normalized_df["holiday_name_ascii"] = _ascii_lower(cast(pd.Series, normalized_df["holiday_name"]))
    public_holiday_series = cast(pd.Series, pd.Series(1, index=normalized_df["date"], dtype=int))
    public_holiday_series = cast(
        pd.Series,
        public_holiday_series[~public_holiday_series.index.duplicated(keep="last")],
    )
    target_index = pd.DatetimeIndex(pd.to_datetime(target_dates, format="%Y-%m-%d"))
    previous_days, next_days = _days_to_previous_and_next_event(
        reference_dates=target_index,
        event_dates=cast(pd.DatetimeIndex, public_holiday_series.index),
    )
    day_before_series = _shifted_binary_feature(public_holiday_series, offset_days=-1)
    day_after_series = _shifted_binary_feature(public_holiday_series, offset_days=1)
    previous_public_holiday = cast(
        pd.Series,
        public_holiday_series.reindex(target_index - pd.Timedelta(days=1)).fillna(0).astype(int),
    )
    next_public_holiday = cast(
        pd.Series,
        public_holiday_series.reindex(target_index + pd.Timedelta(days=1)).fillna(0).astype(int),
    )
    previous_weekend = pd.Series(
        [int(date.weekday() >= 5) for date in (target_index - pd.Timedelta(days=1))],
        index=target_index,
        dtype=int,
    )
    next_weekend = pd.Series(
        [int(date.weekday() >= 5) for date in (target_index + pd.Timedelta(days=1))],
        index=target_index,
        dtype=int,
    )
    previous_day_is_off = previous_public_holiday.to_numpy() | previous_weekend.to_numpy()
    next_day_is_off = next_public_holiday.to_numpy() | next_weekend.to_numpy()
    is_weekend = pd.Series([int(date.weekday() >= 5) for date in target_index], index=target_index, dtype=int).to_numpy()
    is_public_holiday = public_holiday_series.reindex(target_index).fillna(0).astype(int).to_numpy()
    is_bridge_day = (
        (1 - is_public_holiday)
        * (1 - is_weekend)
        * previous_day_is_off
        * next_day_is_off
    )
    return pd.DataFrame(
        {
            "exog_holiday_target_is_public_holiday": is_public_holiday,
            "exog_holiday_target_is_day_before_public_holiday": day_before_series.reindex(target_index)
            .fillna(0)
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_is_day_after_public_holiday": day_after_series.reindex(target_index)
            .fillna(0)
            .astype(int)
            .to_numpy(),
            "exog_holiday_target_days_since_previous_public_holiday": previous_days.to_numpy(),
            "exog_holiday_target_days_until_next_public_holiday": next_days.to_numpy(),
            "exog_holiday_target_is_bridge_day": is_bridge_day.astype(int),
        },
        index=target_dates.index,
    )


def _weather_api_url(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone_name: str,
    weather_api_base_url: str,
) -> str:
    """Construit l'URL Open-Meteo archive pour les variables journalieres."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(WEATHER_DAILY_VARIABLES),
        "timezone": timezone_name,
    }
    return f"{weather_api_base_url}?{urlencode(params, safe=',')}"


def _request_weather_history(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone_name: str,
    weather_api_base_url: str,
    download_json: JsonDownloader,
) -> pd.DataFrame:
    """Telecharge l'historique meteo journalier pour la plage demandee."""

    weather_url = _weather_api_url(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        weather_api_base_url=weather_api_base_url,
    )
    payload = download_json(weather_url)
    daily_payload = payload["daily"]
    weather_df = pd.DataFrame(daily_payload)
    return weather_df.rename(
        columns={
            "time": ORIGIN_DATE_COLUMN,
            "temperature_2m_max": "exog_weather_origin_temperature_2m_max",
            "temperature_2m_min": "exog_weather_origin_temperature_2m_min",
            "temperature_2m_mean": "exog_weather_origin_temperature_2m_mean",
            "apparent_temperature_mean": "exog_weather_origin_apparent_temperature_mean",
            "rain_sum": "exog_weather_origin_rain_sum",
            "snowfall_sum": "exog_weather_origin_snowfall_sum",
            "precipitation_sum": "exog_weather_origin_precipitation_sum",
            "precipitation_hours": "exog_weather_origin_precipitation_hours",
            "sunshine_duration": "exog_weather_origin_sunshine_duration_seconds",
            "shortwave_radiation_sum": "exog_weather_origin_shortwave_radiation_sum",
            "weather_code": "exog_weather_origin_weather_code",
            "wind_speed_10m_max": "exog_weather_origin_wind_speed_10m_max",
            "wind_gusts_10m_max": "exog_weather_origin_wind_gusts_10m_max",
            "cloud_cover_mean": "exog_weather_origin_cloud_cover_mean",
            "relative_humidity_2m_mean": "exog_weather_origin_relative_humidity_2m_mean",
        }
    )


def _load_weather_history(
    origin_dates: pd.Series,
    cache_path: Path,
    latitude: float,
    longitude: float,
    timezone_name: str,
    weather_api_base_url: str,
    download_json: JsonDownloader,
) -> pd.DataFrame:
    """Charge l'historique meteo depuis le cache puis complete si necessaire."""

    normalized_dates = pd.to_datetime(origin_dates, format="%Y-%m-%d")
    start_date = normalized_dates.min().strftime("%Y-%m-%d")
    end_date = normalized_dates.max().strftime("%Y-%m-%d")
    weather_columns = {
        ORIGIN_DATE_COLUMN,
        "exog_weather_origin_temperature_2m_max",
        "exog_weather_origin_temperature_2m_min",
        "exog_weather_origin_temperature_2m_mean",
        "exog_weather_origin_apparent_temperature_mean",
        "exog_weather_origin_rain_sum",
        "exog_weather_origin_snowfall_sum",
        "exog_weather_origin_precipitation_sum",
        "exog_weather_origin_precipitation_hours",
        "exog_weather_origin_sunshine_duration_seconds",
        "exog_weather_origin_shortwave_radiation_sum",
        "exog_weather_origin_weather_code",
        "exog_weather_origin_wind_speed_10m_max",
        "exog_weather_origin_wind_gusts_10m_max",
        "exog_weather_origin_cloud_cover_mean",
        "exog_weather_origin_relative_humidity_2m_mean",
    }
    cached_df = _read_csv_cache(cache_path)
    if cached_df is not None:
        cached_index = pd.to_datetime(cached_df[ORIGIN_DATE_COLUMN], format="%Y-%m-%d")
        has_required_columns = weather_columns.issubset(set(cached_df.columns))
        if (
            has_required_columns
            and cached_index.min() <= pd.Timestamp(start_date)
            and cached_index.max() >= pd.Timestamp(end_date)
        ):
            return cached_df
    weather_df = _request_weather_history(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        weather_api_base_url=weather_api_base_url,
        download_json=download_json,
    )
    merged_df = weather_df if cached_df is None else pd.concat([cached_df, weather_df], ignore_index=True)
    deduped_df = merged_df.drop_duplicates(subset=[ORIGIN_DATE_COLUMN], keep="last").sort_values(ORIGIN_DATE_COLUMN)
    _write_csv_cache(deduped_df, cache_path)
    return deduped_df.reset_index(drop=True)


def _build_weather_feature_frame(
    origin_dates: pd.Series,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reindexe la meteo sur les dates d'origine du dataset."""

    normalized_df = weather_df.copy()
    normalized_df[ORIGIN_DATE_COLUMN] = pd.to_datetime(normalized_df[ORIGIN_DATE_COLUMN], format="%Y-%m-%d")
    weather_index = normalized_df.set_index(ORIGIN_DATE_COLUMN)
    origin_index = pd.DatetimeIndex(pd.to_datetime(origin_dates, format="%Y-%m-%d"))
    weather_columns = [
        "exog_weather_origin_temperature_2m_max",
        "exog_weather_origin_temperature_2m_min",
        "exog_weather_origin_temperature_2m_mean",
        "exog_weather_origin_apparent_temperature_mean",
        "exog_weather_origin_rain_sum",
        "exog_weather_origin_snowfall_sum",
        "exog_weather_origin_precipitation_sum",
        "exog_weather_origin_precipitation_hours",
        "exog_weather_origin_sunshine_duration_seconds",
        "exog_weather_origin_shortwave_radiation_sum",
        "exog_weather_origin_weather_code",
        "exog_weather_origin_wind_speed_10m_max",
        "exog_weather_origin_wind_gusts_10m_max",
        "exog_weather_origin_cloud_cover_mean",
        "exog_weather_origin_relative_humidity_2m_mean",
    ]
    base_weather_df = weather_index.loc[:, weather_columns].reindex(origin_index).reset_index(drop=True)
    derived_weather_df = pd.DataFrame(
        {
            "exog_weather_origin_temperature_range": (
                base_weather_df["exog_weather_origin_temperature_2m_max"]
                - base_weather_df["exog_weather_origin_temperature_2m_min"]
            ),
            "exog_weather_origin_apparent_temperature_gap": (
                base_weather_df["exog_weather_origin_apparent_temperature_mean"]
                - base_weather_df["exog_weather_origin_temperature_2m_mean"]
            ),
            "exog_weather_origin_sunshine_duration_hours": (
                base_weather_df["exog_weather_origin_sunshine_duration_seconds"] / 3600.0
            ),
            "exog_weather_origin_is_rainy_day": (
                base_weather_df["exog_weather_origin_precipitation_sum"] > 0.0
            ).astype(int),
            "exog_weather_origin_is_heavy_rain_day": (
                base_weather_df["exog_weather_origin_precipitation_sum"] >= HEAVY_RAIN_MM_THRESHOLD
            ).astype(int),
            "exog_weather_origin_is_hot_day": (
                base_weather_df["exog_weather_origin_temperature_2m_max"] >= HOT_DAY_TEMPERATURE_THRESHOLD
            ).astype(int),
            "exog_weather_origin_is_cold_day": (
                base_weather_df["exog_weather_origin_temperature_2m_min"] <= COLD_DAY_TEMPERATURE_THRESHOLD
            ).astype(int),
            "exog_weather_origin_is_sunny_day": (
                (base_weather_df["exog_weather_origin_sunshine_duration_seconds"] / 3600.0)
                >= SUNNY_DAY_HOURS_THRESHOLD
            ).astype(int),
            "exog_weather_origin_is_windy_day": (
                base_weather_df["exog_weather_origin_wind_speed_10m_max"] >= WINDY_DAY_KMH_THRESHOLD
            ).astype(int),
        }
    )
    return pd.concat([base_weather_df, derived_weather_df], axis=1)


def build_open_data_features(
    dataset_df: pd.DataFrame,
    holiday_cache_path: Path = SCHOOL_HOLIDAYS_CACHE_CSV,
    public_holiday_cache_path: Path = PUBLIC_HOLIDAYS_CACHE_CSV,
    weather_cache_path: Path = WEATHER_CACHE_CSV,
    school_holidays_url: str = DEFAULT_SCHOOL_HOLIDAYS_URL,
    public_holidays_url: str = DEFAULT_PUBLIC_HOLIDAYS_URL,
    weather_api_base_url: str = DEFAULT_WEATHER_API_BASE_URL,
    school_zone: str = DEFAULT_SCHOOL_ZONE,
    latitude: float = DEFAULT_WEATHER_LATITUDE,
    longitude: float = DEFAULT_WEATHER_LONGITUDE,
    timezone_name: str = DEFAULT_WEATHER_TIMEZONE,
    download_text: TextDownloader = _download_text,
    download_json: JsonDownloader = _download_json,
) -> pd.DataFrame:
    """Construit les features open data a partir des dates d'origine et de cible."""

    origin_dates = cast(pd.Series, dataset_df[ORIGIN_DATE_COLUMN])
    target_dates = cast(pd.Series, dataset_df[TARGET_DATE_COLUMN])
    holidays_df = _load_school_holidays(
        cache_path=holiday_cache_path,
        source_url=school_holidays_url,
        download_text=download_text,
    )
    public_holidays_df = _load_public_holidays(
        cache_path=public_holiday_cache_path,
        source_url=public_holidays_url,
        download_json=download_json,
    )
    weather_df = _load_weather_history(
        origin_dates=origin_dates,
        cache_path=weather_cache_path,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        weather_api_base_url=weather_api_base_url,
        download_json=download_json,
    )
    holiday_features = _build_holiday_feature_frame(
        target_dates=target_dates,
        holidays_df=holidays_df,
        school_zone=school_zone,
    )
    public_holiday_features = _build_public_holiday_feature_frame(
        target_dates=target_dates,
        public_holidays_df=public_holidays_df,
    )
    weather_features = _build_weather_feature_frame(
        origin_dates=origin_dates,
        weather_df=weather_df,
    )
    feature_df = pd.concat([holiday_features, public_holiday_features, weather_features], axis=1)
    LOGGER.info("Built %d open-data feature columns", len(feature_df.columns))
    return feature_df
