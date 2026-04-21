from __future__ import annotations

from datetime import date
from typing import Any, cast

import duckdb
import pandas as pd

from praedixa.platform.signals.open_data.http import http_json
from praedixa.platform.signals.open_data.public_holidays_local import (
    deterministic_public_holiday_rows,
    supported_public_holiday_country_codes,
)
from praedixa.platform.signals.open_data.runtime import DEFAULT_WORLD_BANK_BASE_URL
from praedixa.platform.signals.open_data.runtime import WORLD_BANK_INDICATORS

COUNTRY_CODE_COL = "country_code"
DATASET_SOURCE_COL = "dataset_source"
LOCATION_ID_COL = "location_id"
DT_COL = "dt"
SOURCE_NAME_COL = "source_name"
METRIC_NAME_COL = "metric_name"
HOLIDAY_NAME_COL = "holiday_name"
OBSERVATION_YEAR_COL = "observation_year"
EFFECTIVE_FROM_COL = "effective_from"
METRIC_VALUE_COL = "metric_value"
WORLD_BANK_INDICATORS_SOURCE_NAME = "world_bank_indicators"


def read_silver_location_bounds(duckdb_path: str) -> pd.DataFrame:
    """Return one row per dataset/location with region code and date bounds from silver."""

    connection = duckdb.connect(duckdb_path, read_only=True)
    try:
        return connection.execute(
            """
            select
                dataset_source,
                location_id,
                max(region_id) as region_code,
                min(dt) as min_dt,
                max(dt) as max_dt
            from silver.silver_daily_product_demand
            group by
                dataset_source,
                location_id
            order by
                dataset_source,
                location_id
            """
        ).fetchdf()
    finally:
        connection.close()


def compute_country_date_bounds(
    silver_locations: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    lookback_days: int = 730,
) -> dict[str, tuple[str, str]]:
    """Return bounded date windows per country for lean macro series downloads."""

    merged = silver_locations.merge(
        metadata[[DATASET_SOURCE_COL, LOCATION_ID_COL, COUNTRY_CODE_COL]],
        on=[DATASET_SOURCE_COL, LOCATION_ID_COL],
        how="left",
    )
    merged["min_dt"] = pd.to_datetime(merged["min_dt"])
    merged["max_dt"] = pd.to_datetime(merged["max_dt"])
    bounds: dict[str, tuple[str, str]] = {}
    for country_code, group in merged.dropna(subset=[COUNTRY_CODE_COL]).groupby(COUNTRY_CODE_COL):
        min_dt = group["min_dt"].min() - pd.Timedelta(days=lookback_days)
        max_dt = group["max_dt"].max() + pd.Timedelta(days=1)
        bounds[str(country_code)] = (
            min_dt.date().isoformat(),
            max_dt.date().isoformat(),
        )
    return bounds


def fetch_public_holidays_frame(
    country_years: dict[str, list[int]],
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Build public-holiday rows from deterministic local rules when supported."""

    del timeout_seconds, max_retries, retry_backoff_seconds
    rows: list[dict[str, object]] = []
    for country_code, years in country_years.items():
        normalized_country_code = country_code.upper()
        if normalized_country_code not in supported_public_holiday_country_codes():
            continue
        for year in years:
            rows.extend(deterministic_public_holiday_rows(normalized_country_code, year))
    if not rows:
        return empty_public_holidays_frame()
    return pd.DataFrame(rows).drop_duplicates().sort_values([COUNTRY_CODE_COL, DT_COL, HOLIDAY_NAME_COL]).reset_index(drop=True)


def empty_public_holidays_frame() -> pd.DataFrame:
    """Return an empty public-holidays frame with the canonical CSV columns."""

    return pd.DataFrame(
        columns=[
            COUNTRY_CODE_COL,
            DT_COL,
            HOLIDAY_NAME_COL,
            "holiday_local_name",
            "global_flag",
            "counties_json",
            "holiday_types_json",
            SOURCE_NAME_COL,
        ]
    )


def _world_bank_observations(payload: object) -> list[dict[str, Any]]:
    raw_observations = _world_bank_payload_observations(payload)
    if raw_observations is None:
        return []
    return _world_bank_dict_items(raw_observations)


def _world_bank_payload_observations(payload: object) -> list[object] | None:
    if not isinstance(payload, list):
        return None
    raw_payload = cast(list[object], payload)
    if len(raw_payload) <= 1 or not isinstance(raw_payload[1], list):
        return None
    return cast(list[object], raw_payload[1])


def _world_bank_dict_items(raw_items: list[object]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            observations.append(cast(dict[str, Any], item))
    return observations


def _world_bank_indicator_rows(
    *,
    country_code: str,
    indicator_code: str,
    metric_name: str,
    observations: list[dict[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in observations:
        if item.get("value") is None:
            continue
        observation_year = int(str(item["date"]))
        rows.append(
            {
                COUNTRY_CODE_COL: country_code.upper(),
                "indicator_code": indicator_code,
                METRIC_NAME_COL: metric_name,
                OBSERVATION_YEAR_COL: observation_year,
                EFFECTIVE_FROM_COL: date(observation_year + 1, 1, 1).isoformat(),
                METRIC_VALUE_COL: float(str(item["value"])),
                SOURCE_NAME_COL: WORLD_BANK_INDICATORS_SOURCE_NAME,
            }
        )
    return rows


def fetch_world_bank_indicator(
    *,
    country_code: str,
    indicator_code: str,
    metric_name: str,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Fetch one annual macro series from the World Bank API."""

    payload: object = http_json(
        f"{DEFAULT_WORLD_BANK_BASE_URL}/country/{country_code}/indicator/{indicator_code}?format=json&per_page=20000",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return pd.DataFrame(
        _world_bank_indicator_rows(
            country_code=country_code,
            indicator_code=indicator_code,
            metric_name=metric_name,
            observations=_world_bank_observations(payload),
        )
    )


def _world_bank_indicator_frames(
    *,
    country_codes: list[str],
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[pd.DataFrame]:
    return [
        fetch_world_bank_indicator(
            country_code=country_code,
            indicator_code=indicator_code,
            metric_name=metric_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        for country_code in sorted(country_codes)
        for metric_name, indicator_code in WORLD_BANK_INDICATORS.items()
    ]


def fetch_macro_annual_frame(
    country_codes: list[str],
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Fetch annual macro indicators from the World Bank API for all required countries."""

    frames = _world_bank_indicator_frames(
        country_codes=country_codes,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_macro_annual_frame()

    return pd.concat(frames, axis=0, ignore_index=True).sort_values(
        [COUNTRY_CODE_COL, METRIC_NAME_COL, OBSERVATION_YEAR_COL]
    ).reset_index(drop=True)


def empty_school_holidays_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            DATASET_SOURCE_COL,
            LOCATION_ID_COL,
            DT_COL,
            "school_holiday_name",
            "school_zone",
            SOURCE_NAME_COL,
        ]
    )


def empty_weather_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            DATASET_SOURCE_COL,
            LOCATION_ID_COL,
            DT_COL,
            "latitude",
            "longitude",
            "weather_temperature_mean",
            "weather_temperature_min",
            "weather_temperature_max",
            "weather_precipitation_sum",
            "weather_relative_humidity_mean",
            "weather_wind_speed_mean",
            SOURCE_NAME_COL,
        ]
    )


def empty_macro_annual_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COUNTRY_CODE_COL,
            "indicator_code",
            METRIC_NAME_COL,
            OBSERVATION_YEAR_COL,
            EFFECTIVE_FROM_COL,
            METRIC_VALUE_COL,
            SOURCE_NAME_COL,
        ]
    )
