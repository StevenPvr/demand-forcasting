from __future__ import annotations

from datetime import timedelta
import time
from typing import Any, Callable
from urllib.parse import urlencode

import pandas as pd

from scripts.open_exogenous_school_proxy_data import M5_PROXY_SCHOOL_BREAKS, M5_STATE_PROXY_SCHOOL


M5_STATE_PROXY_WEATHER = {
    "CA": {
        "city_name": "Sacramento",
        "latitude": 38.5816,
        "longitude": -121.4944,
        "weather_location_label": "state_capital_proxy",
        "assumption_source": "m5_state_capital_proxy_weather",
    },
    "TX": {
        "city_name": "Austin",
        "latitude": 30.2672,
        "longitude": -97.7431,
        "weather_location_label": "state_capital_proxy",
        "assumption_source": "m5_state_capital_proxy_weather",
    },
    "WI": {
        "city_name": "Madison",
        "latitude": 43.0731,
        "longitude": -89.4012,
        "weather_location_label": "state_capital_proxy",
        "assumption_source": "m5_state_capital_proxy_weather",
    },
}

def _expand_school_break_rows(
    *,
    dataset_source: str,
    location_id: str,
    school_zone: str,
    start: str,
    end: str,
    holiday_name: str,
    source_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_dt = pd.Timestamp(start).date()
    end_dt = pd.Timestamp(end).date()
    while current_dt <= end_dt:
        rows.append(
            {
                "dataset_source": dataset_source,
                "location_id": location_id,
                "dt": current_dt.isoformat(),
                "school_holiday_name": holiday_name,
                "school_zone": school_zone,
                "source_name": source_name,
            }
        )
        current_dt += timedelta(days=1)
    return rows


def _m5_location_metadata_row(location: object) -> dict[str, object]:
    state_code = str(getattr(location, "region_code") or str(getattr(location, "location_id")).split("_")[0]).upper()
    proxy = M5_STATE_PROXY_WEATHER[state_code]
    school_proxy = M5_STATE_PROXY_SCHOOL[state_code]
    return {
        "dataset_source": getattr(location, "dataset_source"),
        "location_id": getattr(location, "location_id"),
        "country_code": "US",
        "region_code": state_code,
        "city_name": proxy["city_name"],
        "latitude": proxy["latitude"],
        "longitude": proxy["longitude"],
        "school_zone": school_proxy["school_zone"],
        "weather_location_label": proxy["weather_location_label"],
        "assumption_source": f'{proxy["assumption_source"]};{school_proxy["assumption_source"]}',
    }


def _bakery_location_metadata_row(
    location: object,
    *,
    bakery_city_name: str,
    bakery_school_zone: str,
    bakery_latitude: float,
    bakery_longitude: float,
) -> dict[str, object]:
    return {
        "dataset_source": getattr(location, "dataset_source"),
        "location_id": getattr(location, "location_id"),
        "country_code": "FR",
        "region_code": None,
        "city_name": bakery_city_name,
        "latitude": bakery_latitude,
        "longitude": bakery_longitude,
        "school_zone": bakery_school_zone,
        "weather_location_label": "manual_bakery_location_proxy",
        "assumption_source": "bakery_manual_default_location_hypothesis",
    }


def _freshretail_location_metadata_row(location: object) -> dict[str, object]:
    return {
        "dataset_source": getattr(location, "dataset_source"),
        "location_id": getattr(location, "location_id"),
        "country_code": "CN",
        "region_code": getattr(location, "region_code"),
        "city_name": None,
        "latitude": None,
        "longitude": None,
        "school_zone": None,
        "weather_location_label": None,
        "assumption_source": "freshretail_city_id_without_public_geocoding",
    }


def _fixed_country_location_metadata_row(
    location: object,
    *,
    country_code: str | None,
    assumption_source: str,
) -> dict[str, object]:
    return {
        "dataset_source": getattr(location, "dataset_source"),
        "location_id": getattr(location, "location_id"),
        "country_code": country_code,
        "region_code": getattr(location, "region_code"),
        "city_name": None,
        "latitude": None,
        "longitude": None,
        "school_zone": None,
        "weather_location_label": None,
        "assumption_source": assumption_source,
    }


def build_location_metadata_frame(
    silver_locations: pd.DataFrame,
    *,
    bakery_city_name: str,
    bakery_school_zone: str,
    bakery_latitude: float,
    bakery_longitude: float,
) -> pd.DataFrame:
    """Create explicit location metadata and geocoding assumptions for exogenous joins."""

    rows: list[dict[str, object]] = []
    for location in silver_locations.itertuples(index=False):
        if location.dataset_source == "m5":
            rows.append(_m5_location_metadata_row(location))
        elif location.dataset_source == "bakery":
            rows.append(
                _bakery_location_metadata_row(
                    location,
                    bakery_city_name=bakery_city_name,
                    bakery_school_zone=bakery_school_zone,
                    bakery_latitude=bakery_latitude,
                    bakery_longitude=bakery_longitude,
                )
            )
        elif location.dataset_source == "freshretail":
            rows.append(_freshretail_location_metadata_row(location))
        elif location.dataset_source == "freshretail_lt":
            rows.append(
                _fixed_country_location_metadata_row(
                    location,
                    country_code="CN",
                    assumption_source="freshretail_lt_without_public_geocoding",
                )
            )
        elif location.dataset_source in {"uci_online_retail", "uci_online_retail_ii"}:
            rows.append(
                _fixed_country_location_metadata_row(
                    location,
                    country_code="GB",
                    assumption_source="uci_online_retail_single_uk_retailer_hypothesis",
                )
            )
        elif location.dataset_source == "mendeley_pharmacy_id":
            rows.append(
                _fixed_country_location_metadata_row(
                    location,
                    country_code="ID",
                    assumption_source="mendeley_pharmacy_indonesia_country_hypothesis",
                )
            )
        elif location.dataset_source == "mendeley_bangladesh_retail":
            rows.append(
                _fixed_country_location_metadata_row(
                    location,
                    country_code="BD",
                    assumption_source="mendeley_bangladesh_retail_country_hypothesis",
                )
            )
        elif location.dataset_source == "mendeley_ecommerce":
            rows.append(
                _fixed_country_location_metadata_row(
                    location,
                    country_code=None,
                    assumption_source="mendeley_ecommerce_country_unknown",
                )
            )

    return pd.DataFrame(rows).sort_values(["dataset_source", "location_id"]).reset_index(drop=True)


def compute_country_years(
    silver_locations: pd.DataFrame,
    location_metadata: pd.DataFrame,
) -> dict[str, list[int]]:
    """Return required calendar years per country, including target-day spillover."""

    merged = silver_locations.merge(
        location_metadata[["dataset_source", "location_id", "country_code"]],
        on=["dataset_source", "location_id"],
        how="left",
    )
    country_years: dict[str, set[int]] = {}
    for row in merged.itertuples(index=False):
        if pd.isna(row.country_code) or row.country_code is None:
            continue
        min_year = pd.Timestamp(row.min_dt).year
        max_year = (pd.Timestamp(row.max_dt) + pd.Timedelta(days=1)).year
        country_years.setdefault(str(row.country_code), set()).update(range(min_year, max_year + 1))
    return {country: sorted(years) for country, years in country_years.items()}


def parse_school_holiday_ics(ics_text: str, *, dataset_source: str, location_id: str, school_zone: str) -> pd.DataFrame:
    """Parse one zone ICS file from the French official school holiday open data."""

    events: list[dict[str, str]] = []
    current_event: dict[str, str] = {}
    for raw_line in ics_text.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            current_event = {}
        elif line == "END:VEVENT":
            if {"start", "end", "summary"} <= set(current_event):
                events.append(current_event.copy())
            current_event = {}
        elif line.startswith("DTSTART"):
            current_event["start"] = line.split(":")[-1]
        elif line.startswith("DTEND"):
            current_event["end"] = line.split(":")[-1]
        elif line.startswith("SUMMARY:"):
            current_event["summary"] = line.split(":", 1)[1]

    rows: list[dict[str, object]] = []
    for event in events:
        start_dt = pd.Timestamp(event["start"]).date()
        end_dt = pd.Timestamp(event["end"]).date()
        current_dt = start_dt
        while current_dt < end_dt:
            rows.append(
                {
                    "dataset_source": dataset_source,
                    "location_id": location_id,
                    "dt": current_dt.isoformat(),
                    "school_holiday_name": event["summary"],
                    "school_zone": school_zone,
                    "source_name": "fr_education_school_calendar",
                }
            )
            current_dt += timedelta(days=1)

    return pd.DataFrame(rows)


def _location_school_holiday_slice(
    school_frame: pd.DataFrame,
    *,
    dataset_source: str,
    location_id: str,
    school_zone: str,
    min_dt: object,
    max_dt: object,
) -> pd.DataFrame:
    location_days = school_frame.loc[
        school_frame["school_zone"].eq(school_zone)
        & school_frame["dt"].between(pd.Timestamp(min_dt).date().isoformat(), pd.Timestamp(max_dt).date().isoformat())
    ].copy()
    if location_days.empty:
        return location_days

    location_days["dataset_source"] = dataset_source
    location_days["location_id"] = location_id
    return location_days


def _empty_school_holidays_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["dataset_source", "location_id", "dt", "school_holiday_name", "school_zone", "source_name"]
    )


def _school_rows_with_bounds(location_metadata: pd.DataFrame, silver_locations: pd.DataFrame) -> pd.DataFrame:
    school_rows = location_metadata.loc[
        location_metadata["school_zone"].notna()
        & (location_metadata["country_code"].eq("FR") | location_metadata["dataset_source"].eq("m5"))
    ].copy()
    if school_rows.empty:
        return school_rows

    return school_rows.merge(
        silver_locations[["dataset_source", "location_id", "min_dt", "max_dt"]],
        on=["dataset_source", "location_id"],
        how="left",
    )


def _fetch_school_zone_frames(
    school_rows: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: Callable[..., str],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for school_zone in sorted(school_rows["school_zone"].dropna().unique()):
        ics_text = http_text_reader(
            f"https://fr.ftp.opendatasoft.com/openscol/fr-en-calendrier-scolaire/Zone-{school_zone}.ics",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        frames.append(
            parse_school_holiday_ics(
                ics_text,
                dataset_source="bakery",
                location_id="bakery_store_1",
                school_zone=school_zone,
            )
        )
    return frames


def _build_m5_proxy_school_holidays_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state_code, proxy in M5_STATE_PROXY_SCHOOL.items():
        for school_break in M5_PROXY_SCHOOL_BREAKS[state_code]:
            rows.extend(
                _expand_school_break_rows(
                    dataset_source="m5",
                    location_id=f"{state_code}_proxy_school_calendar",
                    school_zone=proxy["school_zone"],
                    start=school_break["start"],
                    end=school_break["end"],
                    holiday_name=school_break["name"],
                    source_name=str(school_break["source_name"]),
                )
            )
    if not rows:
        return _empty_school_holidays_frame()
    return pd.DataFrame(rows).drop_duplicates().sort_values(["school_zone", "dt"]).reset_index(drop=True)


def _school_source_frames(
    school_rows: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: Callable[..., str],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    french_school_rows = school_rows.loc[school_rows["country_code"].eq("FR")].copy()
    if not french_school_rows.empty:
        frames.extend(
            _fetch_school_zone_frames(
                french_school_rows,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                http_text_reader=http_text_reader,
            )
        )

    if school_rows["dataset_source"].eq("m5").any():
        frames.append(_build_m5_proxy_school_holidays_frame())

    return frames


def fetch_school_holidays_frame(
    location_metadata: pd.DataFrame,
    silver_locations: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: Callable[..., str],
) -> pd.DataFrame:
    """Fetch French school holidays for any configured French school zones."""

    school_rows = _school_rows_with_bounds(location_metadata, silver_locations)
    if school_rows.empty:
        return _empty_school_holidays_frame()

    frames = _school_source_frames(
        school_rows,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        http_text_reader=http_text_reader,
    )
    school_frame = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if school_frame.empty:
        return _empty_school_holidays_frame()

    filtered_frames = [
        _location_school_holiday_slice(
            school_frame,
            dataset_source=row.dataset_source,
            location_id=row.location_id,
            school_zone=row.school_zone,
            min_dt=row.min_dt,
            max_dt=row.max_dt,
        )
        for row in school_rows.itertuples(index=False)
    ]
    filtered_frames = [frame for frame in filtered_frames if not frame.empty]
    if not filtered_frames:
        return _empty_school_holidays_frame()

    return pd.concat(filtered_frames, axis=0, ignore_index=True).drop_duplicates().sort_values(
        ["dataset_source", "location_id", "dt"]
    ).reset_index(drop=True)


def _weather_frame_for_location(
    row: object,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    archive_url: str,
    http_json_reader: Callable[..., Any],
) -> pd.DataFrame:
    query = urlencode(
        {
            "latitude": row.latitude,
            "longitude": row.longitude,
            "start_date": pd.Timestamp(row.min_dt).date().isoformat(),
            "end_date": pd.Timestamp(row.max_dt).date().isoformat(),
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "relative_humidity_2m_mean",
                    "wind_speed_10m_mean",
                ]
            ),
            "timezone": "auto",
        }
    )
    payload = http_json_reader(
        f"{archive_url}?{query}",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    daily = payload["daily"]
    return pd.DataFrame(
        {
            "dataset_source": row.dataset_source,
            "location_id": row.location_id,
            "dt": daily["time"],
            "latitude": row.latitude,
            "longitude": row.longitude,
            "weather_temperature_mean": daily.get("temperature_2m_mean"),
            "weather_temperature_min": daily.get("temperature_2m_min"),
            "weather_temperature_max": daily.get("temperature_2m_max"),
            "weather_precipitation_sum": daily.get("precipitation_sum"),
            "weather_relative_humidity_mean": daily.get("relative_humidity_2m_mean"),
            "weather_wind_speed_mean": daily.get("wind_speed_10m_mean"),
            "source_name": "open_meteo_archive",
        }
    )


def _weather_location_batches(weather_locations: pd.DataFrame, max_locations_per_request: int) -> list[pd.DataFrame]:
    if max_locations_per_request <= 0:
        raise ValueError("max_locations_per_request must be strictly positive.")
    return [
        weather_locations.iloc[start_idx : start_idx + max_locations_per_request].copy()
        for start_idx in range(0, len(weather_locations), max_locations_per_request)
    ]


def _weather_payload_to_frames(payload: Any, batch: pd.DataFrame) -> list[pd.DataFrame]:
    payload_items = payload if isinstance(payload, list) else [payload]
    if len(payload_items) != len(batch):
        raise ValueError(
            f"Open-Meteo returned {len(payload_items)} payload blocks for {len(batch)} requested locations."
        )

    frames: list[pd.DataFrame] = []
    for payload_item, row in zip(payload_items, batch.itertuples(index=False), strict=True):
        daily = payload_item["daily"]
        frames.append(
            pd.DataFrame(
                {
                    "dataset_source": row.dataset_source,
                    "location_id": row.location_id,
                    "dt": daily["time"],
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "weather_temperature_mean": daily.get("temperature_2m_mean"),
                    "weather_temperature_min": daily.get("temperature_2m_min"),
                    "weather_temperature_max": daily.get("temperature_2m_max"),
                    "weather_precipitation_sum": daily.get("precipitation_sum"),
                    "weather_relative_humidity_mean": daily.get("relative_humidity_2m_mean"),
                    "weather_wind_speed_mean": daily.get("wind_speed_10m_mean"),
                    "source_name": "open_meteo_archive",
                }
            )
        )
    return frames


def _weather_frames_for_batch(
    batch: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    archive_url: str,
    http_json_reader: Callable[..., Any],
) -> list[pd.DataFrame]:
    query = urlencode(
        {
            "latitude": ",".join(str(value) for value in batch["latitude"].tolist()),
            "longitude": ",".join(str(value) for value in batch["longitude"].tolist()),
            "start_date": pd.Timestamp(batch["min_dt"].min()).date().isoformat(),
            "end_date": pd.Timestamp(batch["max_dt"].max()).date().isoformat(),
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "relative_humidity_2m_mean",
                    "wind_speed_10m_mean",
                ]
            ),
            "timezone": ",".join(["auto"] * len(batch)),
        }
    )
    payload = http_json_reader(
        f"{archive_url}?{query}",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return _weather_payload_to_frames(payload, batch)


def _empty_weather_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset_source",
            "location_id",
            "dt",
            "latitude",
            "longitude",
            "weather_temperature_mean",
            "weather_temperature_min",
            "weather_temperature_max",
            "weather_precipitation_sum",
            "weather_relative_humidity_mean",
            "weather_wind_speed_mean",
            "source_name",
        ]
    )


def _weather_locations_with_bounds(location_metadata: pd.DataFrame, silver_locations: pd.DataFrame) -> pd.DataFrame:
    weather_locations = location_metadata.loc[
        location_metadata["latitude"].notna() & location_metadata["longitude"].notna()
    ].copy()
    if weather_locations.empty:
        return weather_locations

    return weather_locations.merge(
        silver_locations[["dataset_source", "location_id", "min_dt", "max_dt"]],
        on=["dataset_source", "location_id"],
        how="left",
    )


def fetch_weather_frame(
    location_metadata: pd.DataFrame,
    silver_locations: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    max_locations_per_request: int = 25,
    inter_batch_sleep_seconds: float = 0.0,
    archive_url: str,
    http_json_reader: Callable[..., Any],
) -> pd.DataFrame:
    """Fetch historical daily weather from Open-Meteo for locations with known coordinates."""

    weather_locations = _weather_locations_with_bounds(location_metadata, silver_locations)
    if weather_locations.empty:
        return _empty_weather_frame()

    frames: list[pd.DataFrame] = []
    location_batches = _weather_location_batches(weather_locations, max_locations_per_request)
    for batch_index, batch in enumerate(location_batches):
        frames.extend(
            _weather_frames_for_batch(
                batch,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                archive_url=archive_url,
                http_json_reader=http_json_reader,
            )
        )
        if inter_batch_sleep_seconds > 0 and batch_index < len(location_batches) - 1:
            time.sleep(inter_batch_sleep_seconds)

    if not frames:
        return _empty_weather_frame()

    return pd.concat(frames, axis=0, ignore_index=True).sort_values(["dataset_source", "location_id", "dt"]).reset_index(
        drop=True
    )
