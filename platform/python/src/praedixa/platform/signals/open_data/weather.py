from __future__ import annotations

from collections.abc import Hashable
import time
from typing import Any, Mapping, cast
from urllib.parse import urlencode

import pandas as pd

from praedixa.platform.signals.open_data.http import HttpJsonReader


def _weather_daily_fields() -> str:
    return ",".join(
        [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
        ]
    )


def _weather_payload_frame(daily: Mapping[str, object], *, row: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_source": row["dataset_source"],
            "location_id": row["location_id"],
            "dt": daily["time"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "weather_temperature_mean": daily.get("temperature_2m_mean"),
            "weather_temperature_min": daily.get("temperature_2m_min"),
            "weather_temperature_max": daily.get("temperature_2m_max"),
            "weather_precipitation_sum": daily.get("precipitation_sum"),
            "weather_relative_humidity_mean": daily.get("relative_humidity_2m_mean"),
            "weather_wind_speed_mean": daily.get("wind_speed_10m_mean"),
            "source_name": "open_meteo_archive",
        }
    )


def _string_key_row(record: dict[Hashable, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in record.items()}


def _payload_items_list(payload: Any) -> list[object]:
    if isinstance(payload, list):
        items = cast(list[object], payload)
        return list(items)
    return [cast(object, payload)]


def _weather_location_batches(weather_locations: pd.DataFrame, max_locations_per_request: int) -> list[pd.DataFrame]:
    if max_locations_per_request <= 0:
        raise ValueError("max_locations_per_request must be strictly positive.")
    return [
        weather_locations.iloc[start_idx : start_idx + max_locations_per_request].copy()
        for start_idx in range(0, len(weather_locations), max_locations_per_request)
    ]


def _weather_payload_to_frames(payload: Any, batch: pd.DataFrame) -> list[pd.DataFrame]:
    payload_items = _payload_items_list(payload)
    if len(payload_items) != len(batch):
        raise ValueError(
            f"Open-Meteo returned {len(payload_items)} payload blocks for {len(batch)} requested locations."
        )

    frames: list[pd.DataFrame] = []
    batch_records = batch.to_dict(orient="records")
    for raw_payload_item, raw_row in zip(payload_items, batch_records, strict=True):
        row = _string_key_row(raw_row)
        payload_item = cast(dict[str, object], raw_payload_item)
        daily = cast(Mapping[str, object], payload_item["daily"])
        frames.append(_weather_payload_frame(daily, row=row))
    return frames


def _weather_frames_for_batch(
    batch: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    archive_url: str,
    http_json_reader: HttpJsonReader,
) -> list[pd.DataFrame]:
    latitude_values = cast(list[object], batch["latitude"].tolist())
    longitude_values = cast(list[object], batch["longitude"].tolist())
    min_dt_series = batch["min_dt"]
    max_dt_series = batch["max_dt"]
    query = urlencode(
        {
            "latitude": ",".join(str(value) for value in latitude_values),
            "longitude": ",".join(str(value) for value in longitude_values),
            "start_date": pd.Timestamp(min_dt_series.min()).date().isoformat(),
            "end_date": pd.Timestamp(max_dt_series.max()).date().isoformat(),
            "daily": _weather_daily_fields(),
            "timezone": ",".join(["auto"] * len(batch)),
        }
    )
    payload: object = http_json_reader(
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


def _sleep_between_weather_batches(
    *,
    inter_batch_sleep_seconds: float,
    batch_index: int,
    total_batches: int,
) -> None:
    if inter_batch_sleep_seconds <= 0 or batch_index >= total_batches - 1:
        return
    time.sleep(inter_batch_sleep_seconds)


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
    http_json_reader: HttpJsonReader,
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
        _sleep_between_weather_batches(
            inter_batch_sleep_seconds=inter_batch_sleep_seconds,
            batch_index=batch_index,
            total_batches=len(location_batches),
        )

    if not frames:
        return _empty_weather_frame()

    return pd.concat(frames, axis=0, ignore_index=True).sort_values(["dataset_source", "location_id", "dt"]).reset_index(
        drop=True
    )
