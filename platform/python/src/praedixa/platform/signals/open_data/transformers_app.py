from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from praedixa.platform.signals.open_data.annual_public import read_silver_location_bounds
from praedixa.platform.signals.open_data.http import http_json, http_text
from praedixa.platform.signals.open_data.runtime import (
    DEFAULT_OPEN_METEO_ARCHIVE_URL,
    OpenExogenousRuntimeConfig,
    build_runtime_config_from_env,
)
from praedixa.platform.signals.open_data.transformers import (
    COMPETITOR_TAGS_BY_SITE_FORMAT,
    FIXED_SOURCE_METADATA,
    OVERPASS_BASE_URL,
    RESIDENTIAL_POPULATION_WEIGHTS,
    build_location_metadata_frame,
    compute_country_years,
    fetch_location_catchment_frame,
    fetch_school_holidays_frame,
    fetch_weather_frame,
    parse_school_holiday_ics,
)


LOGGER = logging.getLogger(__name__)


def build_open_exogenous_transformer_frames(config: OpenExogenousRuntimeConfig) -> dict[str, pd.DataFrame]:
    silver_locations = read_silver_location_bounds(config.duckdb_path)
    location_metadata = build_location_metadata_frame(
        silver_locations,
        bakery_city_name=config.bakery_city_name,
        bakery_school_zone=config.bakery_school_zone,
        bakery_latitude=config.bakery_latitude,
        bakery_longitude=config.bakery_longitude,
    )
    return {
        "location_metadata": location_metadata,
        "location_catchment": fetch_location_catchment_frame(
            location_metadata,
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
            http_json_reader=http_json,
        ),
        "school_holidays": fetch_school_holidays_frame(
            location_metadata,
            silver_locations,
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
            http_text_reader=http_text,
        ),
        "weather_daily": fetch_weather_frame(
            location_metadata,
            silver_locations,
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
            max_locations_per_request=config.open_meteo_max_locations_per_request,
            inter_batch_sleep_seconds=config.open_meteo_inter_batch_sleep_seconds,
            archive_url=DEFAULT_OPEN_METEO_ARCHIVE_URL,
            http_json_reader=http_json,
        ),
    }


def _write_transformer_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = build_runtime_config_from_env()
    frames = build_open_exogenous_transformer_frames(config)
    outputs = {
        "location_metadata": _write_transformer_csv(frames["location_metadata"], config.output_dir / "location_metadata.csv"),
        "location_catchment": _write_transformer_csv(frames["location_catchment"], config.output_dir / "location_catchment.csv"),
        "school_holidays": _write_transformer_csv(frames["school_holidays"], config.output_dir / "school_holidays.csv"),
        "weather_daily": _write_transformer_csv(frames["weather_daily"], config.output_dir / "weather_daily.csv"),
    }
    LOGGER.info(
        "Open exogenous transformers written: metadata=%s -> %s, catchment=%s -> %s, school=%s -> %s, weather=%s -> %s",
        len(frames["location_metadata"]),
        outputs["location_metadata"],
        len(frames["location_catchment"]),
        outputs["location_catchment"],
        len(frames["school_holidays"]),
        outputs["school_holidays"],
        len(frames["weather_daily"]),
        outputs["weather_daily"],
    )


__all__ = [
    "COMPETITOR_TAGS_BY_SITE_FORMAT",
    "FIXED_SOURCE_METADATA",
    "OVERPASS_BASE_URL",
    "OpenExogenousRuntimeConfig",
    "RESIDENTIAL_POPULATION_WEIGHTS",
    "build_location_metadata_frame",
    "build_open_exogenous_transformer_frames",
    "build_runtime_config_from_env",
    "compute_country_years",
    "fetch_location_catchment_frame",
    "fetch_school_holidays_frame",
    "fetch_weather_frame",
    "main",
    "parse_school_holiday_ics",
]


if __name__ == "__main__":
    main()
