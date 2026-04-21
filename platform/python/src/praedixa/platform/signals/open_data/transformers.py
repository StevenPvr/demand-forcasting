from __future__ import annotations

from praedixa.platform.signals.open_data.location_catchment import (
    COMPETITOR_TAGS_BY_SITE_FORMAT,
    OVERPASS_BASE_URL,
    RESIDENTIAL_POPULATION_WEIGHTS,
    fetch_location_catchment_frame,
)
from praedixa.platform.signals.open_data.location_metadata import (
    FIXED_SOURCE_METADATA,
    build_location_metadata_frame,
    compute_country_years,
)
from praedixa.platform.signals.open_data.school_holidays import (
    parse_school_holiday_ics,
    fetch_school_holidays_frame,
)
from praedixa.platform.signals.open_data.weather import fetch_weather_frame


__all__ = [
    "COMPETITOR_TAGS_BY_SITE_FORMAT",
    "FIXED_SOURCE_METADATA",
    "OVERPASS_BASE_URL",
    "RESIDENTIAL_POPULATION_WEIGHTS",
    "build_location_metadata_frame",
    "compute_country_years",
    "fetch_location_catchment_frame",
    "fetch_school_holidays_frame",
    "fetch_weather_frame",
    "parse_school_holiday_ics",
]
