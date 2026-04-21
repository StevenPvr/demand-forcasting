from __future__ import annotations

import logging

from praedixa.platform.signals.open_data.annual_public import (  # noqa: E402
    compute_country_date_bounds,
)
from praedixa.platform.signals.open_data.catchment_cache import (  # noqa: E402
    fetch_location_catchment_frame_with_cache,
)
from praedixa.platform.signals.open_data.fetcher import (  # noqa: E402
    fetch_open_exogenous_data,
)
from praedixa.platform.signals.open_data.runtime import (  # noqa: E402
    DEFAULT_BAKERY_LATITUDE,
    DEFAULT_BAKERY_LONGITUDE,
    DEFAULT_BAKERY_SCHOOL_ZONE,
    OpenExogenousRuntimeConfig,
    build_runtime_config_from_env,
)
from praedixa.platform.signals.open_data.transformers import (  # noqa: E402
    build_location_metadata_frame,
    compute_country_years,
    fetch_location_catchment_frame,
    fetch_school_holidays_frame,
    fetch_weather_frame,
    parse_school_holiday_ics,
)


__all__ = [
    "DEFAULT_BAKERY_LATITUDE",
    "DEFAULT_BAKERY_LONGITUDE",
    "DEFAULT_BAKERY_SCHOOL_ZONE",
    "OpenExogenousRuntimeConfig",
    "build_location_metadata_frame",
    "build_runtime_config_from_env",
    "compute_country_date_bounds",
    "compute_country_years",
    "fetch_location_catchment_frame",
    "fetch_location_catchment_frame_with_cache",
    "fetch_open_exogenous_data",
    "fetch_school_holidays_frame",
    "fetch_weather_frame",
    "parse_school_holiday_ics",
]


def main() -> None:
    """CLI entrypoint for refreshing the local open-source exogenous datasets."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    fetch_open_exogenous_data(build_runtime_config_from_env())


if __name__ == "__main__":
    main()
