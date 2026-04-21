from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from praedixa.platform.runtime.paths import EXTERNAL_OPEN_DIR
from praedixa.platform.runtime.paths import LOCAL_DUCKDB_PATH
from praedixa.platform.runtime.paths import PROJECT_ROOT


DEFAULT_DUCKDB_PATH = LOCAL_DUCKDB_PATH
DEFAULT_OUTPUT_DIR = EXTERNAL_OPEN_DIR
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_HTTP_MAX_RETRIES = 5
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_OPEN_METEO_MAX_LOCATIONS_PER_REQUEST = 25
DEFAULT_OPEN_METEO_INTER_BATCH_SLEEP_SECONDS = 0.5
DEFAULT_BAKERY_CITY_NAME = "Paris"
DEFAULT_BAKERY_SCHOOL_ZONE = "C"
DEFAULT_BAKERY_LATITUDE = 48.8566
DEFAULT_BAKERY_LONGITUDE = 2.3522
DEFAULT_LOCATION_CATCHMENT_CACHE_RELATIVE_PATH = Path("_cache/location_catchment_snapshot.csv")
DEFAULT_LOCATION_CATCHMENT_SNAPSHOT_RELATIVE_DIR = Path("_snapshots/location_catchment")
DEFAULT_NAGER_BASE_URL = "https://date.nager.at/api/v3"
DEFAULT_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"

WORLD_BANK_INDICATORS = {
    "gdp_growth_latest": "NY.GDP.MKTP.KD.ZG",
    "gdp_current_usd_latest": "NY.GDP.MKTP.CD",
    "lending_interest_rate_latest": "FR.INR.LEND",
    "government_debt_pct_gdp_latest": "GC.DOD.TOTL.GD.ZS",
}


@dataclass(frozen=True)
class OpenExogenousRuntimeConfig:
    """Runtime settings for the open-source exogenous fetcher."""

    duckdb_path: str
    output_dir: Path
    http_timeout_seconds: int
    http_max_retries: int
    http_retry_backoff_seconds: float
    open_meteo_max_locations_per_request: int
    open_meteo_inter_batch_sleep_seconds: float
    bakery_city_name: str
    bakery_school_zone: str
    bakery_latitude: float
    bakery_longitude: float
    location_catchment_cache_path: Path
    location_catchment_snapshot_dir: Path
    allow_contractual_providers: bool


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_output_dir_from_env() -> Path:
    output_dir = Path(os.environ.get("PRAEDIXA_OPEN_EXOGENOUS_DIR", str(DEFAULT_OUTPUT_DIR)))
    return output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir


def _resolve_location_catchment_paths(output_dir: Path) -> tuple[Path, Path]:
    cache_path = Path(
        os.environ.get(
            "PRAEDIXA_LOCATION_CATCHMENT_CACHE_PATH",
            str(output_dir / DEFAULT_LOCATION_CATCHMENT_CACHE_RELATIVE_PATH),
        )
    )
    snapshot_dir = Path(
        os.environ.get(
            "PRAEDIXA_LOCATION_CATCHMENT_SNAPSHOT_DIR",
            str(output_dir / DEFAULT_LOCATION_CATCHMENT_SNAPSHOT_RELATIVE_DIR),
        )
    )
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir
    return cache_path, snapshot_dir


def _resolve_duckdb_target_path() -> str:
    return os.environ.get(
        "PRAEDIXA_DUCKDB_TARGET_PATH",
        os.environ.get("PRAEDIXA_DUCKDB_LOCAL_PATH", str(DEFAULT_DUCKDB_PATH)),
    )


def build_runtime_config_from_env() -> OpenExogenousRuntimeConfig:
    """Build the fetch runtime configuration from environment variables."""

    output_dir = _resolve_output_dir_from_env()
    location_catchment_cache_path, location_catchment_snapshot_dir = _resolve_location_catchment_paths(output_dir)
    return OpenExogenousRuntimeConfig(
        duckdb_path=_resolve_duckdb_target_path(),
        output_dir=output_dir,
        http_timeout_seconds=int(
            os.environ.get("PRAEDIXA_OPEN_EXOGENOUS_HTTP_TIMEOUT_SECONDS", str(DEFAULT_HTTP_TIMEOUT_SECONDS))
        ),
        http_max_retries=int(
            os.environ.get("PRAEDIXA_OPEN_EXOGENOUS_HTTP_MAX_RETRIES", str(DEFAULT_HTTP_MAX_RETRIES))
        ),
        http_retry_backoff_seconds=float(
            os.environ.get(
                "PRAEDIXA_OPEN_EXOGENOUS_HTTP_RETRY_BACKOFF_SECONDS",
                str(DEFAULT_HTTP_RETRY_BACKOFF_SECONDS),
            )
        ),
        open_meteo_max_locations_per_request=int(
            os.environ.get(
                "PRAEDIXA_OPEN_METEO_MAX_LOCATIONS_PER_REQUEST",
                str(DEFAULT_OPEN_METEO_MAX_LOCATIONS_PER_REQUEST),
            )
        ),
        open_meteo_inter_batch_sleep_seconds=float(
            os.environ.get(
                "PRAEDIXA_OPEN_METEO_INTER_BATCH_SLEEP_SECONDS",
                str(DEFAULT_OPEN_METEO_INTER_BATCH_SLEEP_SECONDS),
            )
        ),
        bakery_city_name=os.environ.get("PRAEDIXA_BAKERY_CITY_NAME", DEFAULT_BAKERY_CITY_NAME),
        bakery_school_zone=os.environ.get("PRAEDIXA_BAKERY_SCHOOL_ZONE", DEFAULT_BAKERY_SCHOOL_ZONE),
        bakery_latitude=float(os.environ.get("PRAEDIXA_BAKERY_LATITUDE", str(DEFAULT_BAKERY_LATITUDE))),
        bakery_longitude=float(os.environ.get("PRAEDIXA_BAKERY_LONGITUDE", str(DEFAULT_BAKERY_LONGITUDE))),
        location_catchment_cache_path=location_catchment_cache_path,
        location_catchment_snapshot_dir=location_catchment_snapshot_dir,
        allow_contractual_providers=_env_flag("PRAEDIXA_ALLOW_CONTRACTUAL_PROVIDERS", False),
    )
