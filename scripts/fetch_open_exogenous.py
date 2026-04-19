from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.open_exogenous_macro_timeseries import (  # noqa: E402
    fetch_macro_timeseries_frame,
)
from scripts.open_exogenous_transformers import (  # noqa: E402
    M5_STATE_PROXY_SCHOOL,
    M5_STATE_PROXY_WEATHER,
    build_location_metadata_frame,
    compute_country_years,
    fetch_school_holidays_frame,
    fetch_weather_frame,
    parse_school_holiday_ics,
)

DEFAULT_DUCKDB_PATH = Path("data/warehouse/praedixa.duckdb")
DEFAULT_OUTPUT_DIR = Path("data/external_open")
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_HTTP_MAX_RETRIES = 5
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_OPEN_METEO_MAX_LOCATIONS_PER_REQUEST = 25
DEFAULT_OPEN_METEO_INTER_BATCH_SLEEP_SECONDS = 0.5
DEFAULT_BAKERY_CITY_NAME = "Paris"
DEFAULT_BAKERY_SCHOOL_ZONE = "C"
DEFAULT_BAKERY_LATITUDE = 48.8566
DEFAULT_BAKERY_LONGITUDE = 2.3522
DEFAULT_NAGER_BASE_URL = "https://date.nager.at/api/v3"
DEFAULT_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"

WORLD_BANK_INDICATORS = {
    "gdp_growth_latest": "NY.GDP.MKTP.KD.ZG",
    "gdp_current_usd_latest": "NY.GDP.MKTP.CD",
    "lending_interest_rate_latest": "FR.INR.LEND",
    "government_debt_pct_gdp_latest": "GC.DOD.TOTL.GD.ZS",
}

logger = logging.getLogger(__name__)


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

def build_runtime_config_from_env() -> OpenExogenousRuntimeConfig:
    """Build the fetch runtime configuration from environment variables."""

    return OpenExogenousRuntimeConfig(
        duckdb_path=os.environ.get(
            "PRAEDIXA_DUCKDB_TARGET_PATH",
            os.environ.get("PRAEDIXA_DUCKDB_LOCAL_PATH", str(DEFAULT_DUCKDB_PATH)),
        ),
        output_dir=Path(os.environ.get("PRAEDIXA_OPEN_EXOGENOUS_DIR", str(DEFAULT_OUTPUT_DIR))),
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
    )


def _http_read(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    request = Request(url, headers={"User-Agent": "praedixa-open-exogenous/1.0"})
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                sleep_seconds = retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "HTTP %s on %s, retrying in %.1fs (attempt %s/%s)",
                    exc.code,
                    url,
                    sleep_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_seconds)
                continue
            raise
        except URLError:
            if attempt < max_retries:
                sleep_seconds = retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "Network error on %s, retrying in %.1fs (attempt %s/%s)",
                    url,
                    sleep_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_seconds)
                continue
            raise
    raise RuntimeError(f"Failed to download {url}")


def _http_json(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> Any:
    return json.loads(
        _http_read(
            url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    )

def _http_text(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    return _http_read(
        url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def _curl_text(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    command = ["curl", "-fsSL", "--max-time", str(timeout_seconds), url]
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as exc:
            if attempt < max_retries:
                sleep_seconds = retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "curl failed on %s, retrying in %.1fs (attempt %s/%s)",
                    url,
                    sleep_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"Failed to download {url} with curl") from exc
    raise RuntimeError(f"Failed to download {url} with curl")

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
        metadata[["dataset_source", "location_id", "country_code"]],
        on=["dataset_source", "location_id"],
        how="left",
    )
    merged["min_dt"] = pd.to_datetime(merged["min_dt"])
    merged["max_dt"] = pd.to_datetime(merged["max_dt"])
    bounds: dict[str, tuple[str, str]] = {}
    for country_code, group in merged.dropna(subset=["country_code"]).groupby("country_code"):
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
    """Fetch public holidays from the open-source Nager.Date API."""

    rows: list[dict[str, object]] = []
    for country_code, years in country_years.items():
        for year in years:
            payload = _http_json(
                f"{DEFAULT_NAGER_BASE_URL}/PublicHolidays/{year}/{country_code}",
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            for item in payload:
                rows.append(
                    {
                        "country_code": country_code,
                        "dt": item["date"],
                        "holiday_name": item.get("name"),
                        "holiday_local_name": item.get("localName"),
                        "global_flag": item.get("global"),
                        "counties_json": json.dumps(item.get("counties"), ensure_ascii=True),
                        "holiday_types_json": json.dumps(item.get("types"), ensure_ascii=True),
                        "source_name": "nager_date",
                    }
                )

    return pd.DataFrame(rows).drop_duplicates().sort_values(["country_code", "dt", "holiday_name"]).reset_index(drop=True)


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

    payload = _http_json(
        f"{DEFAULT_WORLD_BANK_BASE_URL}/country/{country_code}/indicator/{indicator_code}?format=json&per_page=20000",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    observations = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    rows: list[dict[str, object]] = []
    for item in observations:
        if item.get("value") is None:
            continue
        observation_year = int(item["date"])
        rows.append(
            {
                "country_code": country_code.upper(),
                "indicator_code": indicator_code,
                "metric_name": metric_name,
                "observation_year": observation_year,
                "effective_from": date(observation_year + 1, 1, 1).isoformat(),
                "metric_value": float(item["value"]),
                "source_name": "world_bank_indicators",
            }
        )
    return pd.DataFrame(rows)


def fetch_macro_annual_frame(
    country_codes: list[str],
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Fetch annual macro indicators from the World Bank API for all required countries."""

    frames = [
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
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "country_code",
                "indicator_code",
                "metric_name",
                "observation_year",
                "effective_from",
                "metric_value",
                "source_name",
            ]
        )

    return pd.concat(frames, axis=0, ignore_index=True).sort_values(
        ["country_code", "metric_name", "observation_year"]
    ).reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_open_exogenous_frames(config: OpenExogenousRuntimeConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[int]]]:
    silver_locations = read_silver_location_bounds(config.duckdb_path)
    location_metadata = build_location_metadata_frame(
        silver_locations,
        bakery_city_name=config.bakery_city_name,
        bakery_school_zone=config.bakery_school_zone,
        bakery_latitude=config.bakery_latitude,
        bakery_longitude=config.bakery_longitude,
    )
    country_years = compute_country_years(silver_locations, location_metadata)
    country_date_bounds = compute_country_date_bounds(silver_locations, location_metadata)

    frames = {
        "location_metadata": location_metadata,
        "public_holidays": fetch_public_holidays_frame(
            country_years,
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
        ),
        "school_holidays": fetch_school_holidays_frame(
            location_metadata,
            silver_locations,
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
            http_text_reader=_http_text,
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
            http_json_reader=_http_json,
        ),
        "macro_annual": fetch_macro_annual_frame(
            country_codes=sorted(location_metadata["country_code"].dropna().unique()),
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
        ),
        "macro_timeseries": fetch_macro_timeseries_frame(
            country_codes=sorted(location_metadata["country_code"].dropna().unique()),
            country_date_bounds=country_date_bounds,
            timeout_seconds=config.http_timeout_seconds,
            max_retries=config.http_max_retries,
            retry_backoff_seconds=config.http_retry_backoff_seconds,
            http_text_reader=_curl_text,
        ),
    }
    return silver_locations, frames, country_years


def _persist_open_exogenous_frames(
    *,
    config: OpenExogenousRuntimeConfig,
    frames: dict[str, pd.DataFrame],
    country_years: dict[str, list[int]],
) -> dict[str, object]:
    output_dir = config.output_dir
    output_paths = {
        "location_metadata_path": output_dir / "location_metadata.csv",
        "public_holidays_path": output_dir / "public_holidays.csv",
        "school_holidays_path": output_dir / "school_holidays.csv",
        "weather_daily_path": output_dir / "weather_daily.csv",
        "macro_annual_path": output_dir / "macro_annual.csv",
        "macro_timeseries_path": output_dir / "macro_timeseries.csv",
    }
    for frame_name, output_key in [
        ("location_metadata", "location_metadata_path"),
        ("public_holidays", "public_holidays_path"),
        ("school_holidays", "school_holidays_path"),
        ("weather_daily", "weather_daily_path"),
        ("macro_annual", "macro_annual_path"),
        ("macro_timeseries", "macro_timeseries_path"),
    ]:
        _write_csv(frames[frame_name], output_paths[output_key])

    manifest = {
        "duckdb_path": config.duckdb_path,
        "output_dir": str(output_dir),
        "location_rows": int(len(frames["location_metadata"])),
        "public_holiday_rows": int(len(frames["public_holidays"])),
        "school_holiday_rows": int(len(frames["school_holidays"])),
        "weather_rows": int(len(frames["weather_daily"])),
        "macro_annual_rows": int(len(frames["macro_annual"])),
        "macro_timeseries_rows": int(len(frames["macro_timeseries"])),
        "country_years": country_years,
        "sources": {
            "public_holidays": "Nager.Date open-source API",
            "school_holidays": "French Ministry of Education open data ICS",
            "weather_daily": "Open-Meteo Historical Weather API",
            "macro_annual": "World Bank Indicators API",
            "macro_timeseries": "FRED public CSV graph endpoint",
        },
    }
    manifest_path = output_dir / "open_exogenous_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    logger.info(
        "Open exogenous data written: locations=%s holidays=%s school=%s weather=%s macro_annual=%s macro_timeseries=%s",
        len(frames["location_metadata"]),
        len(frames["public_holidays"]),
        len(frames["school_holidays"]),
        len(frames["weather_daily"]),
        len(frames["macro_annual"]),
        len(frames["macro_timeseries"]),
    )

    return {**output_paths, "manifest_path": manifest_path}


def fetch_open_exogenous_data(config: OpenExogenousRuntimeConfig) -> dict[str, object]:
    """Fetch and persist the open-source exogenous datasets needed by the gold layer."""

    _, frames, country_years = _build_open_exogenous_frames(config)
    return _persist_open_exogenous_frames(config=config, frames=frames, country_years=country_years)


def main() -> None:
    """CLI entrypoint for refreshing the local open-source exogenous datasets."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    fetch_open_exogenous_data(build_runtime_config_from_env())


if __name__ == "__main__":
    main()
