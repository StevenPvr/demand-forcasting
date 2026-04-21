from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from praedixa.platform.signals.open_data.annual_public import (
    empty_macro_annual_frame,
    empty_public_holidays_frame,
    empty_school_holidays_frame,
    empty_weather_frame,
    fetch_macro_annual_frame,
    fetch_public_holidays_frame,
    read_silver_location_bounds,
)
from praedixa.platform.signals.open_data.france_macro import (
    empty_insee_macro_timeseries_frame,
    fetch_insee_macro_timeseries_frame,
)
from praedixa.platform.signals.open_data.catchment_cache import (
    empty_location_catchment_frame,
    fetch_location_catchment_frame_with_cache,
)
from praedixa.platform.signals.open_data.http import http_json, http_text
from praedixa.platform.signals.open_data.runtime import (
    DEFAULT_OPEN_METEO_ARCHIVE_URL,
    OpenExogenousRuntimeConfig,
)
from praedixa.platform.signals.open_data.transformers import (
    build_location_metadata_frame,
    compute_country_years,
    fetch_school_holidays_frame,
    fetch_weather_frame,
)
from praedixa.platform.governance.source_registry import (
    is_provider_runtime_enabled,
    registry_entry_by_source_id,
)


def provider_policy_snapshot(
    *,
    allow_contractual_providers: bool,
    provider_source_ids: list[str],
) -> dict[str, dict[str, object]]:
    """Return the effective provider-policy snapshot for the current run."""

    snapshot: dict[str, dict[str, object]] = {}
    for source_id in provider_source_ids:
        entry = registry_entry_by_source_id(source_id)
        snapshot[source_id] = {
            "enabled": is_provider_runtime_enabled(
                source_id,
                allow_contractual_providers=allow_contractual_providers,
            ),
            "review_status": entry.review_status,
            "contract_required": entry.contract_required,
            "license_type": entry.license_type,
            "legal_basis": entry.legal_basis,
        }
    return snapshot


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_country_context(
    config: OpenExogenousRuntimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    silver_locations = read_silver_location_bounds(config.duckdb_path)
    location_metadata = build_location_metadata_frame(
        silver_locations,
        bakery_city_name=config.bakery_city_name,
        bakery_school_zone=config.bakery_school_zone,
        bakery_latitude=config.bakery_latitude,
        bakery_longitude=config.bakery_longitude,
    )
    country_years = compute_country_years(silver_locations, location_metadata)
    return silver_locations, location_metadata, country_years


def _build_provider_policy(config: OpenExogenousRuntimeConfig) -> dict[str, dict[str, object]]:
    return provider_policy_snapshot(
        allow_contractual_providers=config.allow_contractual_providers,
        provider_source_ids=[
            "deterministic_public_holidays",
            "french_school_calendar_ics",
            "openstreetmap_odbl",
            "open_meteo_api",
            "world_bank_indicators",
            "insee_bdm",
        ],
    )


def _resolve_location_catchment_outputs(
    *,
    config: OpenExogenousRuntimeConfig,
    location_metadata: pd.DataFrame,
    provider_policy: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    empty_metadata: dict[str, object] = {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_entries": 0,
        "cache_path": str(config.location_catchment_cache_path),
    }
    if not provider_policy["openstreetmap_odbl"]["enabled"]:
        return empty_location_catchment_frame(), empty_metadata
    return fetch_location_catchment_frame_with_cache(
        location_metadata,
        cache_path=config.location_catchment_cache_path,
        snapshot_dir=config.location_catchment_snapshot_dir,
        timeout_seconds=config.http_timeout_seconds,
        max_retries=config.http_max_retries,
        retry_backoff_seconds=config.http_retry_backoff_seconds,
        http_json_reader=http_json,
    )


def _resolve_public_holidays_frame(
    *,
    country_years: dict[str, list[int]],
    provider_policy: dict[str, dict[str, object]],
    config: OpenExogenousRuntimeConfig,
) -> pd.DataFrame:
    if not provider_policy["deterministic_public_holidays"]["enabled"]:
        return empty_public_holidays_frame()
    return fetch_public_holidays_frame(
        country_years,
        timeout_seconds=config.http_timeout_seconds,
        max_retries=config.http_max_retries,
        retry_backoff_seconds=config.http_retry_backoff_seconds,
    )


def _resolve_school_holidays_frame(
    *,
    location_metadata: pd.DataFrame,
    silver_locations: pd.DataFrame,
    provider_policy: dict[str, dict[str, object]],
    config: OpenExogenousRuntimeConfig,
) -> pd.DataFrame:
    if not provider_policy["french_school_calendar_ics"]["enabled"]:
        return empty_school_holidays_frame()
    return fetch_school_holidays_frame(
        location_metadata,
        silver_locations,
        timeout_seconds=config.http_timeout_seconds,
        max_retries=config.http_max_retries,
        retry_backoff_seconds=config.http_retry_backoff_seconds,
        http_text_reader=http_text,
    )


def _resolve_weather_frame(
    *,
    location_metadata: pd.DataFrame,
    silver_locations: pd.DataFrame,
    provider_policy: dict[str, dict[str, object]],
    config: OpenExogenousRuntimeConfig,
) -> pd.DataFrame:
    if not provider_policy["open_meteo_api"]["enabled"]:
        return empty_weather_frame()
    return fetch_weather_frame(
        location_metadata,
        silver_locations,
        timeout_seconds=config.http_timeout_seconds,
        max_retries=config.http_max_retries,
        retry_backoff_seconds=config.http_retry_backoff_seconds,
        max_locations_per_request=config.open_meteo_max_locations_per_request,
        inter_batch_sleep_seconds=config.open_meteo_inter_batch_sleep_seconds,
        archive_url=DEFAULT_OPEN_METEO_ARCHIVE_URL,
        http_json_reader=http_json,
    )


def _resolve_macro_annual_frame(
    *,
    location_metadata: pd.DataFrame,
    provider_policy: dict[str, dict[str, object]],
    config: OpenExogenousRuntimeConfig,
) -> pd.DataFrame:
    if not provider_policy["world_bank_indicators"]["enabled"]:
        return empty_macro_annual_frame()
    return fetch_macro_annual_frame(
        country_codes=sorted(location_metadata["country_code"].dropna().unique()),
        timeout_seconds=config.http_timeout_seconds,
        max_retries=config.http_max_retries,
        retry_backoff_seconds=config.http_retry_backoff_seconds,
    )


def _resolve_macro_timeseries_frame(
    *,
    location_metadata: pd.DataFrame,
    provider_policy: dict[str, dict[str, object]],
    config: OpenExogenousRuntimeConfig,
) -> pd.DataFrame:
    if not provider_policy["insee_bdm"]["enabled"]:
        return empty_insee_macro_timeseries_frame()
    return fetch_insee_macro_timeseries_frame(
        country_codes=sorted(location_metadata["country_code"].dropna().unique()),
        timeout_seconds=config.http_timeout_seconds,
        max_retries=config.http_max_retries,
        retry_backoff_seconds=config.http_retry_backoff_seconds,
    )


def _build_open_exogenous_frames(
    config: OpenExogenousRuntimeConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, list[int]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    silver_locations, location_metadata, country_years = _build_country_context(config)
    provider_policy = _build_provider_policy(config)
    location_catchment_frame, catchment_metadata = _resolve_location_catchment_outputs(
        config=config,
        location_metadata=location_metadata,
        provider_policy=provider_policy,
    )

    frames = {
        "location_metadata": location_metadata,
        "location_catchment": location_catchment_frame,
        "public_holidays": _resolve_public_holidays_frame(
            country_years=country_years,
            provider_policy=provider_policy,
            config=config,
        ),
        "school_holidays": _resolve_school_holidays_frame(
            location_metadata=location_metadata,
            silver_locations=silver_locations,
            provider_policy=provider_policy,
            config=config,
        ),
        "weather_daily": _resolve_weather_frame(
            location_metadata=location_metadata,
            silver_locations=silver_locations,
            provider_policy=provider_policy,
            config=config,
        ),
        "macro_annual": _resolve_macro_annual_frame(
            location_metadata=location_metadata,
            provider_policy=provider_policy,
            config=config,
        ),
        "macro_timeseries": _resolve_macro_timeseries_frame(
            location_metadata=location_metadata,
            provider_policy=provider_policy,
            config=config,
        ),
    }
    return silver_locations, frames, country_years, provider_policy, catchment_metadata


def _open_exogenous_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "location_metadata_path": output_dir / "location_metadata.csv",
        "location_catchment_path": output_dir / "location_catchment.csv",
        "public_holidays_path": output_dir / "public_holidays.csv",
        "school_holidays_path": output_dir / "school_holidays.csv",
        "weather_daily_path": output_dir / "weather_daily.csv",
        "macro_annual_path": output_dir / "macro_annual.csv",
        "macro_timeseries_path": output_dir / "macro_timeseries.csv",
    }


def _persist_named_open_exogenous_frames(frames: dict[str, pd.DataFrame], output_paths: dict[str, Path]) -> None:
    for frame_name, output_key in [
        ("location_metadata", "location_metadata_path"),
        ("location_catchment", "location_catchment_path"),
        ("public_holidays", "public_holidays_path"),
        ("school_holidays", "school_holidays_path"),
        ("weather_daily", "weather_daily_path"),
        ("macro_annual", "macro_annual_path"),
        ("macro_timeseries", "macro_timeseries_path"),
    ]:
        _write_csv(frames[frame_name], output_paths[output_key])


def _build_open_exogenous_manifest(
    *,
    config: OpenExogenousRuntimeConfig,
    frames: dict[str, pd.DataFrame],
    country_years: dict[str, list[int]],
    provider_policy: dict[str, dict[str, object]],
    catchment_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "duckdb_path": config.duckdb_path,
        "output_dir": str(config.output_dir),
        "location_rows": int(len(frames["location_metadata"])),
        "location_catchment_rows": int(len(frames["location_catchment"])),
        "public_holiday_rows": int(len(frames["public_holidays"])),
        "school_holiday_rows": int(len(frames["school_holidays"])),
        "weather_rows": int(len(frames["weather_daily"])),
        "macro_annual_rows": int(len(frames["macro_annual"])),
        "macro_timeseries_rows": int(len(frames["macro_timeseries"])),
        "location_catchment_cache": catchment_metadata,
        "location_catchment_snapshot": catchment_metadata,
        "country_years": country_years,
        "allow_contractual_providers": config.allow_contractual_providers,
        "provider_policy": provider_policy,
        "sources": {
            "location_catchment": "OpenStreetMap ODbL via Overpass-derived POI catchment features",
            "public_holidays": "Deterministic local public-holiday rules for supported countries",
            "school_holidays": "French Ministry of Education open data ICS",
            "weather_daily": "Open-Meteo Historical Weather API (contract required for commercial usage)",
            "macro_annual": "World Bank Indicators API",
            "macro_timeseries": "INSEE BDM SDMX service",
        },
    }


def _log_open_exogenous_row_counts(frames: dict[str, pd.DataFrame]) -> None:
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "Open exogenous data written: locations=%s catchment=%s holidays=%s school=%s weather=%s macro_annual=%s macro_timeseries=%s",
        len(frames["location_metadata"]),
        len(frames["location_catchment"]),
        len(frames["public_holidays"]),
        len(frames["school_holidays"]),
        len(frames["weather_daily"]),
        len(frames["macro_annual"]),
        len(frames["macro_timeseries"]),
    )


def _persist_open_exogenous_frames(
    *,
    config: OpenExogenousRuntimeConfig,
    frames: dict[str, pd.DataFrame],
    country_years: dict[str, list[int]],
    provider_policy: dict[str, dict[str, object]],
    catchment_metadata: dict[str, object],
) -> dict[str, object]:
    output_paths = _open_exogenous_output_paths(config.output_dir)
    _persist_named_open_exogenous_frames(frames, output_paths)
    manifest = _build_open_exogenous_manifest(
        config=config,
        frames=frames,
        country_years=country_years,
        provider_policy=provider_policy,
        catchment_metadata=catchment_metadata,
    )
    manifest_path = config.output_dir / "open_exogenous_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    _log_open_exogenous_row_counts(frames)
    return {**output_paths, "manifest_path": manifest_path}


def fetch_open_exogenous_data(config: OpenExogenousRuntimeConfig) -> dict[str, object]:
    """Fetch and persist the open-source exogenous datasets needed by the gold layer."""

    _, frames, country_years, provider_policy, catchment_metadata = _build_open_exogenous_frames(config)
    return _persist_open_exogenous_frames(
        config=config,
        frames=frames,
        country_years=country_years,
        provider_policy=provider_policy,
        catchment_metadata=catchment_metadata,
    )
