from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from praedixa.platform.signals.open_data.http import HttpJsonReader
from praedixa.platform.signals.open_data.transformers import fetch_location_catchment_frame


LOGGER = logging.getLogger(__name__)


def empty_location_catchment_frame() -> pd.DataFrame:
    """Return an empty catchment frame with the canonical CSV columns."""

    return pd.DataFrame(
        columns=[
            "dataset_source",
            "location_id",
            "population_1km",
            "population_3km",
            "office_poi_count_1km",
            "school_poi_count_1km",
            "transit_station_count_1km",
            "mall_poi_count_1km",
            "tourism_poi_count_1km",
            "competitor_count_500m",
            "competitor_count_1km",
            "parking_score",
            "source_name",
        ]
    )


def empty_location_catchment_cache_frame() -> pd.DataFrame:
    """Return an empty persisted catchment snapshot frame."""

    return pd.DataFrame(
        columns=[
            "dataset_source",
            "location_id",
            "site_format",
            "latitude",
            "longitude",
            "cache_key",
            "cached_at",
            "population_1km",
            "population_3km",
            "office_poi_count_1km",
            "school_poi_count_1km",
            "transit_station_count_1km",
            "mall_poi_count_1km",
            "tourism_poi_count_1km",
            "competitor_count_500m",
            "competitor_count_1km",
            "parking_score",
            "source_name",
        ]
    )


def _location_catchment_cache_key(
    *,
    dataset_source: str,
    location_id: str,
    site_format: str,
    latitude: float,
    longitude: float,
) -> str:
    return "|".join(
        [
            dataset_source,
            location_id,
            site_format or "unknown",
            f"{latitude:.5f}",
            f"{longitude:.5f}",
        ]
    )


def _location_metadata_with_cache_keys(location_metadata: pd.DataFrame) -> pd.DataFrame:
    keyed = location_metadata.copy()
    records = cast(list[dict[str, Any]], keyed.to_dict(orient="records"))
    cache_keys: list[object] = []
    for row in records:
        latitude = row.get("latitude")
        longitude = row.get("longitude")
        if bool(pd.isna(latitude)) or bool(pd.isna(longitude)):
            cache_keys.append(pd.NA)
            continue
        cache_keys.append(
            _location_catchment_cache_key(
                dataset_source=str(row["dataset_source"]),
                location_id=str(row["location_id"]),
                site_format=str(row.get("site_format", "unknown") or "unknown"),
                latitude=float(cast(float | int, latitude)),
                longitude=float(cast(float | int, longitude)),
            )
        )
    keyed["cache_key"] = pd.Series(cache_keys, index=keyed.index, dtype="string")
    return keyed


def _normalize_location_catchment_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy() if not frame.empty else empty_location_catchment_cache_frame()
    expected_columns = empty_location_catchment_cache_frame().columns
    for column in expected_columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for numeric_column in [
        "latitude",
        "longitude",
        "population_1km",
        "population_3km",
        "office_poi_count_1km",
        "school_poi_count_1km",
        "transit_station_count_1km",
        "mall_poi_count_1km",
        "tourism_poi_count_1km",
        "competitor_count_500m",
        "competitor_count_1km",
    ]:
        normalized[numeric_column] = pd.to_numeric(normalized[numeric_column], errors="coerce")
    normalized["parking_score"] = normalized["parking_score"].astype("string")
    normalized["cache_key"] = normalized["cache_key"].astype("string")
    return normalized[list(expected_columns)]


def load_location_catchment_cache(cache_path: Path) -> pd.DataFrame:
    """Load the persisted catchment snapshot if it already exists."""

    if not cache_path.exists():
        return empty_location_catchment_cache_frame()
    try:
        return _normalize_location_catchment_cache_frame(pd.read_csv(cache_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load location catchment cache from %s: %s", cache_path, exc)
        return empty_location_catchment_cache_frame()


def _location_catchment_cache_rows(
    fetched_frame: pd.DataFrame,
    fetchable_locations: pd.DataFrame,
) -> pd.DataFrame:
    if fetched_frame.empty:
        return empty_location_catchment_cache_frame()
    joined = fetchable_locations[
        ["dataset_source", "location_id", "site_format", "latitude", "longitude", "cache_key"]
    ].merge(
        fetched_frame,
        on=["dataset_source", "location_id"],
        how="inner",
    )
    joined["cached_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return _normalize_location_catchment_cache_frame(joined)


def _publish_location_catchment_snapshot(
    *,
    latest_cache_path: Path,
    snapshot_dir: Path,
    cache_frame: pd.DataFrame,
) -> dict[str, str]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_label = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    versioned_snapshot_path = snapshot_dir / f"location_catchment_snapshot_{snapshot_label}.csv"
    cache_frame.to_csv(versioned_snapshot_path, index=False)
    latest_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_frame.to_csv(latest_cache_path, index=False)
    latest_manifest_path = snapshot_dir / "location_catchment_snapshot_latest.json"
    latest_manifest_path.write_text(
        json.dumps(
            {
                "published_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "latest_snapshot_path": str(versioned_snapshot_path),
                "latest_cache_path": str(latest_cache_path),
                "row_count": int(len(cache_frame)),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return {
        "published_snapshot_path": str(versioned_snapshot_path),
        "latest_cache_path": str(latest_cache_path),
        "latest_snapshot_manifest_path": str(latest_manifest_path),
    }


def _empty_location_catchment_cache_stats(cache_path: Path) -> dict[str, object]:
    return {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_entries": 0,
        "cache_path": str(cache_path),
    }


def _location_catchment_updated_cache(
    *,
    cached_frame: pd.DataFrame,
    missing_locations: pd.DataFrame,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> pd.DataFrame:
    fetched_frame = (
        fetch_location_catchment_frame(
            missing_locations,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            http_json_reader=http_json_reader,
        )
        if not missing_locations.empty
        else empty_location_catchment_frame()
    )
    fetched_cache_rows = _location_catchment_cache_rows(fetched_frame, missing_locations)
    updated_cache = pd.concat([cached_frame, fetched_cache_rows], ignore_index=True)
    return (
        updated_cache.drop_duplicates(subset=["cache_key"], keep="last")
        .sort_values(["dataset_source", "location_id", "cache_key"], kind="stable")
        .reset_index(drop=True)
    )


def _resolved_location_catchment_frame(
    *,
    fetchable_locations: pd.DataFrame,
    updated_cache: pd.DataFrame,
) -> pd.DataFrame:
    resolved_frame = fetchable_locations[["dataset_source", "location_id", "cache_key"]].merge(
        updated_cache[
            [
                "dataset_source",
                "location_id",
                "cache_key",
                "population_1km",
                "population_3km",
                "office_poi_count_1km",
                "school_poi_count_1km",
                "transit_station_count_1km",
                "mall_poi_count_1km",
                "tourism_poi_count_1km",
                "competitor_count_500m",
                "competitor_count_1km",
                "parking_score",
                "source_name",
            ]
        ],
        on=["dataset_source", "location_id", "cache_key"],
        how="inner",
    )
    return resolved_frame.drop(columns=["cache_key"]).reset_index(drop=True)


def _location_catchment_cache_stats(
    *,
    cache_path: Path,
    fetchable_locations: pd.DataFrame,
    missing_locations: pd.DataFrame,
    updated_cache: pd.DataFrame,
) -> dict[str, object]:
    return {
        "cache_hits": int(len(fetchable_locations) - len(missing_locations)),
        "cache_misses": int(len(missing_locations)),
        "cache_entries": int(len(updated_cache)),
        "cache_path": str(cache_path),
    }


def fetch_location_catchment_frame_with_cache(
    location_metadata: pd.DataFrame,
    *,
    cache_path: Path,
    snapshot_dir: Path,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_json_reader: HttpJsonReader,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Resolve catchment features using a persisted local snapshot before hitting OSM."""

    keyed_locations = _location_metadata_with_cache_keys(location_metadata)
    fetchable_locations = keyed_locations.loc[keyed_locations["cache_key"].notna()].copy()
    if fetchable_locations.empty:
        return empty_location_catchment_frame(), _empty_location_catchment_cache_stats(cache_path)

    cached_frame = load_location_catchment_cache(cache_path)
    cached_keys = set(cached_frame["cache_key"].dropna().astype(str))
    missing_locations = fetchable_locations.loc[
        ~fetchable_locations["cache_key"].astype(str).isin(cached_keys)
    ].copy()
    updated_cache = _location_catchment_updated_cache(
        cached_frame=cached_frame,
        missing_locations=missing_locations,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        http_json_reader=http_json_reader,
    )
    snapshot_paths = _publish_location_catchment_snapshot(
        latest_cache_path=cache_path,
        snapshot_dir=snapshot_dir,
        cache_frame=updated_cache,
    )
    resolved_frame = _resolved_location_catchment_frame(
        fetchable_locations=fetchable_locations,
        updated_cache=updated_cache,
    )
    stats = _location_catchment_cache_stats(
        cache_path=cache_path,
        fetchable_locations=fetchable_locations,
        missing_locations=missing_locations,
        updated_cache=updated_cache,
    )
    return resolved_frame, {**stats, **snapshot_paths}
