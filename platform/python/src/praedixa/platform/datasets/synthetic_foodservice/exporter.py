from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (
    build_city_weather_frame,
    build_synthetic_calendar,
)
from praedixa.platform.datasets.synthetic_foodservice.config import (
    SyntheticFoodserviceConfig,
)
from praedixa.platform.datasets.synthetic_foodservice.demand import (
    MODEL_FACING_COLUMNS,
    ORACLE_COLUMNS,
    generate_location_batch,
)
from praedixa.platform.datasets.synthetic_foodservice.entities import (
    build_synthetic_foodservice_entities,
)
from praedixa.platform.datasets.synthetic_foodservice.quality import (
    GenerationStatsBuilder,
    SyntheticGenerationStats,
    build_manifest_payload,
    validate_model_facing_frame,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticFoodserviceArtifacts:
    """Files produced by the one-shot synthetic generator."""

    daily_csv_path: Path
    oracle_csv_path: Path | None
    manifest_path: Path
    location_metadata_csv_path: Path
    open_exogenous_location_metadata_csv_path: Path


def generate_synthetic_foodservice_dataset(
    config: SyntheticFoodserviceConfig,
) -> SyntheticFoodserviceArtifacts:
    """Generate stable CSV sources consumed later by the medallion pipeline."""

    _prepare_output_files(config)
    calendar = build_synthetic_calendar(config)
    entities = build_synthetic_foodservice_entities(config)
    city_weather = build_city_weather_frame(
        cities=entities.cities,
        calendar=calendar,
        seed=config.seed,
    )
    _write_location_metadata(config, entities.location_metadata)
    stats_builder = GenerationStatsBuilder(
        site_count=len(entities.locations),
        city_count=len(entities.cities),
    )
    for batch_index, location_batch in enumerate(
        _iter_location_batches(entities.locations, config.site_batch_size)
    ):
        result = generate_location_batch(
            locations=location_batch,
            assortments=entities.assortments,
            calendar=calendar,
            city_weather=city_weather,
            seed=config.seed,
            source_run_id=config.source_run_id,
        )
        validate_model_facing_frame(result.daily)
        _append_csv(result.daily, config.daily_csv_path, columns=MODEL_FACING_COLUMNS)
        if config.write_oracle_debug:
            _append_csv(result.oracle, config.oracle_csv_path, columns=ORACLE_COLUMNS)
        stats_builder.add_batch(daily=result.daily, oracle=result.oracle)
        _log_batch(batch_index, result.daily)
    _write_manifest(config, stats_builder.build())
    return SyntheticFoodserviceArtifacts(
        daily_csv_path=config.daily_csv_path,
        oracle_csv_path=config.oracle_csv_path if config.write_oracle_debug else None,
        manifest_path=config.manifest_path,
        location_metadata_csv_path=config.location_metadata_csv_path,
        open_exogenous_location_metadata_csv_path=config.open_exogenous_location_metadata_csv_path,
    )


def _prepare_output_files(config: SyntheticFoodserviceConfig) -> None:
    for path in _output_paths(config):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and config.overwrite_existing:
            path.unlink()
        elif path.exists():
            raise FileExistsError(f"Synthetic output already exists: {path}")


def _output_paths(config: SyntheticFoodserviceConfig) -> tuple[Path, ...]:
    paths = [
        config.daily_csv_path,
        config.manifest_path,
        config.location_metadata_csv_path,
    ]
    if config.write_oracle_debug:
        paths.append(config.oracle_csv_path)
    return tuple(paths)


def _iter_location_batches(
    locations: pd.DataFrame,
    batch_size: int,
) -> list[pd.DataFrame]:
    return [
        locations.iloc[start : start + batch_size].copy()
        for start in range(0, len(locations), batch_size)
    ]


def _append_csv(frame: pd.DataFrame, path: Path, *, columns: tuple[str, ...]) -> None:
    if frame.empty:
        return
    header = not path.exists()
    frame.loc[:, list(columns)].to_csv(path, mode="a", header=header, index=False)


def _write_location_metadata(
    config: SyntheticFoodserviceConfig,
    metadata: pd.DataFrame,
) -> None:
    ordered = metadata.loc[:, _location_metadata_columns()].copy()
    ordered.to_csv(config.location_metadata_csv_path, index=False)
    config.open_exogenous_location_metadata_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    merged = _merged_open_exogenous_location_metadata(
        existing_path=config.open_exogenous_location_metadata_csv_path,
        synthetic_metadata=ordered,
    )
    merged.to_csv(config.open_exogenous_location_metadata_csv_path, index=False)


def _location_metadata_columns() -> list[str]:
    return [
        "dataset_source",
        "location_id",
        "country_code",
        "region_code",
        "city_name",
        "latitude",
        "longitude",
        "school_zone",
        "weather_location_label",
        "assumption_source",
        "drive_through_flag",
        "delivery_flag",
        "pickup_flag",
        "mall_flag",
        "transit_hub_flag",
        "tourism_flag",
    ]


def _merged_open_exogenous_location_metadata(
    *,
    existing_path: Path,
    synthetic_metadata: pd.DataFrame,
) -> pd.DataFrame:
    if not existing_path.exists():
        return synthetic_metadata
    existing = pd.read_csv(existing_path)
    synthetic_sources = set(synthetic_metadata["dataset_source"].astype(str).unique())
    retained = existing.loc[
        ~existing["dataset_source"].astype(str).isin(synthetic_sources)
    ]
    return pd.concat([retained, synthetic_metadata], ignore_index=True)


def _write_manifest(
    config: SyntheticFoodserviceConfig,
    stats: SyntheticGenerationStats,
) -> None:
    payload = build_manifest_payload(
        seed=config.seed,
        start_date=config.start_date,
        end_date=config.end_date,
        stats=stats,
        location_metadata_path=str(config.location_metadata_csv_path),
        hidden_oracle_columns=ORACLE_COLUMNS,
    )
    config.manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _log_batch(batch_index: int, frame: pd.DataFrame) -> None:
    if frame.empty:
        LOGGER.info("Generated synthetic batch %s with 0 rows", batch_index)
        return
    LOGGER.info(
        "Generated synthetic batch %s rows=%s sites=%s sources=%s",
        batch_index,
        len(frame),
        frame["location_id"].nunique(),
        sorted(frame["dataset_source"].unique().tolist()),
    )
