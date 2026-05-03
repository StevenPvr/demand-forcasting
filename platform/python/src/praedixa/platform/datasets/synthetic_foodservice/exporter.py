"""Orchestrate the full synthetic foodservice generation pipeline.

Generates daily CSV → tickets/lines → time aggregations → stock/staff
→ corruption → validation → manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from praedixa.platform.datasets.synthetic_foodservice.aggregation import (
    aggregate_daily_to_monthly,
    aggregate_daily_to_weekly,
    aggregate_tickets_to_15min,
    aggregate_tickets_to_halfday,
    aggregate_tickets_to_hourly,
)
from praedixa.platform.datasets.synthetic_foodservice.config import (
    SyntheticFoodserviceConfig,
)
from praedixa.platform.datasets.synthetic_foodservice.corruption import (
    assign_quality_levels,
    corrupt_daily_frame,
    corrupt_tickets_frame,
)
from praedixa.platform.datasets.synthetic_foodservice.demand import (
    MODEL_FACING_COLUMNS,
    ORACLE_COLUMNS,
    generate_location_batch,
)
from praedixa.platform.datasets.synthetic_foodservice.entities import (
    SyntheticFoodserviceEntities,
    build_synthetic_foodservice_entities,
)
from praedixa.platform.datasets.synthetic_foodservice.inventory import (
    generate_stock_data,
)
from praedixa.platform.datasets.synthetic_foodservice.quality import (
    GenerationStatsBuilder,
    SyntheticGenerationStats,
    build_manifest_payload,
    validate_model_facing_frame,
)
from praedixa.platform.datasets.synthetic_foodservice.shocks import (
    ShockEvent,
    generate_shock_schedule,
)
from praedixa.platform.datasets.synthetic_foodservice.staffing import (
    generate_staff_schedules,
)
from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (
    SyntheticCalendar,
    build_city_weather_frame,
    build_synthetic_calendar,
)
from praedixa.platform.datasets.synthetic_foodservice.tickets import (
    generate_tickets_from_daily,
)
from praedixa.platform.datasets.synthetic_foodservice.validation import (
    validate_generated_dataset,
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

    _prepare_output_dirs(config)
    calendar = build_synthetic_calendar(config)
    entities = build_synthetic_foodservice_entities(config)
    city_weather = build_city_weather_frame(
        cities=entities.cities,
        calendar=calendar,
        seed=config.seed,
    )
    _write_location_metadata(config, entities.location_metadata)

    rng_corrupt = np.random.default_rng(config.seed + 11111)
    quality_levels = _quality_levels_for_config(config, entities, rng_corrupt)

    shocks_by_location_id = _generate_shocks_by_location_id(config, entities, calendar)

    all_daily, all_oracle = _generate_all_batches(
        config,
        entities,
        calendar,
        city_weather,
        shocks_by_location_id,
    )

    if config.apply_corruption and not all_daily.empty:
        all_daily = corrupt_daily_frame(all_daily, quality_levels, rng_corrupt)

    if not all_daily.empty:
        validate_model_facing_frame(all_daily)
    stats = _build_stats_from_frames(entities, all_daily, all_oracle)

    _write_csv(all_daily, config.daily_csv_path, columns=MODEL_FACING_COLUMNS)
    if config.write_oracle_debug:
        _write_csv(all_oracle, config.oracle_csv_path, columns=ORACLE_COLUMNS)
    _log_core_files(config, all_daily, all_oracle)

    if config.write_granularity_files:
        _write_granularity_files(config, all_daily, quality_levels)

    if config.write_operational_files:
        _write_operational_files(config, all_daily, all_oracle)

    if not all_daily.empty:
        validate_generated_dataset(all_daily)

    _log_generation_summary(stats)
    _write_manifest(config, stats)

    return SyntheticFoodserviceArtifacts(
        daily_csv_path=config.daily_csv_path,
        oracle_csv_path=config.oracle_csv_path if config.write_oracle_debug else None,
        manifest_path=config.manifest_path,
        location_metadata_csv_path=config.location_metadata_csv_path,
        open_exogenous_location_metadata_csv_path=config.open_exogenous_location_metadata_csv_path,
    )


def _generate_all_batches(
    config: SyntheticFoodserviceConfig,
    entities: SyntheticFoodserviceEntities,
    calendar: SyntheticCalendar,
    city_weather: pd.DataFrame,
    shocks_by_location_id: dict[str, list[ShockEvent]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    oracle_frames: list[pd.DataFrame] = []

    for batch_index, batch in enumerate(
        _iter_location_batches(entities.locations, config.site_batch_size),
    ):
        result = generate_location_batch(
            locations=batch,
            assortments=entities.assortments,
            calendar=calendar,
            city_weather=city_weather,
            seed=config.seed,
            source_run_id=config.source_run_id,
            site_shocks_by_location_id=shocks_by_location_id,
            allow_data_gaps=config.apply_corruption,
        )
        validate_model_facing_frame(result.daily)
        daily_frames.append(result.daily)
        oracle_frames.append(result.oracle)
        _log_batch(batch_index, result.daily)

    all_daily = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=MODEL_FACING_COLUMNS)
    )
    all_oracle = (
        pd.concat(oracle_frames, ignore_index=True)
        if oracle_frames
        else pd.DataFrame(columns=ORACLE_COLUMNS)
    )
    return all_daily, all_oracle


def _generate_shocks_by_location_id(
    config: SyntheticFoodserviceConfig,
    entities: SyntheticFoodserviceEntities,
    calendar: SyntheticCalendar,
) -> dict[str, list[ShockEvent]]:
    if not config.inject_shocks:
        return {}
    schedule = generate_shock_schedule(
        calendar=calendar,
        site_count=len(entities.locations),
        seed=config.seed,
    )
    result: dict[str, list[ShockEvent]] = {}
    for site_index, shock in schedule:
        location_id = str(entities.locations.iloc[site_index]["location_id"])
        result.setdefault(location_id, []).append(shock)
    return result


def _quality_levels_for_config(
    config: SyntheticFoodserviceConfig,
    entities: SyntheticFoodserviceEntities,
    rng: np.random.Generator,
) -> pd.DataFrame:
    quality_levels = entities.locations[["location_id"]].copy()
    if config.apply_corruption:
        quality_levels["data_quality_level"] = assign_quality_levels(
            entities.locations, rng
        )
    else:
        quality_levels["data_quality_level"] = "clean"
    return quality_levels


def _build_stats_from_frames(
    entities: SyntheticFoodserviceEntities,
    daily: pd.DataFrame,
    oracle: pd.DataFrame,
) -> SyntheticGenerationStats:
    stats_builder = GenerationStatsBuilder(
        site_count=len(entities.locations),
        city_count=len(entities.cities),
    )
    stats_builder.add_batch(daily=daily, oracle=oracle)
    return stats_builder.build()


def _write_granularity_files(
    config: SyntheticFoodserviceConfig,
    daily: pd.DataFrame,
    quality_levels: pd.DataFrame,
) -> None:
    LOGGER.info("Generating granularity files...")
    rng = np.random.default_rng(config.seed + 22222)

    tickets, lines = generate_tickets_from_daily(daily, seed=config.seed)
    if config.apply_corruption:
        tickets = corrupt_tickets_frame(tickets, quality_levels, rng)
        lines = _duplicate_lines_for_duplicate_tickets(lines, tickets)
        lines = _align_line_slots_to_tickets(lines, tickets)
    _write_csv_simple(tickets, config.tickets_csv_path)
    _write_csv_simple(lines, config.lines_csv_path)
    LOGGER.info("  tickets=%d lines=%d", len(tickets), len(lines))

    agg_15 = aggregate_tickets_to_15min(lines)
    _write_csv_simple(agg_15, config.agg_15min_csv_path)

    agg_h = aggregate_tickets_to_hourly(lines)
    _write_csv_simple(agg_h, config.agg_hourly_csv_path)

    agg_hd = aggregate_tickets_to_halfday(lines)
    _write_csv_simple(agg_hd, config.agg_halfday_csv_path)

    weekly = aggregate_daily_to_weekly(daily)
    _write_csv_simple(weekly, config.agg_weekly_csv_path)

    monthly = aggregate_daily_to_monthly(daily)
    _write_csv_simple(monthly, config.agg_monthly_csv_path)
    LOGGER.info(
        "  15min=%d hourly=%d halfday=%d weekly=%d monthly=%d",
        len(agg_15),
        len(agg_h),
        len(agg_hd),
        len(weekly),
        len(monthly),
    )


def _log_core_files(
    config: SyntheticFoodserviceConfig,
    daily: pd.DataFrame,
    oracle: pd.DataFrame,
) -> None:
    LOGGER.info(
        "Daily model-facing file: rows=%d path=%s", len(daily), config.daily_csv_path
    )
    if config.write_oracle_debug:
        LOGGER.info(
            "Oracle debug file: rows=%d path=%s",
            len(oracle),
            config.oracle_csv_path,
        )


def _duplicate_lines_for_duplicate_tickets(
    lines: pd.DataFrame,
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    if lines.empty or tickets.empty or "ticket_id" not in tickets.columns:
        return lines
    duplicate_tickets = tickets.loc[
        tickets["ticket_id"].astype(str).str.endswith("_dup")
    ].copy()
    if duplicate_tickets.empty:
        return lines
    duplicates: list[pd.DataFrame] = []
    for duplicate_id in duplicate_tickets["ticket_id"].astype(str):
        original_id = duplicate_id.removesuffix("_dup")
        source_lines = lines.loc[lines["ticket_id"].astype(str).eq(original_id)].copy()
        if source_lines.empty:
            continue
        source_lines["ticket_id"] = duplicate_id
        source_lines["sales_line_id"] = (
            source_lines["sales_line_id"].astype(str) + "_dup"
        )
        duplicates.append(source_lines)
    if not duplicates:
        return lines
    return pd.concat([lines, *duplicates], ignore_index=True)


def _align_line_slots_to_tickets(
    lines: pd.DataFrame,
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    if lines.empty or tickets.empty:
        return lines
    if "time_slot" not in lines.columns or "time_slot" not in tickets.columns:
        return lines
    aligned = lines.drop(columns=["time_slot"]).merge(
        tickets[["ticket_id", "time_slot"]],
        on="ticket_id",
        how="left",
    )
    aligned["time_slot"] = (
        pd.to_numeric(aligned["time_slot"], errors="coerce").fillna(0).astype(int)
    )
    return aligned.loc[:, lines.columns]


def _write_operational_files(
    config: SyntheticFoodserviceConfig,
    daily: pd.DataFrame,
    oracle: pd.DataFrame,
) -> None:
    LOGGER.info("Generating operational files...")
    snapshots, movements = generate_stock_data(daily, oracle, seed=config.seed)
    _write_csv_simple(snapshots, config.stock_snapshots_csv_path)
    _write_csv_simple(movements, config.inventory_movements_csv_path)
    LOGGER.info(
        "  stock_snapshots=%d inventory_movements=%d", len(snapshots), len(movements)
    )

    schedules = generate_staff_schedules(daily, oracle, seed=config.seed)
    _write_csv_simple(schedules, config.staff_schedules_csv_path)
    LOGGER.info("  staff_schedules=%d", len(schedules))


# ---------- IO helpers ----------


def _prepare_output_dirs(config: SyntheticFoodserviceConfig) -> None:
    for path in _all_output_paths(config):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and config.overwrite_existing:
            path.unlink()


def _all_output_paths(config: SyntheticFoodserviceConfig) -> list[Path]:
    paths = [
        config.daily_csv_path,
        config.manifest_path,
        config.location_metadata_csv_path,
    ]
    if config.write_oracle_debug:
        paths.append(config.oracle_csv_path)
    if config.write_granularity_files:
        paths.extend(
            [
                config.tickets_csv_path,
                config.lines_csv_path,
                config.agg_15min_csv_path,
                config.agg_hourly_csv_path,
                config.agg_halfday_csv_path,
                config.agg_weekly_csv_path,
                config.agg_monthly_csv_path,
            ]
        )
    if config.write_operational_files:
        paths.extend(
            [
                config.stock_snapshots_csv_path,
                config.inventory_movements_csv_path,
                config.staff_schedules_csv_path,
            ]
        )
    return paths


def _write_csv(frame: pd.DataFrame, path: Path, *, columns: tuple[str, ...]) -> None:
    if frame.empty:
        return
    frame.loc[:, list(columns)].to_csv(path, index=False)


def _write_csv_simple(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _iter_location_batches(
    locations: pd.DataFrame, batch_size: int
) -> list[pd.DataFrame]:
    return [
        locations.iloc[s : s + batch_size].copy()
        for s in range(0, len(locations), batch_size)
    ]


def _write_location_metadata(
    config: SyntheticFoodserviceConfig, metadata: pd.DataFrame
) -> None:
    cols = [
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
    ordered = metadata.loc[:, cols].copy()
    ordered.to_csv(config.location_metadata_csv_path, index=False)
    config.open_exogenous_location_metadata_csv_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    merged = _merge_exogenous_metadata(
        config.open_exogenous_location_metadata_csv_path, ordered
    )
    merged.to_csv(config.open_exogenous_location_metadata_csv_path, index=False)


def _merge_exogenous_metadata(
    existing_path: Path, synthetic: pd.DataFrame
) -> pd.DataFrame:
    if not existing_path.exists():
        return synthetic
    existing = pd.read_csv(existing_path)
    sources = set(synthetic["dataset_source"].astype(str).unique())
    retained = existing.loc[~existing["dataset_source"].astype(str).isin(sources)]
    return pd.concat([retained, synthetic], ignore_index=True)


def _write_manifest(
    config: SyntheticFoodserviceConfig, stats: SyntheticGenerationStats
) -> None:
    payload = build_manifest_payload(
        seed=config.seed,
        start_date=config.start_date,
        end_date=config.end_date,
        stats=stats,
        location_metadata_path=str(config.location_metadata_csv_path),
        hidden_oracle_columns=ORACLE_COLUMNS,
        config_metadata={
            "schema_version": config.schema_version,
            "target_contract": config.target_contract,
            "real_data_policy": config.real_data_policy,
            "panel_mode": config.panel_mode,
            "validation_profile": config.validation_profile,
            "world_count": config.world_count,
            "apply_corruption": config.apply_corruption,
            "inject_shocks": config.inject_shocks,
        },
    )
    config.manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _log_batch(batch_index: int, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    LOGGER.info(
        "Batch %s: rows=%s sites=%s",
        batch_index,
        len(frame),
        frame["location_id"].nunique(),
    )


def _log_generation_summary(stats: SyntheticGenerationStats) -> None:
    if stats.row_count == 0:
        LOGGER.warning("Generation produced 0 rows")
        return
    total = stats.row_count
    LOGGER.info(
        "=== Summary: %d rows, %d series, %d sites ===",
        total,
        stats.series_count,
        stats.site_count,
    )
    LOGGER.info(
        "Rates: censored=%.2f%% zero=%.2f%% promo=%.2f%%",
        100.0 * stats.censored_rows / total,
        100.0 * stats.zero_rows / total,
        100.0 * stats.promo_rows / total,
    )
    for src, ss in sorted(stats.sources.items()):
        st = ss.get("rows", 0)
        if st == 0:
            continue
        LOGGER.info(
            "  %-45s rows=%6d censored=%.2f%%",
            src,
            st,
            100.0 * ss.get("censored_rows", 0) / st,
        )
