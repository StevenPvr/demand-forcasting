from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from praedixa.platform.runtime.constants import RANDOM_SEED
from praedixa.platform.runtime.paths import DATASETS_DIR, SOURCES_DIR


DEFAULT_SYNTHETIC_START_DATE: date = date(2024, 7, 21)
DEFAULT_SYNTHETIC_END_DATE: date = date(2025, 7, 20)
DEFAULT_SYNTHETIC_RAW_DIR: Path = SOURCES_DIR / "commercial_datasets" / "raw"
DEFAULT_BAKERY_PRODUCT_NAMES_CSV: Path = SOURCES_DIR / "bakery_sales" / "Bakery sales.csv"

_PREFIX: str = "synthetic_foodservice"
_RAW: Path = DEFAULT_SYNTHETIC_RAW_DIR


def _raw(name: str) -> Path:
    return _RAW / f"{_PREFIX}_{name}"


DEFAULT_OPEN_EXOGENOUS_LOCATION_METADATA_CSV: Path = (
    DATASETS_DIR / "open_exogenous" / "location_metadata.csv"
)


@dataclass(frozen=True)
class SyntheticFoodserviceConfig:
    """Configuration for the one-shot synthetic foodservice source generator."""

    start_date: date
    end_date: date
    site_count: int
    city_count: int
    seed: int
    # --- core daily outputs ---
    daily_csv_path: Path
    oracle_csv_path: Path
    manifest_path: Path
    location_metadata_csv_path: Path
    open_exogenous_location_metadata_csv_path: Path
    # --- granularity outputs ---
    tickets_csv_path: Path
    lines_csv_path: Path
    agg_15min_csv_path: Path
    agg_hourly_csv_path: Path
    agg_halfday_csv_path: Path
    agg_weekly_csv_path: Path
    agg_monthly_csv_path: Path
    # --- operational outputs ---
    stock_snapshots_csv_path: Path
    inventory_movements_csv_path: Path
    staff_schedules_csv_path: Path
    # --- config flags ---
    site_batch_size: int = 8
    write_oracle_debug: bool = True
    write_granularity_files: bool = True
    write_operational_files: bool = True
    apply_corruption: bool = False
    inject_shocks: bool = True
    overwrite_existing: bool = True
    bakery_product_names_csv_path: Path | None = DEFAULT_BAKERY_PRODUCT_NAMES_CSV
    source_run_id: str = "synthetic_foodservice_v1"
    schema_version: str = "2.0.0"
    target_contract: str = "observed_sales"
    real_data_policy: str = "none"
    panel_mode: str = "complete_product_day"
    validation_profile: str = "strict"
    world_count: int = 1


def build_default_synthetic_foodservice_config() -> SyntheticFoodserviceConfig:
    """Return the production-scale default requested for Praedixa."""

    return SyntheticFoodserviceConfig(
        start_date=DEFAULT_SYNTHETIC_START_DATE,
        end_date=DEFAULT_SYNTHETIC_END_DATE,
        site_count=180,
        city_count=180,
        seed=RANDOM_SEED,
        daily_csv_path=_raw("daily.csv"),
        oracle_csv_path=_raw("oracle_debug.csv"),
        manifest_path=_raw("manifest.json"),
        location_metadata_csv_path=_raw("location_metadata.csv"),
        open_exogenous_location_metadata_csv_path=DEFAULT_OPEN_EXOGENOUS_LOCATION_METADATA_CSV,
        tickets_csv_path=_raw("tickets.csv"),
        lines_csv_path=_raw("lines.csv"),
        agg_15min_csv_path=_raw("15min.csv"),
        agg_hourly_csv_path=_raw("hourly.csv"),
        agg_halfday_csv_path=_raw("halfday.csv"),
        agg_weekly_csv_path=_raw("weekly.csv"),
        agg_monthly_csv_path=_raw("monthly.csv"),
        stock_snapshots_csv_path=_raw("stock_snapshots.csv"),
        inventory_movements_csv_path=_raw("inventory_movements.csv"),
        staff_schedules_csv_path=_raw("staff_schedules.csv"),
    )


def build_smoke_synthetic_foodservice_config(
    output_dir: str | Path,
) -> SyntheticFoodserviceConfig:
    """Return a small config for tests and local smoke validation."""

    d = Path(output_dir)
    return SyntheticFoodserviceConfig(
        start_date=date(2024, 7, 21),
        end_date=date(2024, 10, 15),
        site_count=50,
        city_count=20,
        seed=RANDOM_SEED,
        site_batch_size=5,
        daily_csv_path=d / f"{_PREFIX}_daily.csv",
        oracle_csv_path=d / f"{_PREFIX}_oracle_debug.csv",
        manifest_path=d / f"{_PREFIX}_manifest.json",
        location_metadata_csv_path=d / f"{_PREFIX}_location_metadata.csv",
        open_exogenous_location_metadata_csv_path=d / "location_metadata.csv",
        tickets_csv_path=d / f"{_PREFIX}_tickets.csv",
        lines_csv_path=d / f"{_PREFIX}_lines.csv",
        agg_15min_csv_path=d / f"{_PREFIX}_15min.csv",
        agg_hourly_csv_path=d / f"{_PREFIX}_hourly.csv",
        agg_halfday_csv_path=d / f"{_PREFIX}_halfday.csv",
        agg_weekly_csv_path=d / f"{_PREFIX}_weekly.csv",
        agg_monthly_csv_path=d / f"{_PREFIX}_monthly.csv",
        stock_snapshots_csv_path=d / f"{_PREFIX}_stock_snapshots.csv",
        inventory_movements_csv_path=d / f"{_PREFIX}_inventory_movements.csv",
        staff_schedules_csv_path=d / f"{_PREFIX}_staff_schedules.csv",
        write_granularity_files=True,
        write_operational_files=True,
        apply_corruption=False,
        inject_shocks=True,
    )
