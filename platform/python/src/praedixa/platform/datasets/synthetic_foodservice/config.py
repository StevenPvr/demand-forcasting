from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from praedixa.platform.runtime.constants import RANDOM_SEED
from praedixa.platform.runtime.paths import DATASETS_DIR, SOURCES_DIR


DEFAULT_SYNTHETIC_START_DATE: date = date(2024, 7, 21)
DEFAULT_SYNTHETIC_END_DATE: date = date(2025, 7, 20)
DEFAULT_SYNTHETIC_RAW_DIR: Path = SOURCES_DIR / "commercial_datasets" / "raw"
DEFAULT_SYNTHETIC_DAILY_CSV: Path = (
    DEFAULT_SYNTHETIC_RAW_DIR / "synthetic_foodservice_daily.csv"
)
DEFAULT_SYNTHETIC_ORACLE_CSV: Path = (
    DEFAULT_SYNTHETIC_RAW_DIR / "synthetic_foodservice_oracle_debug.csv"
)
DEFAULT_SYNTHETIC_MANIFEST: Path = (
    DEFAULT_SYNTHETIC_RAW_DIR / "synthetic_foodservice_manifest.json"
)
DEFAULT_SYNTHETIC_LOCATION_METADATA_CSV: Path = (
    DEFAULT_SYNTHETIC_RAW_DIR / "synthetic_foodservice_location_metadata.csv"
)
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
    daily_csv_path: Path
    oracle_csv_path: Path
    manifest_path: Path
    location_metadata_csv_path: Path
    open_exogenous_location_metadata_csv_path: Path
    site_batch_size: int = 8
    write_oracle_debug: bool = True
    overwrite_existing: bool = True
    source_run_id: str = "synthetic_foodservice_v1"


def build_default_synthetic_foodservice_config() -> SyntheticFoodserviceConfig:
    """Return the production-scale default requested for Praedixa."""

    return SyntheticFoodserviceConfig(
        start_date=DEFAULT_SYNTHETIC_START_DATE,
        end_date=DEFAULT_SYNTHETIC_END_DATE,
        site_count=180,
        city_count=180,
        seed=RANDOM_SEED,
        daily_csv_path=DEFAULT_SYNTHETIC_DAILY_CSV,
        oracle_csv_path=DEFAULT_SYNTHETIC_ORACLE_CSV,
        manifest_path=DEFAULT_SYNTHETIC_MANIFEST,
        location_metadata_csv_path=DEFAULT_SYNTHETIC_LOCATION_METADATA_CSV,
        open_exogenous_location_metadata_csv_path=DEFAULT_OPEN_EXOGENOUS_LOCATION_METADATA_CSV,
    )


def build_smoke_synthetic_foodservice_config(
    output_dir: str | Path,
) -> SyntheticFoodserviceConfig:
    """Return a small config for tests and local smoke validation."""

    target_dir = Path(output_dir)
    config = build_default_synthetic_foodservice_config()
    return replace(
        config,
        start_date=date(2024, 7, 21),
        end_date=date(2024, 10, 15),
        site_count=50,
        city_count=20,
        site_batch_size=5,
        daily_csv_path=target_dir / "synthetic_foodservice_daily.csv",
        oracle_csv_path=target_dir / "synthetic_foodservice_oracle_debug.csv",
        manifest_path=target_dir / "synthetic_foodservice_manifest.json",
        location_metadata_csv_path=target_dir
        / "synthetic_foodservice_location_metadata.csv",
        open_exogenous_location_metadata_csv_path=target_dir / "location_metadata.csv",
    )
