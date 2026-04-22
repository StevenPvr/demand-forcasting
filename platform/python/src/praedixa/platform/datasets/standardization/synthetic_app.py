from __future__ import annotations

from dataclasses import dataclass
import logging

from praedixa.platform.datasets.standardization.synthetic import (
    SyntheticColdStartConfig,
    write_synthetic_cold_start_dataset,
)
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_OUTPUT_PATH = SOURCES_DIR / "commercial_datasets" / "raw" / "synthetic_v1.parquet"


@dataclass(frozen=True)
class SyntheticAppConfig:
    output_path: str = str(DEFAULT_OUTPUT_PATH)
    start_date: str = "2024-01-01"
    days: int = 180
    num_locations: int = 12
    products_per_location: int = 24
    random_seed: int = 7


def build_default_synthetic_app_config() -> SyntheticAppConfig:
    return SyntheticAppConfig()


def main() -> None:
    """Generate the synthetic dataset parquet and log the output path."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = build_default_synthetic_app_config()
    output_path = write_synthetic_cold_start_dataset(
        output_path=config.output_path,
        config=SyntheticColdStartConfig(
            start_date=config.start_date,
            days=config.days,
            num_locations=config.num_locations,
            products_per_location=config.products_per_location,
            random_seed=config.random_seed,
        ),
    )
    logging.getLogger(__name__).info("Synthetic cold-start dataset written to %s", output_path)


if __name__ == "__main__":
    main()
