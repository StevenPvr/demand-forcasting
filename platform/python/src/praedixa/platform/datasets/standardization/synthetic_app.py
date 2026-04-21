from __future__ import annotations

import argparse
import logging

from praedixa.platform.datasets.standardization.synthetic import (
    SyntheticColdStartConfig,
    write_synthetic_cold_start_dataset,
)
from praedixa.platform.runtime.paths import SOURCES_DIR


DEFAULT_OUTPUT_PATH = SOURCES_DIR / "commercial_datasets" / "raw" / "synthetic_v1.parquet"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for synthetic cold-start generation."""

    parser = argparse.ArgumentParser(description="Generate a synthetic cold-start daily dataset for Praedixa.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--num-locations", type=int, default=12)
    parser.add_argument("--products-per-location", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """Generate the synthetic dataset parquet and log the output path."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    output_path = write_synthetic_cold_start_dataset(
        output_path=args.output_path,
        config=SyntheticColdStartConfig(
            start_date=args.start_date,
            days=args.days,
            num_locations=args.num_locations,
            products_per_location=args.products_per_location,
            random_seed=args.random_seed,
        ),
    )
    logging.getLogger(__name__).info("Synthetic cold-start dataset written to %s", output_path)


if __name__ == "__main__":
    main()
