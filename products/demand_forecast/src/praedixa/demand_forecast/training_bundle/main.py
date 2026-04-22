from __future__ import annotations

from dataclasses import dataclass
import json
import logging

from praedixa.demand_forecast.training_bundle.bundle_builder import (
    DEFAULT_OUTPUT_DIR,
    build_training_bundle,
)


@dataclass(frozen=True)
class TrainingBundleMainConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    smoke_series_limit: int | None = None
    smoke_dataset_source: str | None = None
    smoke_min_train_rows: int = 56
    smoke_min_tuning_rows: int = 28
    smoke_min_valid_rows: int = 0


def build_default_training_bundle_main_config() -> TrainingBundleMainConfig:
    return TrainingBundleMainConfig()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_default_training_bundle_main_config()
    outputs = build_training_bundle(
        output_dir=config.output_dir,
        smoke_series_limit=config.smoke_series_limit,
        smoke_dataset_source=config.smoke_dataset_source,
        smoke_min_train_rows=config.smoke_min_train_rows,
        smoke_min_tuning_rows=config.smoke_min_tuning_rows,
        smoke_min_valid_rows=config.smoke_min_valid_rows,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
