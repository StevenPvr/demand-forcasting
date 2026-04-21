from __future__ import annotations

import argparse
import json
import logging

from praedixa.demand_forecast.training_bundle.bundle_builder import (
    DEFAULT_OUTPUT_DIR,
    build_training_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TFT training bundle directly from gold or explicit parquet inputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke-series-limit", type=int, default=None)
    parser.add_argument("--smoke-dataset-source", type=str, default=None)
    parser.add_argument("--smoke-min-train-rows", type=int, default=56)
    parser.add_argument("--smoke-min-tuning-rows", type=int, default=28)
    parser.add_argument("--smoke-min-valid-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    outputs = build_training_bundle(
        output_dir=args.output_dir,
        smoke_series_limit=args.smoke_series_limit,
        smoke_dataset_source=args.smoke_dataset_source,
        smoke_min_train_rows=args.smoke_min_train_rows,
        smoke_min_tuning_rows=args.smoke_min_tuning_rows,
        smoke_min_valid_rows=args.smoke_min_valid_rows,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
