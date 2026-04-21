from __future__ import annotations

import json
import logging

from praedixa.demand_forecast.training_bundle.bundle_builder import (
    DEFAULT_OUTPUT_DIR,
    build_training_bundle,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    outputs = build_training_bundle(
        output_dir=DEFAULT_OUTPUT_DIR,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
