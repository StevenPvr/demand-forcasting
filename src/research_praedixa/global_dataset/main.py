from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from research_praedixa.global_dataset.pipeline import build_global_daily_standardization


DEFAULT_OUTPUT_DIR = Path("data/global_dataset")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the global dataset pipeline."""

    parser = argparse.ArgumentParser(description="Build canonical daily demand datasets from local bronze files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    """Build the canonical daily dataset artifacts and log their paths."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    artifacts = build_global_daily_standardization(output_dir=args.output_dir)
    logger.info(
        "Global daily dataset built: %s",
        json.dumps(
            {
                "freshretail_daily": str(artifacts.freshretail_daily) if artifacts.freshretail_daily else None,
                "commercial_external_daily": (
                    str(artifacts.commercial_external_daily) if artifacts.commercial_external_daily else None
                ),
                "bakery_daily": str(artifacts.bakery_daily) if artifacts.bakery_daily else None,
                "global_gold": str(artifacts.global_gold),
                "manifest_path": str(artifacts.manifest_path),
            },
            ensure_ascii=True,
        ),
    )


if __name__ == "__main__":
    main()
