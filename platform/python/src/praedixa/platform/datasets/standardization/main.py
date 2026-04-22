from __future__ import annotations

from dataclasses import dataclass
import json
import logging

from praedixa.platform.datasets.standardization.pipeline import build_global_daily_standardization
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR


DEFAULT_OUTPUT_DIR = GLOBAL_DATASET_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalDatasetMainConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


def build_default_global_dataset_main_config() -> GlobalDatasetMainConfig:
    return GlobalDatasetMainConfig()


def main() -> None:
    """Build the canonical daily dataset artifacts and log their paths."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = build_default_global_dataset_main_config()
    artifacts = build_global_daily_standardization(output_dir=config.output_dir)
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
