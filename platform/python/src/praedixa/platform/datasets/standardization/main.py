from __future__ import annotations

import json
import logging

from praedixa.platform.datasets.standardization.pipeline import build_global_daily_standardization

logger = logging.getLogger(__name__)


def main() -> None:
    """Validate the canonical dataset assembly path without persisting intermediate parquets."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    artifacts = build_global_daily_standardization()
    logger.info(
        "Global daily dataset validation summary: %s",
        json.dumps(
            {
                "sources": artifacts.source_summaries,
                "data_quality": artifacts.data_quality,
            },
            ensure_ascii=True,
        ),
    )


if __name__ == "__main__":
    main()
