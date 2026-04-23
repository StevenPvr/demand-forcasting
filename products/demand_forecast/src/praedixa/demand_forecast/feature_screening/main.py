from __future__ import annotations

import json
import logging

from praedixa.demand_forecast.feature_screening.pipeline import build_lag_selection_outputs
from praedixa.platform.runtime.paths import FEATURE_SELECTION_DIR
from praedixa.demand_forecast.contracts.targets import DEFAULT_ABSOLUTE_TARGET_COL


DEFAULT_OUTPUT_DIR = FEATURE_SELECTION_DIR
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    outputs = build_lag_selection_outputs(
        output_dir=DEFAULT_OUTPUT_DIR,
        target_col=DEFAULT_ABSOLUTE_TARGET_COL,
        correlation_only=True,
        max_selected_lag_features=50,
    )
    logger.info("Feature selection outputs: %s", json.dumps({name: str(path) for name, path in outputs.items()}))


if __name__ == "__main__":
    main()
