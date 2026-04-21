from __future__ import annotations

import logging

from praedixa.demand_forecast.evaluation.pipeline import (
    EvaluationBuildRequest,
    build_evaluation_outputs,
)
from praedixa.demand_forecast.backends.tft.backend import TFTBackendNotReadyError


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        build_evaluation_outputs(EvaluationBuildRequest())
    except TFTBackendNotReadyError:
        logging.getLogger(__name__).error(
            "Le backend TFT unique n'est pas encore branche : l'etape evaluation reste un placeholder structurel."
        )
        raise


if __name__ == "__main__":
    main()
