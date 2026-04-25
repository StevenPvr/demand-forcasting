from __future__ import annotations

"""Point d'entree canonique pour executer tout le pipeline avec ARIMA actif."""

import logging
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning.main import main as run_data_cleaning_main
from src.data_preprocessing.main import main as run_data_preprocessing_main
from src.evaluation.main import main as run_evaluation_main
from src.optimisation_model.main import main as run_optimisation_main

LOGGER: logging.Logger = logging.getLogger(__name__)


def _run_stage(stage_name: str, stage_runner: Callable[[], None]) -> None:
    """Execute une etape du launch avec log de progression."""

    LOGGER.info("Starting launch stage: %s", stage_name)
    stage_runner()


def main() -> None:
    """Execute les etapes canoniques du pipeline de bout en bout."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    _run_stage("data_cleaning", run_data_cleaning_main)
    _run_stage("data_preprocessing", run_data_preprocessing_main)
    _run_stage("optimisation_model", run_optimisation_main)
    _run_stage("evaluation", run_evaluation_main)
    LOGGER.info("Launch pipeline completed successfully")


if __name__ == "__main__":
    main()
