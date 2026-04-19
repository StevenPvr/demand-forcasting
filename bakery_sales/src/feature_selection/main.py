from __future__ import annotations

"""Point d'entree executable pour la selection de features du pipeline SARIMAX."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.paths_sarimax import (
    IMPORTANCES_CSV,
    SUMMARY_JSON,
    TEST_INPUT_CSV,
    TEST_OUTPUT_CSV,
    TUNING_TRIALS_CSV,
    TRAIN_INPUT_CSV,
    TRAIN_OUTPUT_CSV,
    VAL_INPUT_CSV,
    VAL_OUTPUT_CSV,
)
from src.feature_selection.constants import DEFAULT_OPTUNA_TRIALS, DEFAULT_SELECTOR_JOBS
from src.feature_selection.select_features_sarimax import run_feature_selection


def main() -> None:
    """Execute la selection de features entre preprocessing et optimisation."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_feature_selection(
        train_input_csv=TRAIN_INPUT_CSV,
        val_input_csv=VAL_INPUT_CSV,
        test_input_csv=TEST_INPUT_CSV,
        train_output_csv=TRAIN_OUTPUT_CSV,
        val_output_csv=VAL_OUTPUT_CSV,
        test_output_csv=TEST_OUTPUT_CSV,
        summary_json=SUMMARY_JSON,
        importances_csv=IMPORTANCES_CSV,
        tuning_trials_csv=TUNING_TRIALS_CSV,
        n_trials=DEFAULT_OPTUNA_TRIALS,
        n_jobs=DEFAULT_SELECTOR_JOBS,
    )


if __name__ == "__main__":
    main()
