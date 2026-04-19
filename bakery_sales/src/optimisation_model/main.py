from __future__ import annotations

"""Point d'entree executable pour l'optimisation ARIMA par produit."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimisation_model.constants_arima import DEFAULT_FOLD_JOBS, DEFAULT_FOLDS, DEFAULT_TRIALS
from src.optimisation_model.optimize_arima_model import run_optimization
from src.optimisation_model.paths_arima import (
    BEST_MODEL_PKL,
    BEST_PARAMS_JSON,
    BEST_PREDICTIONS_CSV,
    TRAIN_CSV,
    TRIALS_CSV,
    VAL_CSV,
)


def main() -> None:
    """Execute l'optimisation Optuna ARIMA produit par produit."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_optimization(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        best_params_json=BEST_PARAMS_JSON,
        trials_csv=TRIALS_CSV,
        best_model_pkl=BEST_MODEL_PKL,
        best_predictions_csv=BEST_PREDICTIONS_CSV,
        n_trials=DEFAULT_TRIALS,
        n_folds=DEFAULT_FOLDS,
        n_jobs_folds=DEFAULT_FOLD_JOBS,
    )


if __name__ == "__main__":
    main()
