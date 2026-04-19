from __future__ import annotations

"""Point d'entree executable pour l'evaluation ARIMA par produit sur le test."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.constants_arima import DEFAULT_EVALUATION_JOBS
from src.evaluation.evaluate_arima_model import run_evaluation
from src.evaluation.paths_arima import (
    BEST_PARAMS_JSON,
    DIAGNOSTICS_JSON,
    FINAL_MODEL_PKL,
    METRICS_JSON,
    MODEL_CARD_JSON,
    PREDICTIONS_CSV,
    PREDICTIONS_PLOT_PNG,
    RESIDUALS_ACF_PACF_PNG,
    RESIDUALS_PLOT_PNG,
    RESIDUALS_QQ_PNG,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)


def main() -> None:
    """Execute l'evaluation rolling-origin ARIMA produit par produit."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_evaluation(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        best_params_json=BEST_PARAMS_JSON,
        metrics_json=METRICS_JSON,
        predictions_csv=PREDICTIONS_CSV,
        predictions_plot_png=PREDICTIONS_PLOT_PNG,
        diagnostics_json=DIAGNOSTICS_JSON,
        model_card_json=MODEL_CARD_JSON,
        final_model_pkl=FINAL_MODEL_PKL,
        residuals_plot_png=RESIDUALS_PLOT_PNG,
        residuals_qq_png=RESIDUALS_QQ_PNG,
        residuals_acf_pacf_png=RESIDUALS_ACF_PACF_PNG,
        n_jobs=DEFAULT_EVALUATION_JOBS,
    )


if __name__ == "__main__":
    main()
