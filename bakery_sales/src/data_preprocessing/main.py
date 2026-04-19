from __future__ import annotations

"""Point d'entree executable pour les splits ARIMA journaliers par produit."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.paths_arima import (
    DAILY_INPUT_CSV,
    DAILY_TEST_CSV,
    DAILY_TRAIN_CSV,
    DAILY_VAL_CSV,
    PREPROCESSING_BUNDLE_JOBLIB,
    STATIONARITY_REPORT_JSON,
)
from src.data_preprocessing.split_product_arima_dataset import split_product_arima_dataset


def main() -> None:
    """Execute la creation des splits temporels par produit."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    split_product_arima_dataset(
        input_csv=DAILY_INPUT_CSV,
        train_csv=DAILY_TRAIN_CSV,
        val_csv=DAILY_VAL_CSV,
        test_csv=DAILY_TEST_CSV,
        stationarity_report_json=STATIONARITY_REPORT_JSON,
        preprocessing_bundle_path=PREPROCESSING_BUNDLE_JOBLIB,
    )


if __name__ == "__main__":
    main()
