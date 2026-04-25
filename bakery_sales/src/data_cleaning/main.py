from __future__ import annotations

"""Point d'entree executable pour la preparation journaliere ARIMA par produit."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning.paths_arima import (
    DAILY_PRODUCT_ARIMA_DATASET_CSV,
    PRODUCT_ARIMA_SUMMARY_JSON,
    RAW_SALES_CSV,
)
from src.data_cleaning.prepare_product_arima_dataset import prepare_daily_product_arima_dataset


def main() -> None:
    """Execute la preparation du dataset journalier ARIMA par produit."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    prepare_daily_product_arima_dataset(
        input_csv=RAW_SALES_CSV,
        output_csv=DAILY_PRODUCT_ARIMA_DATASET_CSV,
        output_json=PRODUCT_ARIMA_SUMMARY_JSON,
    )


if __name__ == "__main__":
    main()
