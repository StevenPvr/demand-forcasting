from __future__ import annotations

"""Chemins centralises pour les artefacts ARIMA par produit."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_CLEANING_DIR: Path = DATA_DIR / "data_cleaning"

RAW_SALES_CSV: Path = DATA_DIR / "Bakery sales.csv"
DAILY_PRODUCT_ARIMA_DATASET_CSV: Path = DATA_CLEANING_DIR / "daily_product_arima_2021_2022.csv"
PRODUCT_ARIMA_SUMMARY_JSON: Path = DATA_CLEANING_DIR / "daily_product_arima_2021_2022_summary.json"
