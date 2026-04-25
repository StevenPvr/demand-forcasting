from __future__ import annotations

"""Chemins centralises pour les artefacts de preparation journaliere."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_CLEANING_DIR: Path = DATA_DIR / "data_cleaning"

RAW_SALES_CSV: Path = DATA_DIR / "Bakery sales.csv"
DAILY_BAGUETTE_DATASET_CSV: Path = DATA_CLEANING_DIR / "daily_baguette_2021_2022.csv"
SALES_EXOG_MAPPING_JSON: Path = DATA_CLEANING_DIR / "sales_exog_mapping_2021_2022.json"
