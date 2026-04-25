from __future__ import annotations

"""Chemins centralises pour les artefacts de preprocessing journalier."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PREPROCESSING_DIR: Path = DATA_DIR / "data_preprocessing"

DAILY_INPUT_CSV: Path = DATA_DIR / "data_cleaning" / "daily_baguette_2021_2022.csv"
DAILY_TRAIN_CSV: Path = DATA_PREPROCESSING_DIR / "daily_train_2021_2022.csv"
DAILY_VAL_CSV: Path = DATA_PREPROCESSING_DIR / "daily_val_2021_2022.csv"
DAILY_TEST_CSV: Path = DATA_PREPROCESSING_DIR / "daily_test_2021_2022.csv"
STATIONARITY_REPORT_JSON: Path = DATA_PREPROCESSING_DIR / "baguette_stationarity_report_2021_2022.json"
PREPROCESSING_BUNDLE_JOBLIB: Path = DATA_PREPROCESSING_DIR / "preprocessing_bundle_2021_2022.joblib"
SCHOOL_HOLIDAYS_CACHE_CSV: Path = DATA_PREPROCESSING_DIR / "french_school_holidays.csv"
PUBLIC_HOLIDAYS_CACHE_CSV: Path = DATA_PREPROCESSING_DIR / "french_public_holidays_metropole.csv"
WEATHER_CACHE_CSV: Path = DATA_PREPROCESSING_DIR / "weather_guerande_daily_2021_2022.csv"
