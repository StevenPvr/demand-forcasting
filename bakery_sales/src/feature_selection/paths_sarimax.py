from __future__ import annotations

"""Chemins centralises pour les artefacts de selection de variables SARIMAX."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PREPROCESSING_DIR: Path = DATA_DIR / "data_preprocessing"
DATA_FEATURE_SELECTION_DIR: Path = DATA_DIR / "feature_selection"

TRAIN_INPUT_CSV: Path = DATA_PREPROCESSING_DIR / "daily_train_2021_2022.csv"
VAL_INPUT_CSV: Path = DATA_PREPROCESSING_DIR / "daily_val_2021_2022.csv"
TEST_INPUT_CSV: Path = DATA_PREPROCESSING_DIR / "daily_test_2021_2022.csv"

TRAIN_OUTPUT_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_train_selected_2021_2022.csv"
VAL_OUTPUT_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_val_selected_2021_2022.csv"
TEST_OUTPUT_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_test_selected_2021_2022.csv"
SUMMARY_JSON: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_feature_selection_summary.json"
IMPORTANCES_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_feature_importances.csv"
TUNING_TRIALS_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_selector_optuna_trials.csv"
