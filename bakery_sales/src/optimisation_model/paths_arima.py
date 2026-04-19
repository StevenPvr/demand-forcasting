from __future__ import annotations

"""Chemins centralises pour les artefacts d'optimisation ARIMA par produit."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PREPROCESSING_DIR: Path = DATA_DIR / "data_preprocessing"
DATA_OPTIMISATION_DIR: Path = DATA_DIR / "optimisation_model"

TRAIN_CSV: Path = DATA_PREPROCESSING_DIR / "daily_product_arima_train_2021_2022.csv"
VAL_CSV: Path = DATA_PREPROCESSING_DIR / "daily_product_arima_val_2021_2022.csv"
BEST_PARAMS_JSON: Path = DATA_OPTIMISATION_DIR / "arima_by_product_best_params.json"
TRIALS_CSV: Path = DATA_OPTIMISATION_DIR / "arima_by_product_optuna_trials.csv"
BEST_MODEL_PKL: Path = DATA_OPTIMISATION_DIR / "arima_by_product_best_model.pkl"
BEST_PREDICTIONS_CSV: Path = DATA_OPTIMISATION_DIR / "arima_by_product_best_trial_predictions.csv"
