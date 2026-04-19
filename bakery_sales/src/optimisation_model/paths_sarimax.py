from __future__ import annotations

"""Chemins centralises pour les artefacts d'optimisation SARIMAX."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_FEATURE_SELECTION_DIR: Path = DATA_DIR / "feature_selection"
DATA_OPTIMISATION_DIR: Path = DATA_DIR / "optimisation_model"

TRAIN_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_train_selected_2021_2022.csv"
VAL_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_val_selected_2021_2022.csv"
BEST_PARAMS_JSON: Path = DATA_OPTIMISATION_DIR / "sarimax_best_params.json"
TRIALS_CSV: Path = DATA_OPTIMISATION_DIR / "sarimax_optuna_trials.csv"
BEST_MODEL_PKL: Path = DATA_OPTIMISATION_DIR / "sarimax_best_model.pkl"
BEST_PREDICTIONS_CSV: Path = DATA_OPTIMISATION_DIR / "sarimax_best_trial_predictions.csv"
