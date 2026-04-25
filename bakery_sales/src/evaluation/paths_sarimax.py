from __future__ import annotations

"""Chemins centralises pour les artefacts d'evaluation SARIMAX."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_FEATURE_SELECTION_DIR: Path = DATA_DIR / "feature_selection"
DATA_OPTIMISATION_DIR: Path = DATA_DIR / "optimisation_model"
DATA_EVALUATION_DIR: Path = DATA_DIR / "evaluation"

TRAIN_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_train_selected_2021_2022.csv"
VAL_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_val_selected_2021_2022.csv"
TEST_CSV: Path = DATA_FEATURE_SELECTION_DIR / "sarimax_daily_test_selected_2021_2022.csv"
BEST_PARAMS_JSON: Path = DATA_OPTIMISATION_DIR / "sarimax_best_params.json"
METRICS_JSON: Path = DATA_EVALUATION_DIR / "sarimax_test_metrics.json"
PREDICTIONS_CSV: Path = DATA_EVALUATION_DIR / "sarimax_test_predictions.csv"
PREDICTIONS_PLOT_PNG: Path = DATA_EVALUATION_DIR / "sarimax_test_predictions.png"
DIAGNOSTICS_JSON: Path = DATA_EVALUATION_DIR / "sarimax_residual_diagnostics.json"
MODEL_CARD_JSON: Path = DATA_EVALUATION_DIR / "sarimax_model_card.json"
STATISTICAL_BASELINES_JSON: Path = DATA_EVALUATION_DIR / "test_statistical_baselines.json"
FINAL_MODEL_PKL: Path = DATA_EVALUATION_DIR / "sarimax_final_model.pkl"
RESIDUALS_PLOT_PNG: Path = DATA_EVALUATION_DIR / "sarimax_residuals.png"
RESIDUALS_QQ_PNG: Path = DATA_EVALUATION_DIR / "sarimax_residuals_qq.png"
RESIDUALS_ACF_PACF_PNG: Path = DATA_EVALUATION_DIR / "sarimax_residuals_acf_pacf.png"
