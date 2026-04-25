from __future__ import annotations

"""Chemins centralises pour les artefacts d'evaluation ARIMA par produit."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PREPROCESSING_DIR: Path = DATA_DIR / "data_preprocessing"
DATA_OPTIMISATION_DIR: Path = DATA_DIR / "optimisation_model"
DATA_EVALUATION_DIR: Path = DATA_DIR / "evaluation"

TRAIN_CSV: Path = DATA_PREPROCESSING_DIR / "daily_product_arima_train_2021_2022.csv"
VAL_CSV: Path = DATA_PREPROCESSING_DIR / "daily_product_arima_val_2021_2022.csv"
TEST_CSV: Path = DATA_PREPROCESSING_DIR / "daily_product_arima_test_2021_2022.csv"
BEST_PARAMS_JSON: Path = DATA_OPTIMISATION_DIR / "arima_by_product_best_params.json"
METRICS_JSON: Path = DATA_EVALUATION_DIR / "arima_by_product_test_metrics.json"
PREDICTIONS_CSV: Path = DATA_EVALUATION_DIR / "arima_by_product_test_predictions.csv"
PREDICTIONS_PLOT_PNG: Path = DATA_EVALUATION_DIR / "arima_by_product_test_predictions.png"
DIAGNOSTICS_JSON: Path = DATA_EVALUATION_DIR / "arima_by_product_residual_diagnostics.json"
MODEL_CARD_JSON: Path = DATA_EVALUATION_DIR / "arima_by_product_model_card.json"
STATISTICAL_BASELINES_JSON: Path = DATA_EVALUATION_DIR / "arima_by_product_statistical_baselines.json"
ECONOMIC_GAIN_JSON: Path = DATA_EVALUATION_DIR / "arima_by_product_economic_gain_vs_baselines.json"
FINAL_MODEL_PKL: Path = DATA_EVALUATION_DIR / "arima_by_product_final_model.pkl"
RESIDUALS_PLOT_PNG: Path = DATA_EVALUATION_DIR / "arima_by_product_residuals.png"
RESIDUALS_QQ_PNG: Path = DATA_EVALUATION_DIR / "arima_by_product_residuals_qq.png"
RESIDUALS_ACF_PACF_PNG: Path = DATA_EVALUATION_DIR / "arima_by_product_residuals_acf_pacf.png"
