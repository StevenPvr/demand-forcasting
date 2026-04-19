from __future__ import annotations

"""Constantes reutilisables pour l'evaluation ARIMA par produit."""

CSV_ENCODING: str = "utf-8"
DATE_COLUMN: str = "date"
PRODUCT_COLUMN: str = "product"
TARGET_COLUMN: str = "quantity"
DEFAULT_EVALUATION_JOBS: int = 1
DEFAULT_PRODUCTION_COST_RATIO: float = 0.35
FIELD_BASELINE_NAME: str = "blend_lag_1_lag_7_50_50"
