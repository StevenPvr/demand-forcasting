from __future__ import annotations

"""Constantes reutilisables pour le preprocessing ARIMA par produit."""

CSV_ENCODING: str = "utf-8"
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
DATE_COLUMN: str = "date"
PRODUCT_COLUMN: str = "product"
TARGET_COLUMN: str = "quantity"
MISSING_FLAG_COLUMN: str = "is_missing_day"
DAILY_FREQUENCY: str = "D"
