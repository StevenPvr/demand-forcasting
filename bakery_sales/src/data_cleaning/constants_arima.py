from __future__ import annotations

"""Constantes reutilisables pour le dataset ARIMA par produit."""

DATE_COLUMN: str = "date"
PRODUCT_COLUMN: str = "product"
TARGET_COLUMN: str = "quantity"
MISSING_FLAG_COLUMN: str = "is_missing_day"
DAILY_FREQUENCY: str = "D"
CSV_ENCODING: str = "utf-8"
MAX_ZERO_DAY_RATIO: float = 1.0 / 3.0
