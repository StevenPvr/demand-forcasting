from __future__ import annotations

"""Constantes reutilisables pour la preparation du dataset journalier."""

TARGET_ARTICLE: str = "BAGUETTE"
TARGET_COLUMN: str = "target_baguette_t_plus_1"
ORIGIN_DATE_COLUMN: str = "origin_date"
TARGET_DATE_COLUMN: str = "target_date"
ORIGIN_MISSING_FLAG_COLUMN: str = "exog_flag_origin_day_missing"
TARGET_MISSING_FLAG_COLUMN: str = "exog_flag_target_day_missing"
EXOG_SALES_PREFIX: str = "exog_sales_"
EXOG_SALES_SUFFIX: str = "_lag1"
FORECAST_HORIZON_DAYS: int = 1
DAILY_FREQUENCY: str = "D"
CSV_ENCODING: str = "utf-8"
INVALID_ARTICLES: set[str] = {"."}
MORNING_END_MINUTE: int = 11 * 60
LUNCH_END_MINUTE: int = 14 * 60
AFTERNOON_END_MINUTE: int = 18 * 60
