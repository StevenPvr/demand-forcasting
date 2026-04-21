from __future__ import annotations


DEFAULT_BRONZE_SCHEMA: str = "bronze"
DEFAULT_SILVER_SCHEMA: str = "silver"
DEFAULT_GOLD_SCHEMA: str = "gold"

DEFAULT_SILVER_DBT_SELECT: str = "+tag:silver"
DEFAULT_GOLD_DBT_SELECT: str = "tag:gold"

DEFAULT_GOLD_HORIZON_DAYS: str = "1"
DEFAULT_GOLD_BAKERY_TEST_MONTHS: str = "3"
