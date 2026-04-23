from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

from praedixa.platform.runtime.constants import DEFAULT_BRONZE_SCHEMA
from praedixa.platform.runtime.constants import DEFAULT_GOLD_BAKERY_TEST_MONTHS
from praedixa.platform.runtime.constants import DEFAULT_GOLD_HORIZON_DAYS
from praedixa.platform.runtime.constants import DEFAULT_GOLD_SCHEMA
from praedixa.platform.runtime.constants import DEFAULT_SILVER_SCHEMA
from praedixa.platform.runtime.paths import DBT_LOG_DIR
from praedixa.platform.runtime.paths import DBT_TARGET_DIR
from praedixa.platform.runtime.paths import EXTERNAL_OPEN_DIR
from praedixa.platform.runtime.paths import LOCAL_DUCKDB_PATH
from praedixa.platform.runtime.paths import SOURCES_DIR


def _bool_to_env(value: bool) -> str:
    return "true" if value else "false"


def _default_dbt_threads() -> int:
    return os.cpu_count() or 1


@dataclass(frozen=True)
class WarehouseRuntimeConfig:
    """Configuration runtime partagee pour les workflows warehouse locaux."""

    enable_local_warehouse: bool = True
    enable_cloud_warehouse: bool = False
    enable_local_backup: bool = False
    duckdb_local_path: Path = LOCAL_DUCKDB_PATH
    duckdb_target_path: Path | None = None
    bronze_schema: str = DEFAULT_BRONZE_SCHEMA
    silver_schema: str = DEFAULT_SILVER_SCHEMA
    gold_schema: str = DEFAULT_GOLD_SCHEMA
    bronze_data_dir: Path = SOURCES_DIR
    open_exogenous_dir: Path = EXTERNAL_OPEN_DIR
    dbt_threads: int = field(default_factory=_default_dbt_threads)
    gold_horizon_days: str = DEFAULT_GOLD_HORIZON_DAYS
    gold_bakery_test_months: str = DEFAULT_GOLD_BAKERY_TEST_MONTHS

    @property
    def resolved_duckdb_target_path(self) -> Path:
        return self.duckdb_target_path or self.duckdb_local_path

    def to_env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Construit un environnement compatible runners en respectant les overrides explicites."""

        env = dict(os.environ if base_env is None else base_env)
        defaults = {
            "PRAEDIXA_ENABLE_LOCAL_WAREHOUSE": _bool_to_env(self.enable_local_warehouse),
            "PRAEDIXA_ENABLE_CLOUD_WAREHOUSE": _bool_to_env(self.enable_cloud_warehouse),
            "PRAEDIXA_ENABLE_LOCAL_BACKUP": _bool_to_env(self.enable_local_backup),
            "PRAEDIXA_DUCKDB_LOCAL_PATH": str(self.duckdb_local_path),
            "PRAEDIXA_DUCKDB_TARGET_PATH": str(self.resolved_duckdb_target_path),
            "PRAEDIXA_DUCKDB_BRONZE_SCHEMA": self.bronze_schema,
            "PRAEDIXA_DUCKDB_SILVER_SCHEMA": self.silver_schema,
            "PRAEDIXA_DUCKDB_GOLD_SCHEMA": self.gold_schema,
            "PRAEDIXA_BRONZE_DATA_DIR": str(self.bronze_data_dir),
            "PRAEDIXA_OPEN_EXOGENOUS_DIR": str(self.open_exogenous_dir),
            "PRAEDIXA_DBT_THREADS": str(self.dbt_threads),
            "PRAEDIXA_GOLD_HORIZON_DAYS": self.gold_horizon_days,
            "PRAEDIXA_GOLD_BAKERY_TEST_MONTHS": self.gold_bakery_test_months,
            "DBT_TARGET_PATH": str(DBT_TARGET_DIR),
            "DBT_LOG_PATH": str(DBT_LOG_DIR),
        }
        for key, value in defaults.items():
            env.setdefault(key, value)
        return env
