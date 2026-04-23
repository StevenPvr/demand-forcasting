from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.runtime.warehouse import WarehouseRuntimeConfig  # noqa: E402


class RuntimeTests(unittest.TestCase):
    def test_default_dbt_threads_uses_all_detected_cores(self) -> None:
        with mock.patch("praedixa.platform.runtime.warehouse.os.cpu_count", return_value=12):
            config = WarehouseRuntimeConfig()

        self.assertEqual(config.dbt_threads, 12)

    def test_to_env_exposes_local_defaults(self) -> None:
        config = WarehouseRuntimeConfig()

        env = config.to_env(base_env={})

        self.assertEqual(env["PRAEDIXA_ENABLE_LOCAL_WAREHOUSE"], "true")
        self.assertEqual(env["PRAEDIXA_ENABLE_CLOUD_WAREHOUSE"], "false")
        self.assertEqual(env["PRAEDIXA_ENABLE_LOCAL_BACKUP"], "false")
        self.assertEqual(env["PRAEDIXA_DUCKDB_TARGET_PATH"], env["PRAEDIXA_DUCKDB_LOCAL_PATH"])
        self.assertEqual(env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"], "bronze")
        self.assertEqual(env["PRAEDIXA_DUCKDB_SILVER_SCHEMA"], "silver")
        self.assertEqual(env["PRAEDIXA_DUCKDB_GOLD_SCHEMA"], "gold")
        self.assertEqual(env["PRAEDIXA_GOLD_HORIZON_DAYS"], "1")
        self.assertEqual(env["PRAEDIXA_GOLD_BAKERY_TEST_MONTHS"], "3")

    def test_to_env_preserves_explicit_base_env_overrides(self) -> None:
        config = WarehouseRuntimeConfig()

        env = config.to_env(
            base_env={
                "PRAEDIXA_ENABLE_CLOUD_WAREHOUSE": "true",
                "PRAEDIXA_DUCKDB_LOCAL_PATH": "custom.duckdb",
                "PRAEDIXA_DUCKDB_TARGET_PATH": "custom_target.duckdb",
            }
        )

        self.assertEqual(env["PRAEDIXA_ENABLE_CLOUD_WAREHOUSE"], "true")
        self.assertEqual(env["PRAEDIXA_DUCKDB_LOCAL_PATH"], "custom.duckdb")
        self.assertEqual(env["PRAEDIXA_DUCKDB_TARGET_PATH"], "custom_target.duckdb")


if __name__ == "__main__":
    unittest.main()
