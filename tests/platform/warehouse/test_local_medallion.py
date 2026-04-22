from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.warehouse.local_gold import LocalGoldRunConfig  # noqa: E402
from praedixa.platform.warehouse.local_medallion import (  # noqa: E402
    LocalMedallionRunConfig,
    run_local_medallion,
)
from praedixa.platform.warehouse.local_silver import LocalSilverRunConfig  # noqa: E402


class RunLocalMedallionTests(unittest.TestCase):
    def test_run_local_medallion_runs_silver_then_gold(self) -> None:
        config = LocalMedallionRunConfig(
            run_silver=True,
            run_gold=True,
            silver=LocalSilverRunConfig(
                data_dir=Path("var/sources"),
                dbt_select="tag:silver",
                run_bronze_load=True,
                run_dbt_tests=True,
            ),
            gold=LocalGoldRunConfig(
                dbt_select="tag:gold",
                refresh_open_exogenous=True,
                run_dbt_tests=True,
            ),
        )

        with (
            mock.patch(
                "praedixa.platform.warehouse.local_medallion.run_local_silver",
                return_value={"dbt_run": True},
            ) as silver_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_medallion.run_local_gold",
                return_value={"dbt_run": True},
            ) as gold_mock,
        ):
            result = run_local_medallion(config)

        silver_mock.assert_called_once_with(config.silver)
        gold_mock.assert_called_once_with(config.gold)
        self.assertEqual(result["silver"], {"dbt_run": True})
        self.assertEqual(result["gold"], {"dbt_run": True})

    def test_run_local_medallion_rejects_empty_execution_plan(self) -> None:
        config = LocalMedallionRunConfig(
            run_silver=False,
            run_gold=False,
            silver=LocalSilverRunConfig(
                data_dir=Path("var/sources"),
                dbt_select="tag:silver",
                run_bronze_load=True,
                run_dbt_tests=True,
            ),
            gold=LocalGoldRunConfig(
                dbt_select="tag:gold",
                refresh_open_exogenous=True,
                run_dbt_tests=True,
            ),
        )

        with self.assertRaisesRegex(ValueError, "At least one medallion step must be enabled."):
            run_local_medallion(config)


if __name__ == "__main__":
    unittest.main()
