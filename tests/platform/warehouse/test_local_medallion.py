from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists()
)
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
from praedixa.platform.warehouse.pipeline import (  # noqa: E402
    build_medallion_steps,
    validate_medallion_plan,
)


def _config(*, run_silver: bool = True, run_gold: bool = True) -> LocalMedallionRunConfig:
    return LocalMedallionRunConfig(
        run_silver=run_silver,
        run_gold=run_gold,
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


class RunLocalMedallionTests(unittest.TestCase):
    def test_run_local_medallion_delegates_to_explicit_pipeline(self) -> None:
        config = _config()

        with mock.patch(
            "praedixa.platform.warehouse.local_medallion.run_local_medallion_pipeline",
            return_value={"run_silver": {"dbt_run": True}, "run_gold": {"dbt_run": True}},
        ) as pipeline_mock:
            result = run_local_medallion(config)

        pipeline_mock.assert_called_once()
        pipeline_config = pipeline_mock.call_args.args[0]
        self.assertTrue(pipeline_config.load_core_bronze)
        self.assertTrue(pipeline_config.refresh_open_exogenous)
        self.assertEqual(result["silver"], {"dbt_run": True})
        self.assertEqual(result["gold"], {"dbt_run": True})

    def test_medallion_plan_has_explicit_order(self) -> None:
        steps = build_medallion_steps(_config().to_pipeline_config())

        self.assertEqual(
            [step.name for step in steps],
            ["load_core_bronze", "run_silver", "refresh_open_exogenous", "run_gold"],
        )

    def test_run_local_medallion_rejects_empty_execution_plan(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "At least one medallion step must be enabled.",
        ):
            validate_medallion_plan(_config(run_silver=False, run_gold=False).to_pipeline_config())


if __name__ == "__main__":
    unittest.main()
