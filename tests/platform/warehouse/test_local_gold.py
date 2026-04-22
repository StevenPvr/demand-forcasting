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

from praedixa.platform.warehouse.local_gold import (  # noqa: E402
    DEFAULT_GOLD_BAKERY_TEST_MONTHS,
    DEFAULT_GOLD_HORIZON_DAYS,
    DEFAULT_GOLD_SCHEMA,
    LocalGoldRunConfig,
    build_local_gold_env,
    refresh_open_exogenous_inputs,
    run_local_gold,
)


class RunLocalGoldTests(unittest.TestCase):
    def test_run_local_gold_wrapper_uses_no_cli_parser(self) -> None:
        wrapper_path = PROJECT_ROOT / "apps" / "warehouse" / "run_gold" / "main.py"
        source = wrapper_path.read_text(encoding="utf-8")

        self.assertNotIn("argparse", source)
        self.assertNotIn("parse_args", source)

    def test_build_local_gold_env_defaults_to_local_duckdb_gold(self) -> None:
        env = build_local_gold_env(base_env={})

        self.assertEqual(env["PRAEDIXA_ENABLE_LOCAL_WAREHOUSE"], "true")
        self.assertEqual(env["PRAEDIXA_ENABLE_CLOUD_WAREHOUSE"], "false")
        self.assertEqual(env["PRAEDIXA_DUCKDB_GOLD_SCHEMA"], DEFAULT_GOLD_SCHEMA)
        self.assertEqual(env["PRAEDIXA_GOLD_HORIZON_DAYS"], DEFAULT_GOLD_HORIZON_DAYS)
        self.assertEqual(env["PRAEDIXA_GOLD_BAKERY_TEST_MONTHS"], DEFAULT_GOLD_BAKERY_TEST_MONTHS)

    def test_run_local_gold_calls_dbt_run_and_test(self) -> None:
        config = LocalGoldRunConfig(dbt_select="tag:gold", refresh_open_exogenous=True, run_dbt_tests=True)

        with (
            mock.patch("praedixa.platform.warehouse.local_gold.refresh_open_exogenous_inputs", return_value={"fetch": {}, "bronze_load": {}}),
            mock.patch("praedixa.platform.warehouse.local_gold.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_gold.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_gold.dbt_packages_installed", return_value=True),
            mock.patch("praedixa.platform.warehouse.local_gold.run_dbt_command") as run_dbt_command_mock,
        ):
            result = run_local_gold(config)

        self.assertEqual(run_dbt_command_mock.call_count, 3)
        self.assertEqual(run_dbt_command_mock.call_args_list[0].args[1], "seed")
        self.assertEqual(run_dbt_command_mock.call_args_list[1].args[1], "run")
        self.assertEqual(run_dbt_command_mock.call_args_list[1].kwargs["select"], "+tag:gold")
        self.assertEqual(run_dbt_command_mock.call_args_list[2].args[1], "test")
        self.assertEqual(run_dbt_command_mock.call_args_list[2].kwargs["select"], "tag:gold")
        self.assertTrue(result["dbt_run"])
        self.assertTrue(result["dbt_test"])

    def test_run_local_gold_can_skip_dbt_tests(self) -> None:
        config = LocalGoldRunConfig(dbt_select="tag:gold", refresh_open_exogenous=True, run_dbt_tests=False)

        with (
            mock.patch("praedixa.platform.warehouse.local_gold.refresh_open_exogenous_inputs", return_value={"fetch": {}, "bronze_load": {}}),
            mock.patch("praedixa.platform.warehouse.local_gold.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_gold.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_gold.dbt_packages_installed", return_value=True),
            mock.patch("praedixa.platform.warehouse.local_gold.run_dbt_command") as run_dbt_command_mock,
        ):
            result = run_local_gold(config)

        self.assertEqual(run_dbt_command_mock.call_count, 2)
        self.assertEqual(run_dbt_command_mock.call_args_list[0].args[1], "seed")
        self.assertEqual(run_dbt_command_mock.call_args_list[1].args[1], "run")
        self.assertTrue(result["dbt_run"])
        self.assertFalse(result["dbt_test"])

    def test_run_local_gold_bootstraps_dbt_packages_when_missing(self) -> None:
        config = LocalGoldRunConfig(dbt_select="tag:gold", refresh_open_exogenous=True, run_dbt_tests=False)

        with (
            mock.patch("praedixa.platform.warehouse.local_gold.refresh_open_exogenous_inputs", return_value={"fetch": {}, "bronze_load": {}}),
            mock.patch("praedixa.platform.warehouse.local_gold.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_gold.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_gold.dbt_packages_installed", return_value=False),
            mock.patch("praedixa.platform.warehouse.local_gold.run_dbt_command") as run_dbt_command_mock,
        ):
            run_local_gold(config)

        self.assertEqual(run_dbt_command_mock.call_count, 3)
        self.assertEqual(run_dbt_command_mock.call_args_list[0].args[1], "deps")
        self.assertEqual(run_dbt_command_mock.call_args_list[1].args[1], "seed")
        self.assertEqual(run_dbt_command_mock.call_args_list[2].args[1], "run")

    def test_run_local_gold_refreshes_open_exogenous_by_default(self) -> None:
        config = LocalGoldRunConfig(dbt_select="tag:gold", refresh_open_exogenous=True, run_dbt_tests=False)

        with (
            mock.patch("praedixa.platform.warehouse.local_gold.refresh_open_exogenous_inputs", return_value={"fetch": {}, "bronze_load": {}}) as refresh_mock,
            mock.patch("praedixa.platform.warehouse.local_gold.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_gold.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_gold.dbt_packages_installed", return_value=True),
            mock.patch("praedixa.platform.warehouse.local_gold.run_dbt_command"),
        ):
            result = run_local_gold(config)

        refresh_mock.assert_called_once()
        self.assertIsNotNone(result["open_exogenous_refresh"])

    def test_refresh_open_exogenous_inputs_fetches_and_loads_matching_specs(self) -> None:
        env = build_local_gold_env(base_env={})

        with (
            mock.patch("praedixa.platform.warehouse.local_gold.fetch_open_exogenous_data", return_value={"manifest_path": Path("var/datasets/open_exogenous/open_exogenous_manifest.json")}) as fetch_mock,
            mock.patch("praedixa.platform.warehouse.local_gold.build_open_exogenous_runtime_config_from_env", return_value=mock.sentinel.runtime_config),
            mock.patch("praedixa.platform.warehouse.local_gold.default_open_exogenous_bronze_specs", return_value=["spec_a", "spec_b"]) as specs_mock,
            mock.patch("praedixa.platform.warehouse.local_gold.load_selected_bronze_specs", return_value={"local_warehouse_row_counts": {"open_weather_daily": 123}}) as load_mock,
        ):
            result = refresh_open_exogenous_inputs(env)

        fetch_mock.assert_called_once_with(mock.sentinel.runtime_config)
        specs_mock.assert_called_once()
        load_mock.assert_called_once_with(specs=["spec_a", "spec_b"])
        self.assertIn("fetch", result)
        self.assertIn("bronze_load", result)


if __name__ == "__main__":
    unittest.main()
