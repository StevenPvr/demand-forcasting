from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.warehouse.dbt_runner import DbtStageResult  # noqa: E402
from praedixa.platform.warehouse.local_silver import (  # noqa: E402
    DEFAULT_LOCAL_DUCKDB_PATH,
    LocalSilverRunConfig,
    build_local_silver_env,
    dbt_packages_installed,
    ensure_dbt_profiles_file,
    resolve_dbt_selector,
    resolve_dbt_test_exclude,
    resolve_dbt_test_selector,
    resolve_warehouse_project_dir,
    run_local_silver,
)


class RunLocalSilverTests(unittest.TestCase):
    def test_run_local_silver_wrapper_uses_no_cli_parser(self) -> None:
        wrapper_path = PROJECT_ROOT / "apps" / "warehouse" / "run_silver" / "main.py"
        source = wrapper_path.read_text(encoding="utf-8")

        self.assertNotIn("argparse", source)
        self.assertNotIn("parse_args", source)

    def test_build_local_silver_env_defaults_to_local_duckdb(self) -> None:
        env = build_local_silver_env(base_env={})

        self.assertEqual(env["PRAEDIXA_ENABLE_LOCAL_WAREHOUSE"], "true")
        self.assertEqual(env["PRAEDIXA_ENABLE_CLOUD_WAREHOUSE"], "false")
        self.assertEqual(
            env["PRAEDIXA_DUCKDB_LOCAL_PATH"], str(DEFAULT_LOCAL_DUCKDB_PATH)
        )
        self.assertEqual(
            env["PRAEDIXA_DUCKDB_TARGET_PATH"], str(DEFAULT_LOCAL_DUCKDB_PATH)
        )

    def test_build_local_silver_env_preserves_explicit_overrides(self) -> None:
        env = build_local_silver_env(
            base_env={
                "PRAEDIXA_ENABLE_CLOUD_WAREHOUSE": "true",
                "PRAEDIXA_DUCKDB_LOCAL_PATH": "custom.duckdb",
                "PRAEDIXA_DUCKDB_TARGET_PATH": "custom_target.duckdb",
            }
        )

        self.assertEqual(env["PRAEDIXA_ENABLE_CLOUD_WAREHOUSE"], "true")
        self.assertEqual(env["PRAEDIXA_DUCKDB_LOCAL_PATH"], "custom.duckdb")
        self.assertEqual(env["PRAEDIXA_DUCKDB_TARGET_PATH"], "custom_target.duckdb")

    def test_resolve_dbt_selectors(self) -> None:
        self.assertEqual(resolve_dbt_selector("tag:silver"), "+tag:silver")
        self.assertEqual(resolve_dbt_selector("+tag:silver"), "+tag:silver")
        self.assertEqual(resolve_dbt_test_selector("tag:silver"), "tag:silver")
        self.assertEqual(resolve_dbt_test_selector("+tag:silver"), "tag:silver")
        self.assertEqual(resolve_dbt_test_exclude(), "tag:gold")

    def test_resolve_warehouse_project_dir_requires_dbt_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "platform" / "warehouse").mkdir(parents=True)

            self.assertIsNone(resolve_warehouse_project_dir(root))

    def test_run_local_silver_loads_core_bronze_then_runs_stage(self) -> None:
        config = LocalSilverRunConfig(
            data_dir=Path("var/sources"),
            dbt_select="tag:silver",
            run_bronze_load=True,
            run_dbt_tests=True,
        )

        with (
            mock.patch(
                "praedixa.platform.warehouse.local_silver.build_supplemental_corpus_standardized_dataset",
                return_value=Path(
                    "var/datasets/global_dataset/supplemental_corpus_daily.parquet"
                ),
            ) as supplemental_corpus_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.default_active_core_bronze_specs",
                return_value=["core_specs"],
            ) as core_specs_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.default_open_exogenous_bronze_specs",
                return_value=["open_exogenous_specs"],
            ) as open_specs_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.load_selected_bronze_specs",
                return_value={"ok": True},
            ) as load_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.supplemental_corpus_daily_spec",
                return_value="supplemental_spec",
            ) as supplemental_spec_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.DbtStageRunner"
            ) as runner_cls,
        ):
            runner_cls.return_value.run_stage.return_value = DbtStageResult(
                deps=False,
                seed=True,
                run=True,
                test=True,
            )
            result = run_local_silver(config)

        supplemental_corpus_mock.assert_called_once()
        core_specs_mock.assert_called_once()
        open_specs_mock.assert_called_once()
        supplemental_spec_mock.assert_called_once()
        load_mock.assert_called_once_with(
            specs=["core_specs", "open_exogenous_specs", "supplemental_spec"]
        )
        runner_cls.return_value.run_stage.assert_called_once()
        stage_config = runner_cls.return_value.run_stage.call_args.kwargs["config"]
        self.assertEqual(stage_config.selector, "tag:silver")
        self.assertEqual(stage_config.test_selector, "tag:silver")
        self.assertEqual(stage_config.test_exclude, "tag:gold")
        self.assertTrue(stage_config.run_tests)
        self.assertTrue(result["dbt_run"])
        self.assertTrue(result["dbt_test"])

    def test_run_local_silver_can_skip_bronze_reload_and_dbt_tests(self) -> None:
        config = LocalSilverRunConfig(
            data_dir=Path("var/sources"),
            dbt_select="tag:silver",
            run_bronze_load=False,
            run_dbt_tests=False,
        )

        with (
            mock.patch(
                "praedixa.platform.warehouse.local_silver.build_supplemental_corpus_standardized_dataset"
            ) as supplemental_corpus_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.load_selected_bronze_specs"
            ) as load_mock,
            mock.patch(
                "praedixa.platform.warehouse.local_silver.DbtStageRunner"
            ) as runner_cls,
        ):
            runner_cls.return_value.run_stage.return_value = DbtStageResult(
                deps=False,
                seed=True,
                run=True,
                test=False,
            )
            result = run_local_silver(config)

        supplemental_corpus_mock.assert_not_called()
        load_mock.assert_not_called()
        stage_config = runner_cls.return_value.run_stage.call_args.kwargs["config"]
        self.assertFalse(stage_config.run_tests)
        self.assertTrue(result["dbt_run"])
        self.assertFalse(result["dbt_test"])

    def test_ensure_dbt_profiles_file_uses_existing_profile(self) -> None:
        profile_path = ensure_dbt_profiles_file(PROJECT_ROOT)

        self.assertEqual(
            profile_path, PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"
        )
        self.assertTrue(profile_path.exists())

    def test_dbt_packages_installed_reads_expected_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            warehouse_dir = root / "platform" / "warehouse"
            (warehouse_dir / "dbt_packages").mkdir(parents=True)
            (warehouse_dir / "dbt_project.yml").write_text(
                "name: test\n", encoding="utf-8"
            )

            self.assertFalse(dbt_packages_installed(PROJECT_ROOT / "tmp_missing_root"))
            self.assertTrue(dbt_packages_installed(root))


if __name__ == "__main__":
    unittest.main()
