from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.warehouse.local_silver import (  # noqa: E402
    DEFAULT_LOCAL_DUCKDB_PATH,
    LocalSilverRunConfig,
    build_local_silver_env,
    dbt_packages_installed,
    ensure_dbt_profiles_file,
    resolve_warehouse_project_dir,
    resolve_dbt_selector,
    resolve_dbt_test_exclude,
    resolve_dbt_test_selector,
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
        self.assertEqual(env["PRAEDIXA_DUCKDB_LOCAL_PATH"], str(DEFAULT_LOCAL_DUCKDB_PATH))
        self.assertEqual(env["PRAEDIXA_DUCKDB_TARGET_PATH"], str(DEFAULT_LOCAL_DUCKDB_PATH))

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

    def test_resolve_dbt_selector_adds_upstream_dependencies(self) -> None:
        self.assertEqual(resolve_dbt_selector("tag:silver"), "+tag:silver")
        self.assertEqual(resolve_dbt_selector("+tag:silver"), "+tag:silver")

    def test_resolve_dbt_test_selector_drops_upstream_expansion(self) -> None:
        self.assertEqual(resolve_dbt_test_selector("tag:silver"), "tag:silver")
        self.assertEqual(resolve_dbt_test_selector("+tag:silver"), "tag:silver")

    def test_resolve_dbt_test_exclude_targets_gold_tests(self) -> None:
        self.assertEqual(resolve_dbt_test_exclude(), "tag:gold")

    def test_resolve_warehouse_project_dir_requires_dbt_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "platform" / "warehouse").mkdir(parents=True)

            self.assertIsNone(resolve_warehouse_project_dir(root))

    def test_run_local_silver_calls_bronze_loader_then_dbt_run_and_test(self) -> None:
        config = LocalSilverRunConfig(
            data_dir=Path("var/sources"),
            dbt_select="tag:silver",
            run_bronze_load=True,
            run_dbt_tests=True,
        )

        with (
            mock.patch(
                "praedixa.platform.warehouse.local_silver.build_supplemental_corpus_frame",
                return_value=mock.MagicMock(to_pandas=mock.MagicMock(return_value=mock.sentinel.supplemental_frame)),
            ) as supplemental_corpus_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.default_active_bronze_specs", return_value=["active_specs"]) as specs_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.load_selected_bronze_specs", return_value={"ok": True}) as load_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.load_inline_bronze_frame", return_value={"local": 12}) as inline_load_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_silver.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_silver.dbt_packages_installed", return_value=True),
            mock.patch("praedixa.platform.warehouse.local_silver.run_subprocess") as run_mock,
        ):
            result = run_local_silver(config)

        supplemental_corpus_mock.assert_called_once()
        specs_mock.assert_called_once()
        load_mock.assert_called_once_with(specs=["active_specs"])
        inline_load_mock.assert_called_once()
        self.assertEqual(run_mock.call_count, 3)
        seed_command = run_mock.call_args_list[0].args[0]
        first_command = run_mock.call_args_list[1].args[0]
        second_command = run_mock.call_args_list[2].args[0]
        self.assertEqual(seed_command[:2], [".venv/bin/dbt", "seed"])
        self.assertEqual(first_command[:2], [".venv/bin/dbt", "run"])
        self.assertEqual(second_command[:2], [".venv/bin/dbt", "test"])
        self.assertEqual(first_command[-1], "+tag:silver")
        self.assertEqual(second_command[second_command.index("--select") + 1], "tag:silver")
        self.assertEqual(second_command[second_command.index("--exclude") + 1], "tag:gold")
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
            mock.patch("praedixa.platform.warehouse.local_silver.build_supplemental_corpus_frame") as supplemental_corpus_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.load_inline_bronze_frame") as inline_load_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.load_selected_bronze_specs") as load_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_silver.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_silver.dbt_packages_installed", return_value=True),
            mock.patch("praedixa.platform.warehouse.local_silver.run_subprocess") as run_mock,
        ):
            result = run_local_silver(config)

        supplemental_corpus_mock.assert_not_called()
        inline_load_mock.assert_not_called()
        load_mock.assert_not_called()
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args_list[0].args[0][:2], [".venv/bin/dbt", "seed"])
        self.assertEqual(run_mock.call_args_list[1].args[0][:2], [".venv/bin/dbt", "run"])
        self.assertTrue(result["dbt_run"])
        self.assertFalse(result["dbt_test"])

    def test_ensure_dbt_profiles_file_uses_existing_profile(self) -> None:
        profile_path = ensure_dbt_profiles_file(PROJECT_ROOT)

        self.assertEqual(profile_path, PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml")
        self.assertTrue(profile_path.exists())

    def test_dbt_packages_installed_reads_expected_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            warehouse_dir = root / "platform" / "warehouse"
            (warehouse_dir / "dbt_packages").mkdir(parents=True)
            (warehouse_dir / "dbt_project.yml").write_text("name: test\n", encoding="utf-8")

            self.assertFalse(dbt_packages_installed(PROJECT_ROOT / "tmp_missing_root"))
            self.assertTrue(dbt_packages_installed(root))

    def test_run_local_silver_bootstraps_dbt_packages_when_missing(self) -> None:
        config = LocalSilverRunConfig(
            data_dir=Path("var/sources"),
            dbt_select="tag:silver",
            run_bronze_load=False,
            run_dbt_tests=False,
        )

        with (
            mock.patch("praedixa.platform.warehouse.local_silver.build_supplemental_corpus_frame") as supplemental_corpus_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.load_inline_bronze_frame") as inline_load_mock,
            mock.patch("praedixa.platform.warehouse.local_silver.resolve_dbt_executable", return_value=".venv/bin/dbt"),
            mock.patch("praedixa.platform.warehouse.local_silver.ensure_dbt_profiles_file", return_value=PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml"),
            mock.patch("praedixa.platform.warehouse.local_silver.dbt_packages_installed", return_value=False),
            mock.patch("praedixa.platform.warehouse.local_silver.run_subprocess") as run_mock,
        ):
            run_local_silver(config)

        supplemental_corpus_mock.assert_not_called()
        inline_load_mock.assert_not_called()
        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(run_mock.call_args_list[0].args[0][:2], [".venv/bin/dbt", "deps"])
        self.assertEqual(run_mock.call_args_list[1].args[0][:2], [".venv/bin/dbt", "seed"])
        self.assertEqual(run_mock.call_args_list[2].args[0][:2], [".venv/bin/dbt", "run"])
        self.assertEqual(run_mock.call_args_list[2].args[0][-1], "+tag:silver")


if __name__ == "__main__":
    unittest.main()
