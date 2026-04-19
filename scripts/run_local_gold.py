from __future__ import annotations

from dataclasses import dataclass
import argparse
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bronze_duckdb_specs import default_open_exogenous_bronze_specs  # noqa: E402
from scripts.fetch_open_exogenous import (  # noqa: E402
    build_runtime_config_from_env as build_open_exogenous_runtime_config_from_env,
    fetch_open_exogenous_data,
)
from scripts.load_bronze_duckdb import load_selected_bronze_specs  # noqa: E402
from scripts.run_local_silver import (  # noqa: E402
    build_local_silver_env,
    dbt_packages_installed,
    ensure_dbt_profiles_file,
    resolve_dbt_executable,
    resolve_dbt_selector,
    run_dbt_command,
)


logger = logging.getLogger(__name__)
DEFAULT_DBT_SELECT = "tag:gold"
DEFAULT_GOLD_SCHEMA = "gold"
DEFAULT_GOLD_HORIZON_DAYS = "1"
DEFAULT_GOLD_BAKERY_TEST_MONTHS = "3"
DEFAULT_OPEN_EXOGENOUS_DIR = Path("data/external_open")


@dataclass(frozen=True)
class LocalGoldRunConfig:
    """Runtime configuration for the local Praedixa gold workflow."""

    dbt_select: str
    refresh_open_exogenous: bool
    run_dbt_tests: bool


def build_local_gold_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the env vars required for a local DuckDB gold run."""

    env = build_local_silver_env(base_env=base_env)
    env.setdefault("PRAEDIXA_DUCKDB_GOLD_SCHEMA", DEFAULT_GOLD_SCHEMA)
    env.setdefault("PRAEDIXA_GOLD_HORIZON_DAYS", DEFAULT_GOLD_HORIZON_DAYS)
    env.setdefault("PRAEDIXA_GOLD_BAKERY_TEST_MONTHS", DEFAULT_GOLD_BAKERY_TEST_MONTHS)
    env.setdefault("PRAEDIXA_OPEN_EXOGENOUS_DIR", str(DEFAULT_OPEN_EXOGENOUS_DIR))
    return env


@contextmanager
def _patched_environment(env_updates: dict[str, str]) -> object:
    original_environment = os.environ.copy()
    os.environ.update(env_updates)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_environment)


def refresh_open_exogenous_inputs(env: dict[str, str]) -> dict[str, object]:
    """Fetch open-source exogenous data and load only the matching bronze tables."""

    with _patched_environment(env):
        exogenous_fetch_result = fetch_open_exogenous_data(
            build_open_exogenous_runtime_config_from_env()
        )
        exogenous_specs = default_open_exogenous_bronze_specs(
            data_dir=env["PRAEDIXA_BRONZE_DATA_DIR"],
            schema_name=env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"],
            open_exogenous_dir=env["PRAEDIXA_OPEN_EXOGENOUS_DIR"],
        )
        bronze_load_result = load_selected_bronze_specs(specs=exogenous_specs)

    return {
        "fetch": exogenous_fetch_result,
        "bronze_load": bronze_load_result,
    }


def run_local_gold(config: LocalGoldRunConfig) -> dict[str, object]:
    """Materialize the local Praedixa gold workflow in DuckDB with open-source exogenous refresh."""

    env = build_local_gold_env()
    dbt_executable = resolve_dbt_executable(PROJECT_ROOT)
    profiles_path = ensure_dbt_profiles_file(PROJECT_ROOT)
    profiles_dir = profiles_path.parent
    selector = resolve_dbt_selector(config.dbt_select)
    test_selector = config.dbt_select.strip().lstrip("+")

    result: dict[str, object] = {
        "open_exogenous_refresh": None,
        "dbt_run": False,
        "dbt_test": False,
    }

    if config.refresh_open_exogenous:
        result["open_exogenous_refresh"] = refresh_open_exogenous_inputs(env)

    if not dbt_packages_installed(PROJECT_ROOT):
        run_dbt_command(dbt_executable, "deps", profiles_dir=profiles_dir, env=env, cwd=PROJECT_ROOT)

    run_dbt_command(
        dbt_executable,
        "run",
        profiles_dir=profiles_dir,
        env=env,
        cwd=PROJECT_ROOT,
        select=selector,
    )
    result["dbt_run"] = True

    if config.run_dbt_tests:
        run_dbt_command(
            dbt_executable,
            "test",
            profiles_dir=profiles_dir,
            env=env,
            cwd=PROJECT_ROOT,
            select=test_selector,
        )
        result["dbt_test"] = True

    return result


def parse_args() -> LocalGoldRunConfig:
    """Parse CLI args for the local gold runner."""

    parser = argparse.ArgumentParser(description="Run the local Praedixa silver -> gold workflow.")
    parser.add_argument("--select", default=DEFAULT_DBT_SELECT, help="dbt selector to run and test.")
    parser.add_argument(
        "--skip-open-exogenous-refresh",
        action="store_true",
        help="Skip refreshing the open-source exogenous files and bronze tables.",
    )
    parser.add_argument("--skip-dbt-tests", action="store_true", help="Skip dbt test after dbt run.")
    args = parser.parse_args()
    return LocalGoldRunConfig(
        dbt_select=args.select,
        refresh_open_exogenous=not args.skip_open_exogenous_refresh,
        run_dbt_tests=not args.skip_dbt_tests,
    )


def main() -> None:
    """CLI entrypoint for the local Praedixa gold workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_local_gold(parse_args())


if __name__ == "__main__":
    main()
