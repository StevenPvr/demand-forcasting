from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import logging
import os
from pathlib import Path
from typing import Iterator

from praedixa.platform.runtime.constants import DEFAULT_GOLD_BAKERY_TEST_MONTHS
from praedixa.platform.runtime.constants import DEFAULT_GOLD_DBT_SELECT
from praedixa.platform.runtime.constants import DEFAULT_GOLD_HORIZON_DAYS
from praedixa.platform.runtime.constants import DEFAULT_GOLD_SCHEMA
from praedixa.platform.runtime.paths import EXTERNAL_OPEN_DIR
from praedixa.platform.runtime.paths import PROJECT_ROOT
from praedixa.platform.runtime.warehouse import WarehouseRuntimeConfig
from praedixa.platform.warehouse.bronze_specs import default_open_exogenous_bronze_specs
from praedixa.platform.signals.open_data.fetch_app import (
    build_runtime_config_from_env as build_open_exogenous_runtime_config_from_env,
    fetch_open_exogenous_data,
)
from praedixa.platform.warehouse.local_bronze import load_selected_bronze_specs
from praedixa.platform.warehouse.local_silver import (
    build_local_silver_env,
    dbt_packages_installed,
    ensure_dbt_profiles_file,
    resolve_warehouse_project_dir_or_raise,
    resolve_dbt_executable,
    resolve_dbt_selector,
    run_dbt_command,
)


logger = logging.getLogger(__name__)
DEFAULT_DBT_SELECT = DEFAULT_GOLD_DBT_SELECT
DEFAULT_OPEN_EXOGENOUS_DIR = EXTERNAL_OPEN_DIR


@dataclass(frozen=True)
class LocalGoldRunConfig:
    """Runtime configuration for the local Praedixa gold workflow."""

    dbt_select: str
    refresh_open_exogenous: bool
    run_dbt_tests: bool


def build_local_gold_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the env vars required for a local DuckDB gold run."""

    runtime = WarehouseRuntimeConfig(
        gold_schema=DEFAULT_GOLD_SCHEMA,
        open_exogenous_dir=DEFAULT_OPEN_EXOGENOUS_DIR,
        gold_horizon_days=DEFAULT_GOLD_HORIZON_DAYS,
        gold_bakery_test_months=DEFAULT_GOLD_BAKERY_TEST_MONTHS,
    )
    return runtime.to_env(base_env=build_local_silver_env(base_env=base_env))


@contextmanager
def _patched_environment(env_updates: dict[str, str]) -> Iterator[None]:
    original_values = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


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


def _maybe_refresh_open_exogenous(config: LocalGoldRunConfig, env: dict[str, str]) -> dict[str, object] | None:
    if not config.refresh_open_exogenous:
        return None
    return refresh_open_exogenous_inputs(env)


def _run_local_gold_dbt(
    *,
    project_dir: Path,
    dbt_executable: str,
    profiles_dir: Path,
    env: dict[str, str],
    selector: str,
    test_selector: str,
    run_dbt_tests: bool,
) -> dict[str, bool]:
    if not dbt_packages_installed(project_dir):
        run_dbt_command(
            dbt_executable,
            "deps",
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            env=env,
            cwd=PROJECT_ROOT,
        )
    run_dbt_command(
        dbt_executable,
        "seed",
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        env=env,
        cwd=PROJECT_ROOT,
        select="source_registry",
    )
    run_dbt_command(
        dbt_executable,
        "run",
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        env=env,
        cwd=PROJECT_ROOT,
        select=selector,
    )
    if not run_dbt_tests:
        return {"dbt_run": True, "dbt_test": False}
    run_dbt_command(
        dbt_executable,
        "test",
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        env=env,
        cwd=PROJECT_ROOT,
        select=test_selector,
    )
    return {"dbt_run": True, "dbt_test": True}


def run_local_gold(config: LocalGoldRunConfig) -> dict[str, object]:
    """Materialize the local Praedixa gold workflow in DuckDB with open-source exogenous refresh."""

    env = build_local_gold_env()
    warehouse_project_dir = resolve_warehouse_project_dir_or_raise(PROJECT_ROOT)
    dbt_executable = resolve_dbt_executable(PROJECT_ROOT)
    profiles_dir = ensure_dbt_profiles_file(PROJECT_ROOT).parent
    selector = resolve_dbt_selector(config.dbt_select)
    test_selector = config.dbt_select.strip().lstrip("+")
    refresh_result = _maybe_refresh_open_exogenous(config, env)
    dbt_result = _run_local_gold_dbt(
        project_dir=warehouse_project_dir,
        dbt_executable=dbt_executable,
        profiles_dir=profiles_dir,
        env=env,
        selector=selector,
        test_selector=test_selector,
        run_dbt_tests=config.run_dbt_tests,
    )
    return {"open_exogenous_refresh": refresh_result, **dbt_result}


def build_default_local_gold_run_config() -> LocalGoldRunConfig:
    return LocalGoldRunConfig(
        dbt_select=DEFAULT_DBT_SELECT,
        refresh_open_exogenous=True,
        run_dbt_tests=True,
    )


def main() -> None:
    """CLI entrypoint for the local Praedixa gold workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_local_gold(build_default_local_gold_run_config())


if __name__ == "__main__":
    main()
