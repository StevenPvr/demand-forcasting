from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from praedixa.platform.runtime.constants import DEFAULT_BRONZE_SCHEMA
from praedixa.platform.runtime.constants import DEFAULT_SILVER_DBT_SELECT
from praedixa.platform.runtime.constants import DEFAULT_SILVER_SCHEMA
from praedixa.platform.datasets.standardization.supplemental_corpus import build_supplemental_corpus_frame
from praedixa.platform.runtime.paths import LOCAL_DUCKDB_PATH
from praedixa.platform.runtime.paths import SOURCES_DIR
from praedixa.platform.runtime.warehouse import WarehouseRuntimeConfig
from praedixa.platform.warehouse.bronze_specs import default_active_core_bronze_specs
from praedixa.platform.warehouse.bronze_specs import default_open_exogenous_bronze_specs
from praedixa.platform.warehouse.bronze_specs import supplemental_corpus_daily_ddl
from praedixa.platform.warehouse.dbt_runner import dbt_packages_installed
from praedixa.platform.warehouse.dbt_runner import DbtStageRunConfig
from praedixa.platform.warehouse.dbt_runner import DbtStageRunner
from praedixa.platform.warehouse.dbt_runner import ensure_dbt_profiles_file
from praedixa.platform.warehouse.dbt_runner import resolve_dbt_executable
from praedixa.platform.warehouse.dbt_runner import resolve_dbt_selector
from praedixa.platform.warehouse.dbt_runner import resolve_dbt_test_exclude
from praedixa.platform.warehouse.dbt_runner import resolve_dbt_test_selector
from praedixa.platform.warehouse.dbt_runner import resolve_warehouse_project_dir
from praedixa.platform.warehouse.dbt_runner import resolve_warehouse_project_dir_or_raise
from praedixa.platform.warehouse.dbt_runner import run_dbt_command
from praedixa.platform.warehouse.local_bronze import load_inline_bronze_frame
from praedixa.platform.warehouse.local_bronze import load_selected_bronze_specs

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = SOURCES_DIR
DEFAULT_LOCAL_DUCKDB_PATH = LOCAL_DUCKDB_PATH
DEFAULT_DBT_SELECT = DEFAULT_SILVER_DBT_SELECT

__all__ = [
    "LocalSilverRunConfig",
    "build_default_local_silver_run_config",
    "build_local_silver_env",
    "dbt_packages_installed",
    "ensure_dbt_profiles_file",
    "load_core_bronze_sources",
    "resolve_dbt_executable",
    "resolve_dbt_selector",
    "resolve_dbt_test_exclude",
    "resolve_dbt_test_selector",
    "resolve_warehouse_project_dir",
    "resolve_warehouse_project_dir_or_raise",
    "run_dbt_command",
    "run_local_silver",
]


@dataclass(frozen=True)
class LocalSilverRunConfig:
    """Runtime configuration for the local Praedixa silver workflow."""

    data_dir: Path
    dbt_select: str
    run_bronze_load: bool
    run_dbt_tests: bool


def build_local_silver_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the env vars required for a local DuckDB silver run."""

    runtime = WarehouseRuntimeConfig(
        duckdb_local_path=DEFAULT_LOCAL_DUCKDB_PATH,
        bronze_schema=DEFAULT_BRONZE_SCHEMA,
        silver_schema=DEFAULT_SILVER_SCHEMA,
        bronze_data_dir=DEFAULT_DATA_DIR,
    )
    return runtime.to_env(base_env=base_env)


def load_core_bronze_sources(
    config: LocalSilverRunConfig,
    env: dict[str, str],
) -> dict[str, object] | None:
    """Load core local bronze inputs and the inline supplemental corpus."""

    if not config.run_bronze_load:
        return None
    bronze_specs = default_active_core_bronze_specs(
        data_dir=config.data_dir,
        schema_name=env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"],
    )
    bronze_specs = [
        *bronze_specs,
        *default_open_exogenous_bronze_specs(
            data_dir=config.data_dir,
            schema_name=env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"],
            open_exogenous_dir=env.get("PRAEDIXA_OPEN_EXOGENOUS_DIR"),
        ),
    ]
    bronze_load = load_selected_bronze_specs(specs=bronze_specs)
    supplemental_corpus_frame = build_supplemental_corpus_frame(
        raw_dir=config.data_dir / "commercial_datasets" / "raw",
    )
    if supplemental_corpus_frame is None:
        bronze_load["inline_supplemental_corpus_row_counts"] = {}
        return bronze_load
    bronze_load["inline_supplemental_corpus_row_counts"] = load_inline_bronze_frame(
        table_name="bronze_supplemental_corpus_daily",
        ddl=supplemental_corpus_daily_ddl(env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"]),
        frame=supplemental_corpus_frame.to_pandas(),
        source_name="supplemental_corpus_runtime",
    )
    return bronze_load


def run_local_silver(config: LocalSilverRunConfig) -> dict[str, object]:
    """Load local bronze sources into DuckDB and materialize silver with dbt."""

    env = build_local_silver_env()
    env["PRAEDIXA_BRONZE_DATA_DIR"] = str(config.data_dir)
    bronze_load = load_core_bronze_sources(config, env)
    dbt_result = DbtStageRunner().run_stage(
        config=DbtStageRunConfig(
            selector=config.dbt_select,
            test_selector=resolve_dbt_test_selector(config.dbt_select),
            test_exclude=resolve_dbt_test_exclude(),
            run_tests=config.run_dbt_tests,
        ),
        env=env,
    )
    return {"bronze_load": bronze_load, **dbt_result.as_dict()}


def build_default_local_silver_run_config() -> LocalSilverRunConfig:
    return LocalSilverRunConfig(
        data_dir=Path(DEFAULT_DATA_DIR),
        dbt_select=DEFAULT_DBT_SELECT,
        run_bronze_load=True,
        run_dbt_tests=True,
    )


def main() -> None:
    """CLI entrypoint for the local Praedixa silver workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_local_silver(build_default_local_silver_run_config())


if __name__ == "__main__":
    main()
