from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import subprocess

from praedixa.platform.runtime.constants import DEFAULT_BRONZE_SCHEMA
from praedixa.platform.runtime.constants import DEFAULT_SILVER_DBT_SELECT
from praedixa.platform.runtime.constants import DEFAULT_SILVER_SCHEMA
from praedixa.platform.datasets.standardization.commercial_external import build_commercial_external_standardized_dataset
from praedixa.platform.runtime.paths import GLOBAL_DATASET_DIR
from praedixa.platform.runtime.paths import LOCAL_DUCKDB_PATH
from praedixa.platform.runtime.paths import PROJECT_ROOT
from praedixa.platform.runtime.paths import SOURCES_DIR
from praedixa.platform.runtime.paths import WAREHOUSE_PROJECT_DIR
from praedixa.platform.runtime.warehouse import WarehouseRuntimeConfig
from praedixa.platform.warehouse.bronze_specs import default_active_bronze_specs
from praedixa.platform.warehouse.local_bronze import load_selected_bronze_specs

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = SOURCES_DIR
DEFAULT_LOCAL_DUCKDB_PATH = LOCAL_DUCKDB_PATH
DEFAULT_DBT_SELECT = DEFAULT_SILVER_DBT_SELECT


def resolve_warehouse_project_dir(project_root: Path) -> Path | None:
    """Resolve the dbt project directory from either a repo root or a project path."""

    if (project_root / "dbt_project.yml").exists():
        return project_root
    platform_candidate = project_root / "platform" / "warehouse"
    if (platform_candidate / "dbt_project.yml").exists():
        return platform_candidate
    legacy_candidate = project_root / "warehouse"
    if (legacy_candidate / "dbt_project.yml").exists():
        return legacy_candidate
    if project_root == PROJECT_ROOT:
        return WAREHOUSE_PROJECT_DIR
    return None


def resolve_warehouse_project_dir_or_raise(project_root: Path) -> Path:
    """Resolve the dbt project directory or raise a helpful error."""

    warehouse_project_dir = resolve_warehouse_project_dir(project_root)
    if warehouse_project_dir is None:
        raise FileNotFoundError(
            f"Unable to resolve a dbt project directory from {project_root}"
        )
    return warehouse_project_dir


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


def resolve_dbt_executable(project_root: Path) -> str:
    """Prefer the project-local dbt executable when available."""

    candidate = project_root / ".venv" / "bin" / "dbt"
    return str(candidate) if candidate.exists() else "dbt"


def ensure_dbt_profiles_file(project_root: Path) -> Path:
    """Ensure a real dbt profiles file exists for local IDE-friendly execution."""

    warehouse_project_dir = resolve_warehouse_project_dir_or_raise(project_root)
    profiles_path = warehouse_project_dir / "profiles.yml"
    if profiles_path.exists():
        return profiles_path

    example_path = warehouse_project_dir / "profiles.example.yml"
    if not example_path.exists():
        raise FileNotFoundError(f"Missing dbt profiles example at {example_path}")

    shutil.copy2(example_path, profiles_path)
    return profiles_path


def dbt_packages_installed(project_root: Path) -> bool:
    """Return whether dbt dependencies are already installed locally."""

    warehouse_project_dir = resolve_warehouse_project_dir(project_root)
    if warehouse_project_dir is None:
        return False
    return (warehouse_project_dir / "dbt_packages").exists()


def run_dbt_command(
    dbt_executable: str,
    command_name: str,
    *,
    project_dir: Path,
    profiles_dir: Path,
    env: dict[str, str],
    cwd: Path,
    select: str | None = None,
    exclude: str | None = None,
) -> None:
    """Run one dbt command with the standard project/profile arguments."""

    command = [
        dbt_executable,
        command_name,
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(profiles_dir),
    ]
    if select is not None:
        command.extend(["--select", select])
    if exclude is not None:
        command.extend(["--exclude", exclude])
    run_subprocess(command, env=env, cwd=cwd)


def resolve_dbt_selector(selector: str) -> str:
    """Ensure the dbt selector includes upstream parents for runnable silver builds."""

    stripped = selector.strip()
    return stripped if stripped.startswith("+") else f"+{stripped}"


def resolve_dbt_test_selector(selector: str) -> str:
    """Keep silver test selection scoped to silver nodes instead of downstream gold tests."""

    return selector.strip().lstrip("+")


def resolve_dbt_test_exclude() -> str:
    """Exclude gold-tagged data tests when validating only the silver layer."""

    return "tag:gold"


def run_subprocess(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    """Run one checked subprocess with consistent logging."""

    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd, env=env)


def _maybe_load_local_bronze(config: LocalSilverRunConfig, env: dict[str, str]) -> dict[str, object] | None:
    if not config.run_bronze_load:
        return None
    build_commercial_external_standardized_dataset(
        output_path=GLOBAL_DATASET_DIR / "commercial_external_daily.parquet",
        raw_dir=config.data_dir / "commercial_datasets" / "raw",
    )
    bronze_specs = default_active_bronze_specs(
        data_dir=config.data_dir,
        schema_name=env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"],
        open_exogenous_dir=env.get("PRAEDIXA_OPEN_EXOGENOUS_DIR"),
        commercial_external_dir=GLOBAL_DATASET_DIR,
    )
    return load_selected_bronze_specs(specs=bronze_specs)


def _run_local_silver_dbt(
    *,
    project_dir: Path,
    dbt_executable: str,
    profiles_dir: Path,
    env: dict[str, str],
    selector: str,
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
        select=resolve_dbt_test_selector(selector),
        exclude=resolve_dbt_test_exclude(),
    )
    return {"dbt_run": True, "dbt_test": True}


def run_local_silver(config: LocalSilverRunConfig) -> dict[str, object]:
    """Load local bronze sources into DuckDB and materialize silver with dbt."""

    env = build_local_silver_env()
    env["PRAEDIXA_BRONZE_DATA_DIR"] = str(config.data_dir)
    warehouse_project_dir = resolve_warehouse_project_dir_or_raise(PROJECT_ROOT)
    dbt_executable = resolve_dbt_executable(PROJECT_ROOT)
    profiles_dir = ensure_dbt_profiles_file(PROJECT_ROOT).parent
    selector = resolve_dbt_selector(config.dbt_select)
    bronze_load = _maybe_load_local_bronze(config, env)
    dbt_result = _run_local_silver_dbt(
        project_dir=warehouse_project_dir,
        dbt_executable=dbt_executable,
        profiles_dir=profiles_dir,
        env=env,
        selector=selector,
        run_dbt_tests=config.run_dbt_tests,
    )
    return {"bronze_load": bronze_load, **dbt_result}


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
