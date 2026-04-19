from __future__ import annotations

from dataclasses import dataclass
import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_praedixa.global_dataset.commercial_external import build_commercial_external_standardized_dataset
from scripts.bronze_duckdb_specs import default_active_bronze_specs
from scripts.load_bronze_duckdb import load_selected_bronze_specs

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path("data")
DEFAULT_LOCAL_DUCKDB_PATH = Path("data/warehouse/praedixa.duckdb")
DEFAULT_BRONZE_SCHEMA = "bronze"
DEFAULT_SILVER_SCHEMA = "silver"
DEFAULT_DBT_SELECT = "+tag:silver"
DEFAULT_COMMERCIAL_RAW_DIR = Path("data/commercial_datasets/raw")
DEFAULT_GLOBAL_DATASET_OUTPUT_DIR = Path("data/global_dataset")


@dataclass(frozen=True)
class LocalSilverRunConfig:
    """Runtime configuration for the local Praedixa silver workflow."""

    data_dir: Path
    dbt_select: str
    run_bronze_load: bool
    run_dbt_tests: bool


def build_local_silver_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the env vars required for a local DuckDB silver run."""

    env = dict(os.environ if base_env is None else base_env)
    env.setdefault("PRAEDIXA_ENABLE_LOCAL_WAREHOUSE", "true")
    env.setdefault("PRAEDIXA_ENABLE_CLOUD_WAREHOUSE", "false")
    env.setdefault("PRAEDIXA_ENABLE_LOCAL_BACKUP", "true")
    env.setdefault("PRAEDIXA_DUCKDB_LOCAL_PATH", str(DEFAULT_LOCAL_DUCKDB_PATH))
    env.setdefault("PRAEDIXA_DUCKDB_TARGET_PATH", env["PRAEDIXA_DUCKDB_LOCAL_PATH"])
    env.setdefault("PRAEDIXA_DUCKDB_BRONZE_SCHEMA", DEFAULT_BRONZE_SCHEMA)
    env.setdefault("PRAEDIXA_DUCKDB_SILVER_SCHEMA", DEFAULT_SILVER_SCHEMA)
    env.setdefault("PRAEDIXA_BRONZE_DATA_DIR", str(DEFAULT_DATA_DIR))
    env.setdefault("PRAEDIXA_DBT_THREADS", str(min(8, os.cpu_count() or 1)))
    return env


def resolve_dbt_executable(project_root: Path) -> str:
    """Prefer the project-local dbt executable when available."""

    candidate = project_root / ".venv" / "bin" / "dbt"
    return str(candidate) if candidate.exists() else "dbt"


def ensure_dbt_profiles_file(project_root: Path) -> Path:
    """Ensure a real dbt profiles file exists for local IDE-friendly execution."""

    profiles_path = project_root / "warehouse" / "profiles.yml"
    if profiles_path.exists():
        return profiles_path

    example_path = project_root / "warehouse" / "profiles.example.yml"
    if not example_path.exists():
        raise FileNotFoundError(f"Missing dbt profiles example at {example_path}")

    shutil.copy2(example_path, profiles_path)
    return profiles_path


def dbt_packages_installed(project_root: Path) -> bool:
    """Return whether dbt dependencies are already installed locally."""

    return (project_root / "warehouse" / "dbt_packages").exists()


def run_dbt_command(
    dbt_executable: str,
    command_name: str,
    *,
    profiles_dir: Path,
    env: dict[str, str],
    cwd: Path,
    select: str | None = None,
) -> None:
    """Run one dbt command with the standard project/profile arguments."""

    command = [
        dbt_executable,
        command_name,
        "--project-dir",
        "warehouse",
        "--profiles-dir",
        str(profiles_dir),
    ]
    if select is not None:
        command.extend(["--select", select])
    run_subprocess(command, env=env, cwd=cwd)


def resolve_dbt_selector(selector: str) -> str:
    """Ensure the dbt selector includes upstream parents for runnable silver builds."""

    stripped = selector.strip()
    return stripped if stripped.startswith("+") else f"+{stripped}"


def run_subprocess(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    """Run one checked subprocess with consistent logging."""

    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd, env=env)


def run_local_silver(config: LocalSilverRunConfig) -> dict[str, object]:
    """Load local bronze sources into DuckDB and materialize silver with dbt."""

    env = build_local_silver_env()
    env["PRAEDIXA_BRONZE_DATA_DIR"] = str(config.data_dir)
    dbt_executable = resolve_dbt_executable(PROJECT_ROOT)
    profiles_path = ensure_dbt_profiles_file(PROJECT_ROOT)
    profiles_dir = profiles_path.parent
    selector = resolve_dbt_selector(config.dbt_select)

    result: dict[str, object] = {"bronze_load": None, "dbt_run": False, "dbt_test": False}
    if config.run_bronze_load:
        build_commercial_external_standardized_dataset(
            output_path=config.data_dir / "global_dataset" / "commercial_external_daily.parquet",
            raw_dir=config.data_dir / "commercial_datasets" / "raw",
        )
        bronze_specs = default_active_bronze_specs(
            data_dir=config.data_dir,
            schema_name=env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"],
            open_exogenous_dir=env.get("PRAEDIXA_OPEN_EXOGENOUS_DIR"),
        )
        result["bronze_load"] = load_selected_bronze_specs(specs=bronze_specs)

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
            select=selector,
        )
        result["dbt_test"] = True

    return result


def parse_args() -> LocalSilverRunConfig:
    """Parse CLI args for the local silver runner."""

    parser = argparse.ArgumentParser(description="Run the local Praedixa bronze -> silver workflow.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to the local bronze input directory.")
    parser.add_argument("--select", default=DEFAULT_DBT_SELECT, help="dbt selector to run and test.")
    parser.add_argument("--skip-bronze-load", action="store_true", help="Skip reloading the local bronze schema.")
    parser.add_argument("--skip-dbt-tests", action="store_true", help="Skip dbt test after dbt run.")
    args = parser.parse_args()
    return LocalSilverRunConfig(
        data_dir=Path(args.data_dir),
        dbt_select=args.select,
        run_bronze_load=not args.skip_bronze_load,
        run_dbt_tests=not args.skip_dbt_tests,
    )


def main() -> None:
    """CLI entrypoint for the local Praedixa silver workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_local_silver(parse_args())


if __name__ == "__main__":
    main()
