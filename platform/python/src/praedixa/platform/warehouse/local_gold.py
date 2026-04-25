from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from contextlib import contextmanager
import logging
import os
from pathlib import Path
from typing import Generator

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
from praedixa.platform.warehouse.dbt_runner import (
    DbtStageRunConfig,
    DbtStageRunner,
)
from praedixa.platform.warehouse.local_bronze import load_selected_bronze_specs
from praedixa.platform.warehouse.local_silver import (
    build_local_silver_env,
)


logger = logging.getLogger(__name__)
DEFAULT_DBT_SELECT = DEFAULT_GOLD_DBT_SELECT
DEFAULT_OPEN_EXOGENOUS_DIR = EXTERNAL_OPEN_DIR
DEFAULT_SILVER_SCHEMA = "silver"


class MissingSilverDependencyError(RuntimeError):
    """Raised when open exogenous refresh is requested before silver is available."""


class OpenExogenousFallbackUnavailableError(RuntimeError):
    """Raised when provider refresh fails and no complete local fallback exists."""


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
def _patched_environment(env_updates: dict[str, str]) -> Generator[None, None, None]:
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


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_local_duckdb_path(duckdb_path: str) -> bool:
    return not duckdb_path.startswith(("md:", ":memory:"))


def _resolve_local_duckdb_path(duckdb_path: str) -> Path:
    path = Path(duckdb_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def assert_silver_ready_for_open_exogenous(env: dict[str, str]) -> None:
    """Fail fast when silver is missing before fetching point-in-time open data."""

    duckdb_path = env.get("PRAEDIXA_DUCKDB_TARGET_PATH") or env.get(
        "PRAEDIXA_DUCKDB_LOCAL_PATH"
    )
    if duckdb_path is None:
        raise MissingSilverDependencyError(
            "Missing PRAEDIXA_DUCKDB_TARGET_PATH for open exogenous refresh."
        )
    resolved_duckdb_path = (
        str(_resolve_local_duckdb_path(duckdb_path))
        if _is_local_duckdb_path(duckdb_path)
        else duckdb_path
    )
    if _is_local_duckdb_path(duckdb_path) and not Path(resolved_duckdb_path).exists():
        raise MissingSilverDependencyError(
            f"Missing DuckDB warehouse before open exogenous refresh: {duckdb_path}"
        )

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise MissingSilverDependencyError(
            "duckdb is required to verify the silver layer before open exogenous refresh."
        ) from exc

    silver_schema = env.get("PRAEDIXA_DUCKDB_SILVER_SCHEMA", DEFAULT_SILVER_SCHEMA)
    relation = f"{_quote_identifier(silver_schema)}.silver_daily_product_demand"
    try:
        connection = duckdb.connect(resolved_duckdb_path, read_only=True)
        try:
            row = connection.execute(
                f"select count(*) from {relation}"
            ).fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise MissingSilverDependencyError(
            f"Silver dependency unavailable for open exogenous refresh: {relation}"
        ) from exc

    row_count = 0 if row is None else int(row[0])
    if row_count <= 0:
        raise MissingSilverDependencyError(
            f"Silver dependency is empty for open exogenous refresh: {relation}"
        )


def _all_open_exogenous_sources_available(specs: Sequence[object]) -> bool:
    source_paths = [getattr(spec, "source_path", None) for spec in specs]
    return bool(source_paths) and all(
        isinstance(source_path, Path) and source_path.exists()
        for source_path in source_paths
    )


def refresh_open_exogenous_inputs(env: dict[str, str]) -> dict[str, object]:
    """Fetch open-source exogenous data and load only the matching bronze tables."""

    with _patched_environment(env):
        assert_silver_ready_for_open_exogenous(env)
        exogenous_specs = default_open_exogenous_bronze_specs(
            data_dir=env["PRAEDIXA_BRONZE_DATA_DIR"],
            schema_name=env["PRAEDIXA_DUCKDB_BRONZE_SCHEMA"],
            open_exogenous_dir=env["PRAEDIXA_OPEN_EXOGENOUS_DIR"],
        )
        try:
            exogenous_fetch_result: dict[str, object] = fetch_open_exogenous_data(
                build_open_exogenous_runtime_config_from_env()
            )
        except Exception as exc:
            if not _all_open_exogenous_sources_available(exogenous_specs):
                raise OpenExogenousFallbackUnavailableError(
                    "Open exogenous refresh failed and the local fallback file set is incomplete."
                ) from exc
            logger.warning(
                "Open exogenous refresh failed; using existing local open exogenous files: %s: %s",
                type(exc).__name__,
                exc,
            )
            exogenous_fetch_result = {
                "status": "fallback_existing_open_exogenous_files",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        bronze_load_result = load_selected_bronze_specs(specs=exogenous_specs)

    return {
        "fetch": exogenous_fetch_result,
        "bronze_load": bronze_load_result,
    }


def _maybe_refresh_open_exogenous(
    config: LocalGoldRunConfig,
    env: dict[str, str],
) -> dict[str, object] | None:
    if not config.refresh_open_exogenous:
        return None
    return refresh_open_exogenous_inputs(env)


def run_local_gold(config: LocalGoldRunConfig) -> dict[str, object]:
    """Materialize the local Praedixa gold workflow in DuckDB with open-source exogenous refresh."""

    env = build_local_gold_env()
    test_selector = config.dbt_select.strip().lstrip("+")
    refresh_result = _maybe_refresh_open_exogenous(config, env)
    dbt_result = DbtStageRunner().run_stage(
        config=DbtStageRunConfig(
            selector=config.dbt_select,
            test_selector=test_selector,
            run_tests=config.run_dbt_tests,
        ),
        env=env,
    )
    return {"open_exogenous_refresh": refresh_result, **dbt_result.as_dict()}


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
