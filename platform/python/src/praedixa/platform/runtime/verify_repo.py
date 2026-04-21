from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from praedixa.platform.runtime.warehouse import WarehouseRuntimeConfig
from praedixa.platform.runtime.paths import PLATFORM_SRC_ROOT
from praedixa.platform.runtime.paths import PROJECT_ROOT
from praedixa.platform.runtime.paths import WAREHOUSE_PROJECT_DIR
from praedixa.platform.runtime.paths import DEMAND_FORECAST_SRC_ROOT

LOGGER = logging.getLogger(__name__)

STRICT_TYPING_TARGETS: tuple[str, ...] = (
    "platform/python/src/praedixa/platform/runtime",
    "platform/python/src/praedixa/platform/governance",
    "products/demand_forecast/src/praedixa/demand_forecast/contracts",
    "products/demand_forecast/src/praedixa/demand_forecast/backends/tft",
)


def resolve_dbt_executable() -> str:
    local_dbt = PROJECT_ROOT / ".venv" / "bin" / "dbt"
    if local_dbt.exists():
        return str(local_dbt)
    resolved = shutil.which("dbt")
    if resolved is None:
        raise RuntimeError("dbt executable not found in .venv/bin or PATH.")
    return resolved


def _resolve_tool_executable(name: str) -> str:
    local_tool = PROJECT_ROOT / ".venv" / "bin" / name
    if local_tool.exists():
        return str(local_tool)
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"{name} executable not found in .venv/bin or PATH.")
    return resolved


def _run_step(name: str, command: list[str]) -> None:
    LOGGER.info("Running verification step `%s`: %s", name, " ".join(command))
    env = WarehouseRuntimeConfig().to_env(base_env=None)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    pyright_config_path = PROJECT_ROOT / "pyrightconfig.json"
    steps = [
        ("compileall-platform", [sys.executable, "-m", "compileall", str(PLATFORM_SRC_ROOT)]),
        ("compileall-product", [sys.executable, "-m", "compileall", str(DEMAND_FORECAST_SRC_ROOT)]),
        (
            "pyright",
            [
                _resolve_tool_executable("pyright"),
                "--project",
                str(pyright_config_path),
                *STRICT_TYPING_TARGETS,
            ],
        ),
        (
            "mypy",
            [
                _resolve_tool_executable("mypy"),
                "--config-file",
                str(PROJECT_ROOT / "pyproject.toml"),
                *STRICT_TYPING_TARGETS,
            ],
        ),
        ("unittest", [sys.executable, "-m", "unittest", "discover", "-s", "tests"]),
        (
            "dbt-parse",
            [
                resolve_dbt_executable(),
                "parse",
                "--project-dir",
                str(WAREHOUSE_PROJECT_DIR),
                "--profiles-dir",
                str(WAREHOUSE_PROJECT_DIR),
            ],
        ),
    ]
    for name, command in steps:
        _run_step(name, command)
    LOGGER.info("Repository verification completed successfully.")


if __name__ == "__main__":
    main()
