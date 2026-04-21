from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.runtime import verify_repo  # noqa: E402


class VerifyRepoTests(unittest.TestCase):
    def test_resolve_dbt_executable_prefers_local_venv(self) -> None:
        expected = verify_repo.PROJECT_ROOT / ".venv" / "bin" / "dbt"

        def path_exists(path: Path) -> bool:
            return path == expected

        with mock.patch.object(Path, "exists", autospec=True) as exists_mock:
            exists_mock.side_effect = path_exists

            resolved = verify_repo.resolve_dbt_executable()

        self.assertEqual(resolved, str(expected))

    def test_main_runs_compileall_unittest_and_dbt_parse(self) -> None:
        run_calls: list[list[str]] = []
        run_envs: list[dict[str, str]] = []

        def fake_run(command: list[str], *, cwd: Path, check: bool, env: dict[str, str]) -> None:
            self.assertEqual(cwd, verify_repo.PROJECT_ROOT)
            self.assertTrue(check)
            run_calls.append(command)
            run_envs.append(env)

        def resolve_tool(name: str) -> str:
            return f"/tmp/{name}"

        with (
            mock.patch.object(verify_repo, "resolve_dbt_executable", return_value="/tmp/dbt"),
            mock.patch.object(verify_repo, "_resolve_tool_executable", side_effect=resolve_tool),
            mock.patch.object(verify_repo.subprocess, "run", side_effect=fake_run),
        ):
            verify_repo.main()

        self.assertEqual(
            run_calls,
            [
                [sys.executable, "-m", "compileall", str(verify_repo.PLATFORM_SRC_ROOT)],
                [sys.executable, "-m", "compileall", str(verify_repo.DEMAND_FORECAST_SRC_ROOT)],
                ["/tmp/pyright", "--project", str(verify_repo.PROJECT_ROOT / "pyrightconfig.json"), *verify_repo.STRICT_TYPING_TARGETS],
                ["/tmp/mypy", "--config-file", str(verify_repo.PROJECT_ROOT / "pyproject.toml"), *verify_repo.STRICT_TYPING_TARGETS],
                [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                ["/tmp/dbt", "parse", "--project-dir", str(verify_repo.WAREHOUSE_PROJECT_DIR), "--profiles-dir", str(verify_repo.WAREHOUSE_PROJECT_DIR)],
            ],
        )
        self.assertTrue(all("DBT_TARGET_PATH" in env for env in run_envs))
        self.assertTrue(all("DBT_LOG_PATH" in env for env in run_envs))


if __name__ == "__main__":
    unittest.main()
