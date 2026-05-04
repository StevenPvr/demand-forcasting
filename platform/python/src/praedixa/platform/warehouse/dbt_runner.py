from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import subprocess

from praedixa.platform.runtime.paths import PROJECT_ROOT
from praedixa.platform.runtime.paths import WAREHOUSE_PROJECT_DIR


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbtStageRunConfig:
    """One dbt stage invocation with explicit selectors and test policy."""

    selector: str
    test_selector: str | None = None
    test_exclude: str | None = None
    run_tests: bool = True
    seed_selector: str | None = "source_registry feature_registry"


@dataclass(frozen=True)
class DbtStageResult:
    """Structured result for one local dbt stage."""

    deps: bool
    seed: bool
    run: bool
    test: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "dbt_deps": self.deps,
            "dbt_seed": self.seed,
            "dbt_run": self.run,
            "dbt_test": self.test,
        }


class DbtStageRunner:
    """Run dbt stages with one shared deps/seed/run/test implementation."""

    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        project_dir: Path | None = None,
        profiles_dir: Path | None = None,
        dbt_executable: str | None = None,
        cwd: Path = PROJECT_ROOT,
    ) -> None:
        self.project_root = project_root
        self.project_dir = project_dir or resolve_warehouse_project_dir_or_raise(
            project_root
        )
        self.profiles_dir = profiles_dir or ensure_dbt_profiles_file(
            project_root
        ).parent
        self.dbt_executable = dbt_executable or resolve_dbt_executable(project_root)
        self.cwd = cwd

    def run_stage(
        self,
        *,
        config: DbtStageRunConfig,
        env: dict[str, str],
    ) -> DbtStageResult:
        """Run one dbt stage using the configured selector and tests."""

        deps_ran = False
        if not dbt_packages_installed(self.project_dir):
            run_dbt_command(
                self.dbt_executable,
                "deps",
                project_dir=self.project_dir,
                profiles_dir=self.profiles_dir,
                env=env,
                cwd=self.cwd,
            )
            deps_ran = True
        seed_ran = False
        if config.seed_selector is not None:
            run_dbt_command(
                self.dbt_executable,
                "seed",
                project_dir=self.project_dir,
                profiles_dir=self.profiles_dir,
                env=env,
                cwd=self.cwd,
                select=config.seed_selector,
            )
            seed_ran = True
        run_dbt_command(
            self.dbt_executable,
            "run",
            project_dir=self.project_dir,
            profiles_dir=self.profiles_dir,
            env=env,
            cwd=self.cwd,
            select=resolve_dbt_selector(config.selector),
        )
        test_ran = False
        if config.run_tests:
            run_dbt_command(
                self.dbt_executable,
                "test",
                project_dir=self.project_dir,
                profiles_dir=self.profiles_dir,
                env=env,
                cwd=self.cwd,
                select=config.test_selector
                or resolve_dbt_test_selector(config.selector),
                exclude=config.test_exclude,
            )
            test_ran = True
        return DbtStageResult(
            deps=deps_ran,
            seed=seed_ran,
            run=True,
            test=test_ran,
        )


def resolve_warehouse_project_dir(project_root: Path) -> Path | None:
    """Resolve the dbt project directory from either a repo root or a project path."""

    if (project_root / "dbt_project.yml").exists():
        return project_root
    platform_candidate = project_root / "platform" / "warehouse"
    if (platform_candidate / "dbt_project.yml").exists():
        return platform_candidate
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


def run_subprocess(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    """Run one checked subprocess with consistent logging."""

    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd, env=env)


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
    """Ensure a dbt selector includes upstream parents for runnable warehouse builds."""

    stripped = selector.strip()
    return stripped if stripped.startswith("+") else f"+{stripped}"


def resolve_dbt_test_selector(selector: str) -> str:
    """Keep test selection scoped to the requested nodes instead of downstream layers."""

    return selector.strip().lstrip("+")


def resolve_dbt_test_exclude() -> str:
    """Exclude gold-tagged data tests when validating only the silver layer."""

    return "tag:gold"
