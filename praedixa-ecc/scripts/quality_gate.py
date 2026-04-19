from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import python_doctor
import validate_framework


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def gate_python_files(paths: list[str], repo_scope: bool) -> list[Path]:
    """Resolve which Python files should be gated."""
    if paths:
        return python_doctor.iter_python_files(paths, repo_scope=False)
    return python_doctor.iter_python_files([], repo_scope=repo_scope or True)


def collect_test_targets(python_files: list[Path]) -> list[Path]:
    """Map changed Python files to the relevant unittest targets that exist."""
    collected: list[Path] = []
    for path in python_files:
        if python_doctor.is_test_file(path) and path.exists():
            collected.append(path)
            continue
        for candidate in python_doctor.expected_test_candidates(path):
            if candidate.exists():
                collected.append(candidate)
    return sorted(dict.fromkeys(collected))


def module_name_from_test_path(path: Path) -> str:
    """Convert a test file path into a unittest module name."""
    relative = path.resolve().relative_to(REPO_ROOT).with_suffix("")
    return ".".join(relative.parts)


def run_python_doctor_gate(python_files: list[Path], fail_on_warn: bool) -> int:
    """Run python_doctor on the selected files."""
    doctor_args = [str(path) for path in python_files]
    if fail_on_warn:
        doctor_args = ["--fail-on-warn", *doctor_args]
    return python_doctor.main(doctor_args)


def run_test_gate(test_paths: list[Path]) -> int:
    """Run unittest for the resolved test modules."""
    if not test_paths:
        print("Quality gate tests: SKIP")
        print("- test_targets: 0")
        return 0

    module_names = [module_name_from_test_path(path) for path in test_paths]
    print("Quality gate tests: RUN")
    print(f"- test_targets: {len(module_names)}")
    print(f"- modules: {', '.join(module_names)}")

    suite = unittest.defaultTestLoader.loadTestsFromNames(module_names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def run_framework_gate() -> int:
    """Run structural framework validation."""
    try:
        validate_framework.main()
    except SystemExit as error:
        return int(error.code or 1)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the quality gate."""
    parser = argparse.ArgumentParser(description="Praedixa ECC quality gate for Python work.")
    parser.add_argument("paths", nargs="*", help="Python files or directories to gate.")
    parser.add_argument(
        "--repo-scope",
        action="store_true",
        help="Run the gate on the repo Python scope when no explicit paths are provided.",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Do not fail the gate on python_doctor warnings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Praedixa ECC Python quality gate."""
    args = parse_args(argv or sys.argv[1:])
    python_files = gate_python_files(args.paths, repo_scope=args.repo_scope)

    if not python_files:
        print("Quality gate: FAIL")
        print("- no Python files selected")
        return 1

    test_targets = collect_test_targets(python_files)

    python_gate_exit = run_python_doctor_gate(
        python_files,
        fail_on_warn=not args.allow_warnings,
    )
    test_gate_exit = run_test_gate(test_targets)
    framework_gate_exit = run_framework_gate()

    failures = {
        "python_doctor": python_gate_exit,
        "tests": test_gate_exit,
        "framework": framework_gate_exit,
    }
    failing_steps = [name for name, exit_code in failures.items() if exit_code != 0]

    if failing_steps:
        print("Quality gate: FAIL")
        print(f"- failing_steps: {', '.join(failing_steps)}")
        return 1

    print("Quality gate: OK")
    print(f"- scanned_files: {len(python_files)}")
    print(f"- matched_tests: {len(test_targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
