from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIRS = [REPO_ROOT / "praedixa-ecc" / "scripts"]
REPO_SCAN_DIRS = [
    REPO_ROOT / "src" / "research_praedixa",
    REPO_ROOT / "tests",
    REPO_ROOT / "praedixa-ecc" / "scripts",
]
MAX_FILE_LINES = 500
MAX_FUNCTION_LINES = 50
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"^/Users/"),
    re.compile(r"^[A-Za-z]:\\\\"),
    re.compile(r"^/home/"),
)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str
    line: int | None = None


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def is_test_file(path: Path) -> bool:
    """Return whether a path should be treated as test code."""
    return "tests" in path.parts or path.name.startswith("test_")


def _allows_print(path: Path) -> bool:
    return is_test_file(path) or "scripts" in path.parts


def _requires_future_annotations(path: Path) -> bool:
    return not is_test_file(path) and path.suffix == ".py" and path.name != "__init__.py"


def iter_python_files(paths: list[str], repo_scope: bool) -> list[Path]:
    """Expand explicit paths or repo defaults into a deduplicated Python file list."""
    if paths:
        source_paths = [Path(path).resolve() for path in paths]
    elif repo_scope:
        source_paths = REPO_SCAN_DIRS
    else:
        source_paths = DEFAULT_SCAN_DIRS

    files: list[Path] = []
    for source_path in source_paths:
        if source_path.is_dir():
            files.extend(sorted(path for path in source_path.rglob("*.py") if path.is_file()))
        elif source_path.is_file() and source_path.suffix == ".py":
            files.append(source_path)
    return sorted(dict.fromkeys(files))


def _first_meaningful_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return None


def _public_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            found.append(node)
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    found.append(child)
    return found


def _has_complete_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        if argument.arg in {"self", "cls"}:
            continue
        if argument.annotation is None:
            return False
    if node.args.vararg and node.args.vararg.annotation is None:
        return False
    if node.args.kwarg and node.args.kwarg.annotation is None:
        return False
    if node.name != "__init__" and node.returns is None:
        return False
    return True


def _has_bare_except(module: ast.Module) -> list[ast.ExceptHandler]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]


def _print_calls(module: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def _os_path_usages(module: ast.Module) -> list[ast.Attribute]:
    found: list[ast.Attribute] = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            found.append(node)
    return found


def _hardcoded_absolute_paths(module: ast.Module) -> list[tuple[int | None, str]]:
    findings: list[tuple[int | None, str]] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(pattern.match(node.value) for pattern in ABSOLUTE_PATH_PATTERNS):
                findings.append((getattr(node, "lineno", None), node.value))
    return findings


def expected_test_candidates(source_path: Path) -> list[Path]:
    """Return plausible unittest file matches for a source file under src/."""
    if "src" not in source_path.parts or source_path.name == "__init__.py":
        return []

    try:
        relative_source = source_path.relative_to(REPO_ROOT / "src")
    except ValueError:
        return []

    stem = relative_source.stem
    parent_name = relative_source.parent.name
    tests_root = REPO_ROOT / "tests"
    candidates = [tests_root / f"test_{stem}.py"]
    if parent_name and parent_name != "src":
        candidates.append(tests_root / f"test_{parent_name}_{stem}.py")
    return candidates


def _function_findings(
    path: Path,
    display_path: str,
    module: ast.Module,
) -> list[Finding]:
    findings: list[Finding] = []
    for function in _public_functions(module):
        if not is_test_file(path) and not _has_complete_annotations(function):
            findings.append(
                Finding(
                    severity="warning",
                    code="missing-annotations",
                    message=f"Public function `{function.name}` is missing type annotations.",
                    path=display_path,
                    line=function.lineno,
                )
            )
        function_length = (function.end_lineno or function.lineno) - function.lineno + 1
        if function_length > MAX_FUNCTION_LINES:
            findings.append(
                Finding(
                    severity="warning",
                    code="function-too-long",
                    message=f"Function `{function.name}` has {function_length} lines; target is <= {MAX_FUNCTION_LINES}.",
                    path=display_path,
                    line=function.lineno,
                )
            )
        if not is_test_file(path) and ast.get_docstring(function) is None:
            findings.append(
                Finding(
                    severity="warning",
                    code="missing-docstring",
                    message=f"Public function `{function.name}` is missing a docstring.",
                    path=display_path,
                    line=function.lineno,
                )
            )
    return findings


def _module_findings(
    path: Path,
    display_path: str,
    module: ast.Module,
) -> list[Finding]:
    findings: list[Finding] = []
    for handler in _has_bare_except(module):
        findings.append(
            Finding(
                severity="error",
                code="bare-except",
                message="Bare `except:` found; catch a specific exception.",
                path=display_path,
                line=handler.lineno,
            )
        )

    if not _allows_print(path):
        for call in _print_calls(module):
            findings.append(
                Finding(
                    severity="error",
                    code="print-in-prod",
                    message="`print()` found in production code; prefer `logging`.",
                    path=display_path,
                    line=call.lineno,
                )
            )

    for attribute in _os_path_usages(module):
        findings.append(
            Finding(
                severity="warning",
                code="os-path",
                message="`os.path` usage found; prefer `pathlib.Path`.",
                path=display_path,
                line=attribute.lineno,
            )
        )

    for line_number, value in _hardcoded_absolute_paths(module):
        findings.append(
            Finding(
                severity="warning",
                code="absolute-path",
                message=f"Hardcoded absolute path detected: {value}",
                path=display_path,
                line=line_number,
            )
        )
    return findings


def _test_candidate_findings(path: Path, display_path: str) -> list[Finding]:
    findings: list[Finding] = []
    if "src" not in path.parts or is_test_file(path):
        return findings

    candidates = expected_test_candidates(path)
    if candidates and not any(candidate.exists() for candidate in candidates):
        findings.append(
            Finding(
                severity="warning",
                code="missing-test-candidate",
                message="No matching test file candidate found in tests/.",
                path=display_path,
            )
        )
    return findings


def scan_python_file(path: Path) -> list[Finding]:
    """Scan one Python file and return style and quality findings."""
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8")
    display_path = _relative_path(path)

    meaningful_line = _first_meaningful_line(text)
    if _requires_future_annotations(path) and meaningful_line != "from __future__ import annotations":
        findings.append(
            Finding(
                severity="error",
                code="future-annotations",
                message="Missing `from __future__ import annotations` as first meaningful line.",
                path=display_path,
                line=1,
            )
        )

    line_count = len(text.splitlines())
    if line_count > MAX_FILE_LINES:
        findings.append(
            Finding(
                severity="warning",
                code="file-too-long",
                message=f"File has {line_count} lines; target is <= {MAX_FILE_LINES}.",
                path=display_path,
            )
        )

    try:
        module = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        findings.append(
            Finding(
                severity="error",
                code="syntax-error",
                message=error.msg,
                path=display_path,
                line=error.lineno,
            )
        )
        return findings

    findings.extend(_function_findings(path, display_path, module))
    findings.extend(_module_findings(path, display_path, module))
    findings.extend(_test_candidate_findings(path, display_path))
    return findings


def format_finding(finding: Finding) -> str:
    """Format one finding for CLI output."""
    location = f"{finding.path}"
    if finding.line is not None:
        location += f":{finding.line}"
    return f"[{finding.severity}] {finding.code} {location} - {finding.message}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for python_doctor."""
    parser = argparse.ArgumentParser(description="Praedixa ECC Python doctor.")
    parser.add_argument("paths", nargs="*", help="Python files or directories to scan.")
    parser.add_argument(
        "--repo-scope",
        action="store_true",
        help="Scan src/, tests/, and praedixa-ecc/scripts/ instead of framework scripts only.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero on warnings as well as errors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run python_doctor and return a process-style exit code."""
    args = parse_args(argv or sys.argv[1:])
    files = iter_python_files(args.paths, repo_scope=args.repo_scope)

    if not files:
        print("Python doctor: no Python files selected.")
        return 0

    findings: list[Finding] = []
    for file_path in files:
        findings.extend(scan_python_file(file_path))

    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]

    status = "OK"
    if errors or (warnings and args.fail_on_warn):
        status = "FAIL"
    elif warnings:
        status = "WARN"

    print(f"Python doctor: {status}")
    print(f"- scanned_files: {len(files)}")
    print(f"- errors: {len(errors)}")
    print(f"- warnings: {len(warnings)}")

    for finding in findings:
        print(format_finding(finding))

    if errors or (warnings and args.fail_on_warn):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
