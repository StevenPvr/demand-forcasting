from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    current_file = Path(__file__).resolve()
    try:
        return next(
            parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
        )
    except StopIteration as exc:  # pragma: no cover - repository invariant
        raise RuntimeError(
            f"Unable to resolve the Praedixa project root from {current_file}"
        ) from exc


PROJECT_ROOT = _resolve_project_root()
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    from praedixa.platform.warehouse.local_gold import main as run_gold_main

    run_gold_main()


if __name__ == "__main__":
    main()
