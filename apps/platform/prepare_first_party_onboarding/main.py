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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from praedixa.platform.datasets.standardization.first_party_app import (
        main as first_party_onboarding_main,
    )

    first_party_onboarding_main()


if __name__ == "__main__":
    main()
