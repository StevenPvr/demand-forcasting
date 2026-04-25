from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def _bootstrap_import_paths() -> None:
    if __package__ not in {None, ""}:
        return

    current_file = Path(__file__).resolve()
    try:
        project_root = next(
            parent for parent in current_file.parents if (parent / "AGENTS.md").exists()
        )
    except StopIteration as exc:  # pragma: no cover - repository invariant
        raise RuntimeError(
            f"Unable to resolve the Praedixa project root from {current_file}"
        ) from exc

    search_paths: tuple[Path, ...] = (
        project_root,
        project_root / "platform" / "python" / "src",
        project_root / "products" / "demand_forecast" / "src",
    )
    for path in reversed(search_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from praedixa.demand_forecast.contracts.targets import (  # noqa: E402
    DEFAULT_ABSOLUTE_TARGET_COL,
)
from praedixa.demand_forecast.feature_screening.pipeline import (  # noqa: E402
    build_lag_selection_outputs,
)
from praedixa.platform.runtime.paths import FEATURE_SELECTION_DIR  # noqa: E402


DEFAULT_OUTPUT_DIR = FEATURE_SELECTION_DIR
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    outputs = build_lag_selection_outputs(
        output_dir=DEFAULT_OUTPUT_DIR,
        target_col=DEFAULT_ABSOLUTE_TARGET_COL,
        correlation_only=True,
        max_selected_lag_features=50,
    )
    logger.info("Feature selection outputs: %s", json.dumps({name: str(path) for name, path in outputs.items()}))


if __name__ == "__main__":
    main()
