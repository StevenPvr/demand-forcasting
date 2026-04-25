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

from praedixa.platform.datasets.standardization.pipeline import (  # noqa: E402
    build_global_daily_standardization,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Validate the canonical dataset assembly path without persisting intermediate parquets."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    artifacts = build_global_daily_standardization()
    logger.info(
        "Global daily dataset validation summary: %s",
        json.dumps(
            {
                "sources": artifacts.source_summaries,
                "data_quality": artifacts.data_quality,
            },
            ensure_ascii=True,
        ),
    )


if __name__ == "__main__":
    main()
