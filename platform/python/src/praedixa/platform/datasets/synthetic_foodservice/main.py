from __future__ import annotations

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
    )
    for path in reversed(search_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from praedixa.platform.datasets.synthetic_foodservice.config import (  # noqa: E402
    build_default_synthetic_foodservice_config,
)
from praedixa.platform.datasets.synthetic_foodservice.exporter import (  # noqa: E402
    generate_synthetic_foodservice_dataset,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Generate the stable synthetic foodservice CSV sources once."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    artifacts = generate_synthetic_foodservice_dataset(
        build_default_synthetic_foodservice_config()
    )
    LOGGER.info("Synthetic daily CSV written to %s", artifacts.daily_csv_path)
    LOGGER.info("Synthetic manifest written to %s", artifacts.manifest_path)


if __name__ == "__main__":
    main()
