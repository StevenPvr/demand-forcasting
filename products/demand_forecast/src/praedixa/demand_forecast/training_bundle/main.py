from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys


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


def _default_output_dir() -> str:
    from praedixa.demand_forecast.training_bundle.bundle_builder import DEFAULT_OUTPUT_DIR

    return str(DEFAULT_OUTPUT_DIR)


@dataclass(frozen=True)
class TrainingBundleMainConfig:
    output_dir: str = _default_output_dir()
    smoke_series_limit: int | None = None
    smoke_dataset_source: str | None = None
    smoke_min_train_rows: int = 56
    smoke_min_tuning_rows: int = 28
    smoke_min_valid_rows: int = 0


def build_default_training_bundle_main_config() -> TrainingBundleMainConfig:
    return TrainingBundleMainConfig()


def main() -> None:
    from praedixa.demand_forecast.training_bundle.bundle_builder import (
        build_training_bundle,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_default_training_bundle_main_config()
    outputs = build_training_bundle(
        output_dir=config.output_dir,
        smoke_series_limit=config.smoke_series_limit,
        smoke_dataset_source=config.smoke_dataset_source,
        smoke_min_train_rows=config.smoke_min_train_rows,
        smoke_min_tuning_rows=config.smoke_min_tuning_rows,
        smoke_min_valid_rows=config.smoke_min_valid_rows,
    )
    logging.getLogger(__name__).info(
        "Training bundle artifacts written: %s",
        json.dumps({name: str(path) for name, path in outputs.items()}, indent=2),
    )


if __name__ == "__main__":
    main()
