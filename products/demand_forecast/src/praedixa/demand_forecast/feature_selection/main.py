"""Executable entrypoint for training-bundle feature selection."""

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


def _default_bundle_dir() -> str:
    from praedixa.demand_forecast.feature_selection.paths import (
        DEFAULT_BUNDLE_INPUT_DIR,
    )

    return str(DEFAULT_BUNDLE_INPUT_DIR)


def _default_output_dir() -> str:
    from praedixa.demand_forecast.feature_selection.paths import (
        DEFAULT_FEATURE_SELECTION_DIR,
    )

    return str(DEFAULT_FEATURE_SELECTION_DIR)


@dataclass(frozen=True)
class FeatureSelectionMainConfig:
    bundle_dir: str = _default_bundle_dir()
    output_dir: str = _default_output_dir()
    n_trials: int = 20
    n_folds: int = 5
    n_fold_workers: int = 5


def build_default_feature_selection_main_config() -> FeatureSelectionMainConfig:
    return FeatureSelectionMainConfig()


def main() -> None:
    from praedixa.demand_forecast.feature_selection.linear import run_feature_selection

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_default_feature_selection_main_config()
    result = run_feature_selection(
        bundle_dir=config.bundle_dir,
        output_dir=config.output_dir,
        n_trials=config.n_trials,
        n_folds=config.n_folds,
        n_fold_workers=config.n_fold_workers,
    )
    logging.getLogger(__name__).info(
        "Feature selection artifacts written: %s",
        json.dumps(result.__dict__, default=str, indent=2),
    )


if __name__ == "__main__":
    main()
