"""Centralized paths for demand-forecast feature selection artifacts."""

from __future__ import annotations

from pathlib import Path

from praedixa.platform.runtime.paths import TRAINING_BUNDLE_DIR, VAR_DIR


DEFAULT_BUNDLE_INPUT_DIR: Path = TRAINING_BUNDLE_DIR
DEFAULT_FEATURE_SELECTION_DIR: Path = (
    VAR_DIR / "experiments" / "demand_forecast" / "feature_selection"
)
