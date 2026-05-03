"""Reusable constants for bundle feature selection."""

from __future__ import annotations

RANDOM_SEED: int = 7
DEFAULT_OPTUNA_TRIALS: int = 20
DEFAULT_FOLDS: int = 5
DEFAULT_FOLD_WORKERS: int = 5
DEFAULT_SELECTOR_MAX_ITER: int = 5_000
MAX_SELECTED_MODEL_FEATURES: int = 50
CORRELATION_THRESHOLD: float = 0.95
COEFFICIENT_SELECTION_THRESHOLD: float = 1e-6
