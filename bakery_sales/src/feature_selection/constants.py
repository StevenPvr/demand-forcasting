from __future__ import annotations

"""Constantes reutilisables pour la selection de variables."""

CSV_ENCODING: str = "utf-8"
TARGET_COLUMN: str = "target_baguette_t_plus_1"
RANDOM_SEED: int = 7
DEFAULT_SELECTOR_JOBS: int = -1
DEFAULT_OPTUNA_TRIALS: int = 50
DEFAULT_FOLDS: int = 10
DEFAULT_SELECTOR_MAX_ITER: int = 5_000
CORRELATION_THRESHOLD: float = 0.95
COEFFICIENT_SELECTION_THRESHOLD: float = 1e-6
SARIMAX_MAX_SELECTED_EXOGENOUS_FEATURES: int = 5
