from __future__ import annotations

"""Constantes reutilisables pour l'optimisation des modeles."""

CSV_ENCODING: str = "utf-8"
TARGET_COLUMN: str = "target_baguette_t_plus_1"
DEFAULT_TRIALS: int = 100
DEFAULT_FOLDS: int = 10
DEFAULT_FOLD_JOBS: int = -1
DEFAULT_ELASTICNET_MAX_ITER: int = 5_000
RANDOM_SEED: int = 7
MIN_ALPHA: float = 1e-4
MAX_ALPHA: float = 10.0
MIN_L1_RATIO: float = 0.05
MAX_L1_RATIO: float = 1.0
