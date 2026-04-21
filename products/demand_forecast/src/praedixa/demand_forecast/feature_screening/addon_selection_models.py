from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np
import pandas as pd


CheckpointCallback = Callable[[list[dict[str, object]], list[str]], None] | None


@dataclass(frozen=True)
class AddonSelectionContext:
    base_frame: pd.DataFrame
    folds: list[dict[str, object]]
    base_feature_cols: list[str]
    candidate_cols: list[str]
    candidate_matrix: np.memmap
    candidate_to_index: dict[str, int]
    target_col: str
    model_params: dict[str, object] | None
    threads_per_worker: int
    base_mean_wape: float
    base_scores: list[float]
    checkpoint_every: int
    checkpoint_callback: CheckpointCallback
    logger: logging.Logger
