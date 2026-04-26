from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from praedixa.demand_forecast.training.config.constants import SUPPORTED_MODEL_BACKENDS


OptimiseFn = Callable[..., tuple[dict[str, object], pd.DataFrame, dict[str, object]]]


@dataclass(frozen=True)
class OptimisationModelBackend:
    name: str
    optimize_fn: OptimiseFn
    require_available: Callable[[str], None]


def resolve_optimisation_model_backend(
    model_backend: str,
) -> OptimisationModelBackend:
    normalized_backend = model_backend.strip().lower()
    if normalized_backend == "xgboost":
        from praedixa.demand_forecast.backends.xgboost.backend import (
            raise_if_xgboost_backend_required,
        )
        from praedixa.demand_forecast.training.xgboost.tuning import (
            optimize_xgboost_model_params,
        )

        return OptimisationModelBackend(
            name="xgboost",
            optimize_fn=optimize_xgboost_model_params,
            require_available=raise_if_xgboost_backend_required,
        )
    if normalized_backend == "tft":
        from praedixa.demand_forecast.backends.tft.backend import (
            raise_if_tft_backend_required,
        )
        from praedixa.demand_forecast.training.tft.tuning import (
            optimize_tft_model_params,
        )

        return OptimisationModelBackend(
            name="tft",
            optimize_fn=optimize_tft_model_params,
            require_available=raise_if_tft_backend_required,
        )
    if normalized_backend == "chronos2_finetune":
        from praedixa.demand_forecast.backends.chronos2.backend import (
            raise_if_chronos2_finetune_backend_required,
        )
        from praedixa.demand_forecast.training.chronos2.tuning import (
            optimize_chronos2_finetune_params,
        )

        return OptimisationModelBackend(
            name="chronos2_finetune",
            optimize_fn=optimize_chronos2_finetune_params,
            require_available=raise_if_chronos2_finetune_backend_required,
        )
    raise ValueError(
        f"Unsupported optimisation model backend `{model_backend}`. "
        f"Supported backends: {', '.join(SUPPORTED_MODEL_BACKENDS)}."
    )
