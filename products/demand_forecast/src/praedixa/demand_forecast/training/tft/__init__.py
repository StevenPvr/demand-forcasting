from __future__ import annotations

from typing import Any


_EXPORTED_TUNING_NAMES: tuple[str, ...] = (
    "fit_and_score_tft_model_on_tuning",
    "optimize_tft_model_params",
    "prewarm_tft_fold_cores_for_optuna",
    "resolve_hpo_execution_policy",
)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTED_TUNING_NAMES:
        raise AttributeError(name)
    from praedixa.demand_forecast.training.tft import tuning

    return getattr(tuning, name)

