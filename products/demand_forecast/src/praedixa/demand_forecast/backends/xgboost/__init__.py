from __future__ import annotations

from praedixa.demand_forecast.backends.xgboost.backend import (
    XGBoostBackendNotReadyError,
    get_xgboost_backend_availability,
    raise_if_xgboost_backend_required,
)
from praedixa.demand_forecast.backends.xgboost.model_common import (
    DEFAULT_XGBOOST_MODEL_PARAMS,
    resolve_xgboost_model_params,
)
from praedixa.demand_forecast.backends.xgboost.model_fit import (
    FittedXGBoostModel,
    fit_xgboost_model,
    fit_xgboost_model_and_predict_validation,
    predict_with_xgboost_model,
)
from praedixa.demand_forecast.backends.xgboost.runtime import (
    XGBoostRuntimeResolution,
    resolve_xgboost_runtime_profile,
    xgboost_cuda_preflight_available,
)

__all__ = [
    "DEFAULT_XGBOOST_MODEL_PARAMS",
    "FittedXGBoostModel",
    "XGBoostRuntimeResolution",
    "XGBoostBackendNotReadyError",
    "fit_xgboost_model",
    "fit_xgboost_model_and_predict_validation",
    "get_xgboost_backend_availability",
    "predict_with_xgboost_model",
    "raise_if_xgboost_backend_required",
    "resolve_xgboost_model_params",
    "resolve_xgboost_runtime_profile",
    "xgboost_cuda_preflight_available",
]
