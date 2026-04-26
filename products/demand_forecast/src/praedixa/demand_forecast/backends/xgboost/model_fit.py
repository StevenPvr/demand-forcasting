from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import time
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.backends.xgboost.backend import (
    raise_if_xgboost_backend_required,
)
from praedixa.demand_forecast.backends.xgboost.gpu_input import (
    build_xgboost_matrix_input_payload,
)
from praedixa.demand_forecast.backends.xgboost.model_common import (
    DEFAULT_XGBOOST_MODEL_PARAMS,
    resolve_xgboost_model_params,
    xgboost_early_stopping_rounds,
    xgboost_num_boost_round,
    xgboost_train_params,
)
from praedixa.demand_forecast.backends.xgboost.runtime import (
    xgboost_cuda_preflight_available,
    xgboost_matrix_type,
    xgboost_uses_cuda,
)
from praedixa.demand_forecast.backends.xgboost.preprocessing import (
    XGBoostFeatureSpec,
    XGBoostPreparedMatrixData,
    prepare_xgboost_matrix_data,
    to_xgboost_feature_data,
    transform_xgboost_features,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedXGBoostModel:
    model: Any
    feature_spec: XGBoostFeatureSpec
    params: dict[str, object]


@dataclass(frozen=True)
class _XGBoostTrainingPlan:
    resolved_params: dict[str, object]
    matrices: XGBoostPreparedMatrixData
    train_params: dict[str, object]
    num_boost_round: int
    early_stopping_rounds: int | None
    use_early_stopping: bool


@dataclass(frozen=True)
class XGBoostPreparedTrainingMatrices:
    resolved_params: dict[str, object]
    matrices: XGBoostPreparedMatrixData
    train_matrix: Any
    valid_matrix: Any | None


def _xgboost_module() -> Any:
    raise_if_xgboost_backend_required("xgboost.fit")
    import xgboost as xgb

    return xgb


@lru_cache(maxsize=None)
def warm_up_xgboost_runtime(nthread: int) -> None:
    """Initialise le runtime natif XGBoost/OpenMP avant les fits concurrents."""

    threads = max(1, int(nthread))
    xgb_module = _xgboost_module()
    data = np.asarray([[0.0], [1.0]], dtype=np.float32)
    label = np.asarray([0.0, 1.0], dtype=np.float32)
    matrix = xgb_module.DMatrix(
        data=data,
        label=label,
        feature_names=["warmup_feature"],
        nthread=threads,
    )
    LOGGER.debug("XGBoost native runtime warm-up starting: nthread=%s", threads)
    xgb_module.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_depth": 1,
            "nthread": threads,
            "verbosity": 0,
        },
        dtrain=matrix,
        num_boost_round=1,
        verbose_eval=False,
    )
    LOGGER.debug("XGBoost native runtime warm-up completed: nthread=%s", threads)


def _build_dmatrix(
    xgb_module: Any,
    features: pd.DataFrame | np.ndarray,
    feature_spec: XGBoostFeatureSpec,
    *,
    params: dict[str, object],
    label: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    ref: Any | None = None,
) -> Any:
    input_payload = build_xgboost_matrix_input_payload(
        data=to_xgboost_feature_data(features, feature_spec),
        label=label,
        weight=weight,
        feature_spec=feature_spec,
        model_params=params,
    )
    kwargs: dict[str, object] = {
        "data": input_payload.data,
        "enable_categorical": feature_spec.native_categorical,
        "feature_names": feature_spec.feature_cols,
        "nthread": int(cast(Any, params["n_jobs"])),
    }
    if input_payload.label is not None:
        kwargs["label"] = input_payload.label
    if input_payload.weight is not None:
        kwargs["weight"] = input_payload.weight
    LOGGER.debug(
        "XGBoost native matrix input resolved: matrix_type=%s device=%s requested_gpu_input_backend=%s resolved_gpu_input_backend=%s native_categorical=%s",
        xgboost_matrix_type(params),
        params.get("device"),
        input_payload.resolution.requested_backend,
        input_payload.resolution.resolved_backend,
        feature_spec.native_categorical,
    )
    if xgboost_matrix_type(params) != "quantile":
        return xgb_module.DMatrix(**kwargs)
    quantile_kwargs = {
        **kwargs,
        "max_bin": int(cast(Any, params.get("max_bin", 256))),
    }
    if ref is not None:
        quantile_kwargs["ref"] = ref
    try:
        return xgb_module.QuantileDMatrix(**quantile_kwargs)
    except Exception as exc:
        if xgboost_uses_cuda(params):
            raise RuntimeError(
                "XGBoost CUDA training requires QuantileDMatrix; refusing to "
                "fall back to CPU-style DMatrix for a CUDA run."
            ) from exc
        LOGGER.warning(
            "XGBoost QuantileDMatrix build failed; falling back to DMatrix on the same device: %s",
            exc,
        )
        return xgb_module.DMatrix(**kwargs)


def _param_snapshot(params: dict[str, object]) -> dict[str, object]:
    keys = (
        "colsample_bytree",
        "early_stopping_rounds",
        "enable_categorical",
        "enable_early_stopping",
        "learning_rate",
        "max_bin",
        "max_cat_threshold",
        "max_cat_to_onehot",
        "max_depth",
        "min_child_weight",
        "nthread",
        "n_estimators",
        "n_jobs",
        "reg_alpha",
        "reg_lambda",
        "alpha",
        "lambda",
        "seed",
        "subsample",
        "tree_method",
        "device",
        "xgboost_matrix_type",
        "xgboost_gpu_input_backend",
    )
    return {key: params[key] for key in keys if key in params}


def _model_attr(model: object, name: str) -> object | None:
    try:
        return getattr(model, name, None)
    except AttributeError:
        return None


def _prediction_iteration_range(model: object) -> tuple[int, int] | None:
    best_iteration = _model_attr(model, "best_iteration")
    if isinstance(best_iteration, bool) or not isinstance(best_iteration, int):
        return None
    return (0, best_iteration + 1)


def _log_matrix_summary(matrices: XGBoostPreparedMatrixData) -> None:
    LOGGER.debug(
        "XGBoost fit matrices prepared: train_shape=%s valid_shape=%s "
        "categorical_features=%s numeric_features=%s native_categorical=%s "
        "train_target_dtype=%s valid_target_dtype=%s train_target_c_contiguous=%s "
        "valid_target_c_contiguous=%s train_target_aligned=%s valid_target_aligned=%s "
        "train_target_nan=%s valid_target_nan=%s",
        matrices.train_x.shape,
        matrices.valid_x.shape,
        len(matrices.feature_spec.categorical_feature_cols),
        len(matrices.feature_spec.numeric_feature_cols),
        matrices.feature_spec.native_categorical,
        matrices.train_y.dtype,
        matrices.valid_y.dtype,
        bool(matrices.train_y.flags.c_contiguous),
        bool(matrices.valid_y.flags.c_contiguous),
        bool(matrices.train_y.flags.aligned),
        bool(matrices.valid_y.flags.aligned),
        int(np.isnan(matrices.train_y).sum()),
        int(np.isnan(matrices.valid_y).sum()),
    )


def _build_training_plan(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    train_sample_weight: np.ndarray | None = None,
) -> _XGBoostTrainingPlan:
    resolved_params = resolve_xgboost_model_params(
        model_params,
        default_params=default_params or DEFAULT_XGBOOST_MODEL_PARAMS,
    )
    native_categorical = bool(resolved_params.get("enable_categorical", False))
    use_early_stopping = (
        bool(resolved_params.get("enable_early_stopping", False))
        and len(valid_frame) > 0
    )
    LOGGER.debug(
        "XGBoost fit preparing matrices: train_rows=%s valid_rows=%s feature_count=%s "
        "target_col=%s native_categorical=%s use_early_stopping=%s params=%s sample_weight=%s",
        len(train_frame),
        len(valid_frame),
        len(feature_cols),
        target_col,
        native_categorical,
        use_early_stopping,
        _param_snapshot(resolved_params),
        train_sample_weight is not None,
    )
    matrices = prepare_xgboost_matrix_data(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_cols=feature_cols,
        target_col=target_col,
        train_sample_weight=train_sample_weight,
        native_categorical=native_categorical,
    )
    _log_matrix_summary(matrices)
    train_params = xgboost_train_params(resolved_params)
    num_boost_round = xgboost_num_boost_round(resolved_params)
    early_stopping_rounds = (
        xgboost_early_stopping_rounds(resolved_params) if use_early_stopping else None
    )
    LOGGER.debug(
        "XGBoost native train params resolved: num_boost_round=%s early_stopping_rounds=%s train_params=%s",
        num_boost_round,
        early_stopping_rounds,
        _param_snapshot(train_params),
    )
    return _XGBoostTrainingPlan(
        resolved_params=resolved_params,
        matrices=matrices,
        train_params=train_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        use_early_stopping=use_early_stopping,
    )


def _train_booster_with_validation_matrix(
    xgb_module: Any,
    plan: _XGBoostTrainingPlan,
    *,
    build_valid_matrix: bool = False,
) -> tuple[Any, Any | None]:
    train_matrix, valid_matrix = _build_native_matrices(
        xgb_module,
        plan,
        build_valid_matrix=build_valid_matrix,
    )
    return _train_booster_with_prebuilt_matrices(
        xgb_module,
        plan,
        train_matrix=train_matrix,
        valid_matrix=valid_matrix,
    )


def _build_native_matrices(
    xgb_module: Any,
    plan: _XGBoostTrainingPlan,
    *,
    build_valid_matrix: bool,
) -> tuple[Any, Any | None]:
    LOGGER.debug(
        "XGBoost native train matrix build starting: train_shape=%s categorical=%s",
        plan.matrices.train_x.shape,
        plan.matrices.feature_spec.native_categorical,
    )
    train_matrix = _build_dmatrix(
        xgb_module,
        plan.matrices.train_x,
        plan.matrices.feature_spec,
        params=plan.resolved_params,
        label=plan.matrices.train_y,
        weight=plan.matrices.train_sample_weight,
    )
    LOGGER.debug(
        "XGBoost native train matrix build completed: train_rows=%s",
        len(plan.matrices.train_x),
    )
    valid_matrix = None
    if plan.use_early_stopping or build_valid_matrix:
        LOGGER.debug(
            "XGBoost native validation matrix build starting: valid_shape=%s categorical=%s",
            plan.matrices.valid_x.shape,
            plan.matrices.feature_spec.native_categorical,
        )
        valid_matrix = _build_dmatrix(
            xgb_module,
            plan.matrices.valid_x,
            plan.matrices.feature_spec,
            params=plan.resolved_params,
            label=plan.matrices.valid_y,
            ref=train_matrix,
        )
        LOGGER.debug(
            "XGBoost native validation matrix build completed: valid_rows=%s",
            len(plan.matrices.valid_x),
        )
    return train_matrix, valid_matrix


def _train_booster_with_prebuilt_matrices(
    xgb_module: Any,
    plan: _XGBoostTrainingPlan,
    *,
    train_matrix: Any,
    valid_matrix: Any | None,
) -> tuple[Any, Any | None]:
    evals = [(valid_matrix, "validation")] if valid_matrix is not None else []
    LOGGER.debug(
        "XGBoost native fit starting: train_rows=%s valid_rows=%s eval_metric=%s early_stopping_rounds=%s matrix_type=%s device=%s",
        len(plan.matrices.train_x),
        len(plan.matrices.valid_x),
        plan.resolved_params.get("eval_metric"),
        plan.early_stopping_rounds,
        xgboost_matrix_type(plan.resolved_params),
        plan.resolved_params.get("device"),
    )
    model = xgb_module.train(
        params=plan.train_params,
        dtrain=train_matrix,
        num_boost_round=plan.num_boost_round,
        evals=evals,
        early_stopping_rounds=plan.early_stopping_rounds,
        verbose_eval=False,
    )
    return model, valid_matrix


def _train_booster(xgb_module: Any, plan: _XGBoostTrainingPlan) -> Any:
    model, _ = _train_booster_with_validation_matrix(xgb_module, plan)
    return model


def _prepared_training_plan(
    prepared_matrices: XGBoostPreparedTrainingMatrices,
    *,
    model_params: dict[str, object] | None,
    default_params: dict[str, object] | None,
) -> _XGBoostTrainingPlan:
    resolved_params = resolve_xgboost_model_params(
        model_params,
        default_params=default_params or prepared_matrices.resolved_params,
    )
    _assert_prepared_matrix_params_compatible(
        prepared_matrices.resolved_params,
        resolved_params,
    )
    use_early_stopping = (
        bool(resolved_params.get("enable_early_stopping", False))
        and prepared_matrices.valid_matrix is not None
    )
    return _XGBoostTrainingPlan(
        resolved_params=resolved_params,
        matrices=prepared_matrices.matrices,
        train_params=xgboost_train_params(resolved_params),
        num_boost_round=xgboost_num_boost_round(resolved_params),
        early_stopping_rounds=xgboost_early_stopping_rounds(resolved_params)
        if use_early_stopping
        else None,
        use_early_stopping=use_early_stopping,
    )


def _assert_prepared_matrix_params_compatible(
    prepared_params: dict[str, object],
    resolved_params: dict[str, object],
) -> None:
    for key in (
        "enable_categorical",
        "enable_dataset_sample_weight",
        "max_bin",
        "xgboost_matrix_type",
        "xgboost_gpu_input_backend",
    ):
        if _matrix_param_value(prepared_params, key) != _matrix_param_value(
            resolved_params,
            key,
        ):
            raise ValueError(
                "Cannot reuse prepared XGBoost matrices because parameter "
                f"`{key}` changed between cache build and trial fit."
            )


def _matrix_param_value(params: dict[str, object], key: str) -> object:
    value = params.get(key)
    if key == "max_bin" and value is not None:
        return int(cast(Any, value))
    if key in {"enable_categorical", "enable_dataset_sample_weight"}:
        return bool(value)
    if key == "xgboost_matrix_type":
        return str(value)
    if key == "xgboost_gpu_input_backend":
        return str(value)
    return value


def _predict_booster(
    *,
    model: Any,
    matrix: Any,
) -> np.ndarray:
    iteration_range = _prediction_iteration_range(model)
    if iteration_range is None:
        predictions = model.predict(matrix)
    else:
        predictions = model.predict(matrix, iteration_range=iteration_range)
    return np.asarray(predictions, dtype=float)


def _predict_booster_inplace(
    *,
    model: Any,
    data: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    iteration_range = _prediction_iteration_range(model)
    if iteration_range is None:
        predictions = model.inplace_predict(data, validate_features=False)
    else:
        predictions = model.inplace_predict(
            data,
            iteration_range=iteration_range,
            validate_features=False,
        )
    return np.asarray(predictions, dtype=float)


def _prediction_params(fitted_model: FittedXGBoostModel) -> dict[str, object]:
    params = dict(fitted_model.params)
    if not xgboost_uses_cuda(params) or xgboost_cuda_preflight_available():
        return params
    LOGGER.warning(
        "XGBoost CUDA prediction requested but CUDA preflight is unavailable; "
        "switching booster prediction to CPU."
    )
    set_param = getattr(fitted_model.model, "set_param", None)
    if callable(set_param):
        set_param({"device": "cpu"})
    params["runtime_profile"] = "local_cpu"
    params["device"] = "cpu"
    params["xgboost_matrix_type"] = "dmatrix"
    return params


def fit_xgboost_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    train_sample_weight: np.ndarray | None = None,
) -> FittedXGBoostModel:
    fit_start = time.perf_counter()
    plan = _build_training_plan(
        train_frame,
        valid_frame,
        feature_cols,
        target_col=target_col,
        model_params=model_params,
        default_params=default_params,
        train_sample_weight=train_sample_weight,
    )
    LOGGER.debug("XGBoost native booster training about to start.")
    model = _train_booster(_xgboost_module(), plan)
    LOGGER.debug(
        "XGBoost native fit completed: duration_seconds=%.3f best_iteration=%s best_score=%s",
        time.perf_counter() - fit_start,
        _model_attr(model, "best_iteration"),
        _model_attr(model, "best_score"),
    )
    return FittedXGBoostModel(
        model=model,
        feature_spec=plan.matrices.feature_spec,
        params=plan.resolved_params,
    )


def prepare_xgboost_training_matrices(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    train_sample_weight: np.ndarray | None = None,
) -> XGBoostPreparedTrainingMatrices:
    plan = _build_training_plan(
        train_frame,
        valid_frame,
        feature_cols,
        target_col=target_col,
        model_params=model_params,
        default_params=default_params,
        train_sample_weight=train_sample_weight,
    )
    train_matrix, valid_matrix = _build_native_matrices(
        _xgboost_module(),
        plan,
        build_valid_matrix=True,
    )
    return XGBoostPreparedTrainingMatrices(
        resolved_params=plan.resolved_params,
        matrices=plan.matrices,
        train_matrix=train_matrix,
        valid_matrix=valid_matrix,
    )


def fit_prepared_xgboost_matrices_and_predict_validation(
    prepared_matrices: XGBoostPreparedTrainingMatrices,
    *,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
) -> tuple[FittedXGBoostModel, np.ndarray]:
    fit_start = time.perf_counter()
    plan = _prepared_training_plan(
        prepared_matrices,
        model_params=model_params,
        default_params=default_params,
    )
    xgb_module = _xgboost_module()
    model, valid_matrix = _train_booster_with_prebuilt_matrices(
        xgb_module,
        plan,
        train_matrix=prepared_matrices.train_matrix,
        valid_matrix=prepared_matrices.valid_matrix,
    )
    if valid_matrix is None:
        valid_matrix = _build_dmatrix(
            xgb_module,
            plan.matrices.valid_x,
            plan.matrices.feature_spec,
            params=plan.resolved_params,
            label=plan.matrices.valid_y,
        )
    predictions = _predict_booster(model=model, matrix=valid_matrix)
    LOGGER.debug(
        "XGBoost prebuilt fit+validation predict completed: rows=%s duration_seconds=%.3f best_iteration=%s best_score=%s",
        len(predictions),
        time.perf_counter() - fit_start,
        _model_attr(model, "best_iteration"),
        _model_attr(model, "best_score"),
    )
    return (
        FittedXGBoostModel(
            model=model,
            feature_spec=plan.matrices.feature_spec,
            params=plan.resolved_params,
        ),
        predictions,
    )


def fit_xgboost_model_and_predict_validation(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    model_params: dict[str, object] | None = None,
    default_params: dict[str, object] | None = None,
    train_sample_weight: np.ndarray | None = None,
) -> tuple[FittedXGBoostModel, np.ndarray]:
    fit_start = time.perf_counter()
    plan = _build_training_plan(
        train_frame,
        valid_frame,
        feature_cols,
        target_col=target_col,
        model_params=model_params,
        default_params=default_params,
        train_sample_weight=train_sample_weight,
    )
    LOGGER.debug(
        "XGBoost native booster training with validation prediction about to start."
    )
    xgb_module = _xgboost_module()
    model, valid_matrix = _train_booster_with_validation_matrix(xgb_module, plan)
    if valid_matrix is None:
        valid_matrix = _build_dmatrix(
            xgb_module,
            plan.matrices.valid_x,
            plan.matrices.feature_spec,
            params=plan.resolved_params,
            label=plan.matrices.valid_y,
        )
    predictions = _predict_booster(model=model, matrix=valid_matrix)
    LOGGER.debug(
        "XGBoost native fit+validation predict completed: rows=%s duration_seconds=%.3f best_iteration=%s best_score=%s",
        len(predictions),
        time.perf_counter() - fit_start,
        _model_attr(model, "best_iteration"),
        _model_attr(model, "best_score"),
    )
    return (
        FittedXGBoostModel(
            model=model,
            feature_spec=plan.matrices.feature_spec,
            params=plan.resolved_params,
        ),
        predictions,
    )


def predict_with_xgboost_model(
    fitted_model: FittedXGBoostModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    predict_start = time.perf_counter()
    prediction_params = _prediction_params(fitted_model)
    LOGGER.debug(
        "XGBoost prediction preparing features: rows=%s feature_count=%s",
        len(frame),
        len(fitted_model.feature_spec.feature_cols),
    )
    features = transform_xgboost_features(frame, fitted_model.feature_spec)
    LOGGER.debug(
        "XGBoost native prediction starting: feature_shape=%s iteration_range=%s native_categorical=%s device=%s",
        features.shape,
        _prediction_iteration_range(fitted_model.model),
        fitted_model.feature_spec.native_categorical,
        prediction_params.get("device"),
    )
    if not fitted_model.feature_spec.native_categorical and not xgboost_uses_cuda(
        prediction_params
    ):
        predictions = _predict_booster_inplace(
            model=fitted_model.model,
            data=to_xgboost_feature_data(features, fitted_model.feature_spec),
        )
    else:
        matrix = _build_dmatrix(
            _xgboost_module(),
            features,
            fitted_model.feature_spec,
            params=prediction_params,
        )
        predictions = _predict_booster(model=fitted_model.model, matrix=matrix)
    LOGGER.debug(
        "XGBoost native prediction completed: rows=%s duration_seconds=%.3f",
        len(predictions),
        time.perf_counter() - predict_start,
    )
    return predictions
