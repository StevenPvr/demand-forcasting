from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from importlib import import_module
import logging
import math
import resource
import sys
from tempfile import TemporaryDirectory, mkdtemp
from threading import Lock
import time
from typing import Any, Mapping, cast


LOGGER = logging.getLogger(__name__)
_CPU_INTEROP_THREADS_LOCK = Lock()
_cpu_interop_threads_configured = False
_VISIBLE_VALIDATION_METRIC_ORDER: tuple[str, ...] = (
    "business_val_wape",
    "business_val_abs_bias",
    "val_wape",
    "val_loss",
    "train_loss_epoch",
)
_VISIBLE_VALIDATION_METRIC_LABELS: dict[str, str] = {
    "business_val_wape": "last_business_val_wape",
    "business_val_abs_bias": "last_business_val_abs_bias",
    "val_wape": "last_model_space_val_wape",
    "val_loss": "last_model_space_val_quantile_loss",
    "train_loss_epoch": "last_model_space_train_quantile_loss",
}
_LIVE_PROGRESS_METRIC_CANDIDATES: tuple[str, ...] = ("train_loss_step", "loss")
_LIVE_PROGRESS_METRIC_LABEL = "live_model_space_train_quantile_loss"
_HIDDEN_PROGRESS_BAR_METRICS: tuple[str, ...] = (
    "val_wape",
    "val_loss",
    "train_loss_epoch",
    "train_loss_step",
    "loss",
)


def _callable_int_value(func: Any, *, default: int) -> int:
    if not callable(func):
        return default
    return int(cast(Any, func()))


def _configure_cpu_parallelism(
    imports: dict[str, Any], resolved_params: dict[str, object]
) -> None:
    if str(cast(Any, resolved_params.get("accelerator", "cpu"))) != "cpu":
        return
    torch = imports["torch"]
    requested_threads = max(1, int(cast(Any, resolved_params.get("n_jobs", 1))))
    set_num_threads = getattr(torch, "set_num_threads", None)
    get_num_threads = getattr(torch, "get_num_threads", None)
    if callable(set_num_threads):
        set_num_threads(requested_threads)
    resolved_threads = _callable_int_value(get_num_threads, default=requested_threads)
    requested_interop_threads = max(1, min(4, requested_threads))
    set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
    get_num_interop_threads = getattr(torch, "get_num_interop_threads", None)
    global _cpu_interop_threads_configured
    with _CPU_INTEROP_THREADS_LOCK:
        if not _cpu_interop_threads_configured and callable(set_num_interop_threads):
            try:
                set_num_interop_threads(requested_interop_threads)
            except RuntimeError:
                pass
            _cpu_interop_threads_configured = True
    resolved_interop_threads = _callable_int_value(
        get_num_interop_threads,
        default=requested_interop_threads,
    )
    LOGGER.info(
        "Configured TFT CPU runtime threading: n_jobs=%s torch_num_threads=%s torch_num_interop_threads=%s dataloader_num_workers=%s",
        requested_threads,
        resolved_threads,
        resolved_interop_threads,
        int(cast(Any, resolved_params.get("num_workers", 0))),
    )


def _configure_gpu_parallelism(
    imports: dict[str, Any], resolved_params: dict[str, object]
) -> None:
    if str(cast(Any, resolved_params.get("accelerator", "cpu"))) != "gpu":
        return
    torch = imports["torch"]
    cuda_backends = getattr(torch.backends, "cuda", None)
    cuda_matmul = (
        getattr(cuda_backends, "matmul", None) if cuda_backends is not None else None
    )
    if cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
        setattr(cuda_matmul, "allow_tf32", True)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        setattr(cudnn_backend, "allow_tf32", True)
    determinism_mode = str(cast(Any, resolved_params.get("determinism_mode", "strict")))
    benchmark_enabled = determinism_mode != "strict"
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        setattr(cudnn_backend, "benchmark", benchmark_enabled)
    LOGGER.info(
        "Configured TFT GPU runtime acceleration: precision=%s determinism_mode=%s matmul_precision=%s allow_tf32=%s cudnn_benchmark=%s dataloader_num_workers=%s prefetch_factor=%s pin_memory=%s",
        str(cast(Any, resolved_params.get("precision", "32-true"))),
        str(cast(Any, resolved_params.get("determinism_mode", "strict"))),
        str(cast(Any, resolved_params.get("matmul_precision", "high"))),
        True,
        benchmark_enabled,
        int(cast(Any, resolved_params.get("num_workers", 0))),
        resolved_params.get("prefetch_factor"),
        bool(cast(Any, resolved_params.get("pin_memory", False))),
    )


def seed_tft_runtime(
    imports: dict[str, Any], resolved_params: dict[str, object]
) -> None:
    random_state = int(cast(Any, resolved_params.get("random_state", 7)))
    imports["seed_everything"](random_state, workers=True)
    determinism_mode = str(cast(Any, resolved_params.get("determinism_mode", "strict")))
    if determinism_mode != "off":
        warn_only = determinism_mode == "warn_only"
        imports["torch"].use_deterministic_algorithms(True, warn_only=warn_only)
    set_matmul_precision = getattr(
        imports["torch"], "set_float32_matmul_precision", None
    )
    if callable(set_matmul_precision):
        set_matmul_precision(
            str(cast(Any, resolved_params.get("matmul_precision", "highest")))
        )
    _configure_cpu_parallelism(imports, resolved_params)
    _configure_gpu_parallelism(imports, resolved_params)


def _trainer_benchmark_enabled(resolved_params: dict[str, object]) -> bool:
    return (
        str(cast(Any, resolved_params.get("accelerator", "cpu"))) == "gpu"
        and str(cast(Any, resolved_params.get("determinism_mode", "strict")))
        != "strict"
    )


def _training_logger_root(resolved_params: dict[str, object]) -> str:
    tensorboard_logdir = resolved_params.get("tensorboard_logdir")
    if isinstance(tensorboard_logdir, str) and tensorboard_logdir:
        return tensorboard_logdir
    return mkdtemp(prefix="praedixa-tft-logs-")


def _metric_float_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(cast(Any, item)())
        except (TypeError, ValueError):
            return None
    return None


def _visible_validation_metrics(metrics: Mapping[str, object]) -> dict[str, str]:
    formatted: dict[str, str] = {}
    for metric_name in _VISIBLE_VALIDATION_METRIC_ORDER:
        metric_value = _metric_float_value(metrics.get(metric_name))
        if metric_value is None:
            continue
        formatted[_VISIBLE_VALIDATION_METRIC_LABELS[metric_name]] = (
            f"{metric_value:.6f}"
            if metric_name in {"business_val_wape", "business_val_abs_bias", "val_wape"}
            else f"{metric_value:.6g}"
        )
    return formatted


def _live_progress_metrics(metrics: Mapping[str, object]) -> dict[str, str]:
    for metric_name in _LIVE_PROGRESS_METRIC_CANDIDATES:
        metric_value = _metric_float_value(metrics.get(metric_name))
        if metric_value is None:
            continue
        return {_LIVE_PROGRESS_METRIC_LABEL: f"{metric_value:.6g}"}
    return {}


def _build_loggers(
    imports: dict[str, Any], resolved_params: dict[str, object]
) -> tuple[list[Any], dict[str, str | None]]:
    loggers: list[Any] = []
    logger_root: str | None = None
    if bool(cast(Any, resolved_params.get("enable_csv_logger", True))):
        logger_root = _training_logger_root(resolved_params)
        loggers.append(
            imports["CSVLogger"](save_dir=logger_root, name="csv", version=None),
        )
    tensorboard_logdir = resolved_params.get("tensorboard_logdir")
    if isinstance(tensorboard_logdir, str) and tensorboard_logdir:
        loggers.append(
            imports["TensorBoardLogger"](
                save_dir=tensorboard_logdir,
                name="tensorboard",
                version=None,
            )
        )
    return loggers, {
        "csv_log_dir": logger_root,
        "tensorboard_log_dir": tensorboard_logdir
        if isinstance(tensorboard_logdir, str)
        else None,
    }


def _validation_metrics_logging_callback(imports: dict[str, Any]) -> Any:
    callback_base: type[Any] = cast(type[Any], imports["Callback"])

    class _ValidationMetricsLoggingCallback(callback_base):
        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            _ = pl_module
            visible_metrics = _visible_validation_metrics(
                cast(Mapping[str, object], getattr(trainer, "callback_metrics", {}))
            )
            if not visible_metrics:
                return
            metrics_summary = " ".join(
                f"{metric_name}={metric_value}"
                for metric_name, metric_value in visible_metrics.items()
            )
            LOGGER.info(
                "TFT validation metrics: epoch=%s %s",
                int(getattr(trainer, "current_epoch", 0)),
                metrics_summary,
            )

    return _ValidationMetricsLoggingCallback()


def _progress_bar_callback(
    imports: dict[str, Any], resolved_params: dict[str, object]
) -> Any | None:
    if not bool(cast(Any, resolved_params.get("enable_progress_bar", True))):
        return None
    progress_bar_base: type[Any] = cast(type[Any], imports["TQDMProgressBar"])

    class _VisibleMetricsProgressBar(progress_bar_base):
        def get_metrics(self, trainer: Any, model: Any) -> dict[str, object]:
            progress_bar_super: Any = super()
            raw_metrics = dict(
                cast(
                    Mapping[str, object], progress_bar_super.get_metrics(trainer, model)
                )
            )
            metrics = dict(raw_metrics)
            for hidden_metric in _HIDDEN_PROGRESS_BAR_METRICS:
                metrics.pop(hidden_metric, None)
            for metric_name, metric_value in _live_progress_metrics(raw_metrics).items():
                metrics[metric_name] = metric_value
            for metric_name, metric_value in _visible_validation_metrics(
                cast(Mapping[str, object], getattr(trainer, "callback_metrics", {}))
            ).items():
                metrics[metric_name] = metric_value
            return metrics

    return _VisibleMetricsProgressBar(
        refresh_rate=max(
            1, int(cast(Any, resolved_params.get("progress_bar_refresh_rate", 1)))
        ),
    )


def _early_stopping_callbacks(
    imports: dict[str, Any],
    *,
    resolved_params: dict[str, object],
) -> list[Any]:
    monitor_metric = _validation_monitor_metric(resolved_params)
    callbacks = [
        imports["EarlyStopping"](
            monitor=monitor_metric,
            mode="min",
            patience=int(cast(Any, resolved_params["patience"])),
            strict=True,
            check_finite=True,
        )
    ]
    loss_patience = int(
        cast(Any, resolved_params.get("loss_patience", resolved_params["patience"]))
    )
    if loss_patience > 0:
        callbacks.append(
            imports["EarlyStopping"](
                monitor="val_loss",
                mode="min",
                patience=loss_patience,
                strict=True,
                check_finite=True,
            )
        )
    return callbacks


def _validation_monitor_metric(resolved_params: dict[str, object]) -> str:
    monitor_metric = str(
        cast(Any, resolved_params.get("validation_monitor_metric", "business_val_wape"))
    )
    if monitor_metric == "business_val_wape" and not bool(
        cast(Any, resolved_params.get("enable_business_validation_metrics", True))
    ):
        return "val_wape"
    return monitor_metric


def _validation_callbacks(
    imports: dict[str, Any],
    *,
    checkpoint_dir: str,
    resolved_params: dict[str, object],
) -> tuple[list[Any], Any]:
    monitor_metric = _validation_monitor_metric(resolved_params)
    checkpoint_callback = imports["ModelCheckpoint"](
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    return [
        *_early_stopping_callbacks(imports, resolved_params=resolved_params),
        checkpoint_callback,
    ], checkpoint_callback


def _trainer_deterministic_setting(
    resolved_params: dict[str, object],
) -> str | bool:
    determinism_mode = str(cast(Any, resolved_params.get("determinism_mode", "strict")))
    if determinism_mode == "warn_only":
        return "warn"
    if determinism_mode == "off":
        return False
    return True


def _should_run_learning_rate_finder(
    resolved_params: dict[str, object], valid_loader: Any
) -> bool:
    if not bool(cast(Any, resolved_params.get("use_learning_rate_finder", False))):
        return False
    if valid_loader is not None:
        return True
    LOGGER.info(
        "Skipping TFT learning rate finder because no validation loader is available."
    )
    return False


def _build_lr_finder_trainer(
    imports: dict[str, Any], resolved_params: dict[str, object]
) -> Any:
    return imports["Trainer"](
        accelerator=str(cast(Any, resolved_params["accelerator"])),
        devices=int(cast(Any, resolved_params["devices"])),
        precision=str(cast(Any, resolved_params["precision"])),
        max_epochs=int(cast(Any, resolved_params["max_epochs"])),
        deterministic=_trainer_deterministic_setting(resolved_params),
        benchmark=_trainer_benchmark_enabled(resolved_params),
        gradient_clip_val=float(cast(Any, resolved_params["gradient_clip_val"])),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )


def _weights_only_safe_globals(imports: dict[str, Any]) -> list[object]:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from torchmetrics.metric import jit_distributed_available
    from torchmetrics.utilities.data import dim_zero_sum

    numpy_multiarray = import_module("numpy._core.multiarray")
    pandas_internals = import_module("pandas._libs.internals")
    pandas_indexes_base = import_module("pandas.core.indexes.base")
    pandas_block_manager = import_module("pandas.core.internals.managers")
    pyarrow_lib = import_module("pyarrow.lib")

    safe_globals: list[object | None] = [
        imports.get("GroupNormalizer"),
        imports.get("NaNLabelEncoder"),
        imports.get("QuantileLoss"),
        imports.get("ModuleList"),
        imports.get("WAPEMetric"),
        StandardScaler,
        pd.DataFrame,
        pd.Index,
        pd.StringDtype,
        pd.arrays.ArrowStringArray,
        getattr(pandas_internals, "_unpickle_block", None),
        getattr(pandas_indexes_base, "_new_Index", None),
        getattr(pandas_block_manager, "BlockManager", None),
        np.int32,
        np.ndarray,
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("float64")),
        type(np.dtype("int32")),
        type(np.dtype("int64")),
        type(np.dtype("bool")),
        type(np.dtype("str")),
        type(np.dtype("O")),
        getattr(numpy_multiarray, "_reconstruct", None),
        getattr(numpy_multiarray, "scalar", None),
        OrderedDict,
        slice,
        jit_distributed_available,
        dim_zero_sum,
        getattr(pyarrow_lib, "type_for_alias", None),
        getattr(pyarrow_lib, "py_buffer", None),
        getattr(pyarrow_lib, "_restore_array", None),
    ]
    return [safe_global for safe_global in safe_globals if safe_global is not None]


def _lr_finder_safe_globals_context(imports: dict[str, Any]) -> Any:
    torch_module = imports.get("torch")
    serialization = getattr(torch_module, "serialization", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    if not callable(safe_globals):
        return nullcontext()
    return safe_globals(_weights_only_safe_globals(imports))


def _finite_numeric_value(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _lr_finder_results(lr_finder: object) -> Mapping[str, object] | None:
    raw_results = getattr(lr_finder, "results", None)
    if not isinstance(raw_results, Mapping):
        return None
    return cast(Mapping[str, object], raw_results)


def _lr_finder_values(results: Mapping[str, object], key: str) -> tuple[object, ...]:
    raw_values = results.get(key)
    if not isinstance(raw_values, (list, tuple)):
        return ()
    return tuple(cast(list[object] | tuple[object, ...], raw_values))


def _lr_finder_result_suggestion(lr_finder: object) -> float | None:
    results = _lr_finder_results(lr_finder)
    if results is None:
        return None
    finite_pairs: list[tuple[float, float]] = []
    for loss_value, lr_value in zip(
        _lr_finder_values(results, "loss"),
        _lr_finder_values(results, "lr"),
        strict=False,
    ):
        numeric_loss = _finite_numeric_value(loss_value)
        numeric_lr = _finite_numeric_value(lr_value)
        if numeric_loss is None or numeric_lr is None:
            continue
        finite_pairs.append((numeric_loss, numeric_lr))
    if not finite_pairs:
        return None
    _, suggested_learning_rate = min(finite_pairs, key=lambda pair: pair[0])
    return suggested_learning_rate


def _extract_lr_finder_suggestion(lr_finder: object) -> float | None:
    suggestion = getattr(lr_finder, "suggestion", None)
    if callable(suggestion):
        numeric_suggestion = _finite_numeric_value(cast(Any, suggestion)())
        if numeric_suggestion is not None:
            return numeric_suggestion
    return _lr_finder_result_suggestion(lr_finder)


def find_learning_rate(
    imports: dict[str, Any],
    *,
    model: Any,
    resolved_params: dict[str, object],
    train_loader: Any,
    valid_loader: Any,
) -> float:
    current_learning_rate = float(cast(Any, resolved_params["learning_rate"]))
    if not _should_run_learning_rate_finder(resolved_params, valid_loader):
        return current_learning_rate
    tuner_trainer = _build_lr_finder_trainer(imports, resolved_params)
    with _lr_finder_safe_globals_context(imports):
        lr_finder = imports["Tuner"](tuner_trainer).lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            min_lr=float(cast(Any, resolved_params["lr_find_min_lr"])),
            max_lr=float(cast(Any, resolved_params["lr_find_max_lr"])),
            num_training=int(cast(Any, resolved_params["lr_find_num_training"])),
            early_stop_threshold=float(
                cast(Any, resolved_params["lr_find_early_stop_threshold"])
            ),
            update_attr=False,
        )
    if lr_finder is None:
        return current_learning_rate
    suggested_learning_rate = _extract_lr_finder_suggestion(lr_finder)
    if suggested_learning_rate is None or suggested_learning_rate <= 0.0:
        LOGGER.warning(
            "TFT learning rate finder did not return a usable suggestion; falling back to configured learning_rate=%s",
            current_learning_rate,
        )
        return current_learning_rate
    LOGGER.info(
        "TFT learning rate finder selected learning_rate=%s",
        suggested_learning_rate,
    )
    return suggested_learning_rate


def build_trainer(
    imports: dict[str, Any],
    resolved_params: dict[str, object],
    *,
    checkpoint_dir: str | None,
    has_validation: bool,
    extra_callbacks: list[Any] | None = None,
) -> tuple[Any, Any | None, dict[str, str | None]]:
    loggers, logger_paths = _build_loggers(imports, resolved_params)
    callbacks: list[Any] = list(extra_callbacks or [])
    if bool(cast(Any, resolved_params.get("enable_lr_monitor", True))):
        callbacks.append(
            imports["LearningRateMonitor"](
                logging_interval="epoch", log_weight_decay=True
            )
        )
    if bool(
        cast(Any, resolved_params.get("enable_validation_metric_logging", True))
    ):
        callbacks.append(_validation_metrics_logging_callback(imports))
    progress_bar = _progress_bar_callback(imports, resolved_params)
    if progress_bar is not None:
        callbacks.append(progress_bar)
    if str(cast(Any, resolved_params["accelerator"])) == "gpu" and bool(
        cast(Any, resolved_params.get("enable_device_stats_monitor", True))
    ):
        callbacks.append(imports["DeviceStatsMonitor"]())
    checkpoint_callback = None
    if has_validation and checkpoint_dir is not None:
        validation_callbacks, checkpoint_callback = _validation_callbacks(
            imports,
            checkpoint_dir=checkpoint_dir,
            resolved_params=resolved_params,
        )
        callbacks.extend(validation_callbacks)
    trainer = imports["Trainer"](
        accelerator=str(cast(Any, resolved_params["accelerator"])),
        devices=int(cast(Any, resolved_params["devices"])),
        precision=str(cast(Any, resolved_params["precision"])),
        max_epochs=int(cast(Any, resolved_params["max_epochs"])),
        gradient_clip_val=float(cast(Any, resolved_params["gradient_clip_val"])),
        deterministic=_trainer_deterministic_setting(resolved_params),
        benchmark=_trainer_benchmark_enabled(resolved_params),
        enable_checkpointing=bool(checkpoint_callback),
        enable_progress_bar=bool(
            cast(Any, resolved_params.get("enable_progress_bar", True))
        ),
        enable_model_summary=False,
        logger=loggers or False,
        log_every_n_steps=max(
            1, int(cast(Any, resolved_params.get("log_every_n_steps", 1)))
        ),
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    return trainer, checkpoint_callback, logger_paths


def maybe_compile_model(
    imports: dict[str, Any], model: Any, resolved_params: dict[str, object]
) -> tuple[Any, bool]:
    compile_mode = str(cast(Any, resolved_params.get("compile_mode", "off")))
    if compile_mode == "off":
        return model, False
    torch_compile = getattr(imports["torch"], "compile", None)
    if not callable(torch_compile):
        return model, False
    return torch_compile(model, mode=compile_mode), True


def unwrap_compiled_model(model: Any) -> Any:
    return getattr(model, "_orig_mod", model)


def _fit_compiled_model(
    imports: dict[str, Any],
    *,
    compiled_model: Any,
    resolved_params: dict[str, object],
    train_loader: Any,
    valid_loader: Any,
    extra_callbacks: list[Any] | None = None,
) -> tuple[Any, int, dict[str, str | None]]:
    with TemporaryDirectory(prefix="tft-backend-") as checkpoint_dir:
        trainer, checkpoint_callback, logger_paths = build_trainer(
            imports,
            resolved_params,
            checkpoint_dir=checkpoint_dir,
            has_validation=valid_loader is not None,
            extra_callbacks=extra_callbacks,
        )
        trainer.fit(compiled_model, train_loader, valid_loader)
        final_model = unwrap_compiled_model(compiled_model)
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            final_model = imports["TemporalFusionTransformer"].load_from_checkpoint(
                checkpoint_callback.best_model_path
            )
        best_iteration = max(0, int(getattr(trainer, "current_epoch", 1)) - 1)
    return final_model, best_iteration, logger_paths


def _peak_ram_mb() -> float:
    peak_ram_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return (
        peak_ram_raw / (1024.0 * 1024.0)
        if sys.platform == "darwin"
        else peak_ram_raw / 1024.0
    )


def _runtime_metrics_payload(
    *,
    resolved_params: dict[str, object],
    fit_duration_seconds: float,
    best_iteration: int,
    rows_seen: int,
    peak_vram_bytes: int,
    compile_applied: bool,
    logger_paths: dict[str, str | None],
) -> dict[str, float | int | str | None]:
    epochs_completed = max(1, best_iteration + 1)
    return {
        "fit_duration_seconds": fit_duration_seconds,
        "epochs_completed": epochs_completed,
        "epoch_duration_seconds": fit_duration_seconds / float(epochs_completed),
        "rows_per_second": float(rows_seen) / fit_duration_seconds
        if fit_duration_seconds > 0
        else 0.0,
        "loader_worker_count": int(cast(Any, resolved_params["num_workers"])),
        "peak_ram_mb": _peak_ram_mb(),
        "peak_vram_bytes": peak_vram_bytes,
        "compile_mode": str(cast(Any, resolved_params.get("compile_mode", "off"))),
        "compile_applied": int(compile_applied),
        "csv_log_dir": logger_paths["csv_log_dir"],
        "tensorboard_log_dir": logger_paths["tensorboard_log_dir"],
    }


def fit_trainer_model(
    imports: dict[str, Any],
    *,
    model: Any,
    resolved_params: dict[str, object],
    train_loader: Any,
    valid_loader: Any,
    extra_callbacks: list[Any] | None = None,
) -> tuple[Any, int, dict[str, float | int | str | None]]:
    torch = imports["torch"]
    compiled_model, compile_applied = maybe_compile_model(
        imports, model, resolved_params
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    final_model, best_iteration, logger_paths = _fit_compiled_model(
        imports,
        compiled_model=compiled_model,
        resolved_params=resolved_params,
        train_loader=train_loader,
        valid_loader=valid_loader,
        extra_callbacks=extra_callbacks,
    )
    fit_duration_seconds = max(0.0, time.perf_counter() - fit_start)
    rows_seen = len(train_loader.dataset) * max(1, best_iteration + 1)
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    return (
        final_model,
        best_iteration,
        _runtime_metrics_payload(
            resolved_params=resolved_params,
            fit_duration_seconds=fit_duration_seconds,
            best_iteration=best_iteration,
            rows_seen=rows_seen,
            peak_vram_bytes=peak_vram_bytes,
            compile_applied=compile_applied,
            logger_paths=logger_paths,
        ),
    )
