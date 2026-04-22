from __future__ import annotations

import logging
import resource
import sys
from tempfile import TemporaryDirectory, mkdtemp
from threading import Lock
import time
from typing import Any, cast


LOGGER = logging.getLogger(__name__)
_CPU_INTEROP_THREADS_LOCK = Lock()
_cpu_interop_threads_configured = False


def _callable_int_value(func: Any, *, default: int) -> int:
    if not callable(func):
        return default
    return int(cast(Any, func()))


def _configure_cpu_parallelism(imports: dict[str, Any], resolved_params: dict[str, object]) -> None:
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


def _configure_gpu_parallelism(imports: dict[str, Any], resolved_params: dict[str, object]) -> None:
    if str(cast(Any, resolved_params.get("accelerator", "cpu"))) != "gpu":
        return
    torch = imports["torch"]
    cuda_backends = getattr(torch.backends, "cuda", None)
    cuda_matmul = getattr(cuda_backends, "matmul", None) if cuda_backends is not None else None
    if cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
        setattr(cuda_matmul, "allow_tf32", True)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        setattr(cudnn_backend, "allow_tf32", True)
    benchmark_enabled = str(cast(Any, resolved_params.get("determinism_mode", "strict"))) == "warn_only"
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        setattr(cudnn_backend, "benchmark", benchmark_enabled)
    LOGGER.info(
        "Configured TFT GPU runtime acceleration: precision=%s matmul_precision=%s allow_tf32=%s cudnn_benchmark=%s dataloader_num_workers=%s prefetch_factor=%s pin_memory=%s",
        str(cast(Any, resolved_params.get("precision", "32-true"))),
        str(cast(Any, resolved_params.get("matmul_precision", "high"))),
        True,
        benchmark_enabled,
        int(cast(Any, resolved_params.get("num_workers", 0))),
        resolved_params.get("prefetch_factor"),
        bool(cast(Any, resolved_params.get("pin_memory", False))),
    )


def seed_tft_runtime(imports: dict[str, Any], resolved_params: dict[str, object]) -> None:
    random_state = int(cast(Any, resolved_params.get("random_state", 7)))
    imports["seed_everything"](random_state, workers=True)
    warn_only = str(cast(Any, resolved_params.get("determinism_mode", "strict"))) == "warn_only"
    imports["torch"].use_deterministic_algorithms(True, warn_only=warn_only)
    set_matmul_precision = getattr(imports["torch"], "set_float32_matmul_precision", None)
    if callable(set_matmul_precision):
        set_matmul_precision(str(cast(Any, resolved_params.get("matmul_precision", "highest"))))
    _configure_cpu_parallelism(imports, resolved_params)
    _configure_gpu_parallelism(imports, resolved_params)


def _trainer_benchmark_enabled(resolved_params: dict[str, object]) -> bool:
    return (
        str(cast(Any, resolved_params.get("accelerator", "cpu"))) == "gpu"
        and str(cast(Any, resolved_params.get("determinism_mode", "strict"))) == "warn_only"
    )


def _training_logger_root(resolved_params: dict[str, object]) -> str:
    tensorboard_logdir = resolved_params.get("tensorboard_logdir")
    if isinstance(tensorboard_logdir, str) and tensorboard_logdir:
        return tensorboard_logdir
    return mkdtemp(prefix="praedixa-tft-logs-")


def _build_loggers(imports: dict[str, Any], resolved_params: dict[str, object]) -> tuple[list[Any], dict[str, str | None]]:
    logger_root = _training_logger_root(resolved_params)
    loggers: list[Any] = [
        imports["CSVLogger"](save_dir=logger_root, name="csv", version=None),
    ]
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
        "tensorboard_log_dir": tensorboard_logdir if isinstance(tensorboard_logdir, str) else None,
    }


def _progress_bar_callback(imports: dict[str, Any], resolved_params: dict[str, object]) -> Any | None:
    if not bool(cast(Any, resolved_params.get("enable_progress_bar", True))):
        return None
    return imports["TQDMProgressBar"](
        refresh_rate=max(1, int(cast(Any, resolved_params.get("progress_bar_refresh_rate", 1)))),
    )


def _validation_callbacks(
    imports: dict[str, Any],
    *,
    checkpoint_dir: str,
    patience: int,
) -> tuple[list[Any], Any]:
    early_stopping = imports["EarlyStopping"](
        monitor="val_wape",
        mode="min",
        patience=patience,
        strict=True,
        check_finite=True,
    )
    checkpoint_callback = imports["ModelCheckpoint"](
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{val_wape:.4f}",
        monitor="val_wape",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    return [early_stopping, checkpoint_callback], checkpoint_callback


def build_trainer(
    imports: dict[str, Any],
    resolved_params: dict[str, object],
    *,
    checkpoint_dir: str | None,
    has_validation: bool,
) -> tuple[Any, Any | None, dict[str, str | None]]:
    loggers, logger_paths = _build_loggers(imports, resolved_params)
    callbacks: list[Any] = [
        imports["LearningRateMonitor"](logging_interval="epoch", log_weight_decay=True),
    ]
    progress_bar = _progress_bar_callback(imports, resolved_params)
    if progress_bar is not None:
        callbacks.append(progress_bar)
    if str(cast(Any, resolved_params["accelerator"])) == "gpu":
        callbacks.append(imports["DeviceStatsMonitor"]())
    checkpoint_callback = None
    if has_validation and checkpoint_dir is not None:
        validation_callbacks, checkpoint_callback = _validation_callbacks(
            imports,
            checkpoint_dir=checkpoint_dir,
            patience=int(cast(Any, resolved_params["patience"])),
        )
        callbacks.extend(validation_callbacks)
    deterministic_mode = str(cast(Any, resolved_params.get("determinism_mode", "strict")))
    trainer = imports["Trainer"](
        accelerator=str(cast(Any, resolved_params["accelerator"])),
        devices=int(cast(Any, resolved_params["devices"])),
        precision=str(cast(Any, resolved_params["precision"])),
        max_epochs=int(cast(Any, resolved_params["max_epochs"])),
        gradient_clip_val=float(cast(Any, resolved_params["gradient_clip_val"])),
        deterministic="warn" if deterministic_mode == "warn_only" else True,
        benchmark=_trainer_benchmark_enabled(resolved_params),
        enable_checkpointing=bool(checkpoint_callback),
        enable_progress_bar=bool(cast(Any, resolved_params.get("enable_progress_bar", True))),
        enable_model_summary=False,
        logger=loggers,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    return trainer, checkpoint_callback, logger_paths


def maybe_compile_model(imports: dict[str, Any], model: Any, resolved_params: dict[str, object]) -> tuple[Any, bool]:
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
) -> tuple[Any, int, dict[str, str | None]]:
    with TemporaryDirectory(prefix="tft-backend-") as checkpoint_dir:
        trainer, checkpoint_callback, logger_paths = build_trainer(
            imports,
            resolved_params,
            checkpoint_dir=checkpoint_dir,
            has_validation=valid_loader is not None,
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
    return peak_ram_raw / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_ram_raw / 1024.0


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
        "rows_per_second": float(rows_seen) / fit_duration_seconds if fit_duration_seconds > 0 else 0.0,
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
) -> tuple[Any, int, dict[str, float | int | str | None]]:
    torch = imports["torch"]
    compiled_model, compile_applied = maybe_compile_model(imports, model, resolved_params)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    final_model, best_iteration, logger_paths = _fit_compiled_model(
        imports,
        compiled_model=compiled_model,
        resolved_params=resolved_params,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    fit_duration_seconds = max(0.0, time.perf_counter() - fit_start)
    rows_seen = len(train_loader.dataset) * max(1, best_iteration + 1)
    peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return final_model, best_iteration, _runtime_metrics_payload(
        resolved_params=resolved_params,
        fit_duration_seconds=fit_duration_seconds,
        best_iteration=best_iteration,
        rows_seen=rows_seen,
        peak_vram_bytes=peak_vram_bytes,
        compile_applied=compile_applied,
        logger_paths=logger_paths,
    )
