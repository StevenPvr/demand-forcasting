from __future__ import annotations

import resource
import sys
from tempfile import TemporaryDirectory, mkdtemp
import time
from typing import Any, cast


def seed_tft_runtime(imports: dict[str, Any], resolved_params: dict[str, object]) -> None:
    random_state = int(cast(Any, resolved_params.get("random_state", 7)))
    imports["seed_everything"](random_state, workers=True)
    warn_only = str(cast(Any, resolved_params.get("determinism_mode", "strict"))) == "warn_only"
    imports["torch"].use_deterministic_algorithms(True, warn_only=warn_only)
    set_matmul_precision = getattr(imports["torch"], "set_float32_matmul_precision", None)
    if callable(set_matmul_precision):
        set_matmul_precision(str(cast(Any, resolved_params.get("matmul_precision", "highest"))))


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
    if str(cast(Any, resolved_params["accelerator"])) == "gpu":
        callbacks.append(imports["DeviceStatsMonitor"]())
    checkpoint_callback = None
    if has_validation and checkpoint_dir is not None:
        callbacks.append(
            imports["EarlyStopping"](
                monitor="val_loss",
                mode="min",
                patience=int(cast(Any, resolved_params["patience"])),
                strict=True,
                check_finite=True,
            )
        )
        checkpoint_callback = imports["ModelCheckpoint"](
            dirpath=checkpoint_dir,
            filename="{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    deterministic_mode = str(cast(Any, resolved_params.get("determinism_mode", "strict")))
    trainer = imports["Trainer"](
        accelerator=str(cast(Any, resolved_params["accelerator"])),
        devices=int(cast(Any, resolved_params["devices"])),
        precision=str(cast(Any, resolved_params["precision"])),
        max_epochs=int(cast(Any, resolved_params["max_epochs"])),
        gradient_clip_val=float(cast(Any, resolved_params["gradient_clip_val"])),
        deterministic="warn" if deterministic_mode == "warn_only" else True,
        benchmark=False,
        enable_checkpointing=bool(checkpoint_callback),
        enable_progress_bar=False,
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
            final_model = imports["TemporalFusionTransformer"].load_from_checkpoint(checkpoint_callback.best_model_path)
        best_iteration = max(0, int(getattr(trainer, "current_epoch", 1)) - 1)
    fit_duration_seconds = max(0.0, time.perf_counter() - fit_start)
    epochs_completed = max(1, best_iteration + 1)
    rows_seen = len(train_loader.dataset) * epochs_completed
    peak_ram_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_ram_mb = peak_ram_raw / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_ram_raw / 1024.0
    peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return final_model, best_iteration, {
        "fit_duration_seconds": fit_duration_seconds,
        "epochs_completed": epochs_completed,
        "epoch_duration_seconds": fit_duration_seconds / float(epochs_completed),
        "rows_per_second": float(rows_seen) / fit_duration_seconds if fit_duration_seconds > 0 else 0.0,
        "loader_worker_count": int(cast(Any, resolved_params["num_workers"])),
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_bytes": peak_vram_bytes,
        "compile_mode": str(cast(Any, resolved_params.get("compile_mode", "off"))),
        "compile_applied": int(compile_applied),
        "csv_log_dir": logger_paths["csv_log_dir"],
        "tensorboard_log_dir": logger_paths["tensorboard_log_dir"],
    }
