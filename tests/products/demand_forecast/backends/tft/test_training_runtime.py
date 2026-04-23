from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, cast
import unittest
from unittest.mock import patch


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
if str(PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC))
if str(PRODUCT_SRC) not in sys.path:
    sys.path.insert(0, str(PRODUCT_SRC))

import praedixa.demand_forecast.backends.tft.training_runtime as training_runtime_module  # noqa: E402
from praedixa.demand_forecast.backends.tft.training_runtime import seed_tft_runtime  # noqa: E402


class _FakeCudaMatmul:
    def __init__(self) -> None:
        self.allow_tf32 = False


class _FakeCudaBackend:
    def __init__(self) -> None:
        self.matmul = _FakeCudaMatmul()


class _FakeCuDNNBackend:
    def __init__(self) -> None:
        self.allow_tf32 = False
        self.benchmark = False


class _FakeBackends:
    def __init__(self) -> None:
        self.cuda = _FakeCudaBackend()
        self.cudnn = _FakeCuDNNBackend()


class _FakeTorch:
    def __init__(self) -> None:
        self.warn_only: bool | None = None
        self.matmul_precision: str | None = None
        self.num_threads = 1
        self.num_interop_threads = 1
        self.backends = _FakeBackends()

    def use_deterministic_algorithms(self, _: bool, *, warn_only: bool) -> None:
        self.warn_only = warn_only

    def set_float32_matmul_precision(self, precision: str) -> None:
        self.matmul_precision = precision

    def set_num_threads(self, count: int) -> None:
        self.num_threads = int(count)

    def get_num_threads(self) -> int:
        return self.num_threads

    def set_num_interop_threads(self, count: int) -> None:
        self.num_interop_threads = int(count)

    def get_num_interop_threads(self) -> int:
        return self.num_interop_threads


class _FakeLRFinder:
    def __init__(self, suggestion_value: float | None) -> None:
        self._suggestion_value = suggestion_value

    def suggestion(self) -> float | None:
        return self._suggestion_value


class _FakeLRFinderFromResults:
    def __init__(self) -> None:
        self.results: dict[str, list[float]] = {
            "loss": [1.2, 0.7, 0.9],
            "lr": [0.001, 0.004, 0.002],
        }

    def suggestion(self) -> None:
        return None


class _FakeSafeGlobalsContext:
    def __init__(self, owner: "_FakeSafeGlobals") -> None:
        self._owner = owner

    def __enter__(self) -> None:
        self._owner.entered += 1
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        _ = exc_type
        _ = exc
        _ = traceback
        self._owner.exited += 1
        return False


class _FakeSafeGlobals:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.entered = 0
        self.exited = 0

    def __call__(self, values: list[object]) -> _FakeSafeGlobalsContext:
        self.calls.append(tuple(values))
        return _FakeSafeGlobalsContext(self)


class TFTTrainingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        cast(Any, training_runtime_module)._cpu_interop_threads_configured = False

    def tearDown(self) -> None:
        cast(Any, training_runtime_module)._cpu_interop_threads_configured = False

    def test_seed_tft_runtime_configures_cpu_threads_from_n_jobs(self) -> None:
        fake_torch = _FakeTorch()
        seeded: list[tuple[int, bool]] = []

        def _fake_seed_everything(seed: int, workers: bool) -> None:
            seeded.append((int(seed), bool(workers)))

        imports: dict[str, Any] = {
            "torch": fake_torch,
            "seed_everything": _fake_seed_everything,
        }

        seed_tft_runtime(
            imports,
            {
                "accelerator": "cpu",
                "n_jobs": 6,
                "random_state": 13,
                "determinism_mode": "strict",
                "matmul_precision": "high",
                "num_workers": 4,
            },
        )

        self.assertEqual(seeded, [(13, True)])
        self.assertFalse(fake_torch.warn_only)
        self.assertEqual(fake_torch.matmul_precision, "high")
        self.assertEqual(fake_torch.num_threads, 6)
        self.assertEqual(fake_torch.num_interop_threads, 4)

    def test_seed_tft_runtime_configures_gpu_acceleration_flags(self) -> None:
        fake_torch = _FakeTorch()
        seeded: list[tuple[int, bool]] = []

        def _fake_seed_everything(seed: int, workers: bool) -> None:
            seeded.append((int(seed), bool(workers)))

        imports: dict[str, Any] = {
            "torch": fake_torch,
            "seed_everything": _fake_seed_everything,
        }

        seed_tft_runtime(
            imports,
            {
                "accelerator": "gpu",
                "random_state": 21,
                "determinism_mode": "warn_only",
                "matmul_precision": "high",
                "precision": "bf16-mixed",
                "num_workers": 8,
                "prefetch_factor": 4,
                "pin_memory": True,
            },
        )

        self.assertEqual(seeded, [(21, True)])
        self.assertTrue(fake_torch.warn_only)
        self.assertEqual(fake_torch.matmul_precision, "high")
        self.assertTrue(fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(fake_torch.backends.cudnn.allow_tf32)
        self.assertTrue(fake_torch.backends.cudnn.benchmark)

    def test_gpu_trainer_benchmark_is_enabled_in_warn_only_mode(self) -> None:
        benchmark_enabled = cast(
            Any, training_runtime_module
        )._trainer_benchmark_enabled(
            {"accelerator": "gpu", "determinism_mode": "warn_only"}
        )

        self.assertTrue(benchmark_enabled)

    def test_gpu_trainer_benchmark_is_enabled_when_determinism_is_off(self) -> None:
        benchmark_enabled = cast(
            Any, training_runtime_module
        )._trainer_benchmark_enabled(
            {"accelerator": "gpu", "determinism_mode": "off"}
        )

        self.assertTrue(benchmark_enabled)

    def test_seed_tft_runtime_skips_deterministic_algorithms_when_off(self) -> None:
        fake_torch = _FakeTorch()
        seeded: list[tuple[int, bool]] = []

        def _fake_seed_everything(seed: int, workers: bool) -> None:
            seeded.append((int(seed), bool(workers)))

        imports: dict[str, Any] = {
            "torch": fake_torch,
            "seed_everything": _fake_seed_everything,
        }

        seed_tft_runtime(
            imports,
            {
                "accelerator": "gpu",
                "random_state": 9,
                "determinism_mode": "off",
                "matmul_precision": "high",
                "precision": "bf16-mixed",
                "num_workers": 8,
                "prefetch_factor": 4,
                "pin_memory": True,
            },
        )

        self.assertEqual(seeded, [(9, True)])
        self.assertIsNone(fake_torch.warn_only)
        self.assertTrue(fake_torch.backends.cudnn.benchmark)

    def test_early_stopping_callbacks_monitor_wape_and_loss(self) -> None:
        callbacks = cast(Any, training_runtime_module)._early_stopping_callbacks(
            {
                "EarlyStopping": lambda **kwargs: kwargs,
            },
            resolved_params={"patience": 8, "loss_patience": 6},
        )

        self.assertEqual(
            [(callback["monitor"], callback["patience"]) for callback in callbacks],
            [("val_wape", 8), ("val_loss", 6)],
        )

    def test_find_learning_rate_uses_tuner_suggestion_when_enabled(self) -> None:
        tuner_calls: list[dict[str, object]] = []
        trainer_calls: list[dict[str, object]] = []

        class _FakeTuner:
            def __init__(self, trainer: object) -> None:
                _ = trainer

            def lr_find(self, model: object, **kwargs: object) -> _FakeLRFinder:
                _ = model
                tuner_calls.append(dict(kwargs))
                return _FakeLRFinder(0.0042)

        learning_rate = training_runtime_module.find_learning_rate(
            {
                "Trainer": lambda **kwargs: trainer_calls.append(dict(kwargs))
                or dict(kwargs),
                "Tuner": _FakeTuner,
            },
            model=object(),
            resolved_params={
                "use_learning_rate_finder": True,
                "learning_rate": 0.01,
                "max_epochs": 2,
                "lr_find_min_lr": 1e-5,
                "lr_find_max_lr": 1.0,
                "lr_find_num_training": 100,
                "lr_find_early_stop_threshold": 10000.0,
                "accelerator": "gpu",
                "devices": 1,
                "precision": "bf16-mixed",
                "determinism_mode": "warn_only",
                "gradient_clip_val": 0.1,
            },
            train_loader=object(),
            valid_loader=object(),
        )

        self.assertAlmostEqual(learning_rate, 0.0042)
        self.assertEqual(len(tuner_calls), 1)
        self.assertEqual(len(trainer_calls), 1)
        self.assertEqual(trainer_calls[0]["max_epochs"], 2)
        self.assertEqual(tuner_calls[0]["min_lr"], 1e-5)
        self.assertEqual(tuner_calls[0]["max_lr"], 1.0)

    def test_find_learning_rate_falls_back_to_results_mapping(self) -> None:
        class _FakeTuner:
            def __init__(self, trainer: object) -> None:
                _ = trainer

            def lr_find(
                self, model: object, **kwargs: object
            ) -> _FakeLRFinderFromResults:
                _ = model
                _ = kwargs
                return _FakeLRFinderFromResults()

        learning_rate = training_runtime_module.find_learning_rate(
            {
                "Trainer": lambda **kwargs: dict(kwargs),
                "Tuner": _FakeTuner,
            },
            model=object(),
            resolved_params={
                "use_learning_rate_finder": True,
                "learning_rate": 0.01,
                "max_epochs": 2,
                "lr_find_min_lr": 1e-5,
                "lr_find_max_lr": 1.0,
                "lr_find_num_training": 100,
                "lr_find_early_stop_threshold": 10000.0,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "determinism_mode": "strict",
                "gradient_clip_val": 0.1,
            },
            train_loader=object(),
            valid_loader=object(),
        )

        self.assertAlmostEqual(learning_rate, 0.004)

    def test_find_learning_rate_uses_safe_globals_context_when_available(self) -> None:
        safe_globals = _FakeSafeGlobals()

        class _FakeTuner:
            def __init__(self, trainer: object) -> None:
                _ = trainer

            def lr_find(self, model: object, **kwargs: object) -> _FakeLRFinder:
                _ = model
                _ = kwargs
                return _FakeLRFinder(0.0042)

        class _FakeGroupNormalizer:
            pass

        class _FakeNaNLabelEncoder:
            pass

        class _FakeQuantileLoss:
            pass

        class _FakeModuleList:
            pass

        class _FakeWAPEMetric:
            pass

        learning_rate = training_runtime_module.find_learning_rate(
            {
                "torch": SimpleNamespace(
                    serialization=SimpleNamespace(safe_globals=safe_globals)
                ),
                "Trainer": lambda **kwargs: dict(kwargs),
                "Tuner": _FakeTuner,
                "GroupNormalizer": _FakeGroupNormalizer,
                "NaNLabelEncoder": _FakeNaNLabelEncoder,
                "QuantileLoss": _FakeQuantileLoss,
                "ModuleList": _FakeModuleList,
                "WAPEMetric": _FakeWAPEMetric,
            },
            model=object(),
            resolved_params={
                "use_learning_rate_finder": True,
                "learning_rate": 0.01,
                "max_epochs": 2,
                "lr_find_min_lr": 1e-5,
                "lr_find_max_lr": 1.0,
                "lr_find_num_training": 100,
                "lr_find_early_stop_threshold": 10000.0,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "determinism_mode": "strict",
                "gradient_clip_val": 0.1,
            },
            train_loader=object(),
            valid_loader=object(),
        )

        self.assertAlmostEqual(learning_rate, 0.0042)
        self.assertEqual(len(safe_globals.calls), 1)
        self.assertEqual(safe_globals.entered, 1)
        self.assertEqual(safe_globals.exited, 1)
        safe_global_names = {getattr(value, "__name__", type(value).__name__) for value in safe_globals.calls[0]}
        self.assertIn("_FakeGroupNormalizer", safe_global_names)
        self.assertIn("_FakeNaNLabelEncoder", safe_global_names)
        self.assertIn("_FakeQuantileLoss", safe_global_names)
        self.assertIn("_FakeModuleList", safe_global_names)
        self.assertIn("_FakeWAPEMetric", safe_global_names)

    def test_visible_validation_metrics_formats_last_validation_metrics_for_logs(
        self,
    ) -> None:
        class _FakeScalar:
            def __init__(self, value: float) -> None:
                self._value = value

            def item(self) -> float:
                return self._value

        visible_metrics = cast(
            Any, training_runtime_module
        )._visible_validation_metrics(
            {
                "business_val_wape": _FakeScalar(0.083333333),
                "business_val_abs_bias": _FakeScalar(2.0),
                "val_wape": _FakeScalar(0.123456789),
                "val_loss": _FakeScalar(0.212345678),
                "train_loss_epoch": _FakeScalar(0.000000865),
                "train_loss_step": _FakeScalar(0.000000123),
            }
        )

        self.assertEqual(
            visible_metrics,
            {
                "last_business_val_wape": "0.083333",
                "last_business_val_abs_bias": "2.000000",
                "last_model_space_val_wape": "0.123457",
                "last_model_space_val_quantile_loss": "0.212346",
                "last_model_space_train_quantile_loss": "8.65e-07",
            },
        )

    def test_live_progress_metrics_formats_batch_level_train_loss(self) -> None:
        class _FakeScalar:
            def __init__(self, value: float) -> None:
                self._value = value

            def item(self) -> float:
                return self._value

        live_metrics = cast(Any, training_runtime_module)._live_progress_metrics(
            {"train_loss_step": _FakeScalar(0.004321)}
        )

        self.assertEqual(
            live_metrics,
            {"live_model_space_train_quantile_loss": "0.004321"},
        )

    def test_fit_trainer_model_forwards_extra_callbacks_to_compiled_fit(self) -> None:
        forwarded_callbacks: list[list[object] | None] = []

        def _fake_fit_compiled_model(
            imports: dict[str, Any],
            *,
            compiled_model: object,
            resolved_params: dict[str, object],
            train_loader: object,
            valid_loader: object,
            extra_callbacks: list[object] | None = None,
        ) -> tuple[object, int, dict[str, str | None]]:
            _ = imports
            _ = compiled_model
            _ = resolved_params
            _ = train_loader
            _ = valid_loader
            forwarded_callbacks.append(extra_callbacks)
            return object(), 1, {"csv_log_dir": None, "tensorboard_log_dir": None}

        extra_callbacks: list[object] = [object()]
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0,
            )
        )

        with patch.object(
            training_runtime_module,
            "_fit_compiled_model",
            side_effect=_fake_fit_compiled_model,
        ):
            _ = training_runtime_module.fit_trainer_model(
                {"torch": fake_torch},
                model=object(),
                resolved_params={
                    "compile_mode": "off",
                    "num_workers": 0,
                },
                train_loader=SimpleNamespace(dataset=[1, 2, 3]),
                valid_loader=None,
                extra_callbacks=extra_callbacks,
            )

        self.assertEqual(forwarded_callbacks, [extra_callbacks])

    def test_build_trainer_can_disable_batch_level_overhead_for_hpo(self) -> None:
        trainer_calls: list[dict[str, object]] = []

        trainer, checkpoint_callback, logger_paths = training_runtime_module.build_trainer(
            {
                "CSVLogger": lambda **kwargs: ("csv", kwargs),
                "TensorBoardLogger": lambda **kwargs: ("tb", kwargs),
                "LearningRateMonitor": lambda **kwargs: ("lr", kwargs),
                "DeviceStatsMonitor": lambda: "device-stats",
                "Callback": object,
                "TQDMProgressBar": object,
                "EarlyStopping": lambda **kwargs: ("es", kwargs),
                "ModelCheckpoint": lambda **kwargs: SimpleNamespace(
                    best_model_path="", **kwargs
                ),
                "Trainer": lambda **kwargs: trainer_calls.append(dict(kwargs))
                or SimpleNamespace(**kwargs),
            },
            {
                "accelerator": "gpu",
                "devices": 1,
                "precision": "bf16-mixed",
                "max_epochs": 2,
                "gradient_clip_val": 0.1,
                "determinism_mode": "off",
                "enable_progress_bar": False,
                "enable_csv_logger": False,
                "enable_lr_monitor": False,
                "enable_validation_metric_logging": False,
                "enable_device_stats_monitor": False,
                "log_every_n_steps": 50,
                "tensorboard_logdir": None,
            },
            checkpoint_dir=None,
            has_validation=False,
        )

        _ = trainer
        self.assertIsNone(checkpoint_callback)
        self.assertEqual(logger_paths["csv_log_dir"], None)
        self.assertEqual(logger_paths["tensorboard_log_dir"], None)
        self.assertEqual(len(trainer_calls), 1)
        self.assertFalse(bool(trainer_calls[0]["logger"]))
        self.assertEqual(int(cast(Any, trainer_calls[0]["log_every_n_steps"])), 50)
        self.assertFalse(bool(trainer_calls[0]["enable_progress_bar"]))
        self.assertEqual(cast(list[object], trainer_calls[0]["callbacks"]), [])


if __name__ == "__main__":
    unittest.main()
