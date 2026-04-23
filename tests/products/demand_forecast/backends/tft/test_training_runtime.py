from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
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
        benchmark_enabled = cast(Any, training_runtime_module)._trainer_benchmark_enabled(
            {"accelerator": "gpu", "determinism_mode": "warn_only"}
        )

        self.assertTrue(benchmark_enabled)

    def test_visible_validation_metrics_formats_wape_and_losses_for_logs(self) -> None:
        class _FakeScalar:
            def __init__(self, value: float) -> None:
                self._value = value

            def item(self) -> float:
                return self._value

        visible_metrics = cast(Any, training_runtime_module)._visible_validation_metrics(
            {
                "val_wape": _FakeScalar(0.123456789),
                "val_loss": _FakeScalar(0.212345678),
                "train_loss_epoch": _FakeScalar(0.000000865),
            }
        )

        self.assertEqual(
            visible_metrics,
            {
                "val_wape": "0.123457",
                "val_loss": "0.212346",
                "train_loss_epoch": "8.65e-07",
            },
        )


if __name__ == "__main__":
    unittest.main()
