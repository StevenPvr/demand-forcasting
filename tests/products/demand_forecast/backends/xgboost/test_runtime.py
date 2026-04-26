from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.backends.xgboost.runtime import (  # noqa: E402
    resolve_xgboost_runtime_profile,
    xgboost_cuda_preflight_available,
    xgboost_matrix_type,
)


class XGBoostRuntimeTests(unittest.TestCase):
    def test_auto_profile_selects_l40s_when_xgboost_cuda_preflight_passes(
        self,
    ) -> None:
        with (
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.runtime.xgboost_cuda_preflight_available",
                return_value=True,
            ),
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.runtime._cuda_device_name",
                return_value="NVIDIA L40S",
            ),
        ):
            resolution = resolve_xgboost_runtime_profile("auto")

        self.assertEqual(resolution.runtime_profile, "scaleway_l40s")
        self.assertEqual(resolution.device, "cuda")
        self.assertEqual(resolution.accelerator, "gpu")
        self.assertTrue(resolution.cuda_available)

    def test_auto_profile_falls_back_to_cpu_when_preflight_fails(self) -> None:
        with mock.patch(
            "praedixa.demand_forecast.backends.xgboost.runtime.xgboost_cuda_preflight_available",
            return_value=False,
        ):
            resolution = resolve_xgboost_runtime_profile("auto")

        self.assertEqual(resolution.runtime_profile, "local_cpu")
        self.assertEqual(resolution.device, "cpu")
        self.assertFalse(resolution.cuda_available)
        self.assertIsNotNone(resolution.fallback_reason)

    def test_cuda_preflight_requires_visible_cuda_device(self) -> None:
        xgboost_cuda_preflight_available.cache_clear()
        with (
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.runtime._xgboost_module",
                return_value=object(),
            ),
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.runtime._cuda_device_name",
                return_value=None,
            ),
        ):
            self.assertFalse(xgboost_cuda_preflight_available())
        xgboost_cuda_preflight_available.cache_clear()

    def test_explicit_cuda_profile_fails_when_preflight_fails(self) -> None:
        with (
            mock.patch(
                "praedixa.demand_forecast.backends.xgboost.runtime.xgboost_cuda_preflight_available",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "requires a working XGBoost CUDA"),
        ):
            resolve_xgboost_runtime_profile("scaleway_l40s")

    def test_mac_metal_is_rejected_for_xgboost(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not support the `mac_metal`"):
            resolve_xgboost_runtime_profile("mac_metal")

    def test_matrix_type_uses_quantile_only_for_cuda(self) -> None:
        self.assertEqual(xgboost_matrix_type({"device": "cpu"}), "dmatrix")
        self.assertEqual(
            xgboost_matrix_type({"runtime_profile": "scaleway_l40s"}),
            "quantile",
        )


if __name__ == "__main__":
    unittest.main()
