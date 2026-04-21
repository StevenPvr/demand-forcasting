from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.runtime_profile import (  # noqa: E402
    DEFAULT_RUNTIME_PROFILE_NAME,
    resolve_runtime_profile,
)


class TFTRuntimeProfileTests(unittest.TestCase):
    def test_resolve_runtime_profile_returns_local_cpu_defaults(self) -> None:
        profile = resolve_runtime_profile(DEFAULT_RUNTIME_PROFILE_NAME)

        self.assertEqual(profile["accelerator"], "cpu")
        self.assertEqual(profile["devices"], 1)
        self.assertEqual(profile["precision"], "32-true")
        self.assertEqual(profile["num_workers"], 0)
        self.assertFalse(profile["pin_memory"])
        self.assertFalse(profile["persistent_workers"])
        self.assertEqual(profile["compile_mode"], "off")
        self.assertEqual(profile["determinism_mode"], "strict")
        self.assertFalse(profile["torch_compile"])

    def test_resolve_runtime_profile_returns_scaleway_l40s_defaults(self) -> None:
        profile = resolve_runtime_profile("scaleway_l40s", bf16_supported=True)

        self.assertEqual(profile["accelerator"], "gpu")
        self.assertEqual(profile["devices"], 1)
        self.assertEqual(profile["precision"], "bf16-mixed")
        self.assertEqual(profile["num_workers"], 4)
        self.assertTrue(profile["pin_memory"])
        self.assertTrue(profile["persistent_workers"])
        self.assertEqual(profile["compile_mode"], "off")
        self.assertEqual(profile["determinism_mode"], "warn_only")
        self.assertEqual(profile["matmul_precision"], "high")

    def test_resolve_runtime_profile_returns_mac_metal_defaults(self) -> None:
        profile = resolve_runtime_profile("mac_metal")

        self.assertIn(profile["accelerator"], {"mps", "cpu"})
        self.assertEqual(profile["devices"], 1)
        self.assertEqual(profile["precision"], "32-true")
        self.assertEqual(profile["num_workers"], 0)
        self.assertFalse(profile["pin_memory"])
        self.assertFalse(profile["persistent_workers"])
        self.assertEqual(profile["compile_mode"], "off")
        self.assertEqual(profile["determinism_mode"], "warn_only")
        self.assertEqual(profile["matmul_precision"], "high")

    def test_resolve_runtime_profile_rejects_unknown_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_runtime_profile("unknown")


if __name__ == "__main__":
    unittest.main()
