from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.backends.tft.backend import (  # noqa: E402
    get_tft_backend_availability,
    raise_if_tft_backend_required,
)


class TFTBackendTests(unittest.TestCase):
    def test_backend_availability_reports_ready(self) -> None:
        availability = get_tft_backend_availability()

        self.assertTrue(availability.is_ready)
        self.assertEqual(availability.backend_name, "TFT")
        self.assertIn("disponible", availability.reason.lower())

    def test_raise_if_tft_backend_required_is_noop_when_backend_is_ready(self) -> None:
        self.assertIsNone(raise_if_tft_backend_required("optimisation.build_optimisation_outputs"))


if __name__ == "__main__":
    unittest.main()
