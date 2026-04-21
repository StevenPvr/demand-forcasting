from __future__ import annotations

from pathlib import Path
import runpy
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class EvaluationMainTests(unittest.TestCase):
    def test_main_exposes_explicit_tft_not_ready_error(self) -> None:
        from praedixa.demand_forecast.backends.tft.backend import (
            TFTBackendNotReadyError,
        )

        with (
            mock.patch(
                "praedixa.demand_forecast.evaluation.pipeline.build_evaluation_outputs",
                side_effect=TFTBackendNotReadyError("tft missing"),
            ),
            mock.patch.object(sys, "argv", ["evaluation.main"]),
            self.assertRaises(TFTBackendNotReadyError),
        ):
            runpy.run_module(
                "praedixa.demand_forecast.evaluation.main", run_name="__main__"
            )


if __name__ == "__main__":
    unittest.main()
