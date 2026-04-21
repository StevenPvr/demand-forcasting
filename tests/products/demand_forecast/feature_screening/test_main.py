from __future__ import annotations

from pathlib import Path
import runpy
import sys
import unittest
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FeatureSelectionLagMainTests(unittest.TestCase):
    def test_main_logs_outputs_without_running_real_pipeline(self) -> None:
        fake_outputs = {
            "train_selected": PROJECT_ROOT
            / "var"
            / "experiments"
            / "demand_forecast"
            / "feature_screening"
            / "train_selection.parquet"
        }
        with (
            mock.patch(
                "praedixa.demand_forecast.feature_screening.pipeline.build_lag_selection_outputs",
                return_value=fake_outputs,
            ),
            mock.patch.object(sys, "argv", ["features_selection_lag.main"]),
        ):
            runpy.run_module("praedixa.demand_forecast.feature_screening.main", run_name="__main__")


if __name__ == "__main__":
    unittest.main()
