from __future__ import annotations

from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from typing import Callable, cast
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import praedixa.demand_forecast.backends.moirai.model as moirai_model  # noqa: E402
import praedixa.demand_forecast.backends.timesfm.model as timesfm_model  # noqa: E402
from praedixa.demand_forecast.backends.foundation_covariates import (  # noqa: E402
    aggregate_cold_start_row,
    build_covariate_schema,
)
from praedixa.demand_forecast.backends.timesfm.backend import (  # noqa: E402
    get_timesfm_backend_availability,
)


FutureFrameBuilder = Callable[..., pd.DataFrame]


class FoundationRuntimeTest(unittest.TestCase):
    def test_timesfm_backend_requires_jax_for_multivariate_covariates(self) -> None:
        def fake_find_spec(package: str) -> ModuleSpec | None:
            if package == "jax":
                return None
            return ModuleSpec(package, loader=None)

        with (
            patch(
                "praedixa.demand_forecast.backends.timesfm.backend._runtime_python_major_minor",
                return_value=(3, 11),
            ),
            patch(
                "praedixa.demand_forecast.backends.timesfm.backend.find_spec",
                side_effect=fake_find_spec,
            ),
        ):
            availability = get_timesfm_backend_availability()

        self.assertFalse(availability.is_ready)
        self.assertIn("jax", availability.reason)
        self.assertIn("timesfm[torch] jax", availability.next_step)

    def test_cold_start_aggregation_avoids_fragmentation_warning(self) -> None:
        feature_cols = [f"feature_{index}" for index in range(180)]
        dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
        frame = pd.DataFrame({"date": dates, "target": [10.0, 20.0, 30.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for index, column in enumerate(feature_cols):
                frame[column] = np.asarray(
                    [float(index), float(index + 1), float(index + 2)]
                )
        schema = build_covariate_schema(frame, feature_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            aggregated = aggregate_cold_start_row(
                frame,
                date_col="date",
                id_col="series_id",
                target_col="target",
                cold_start_series_id="cold",
                schema=schema,
            )

        self.assertEqual(aggregated["series_id"].tolist(), ["cold", "cold"])
        self.assertEqual(aggregated["target"].tolist(), [15.0, 30.0])

    def test_foundation_future_frames_avoid_fragmentation_warning(self) -> None:
        feature_cols = [f"feature_{index}" for index in range(180)]
        frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "client_id": ["store_a", "store_a"],
                "product_id": ["sku_a", "sku_a"],
                **{
                    column: [float(index), float(index + 1)]
                    for index, column in enumerate(feature_cols)
                },
            }
        )
        schema = build_covariate_schema(frame, feature_cols)
        timesfm_future_frame = cast(
            FutureFrameBuilder,
            getattr(timesfm_model, "_future_frame"),
        )
        moirai_future_frame = cast(
            FutureFrameBuilder,
            getattr(moirai_model, "_future_frame"),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            timesfm_future = timesfm_future_frame(frame, covariate_schema=schema)
            moirai_future = moirai_future_frame(frame, covariate_schema=schema)

        self.assertEqual(timesfm_future["_horizon_step"].tolist(), [0, 1])
        self.assertEqual(moirai_future["_horizon_step"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
