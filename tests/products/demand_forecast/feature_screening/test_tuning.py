from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.feature_screening.tft_fold_cache import (  # noqa: E402
    clear_screening_fold_cache,
)
from praedixa.demand_forecast.feature_screening.tuning import (  # noqa: E402
    fit_and_score_tft_model,
)


def _screening_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["series_1"] * 8,
            "dt": pd.date_range("2024-01-01", periods=8, freq="D"),
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "rolling_mean_7": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        }
    )


class FeatureScreeningTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_screening_fold_cache()

    def tearDown(self) -> None:
        clear_screening_fold_cache()

    def test_fit_and_score_reuses_cached_fold_dataset_artifacts(self) -> None:
        frame = _screening_frame()
        folds = [{"fold": 0, "train_idx": [0, 1, 2, 3], "valid_idx": [4, 5, 6, 7]}]

        with (
            patch(
                "praedixa.demand_forecast.feature_screening.tft_fold_cache.build_training_dataset_artifacts",
                return_value=object(),
            ) as mocked_build_dataset_artifacts,
            patch(
                "praedixa.demand_forecast.feature_screening.tuning.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.feature_screening.tuning.predict_with_tft_model",
                return_value=np.asarray([14.0, 15.0, 16.0, 17.0], dtype=float),
            ),
        ):
            fit_and_score_tft_model(
                frame=frame,
                folds=folds,
                feature_cols=["rolling_mean_7"],
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 1, "max_encoder_length": 28},
            )
            fit_and_score_tft_model(
                frame=frame,
                folds=folds,
                feature_cols=["rolling_mean_7"],
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 1, "max_encoder_length": 28},
            )

        self.assertEqual(mocked_build_dataset_artifacts.call_count, 1)

    def test_fit_and_score_rebuilds_cache_when_encoder_length_changes(self) -> None:
        frame = _screening_frame()
        folds = [{"fold": 0, "train_idx": [0, 1, 2, 3], "valid_idx": [4, 5, 6, 7]}]

        with (
            patch(
                "praedixa.demand_forecast.feature_screening.tft_fold_cache.build_training_dataset_artifacts",
                return_value=object(),
            ) as mocked_build_dataset_artifacts,
            patch(
                "praedixa.demand_forecast.feature_screening.tuning.fit_tft_model",
                return_value=object(),
            ),
            patch(
                "praedixa.demand_forecast.feature_screening.tuning.predict_with_tft_model",
                return_value=np.asarray([14.0, 15.0, 16.0, 17.0], dtype=float),
            ),
        ):
            fit_and_score_tft_model(
                frame=frame,
                folds=folds,
                feature_cols=["rolling_mean_7"],
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 1, "max_encoder_length": 14},
            )
            fit_and_score_tft_model(
                frame=frame,
                folds=folds,
                feature_cols=["rolling_mean_7"],
                logger=__import__("logging").getLogger(__name__),
                model_params={"runtime_profile": "local_cpu", "n_jobs": 1, "max_encoder_length": 56},
            )

        self.assertEqual(mocked_build_dataset_artifacts.call_count, 2)


if __name__ == "__main__":
    unittest.main()
