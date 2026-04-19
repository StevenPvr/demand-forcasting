from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.xgboost_utils import fit_xgboost_booster, predict_with_xgboost_booster  # noqa: E402


class XGBoostUtilsTests(unittest.TestCase):
    def test_fit_xgboost_booster_accepts_nullable_boolean_features(self) -> None:
        train_frame = pd.DataFrame(
            {
                "flag": pd.Series([True, False, pd.NA, True, False], dtype="boolean"),
                "series_id": ["s1", "s1", "s2", "s2", "s3"],
                "location_id": ["l1", "l1", "l2", "l2", "l3"],
                "lag_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "target": [1.1, 1.9, 3.2, 4.1, 4.8],
            }
        )
        valid_frame = pd.DataFrame(
            {
                "flag": pd.Series([pd.NA, True], dtype="boolean"),
                "series_id": ["s4", "s1"],
                "location_id": ["l4", "l1"],
                "lag_1": [6.0, 7.0],
                "target": [6.2, 6.9],
            }
        )

        model = fit_xgboost_booster(
            train_frame,
            ["flag", "series_id", "location_id", "lag_1"],
            target_col="target",
            model_params={
                "n_estimators": 5,
                "learning_rate": 0.1,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "n_jobs": 1,
                "verbosity": 0,
            },
            valid_frame=valid_frame,
            early_stopping_rounds=2,
        )

        predictions = predict_with_xgboost_booster(model, valid_frame, ["flag", "series_id", "location_id", "lag_1"])
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
