from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.feature_engineering.calendar_features import (  # noqa: E402
    add_calendar_features,
)


class CalendarFeaturesTests(unittest.TestCase):
    def test_add_calendar_features_creates_expected_daily_signals(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-03-31", "2024-04-01", "2024-04-30"],
                "holiday_flag": [1, 0, 0],
            }
        )

        featured = add_calendar_features(frame, date_col="dt", holiday_col="holiday_flag")

        self.assertEqual(
            list(featured["day_of_week"]),
            [6, 0, 1],
        )
        self.assertEqual(list(featured["is_weekend"]), [1, 0, 0])
        self.assertEqual(list(featured["is_month_end"]), [1, 0, 1])
        self.assertEqual(list(featured["is_month_start"]), [0, 1, 0])
        self.assertEqual(list(featured["is_quarter_end"]), [1, 0, 0])
        self.assertEqual(list(featured["is_pre_holiday"]), [0, 0, 0])
        self.assertEqual(list(featured["is_post_holiday"]), [0, 1, 0])
        self.assertEqual(list(featured["is_payday_start_window"]), [0, 1, 0])
        self.assertEqual(list(featured["is_payday_end_window"]), [1, 0, 1])
        self.assertEqual(list(featured["is_bridge_day"]), [0, 0, 0])
        self.assertIn("month_sin", featured.columns)
        self.assertIn("day_of_week_cos", featured.columns)
        self.assertIn("day_of_year_sin", featured.columns)

    def test_add_calendar_features_preserves_existing_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "dt": ["2024-04-05"],
                "store_id": [7],
                "sale_amount": [2.5],
            }
        )

        featured = add_calendar_features(frame, date_col="dt")

        self.assertEqual(featured.loc[0, "store_id"], 7)
        self.assertEqual(featured.loc[0, "sale_amount"], 2.5)
        self.assertEqual(featured.loc[0, "is_friday"], 1)


if __name__ == "__main__":
    unittest.main()
