from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.signals.open_data.annual_public import fetch_public_holidays_frame  # noqa: E402


class PublicHolidaysFrameTests(unittest.TestCase):
    def test_fetch_public_holidays_frame_generates_deterministic_rows_for_fr_and_gb(self) -> None:
        frame = fetch_public_holidays_frame(
            {"FR": [2024], "GB": [2024]},
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
        )

        self.assertFalse(frame.empty)
        self.assertTrue(
            (
                (frame["country_code"] == "FR")
                & (frame["dt"] == "2024-01-01")
                & (frame["holiday_name"] == "Jour de l'an")
            ).any()
        )
        self.assertTrue(
            (
                (frame["country_code"] == "GB")
                & (frame["dt"] == "2024-12-25")
                & (frame["holiday_name"] == "Christmas Day")
            ).any()
        )

    def test_fetch_public_holidays_frame_returns_empty_for_unsupported_country(self) -> None:
        frame = fetch_public_holidays_frame(
            {"ZZ": [2024]},
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
        )

        self.assertTrue(frame.empty)


if __name__ == "__main__":
    unittest.main()
