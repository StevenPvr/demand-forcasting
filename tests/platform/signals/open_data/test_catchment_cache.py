from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from collections.abc import Mapping, Sequence
import unittest

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.signals.open_data.catchment_cache import (  # noqa: E402
    fetch_location_catchment_frame_with_cache,
)


def _build_catchment_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_source": "bakery",
                "location_id": "bakery_store_1",
                "latitude": 48.8566,
                "longitude": 2.3522,
            }
        ]
    )


def _build_catchment_reader(
    payloads: Sequence[Mapping[str, object]],
    counter: dict[str, int],
):
    def fake_http_json_reader(
        url: str,
        *,
        timeout_seconds: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> dict[str, object]:
        del url, timeout_seconds, max_retries, retry_backoff_seconds
        payload = payloads[counter["value"]]
        counter["value"] += 1
        return dict(payload)

    return fake_http_json_reader


class OpenExogenousCatchmentCacheTests(unittest.TestCase):
    def test_fetch_location_catchment_frame_with_cache_reuses_snapshot(self) -> None:
        metadata = _build_catchment_metadata()
        payloads: list[Mapping[str, object]] = [
            {"elements": [{"type": "node", "id": 1, "tags": {"shop": "bakery"}}]},
            {"elements": [{"type": "node", "id": 2, "tags": {"amenity": "parking"}}]},
            {"elements": [{"type": "node", "id": 3, "tags": {"landuse": "residential"}}]},
        ]
        call_count = {"value": 0}
        fake_http_json_reader = _build_catchment_reader(payloads, call_count)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "location_catchment_snapshot.csv"
            snapshot_dir = Path(temp_dir) / "snapshots"

            first_frame, first_metadata = fetch_location_catchment_frame_with_cache(
                metadata,
                cache_path=cache_path,
                snapshot_dir=snapshot_dir,
                timeout_seconds=1,
                max_retries=0,
                retry_backoff_seconds=0.0,
                http_json_reader=fake_http_json_reader,
            )
            second_frame, second_metadata = fetch_location_catchment_frame_with_cache(
                metadata,
                cache_path=cache_path,
                snapshot_dir=snapshot_dir,
                timeout_seconds=1,
                max_retries=0,
                retry_backoff_seconds=0.0,
                http_json_reader=fake_http_json_reader,
            )

        self.assertEqual(call_count["value"], 3)
        self.assertEqual(first_metadata["cache_hits"], 0)
        self.assertEqual(first_metadata["cache_misses"], 1)
        self.assertEqual(second_metadata["cache_hits"], 1)
        self.assertEqual(second_metadata["cache_misses"], 0)
        pd.testing.assert_frame_equal(first_frame, second_frame)


if __name__ == "__main__":
    unittest.main()
