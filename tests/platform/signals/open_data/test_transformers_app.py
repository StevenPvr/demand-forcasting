from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.signals.open_data.runtime import OpenExogenousRuntimeConfig  # noqa: E402
from praedixa.platform.signals.open_data.transformers_app import (  # noqa: E402
    build_open_exogenous_transformer_frames,
    main,
)


def _build_runtime_config(output_dir: Path) -> OpenExogenousRuntimeConfig:
    return OpenExogenousRuntimeConfig(
        duckdb_path="ignored.duckdb",
        output_dir=output_dir,
        http_timeout_seconds=1,
        http_max_retries=0,
        http_retry_backoff_seconds=0.0,
        open_meteo_max_locations_per_request=1,
        open_meteo_inter_batch_sleep_seconds=0.0,
        bakery_city_name="Paris",
        bakery_school_zone="C",
        bakery_latitude=48.8566,
        bakery_longitude=2.3522,
        location_catchment_cache_path=output_dir / "_cache" / "location_catchment_snapshot.csv",
        location_catchment_snapshot_dir=output_dir / "_snapshots" / "location_catchment",
        allow_contractual_providers=False,
    )


class OpenExogenousTransformersAppTests(unittest.TestCase):
    def test_build_open_exogenous_transformer_frames_returns_expected_keys(self) -> None:
        with (
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.read_silver_location_bounds",
                return_value=pd.DataFrame(),
            ),
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.build_location_metadata_frame",
                return_value=pd.DataFrame(),
            ),
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.fetch_location_catchment_frame",
                return_value=pd.DataFrame(),
            ),
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.fetch_school_holidays_frame",
                return_value=pd.DataFrame(),
            ),
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.fetch_weather_frame",
                return_value=pd.DataFrame(),
            ),
        ):
            frames = build_open_exogenous_transformer_frames(_build_runtime_config(Path("/tmp/ignored")))

        self.assertEqual(
            sorted(frames.keys()),
            [
                "location_catchment",
                "location_metadata",
                "school_holidays",
                "weather_daily",
            ],
        )

    def test_main_delegates_to_governed_fetcher(self) -> None:
        config = _build_runtime_config(Path("/tmp/ignored"))

        with (
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.build_runtime_config_from_env",
                return_value=config,
            ) as build_config,
            mock.patch(
                "praedixa.platform.signals.open_data.transformers_app.fetch_open_exogenous_data",
                return_value={"manifest_path": "/tmp/manifest.json"},
            ) as fetch_open_exogenous,
        ):
            main()

        build_config.assert_called_once_with()
        fetch_open_exogenous.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
