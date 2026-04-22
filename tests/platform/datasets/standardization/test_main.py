from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.datasets.standardization.main import main  # noqa: E402


class GlobalDatasetMainTests(unittest.TestCase):
    def test_main_runs_validation_without_persisting_intermediate_artifacts(self) -> None:
        with mock.patch(
            "praedixa.platform.datasets.standardization.main.build_global_daily_standardization",
            return_value=SimpleNamespace(source_summaries={"combined": {"rows": 1}}, data_quality={"combined": {"error_count": 0}}),
        ) as build_mock:
            main()

        build_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
