from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
if str(PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC))
if str(PRODUCT_SRC) not in sys.path:
    sys.path.insert(0, str(PRODUCT_SRC))

from praedixa.demand_forecast.backends.tft.model_common import (  # noqa: E402
    lazy_import_tft_dependencies,
)


class _FakeTuner:
    def __init__(self, trainer: object) -> None:
        self.trainer = trainer


class TFTModelCommonTests(unittest.TestCase):
    def test_lazy_import_tft_dependencies_exposes_callable_tuner(self) -> None:
        with patch(
            "praedixa.demand_forecast.backends.tft.model_common.import_module",
            return_value=SimpleNamespace(Tuner=_FakeTuner),
        ):
            imports = lazy_import_tft_dependencies()

        self.assertIs(imports["Tuner"], _FakeTuner)
        self.assertTrue(callable(imports["Tuner"]))


if __name__ == "__main__":
    unittest.main()
