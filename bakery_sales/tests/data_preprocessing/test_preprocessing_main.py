from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data_preprocessing.main as preprocessing_main


def test_main_delegates_to_split_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_split_product_arima_dataset(**_: object) -> None:
        calls.append("called")

    monkeypatch.setattr(
        preprocessing_main,
        "split_product_arima_dataset",
        fake_split_product_arima_dataset,
    )

    preprocessing_main.main()

    assert calls == ["called"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
