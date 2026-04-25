from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data_analyse.main as analysis_main


def test_main_delegates_to_target_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_analyze_target_series(**_: object) -> None:
        calls.append("called")

    monkeypatch.setattr(
        analysis_main,
        "analyze_target_series",
        fake_analyze_target_series,
    )

    analysis_main.main()

    assert calls == ["called"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
