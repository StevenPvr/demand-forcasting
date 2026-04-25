from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.feature_selection.main as feature_selection_main


def test_main_delegates_to_feature_selection_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_feature_selection(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        feature_selection_main,
        "run_feature_selection",
        fake_run_feature_selection,
    )

    feature_selection_main.main()

    assert len(calls) == 1
    assert calls[0]["n_jobs"] == feature_selection_main.DEFAULT_SELECTOR_JOBS
    assert calls[0]["n_trials"] == feature_selection_main.DEFAULT_OPTUNA_TRIALS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
