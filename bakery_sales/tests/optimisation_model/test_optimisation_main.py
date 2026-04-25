from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.optimisation_model.main as optimisation_main


def test_main_delegates_to_optimization_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_optimization(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        optimisation_main,
        "run_optimization",
        fake_run_optimization,
    )

    optimisation_main.main()

    assert len(calls) == 1
    assert calls[0]["n_jobs_folds"] == optimisation_main.DEFAULT_FOLD_JOBS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
