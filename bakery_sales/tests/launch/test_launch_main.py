from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.launch.main as launch_main


def test_launch_main_runs_pipeline_steps_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    logs: list[str] = []

    monkeypatch.setattr(launch_main, "run_data_cleaning_main", lambda: calls.append("cleaning"))
    monkeypatch.setattr(launch_main, "run_data_preprocessing_main", lambda: calls.append("preprocessing"))
    monkeypatch.setattr(launch_main, "run_optimisation_main", lambda: calls.append("optimisation"))
    monkeypatch.setattr(launch_main, "run_evaluation_main", lambda: calls.append("evaluation"))
    monkeypatch.setattr(
        launch_main.LOGGER,
        "info",
        lambda message, *args: logs.append(message % args),
    )

    launch_main.main()

    assert calls == ["cleaning", "preprocessing", "optimisation", "evaluation"]
    assert logs == [
        "Starting launch stage: data_cleaning",
        "Starting launch stage: data_preprocessing",
        "Starting launch stage: optimisation_model",
        "Starting launch stage: evaluation",
        "Launch pipeline completed successfully",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
