from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as root_main


def test_main_delegates_to_launch_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_launch_main() -> None:
        calls.append("called")

    monkeypatch.setattr(root_main, "run_launch_main", fake_run_launch_main)

    root_main.main()

    assert calls == ["called"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
