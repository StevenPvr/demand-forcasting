from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_analyse.analyze_target_series import (
    _build_target_decomposition_figure,
    analyze_target_series,
    infer_seasonal_period,
)


def test_infer_seasonal_period_returns_weekly_default() -> None:
    target_df = pd.DataFrame(
        {
            "target_date": [f"2021-01-{index + 1:02d}" for index in range(14)],
            "target_baguette_t_plus_1": [float(index) for index in range(14)],
        }
    )

    assert infer_seasonal_period(target_df) == 7


def test_analyze_target_series_writes_summary_and_figures(tmp_path: Path) -> None:
    input_csv = tmp_path / "train.csv"
    output_dir = tmp_path / "analysis"
    timestamps = pd.date_range("2021-01-02", periods=21, freq="D")
    target_values = [float((index % 7) + (index // 7)) for index in range(21)]
    pd.DataFrame(
        {
            "target_date": timestamps.strftime("%Y-%m-%d"),
            "target_baguette_t_plus_1": target_values,
        }
    ).to_csv(input_csv, index=False)

    summary = analyze_target_series(
        input_csv=input_csv,
        output_dir=output_dir,
        seasonal_period=7,
        max_lags=7,
    )

    summary_payload = json.loads((output_dir / "target_summary.json").read_text())

    assert summary["row_count"] == 21
    assert summary["seasonal_period"] == 7
    assert summary["max_lags"] == 7
    assert summary_payload["row_count"] == 21
    assert (output_dir / "target_timeseries.png").exists()
    assert (output_dir / "target_distribution.png").exists()
    assert (output_dir / "target_decomposition.png").exists()
    assert (output_dir / "target_acf_pacf.png").exists()


def test_build_target_decomposition_figure_uses_timestamp_axis() -> None:
    timestamps = pd.date_range("2021-01-02", periods=21, freq="D")
    target_values = [float((index % 7) + (index // 7)) for index in range(21)]
    target_df = pd.DataFrame(
        {
            "target_date": timestamps.strftime("%Y-%m-%d"),
            "target_baguette_t_plus_1": target_values,
            "timestamp": timestamps,
        }
    )

    figure = _build_target_decomposition_figure(target_df, seasonal_period=7)

    axes = figure.axes
    assert axes
    assert axes[-1].get_xlabel() == "Target date"
    assert any(label.get_text() for label in axes[-1].get_xticklabels())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
