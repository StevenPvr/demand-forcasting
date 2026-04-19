from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluation.generate_statistical_baselines as generate_statistical_baselines


def _product_frame(product_name: str, start_date: str, periods: int, offset: float) -> pd.DataFrame:
    date_index = pd.date_range(start_date, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": date_index.strftime("%Y-%m-%d"),
            "product": [product_name] * periods,
            "quantity": [offset + float(20 + (index % 7)) for index in range(periods)],
            "is_missing_day": [0] * periods,
        }
    )


def test_run_statistical_baselines_writes_global_and_per_product_best_rows(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    output_json = tmp_path / "baselines.json"

    train_df = pd.concat(
        [
            _product_frame("BAGUETTE", "2021-01-01", 35, 0.0),
            _product_frame("CROISSANT", "2021-01-01", 35, 5.0),
        ],
        ignore_index=True,
    )
    val_df = pd.concat(
        [
            _product_frame("BAGUETTE", "2021-02-05", 10, 0.0),
            _product_frame("CROISSANT", "2021-02-05", 10, 5.0),
        ],
        ignore_index=True,
    )
    test_df = pd.concat(
        [
            _product_frame("BAGUETTE", "2021-03-01", 3, 0.0),
            _product_frame("CROISSANT", "2021-03-01", 3, 5.0),
        ],
        ignore_index=True,
    )
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    payload = generate_statistical_baselines.run_statistical_baselines(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_json=output_json,
    )

    persisted_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["baseline_count"] == persisted_payload["baseline_count"]
    assert persisted_payload["baseline_count"] > 0
    assert persisted_payload["product_count"] == 2
    assert set(persisted_payload["per_product_best_baselines"]) == {"BAGUETTE", "CROISSANT"}
    assert set(persisted_payload["per_product_field_baselines"]) == {"BAGUETTE", "CROISSANT"}
    assert persisted_payload["field_baseline_name"] == "blend_lag_1_lag_7_50_50"
    assert persisted_payload["best_baseline"]["name"]
    assert persisted_payload["per_product_field_baselines"]["BAGUETTE"]["name"] == "blend_lag_1_lag_7_50_50"
    assert len(persisted_payload["best_baseline"]["prediction_rows"]) == len(test_df)
    assert set(row["product"] for row in persisted_payload["best_baseline"]["prediction_rows"]) == {
        "BAGUETTE",
        "CROISSANT",
    }
    assert persisted_payload["best_baseline"]["prediction_rows"][0]["target_date"] == "2021-03-01"


def test_main_delegates_to_statistical_baselines_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_statistical_baselines(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        generate_statistical_baselines,
        "run_statistical_baselines",
        fake_run_statistical_baselines,
    )

    generate_statistical_baselines.main()

    assert len(calls) == 1
    assert calls[0]["train_csv"] == generate_statistical_baselines.TRAIN_CSV
    assert calls[0]["val_csv"] == generate_statistical_baselines.VAL_CSV
    assert calls[0]["test_csv"] == generate_statistical_baselines.TEST_CSV
    assert calls[0]["output_json"] == generate_statistical_baselines.STATISTICAL_BASELINES_JSON


def test_run_statistical_baselines_raises_clear_error_when_inputs_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="Missing baseline input split"):
        generate_statistical_baselines.run_statistical_baselines(
            train_csv=tmp_path / "missing_train.csv",
            val_csv=tmp_path / "missing_val.csv",
            test_csv=tmp_path / "missing_test.csv",
            output_json=tmp_path / "baselines.json",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
