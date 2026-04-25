from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compute_economic_gain_by_product import compute_economic_gain_report


def test_compute_economic_gain_report_writes_per_product_and_global_payload(tmp_path: Path) -> None:
    arima_metrics_json = tmp_path / "arima_metrics.json"
    baselines_json = tmp_path / "baselines.json"
    predictions_csv = tmp_path / "predictions.csv"
    test_csv = tmp_path / "test.csv"
    raw_sales_csv = tmp_path / "raw_sales.csv"
    output_json = tmp_path / "economic_gain.json"

    arima_metrics_json.write_text(
        json.dumps(
            {
                "model_family": "ARIMA_BY_PRODUCT",
                "product_count": 2,
                "per_product_metrics": {
                    "BAGUETTE": {"mae": 2.0, "test_rows": 3},
                    "CROISSANT": {"mae": 1.5, "test_rows": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    baselines_json.write_text(
        json.dumps(
            {
                "field_baseline_name": "blend_lag_1_lag_7_50_50",
                "per_product_field_baselines": {
                    "BAGUETTE": {
                        "name": "blend_lag_1_lag_7_50_50",
                        "mae": 3.0,
                        "prediction_rows": [
                            {"product": "BAGUETTE", "target_date": "2021-03-01", "actual": 10.0, "prediction": 8.0},
                            {"product": "BAGUETTE", "target_date": "2021-03-02", "actual": 10.0, "prediction": 12.0},
                            {"product": "BAGUETTE", "target_date": "2021-03-03", "actual": 10.0, "prediction": 10.0},
                        ],
                    },
                    "CROISSANT": {
                        "name": "blend_lag_1_lag_7_50_50",
                        "mae": 2.0,
                        "prediction_rows": [
                            {"product": "CROISSANT", "target_date": "2021-03-01", "actual": 5.0, "prediction": 4.0},
                            {"product": "CROISSANT", "target_date": "2021-03-02", "actual": 5.0, "prediction": 7.0},
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    predictions_csv.write_text(
        "\n".join(
            [
                "origin_date,target_date,actual,prediction_raw,prediction_rounded,lower_80,upper_80,lower_95,upper_95,train_rows_used,used_order,used_seasonal_order,used_trend,product",
                '2021-02-28,2021-03-01,10.0,9.0,9.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-03-01,2021-03-02,10.0,11.0,11.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-03-02,2021-03-03,10.0,10.0,10.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-02-28,2021-03-01,5.0,5.0,5.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,CROISSANT',
                '2021-03-01,2021-03-02,5.0,6.0,6.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,CROISSANT',
            ]
        ),
        encoding="utf-8",
    )
    test_csv.write_text(
        "\n".join(
            [
                "date,product,quantity,is_missing_day",
                "2021-03-01,BAGUETTE,10.0,0",
                "2021-03-02,BAGUETTE,10.0,0",
                "2021-03-03,BAGUETTE,10.0,0",
                "2021-03-01,CROISSANT,5.0,0",
                "2021-03-02,CROISSANT,5.0,0",
            ]
        ),
        encoding="utf-8",
    )
    raw_sales_csv.write_text(
        "\n".join(
            [
                "index,date,time,ticket_number,article,Quantity,unit_price",
                '0,2021-03-01,08:00,1,BAGUETTE,2.0,"0,90 €"',
                '1,2021-03-02,08:00,2,BAGUETTE,2.0,"0,90 €"',
                '2,2021-03-03,08:00,3,BAGUETTE,2.0,"0,90 €"',
                '3,2021-03-01,08:00,4,CROISSANT,1.0,"1,20 €"',
                '4,2021-03-02,08:00,5,CROISSANT,1.0,"1,20 €"',
                '5,2021-03-03,08:00,6,CROISSANT,1.0,"1,20 €"',
                '6,2021-03-04,08:00,7,CROISSANT,1.0,"1,20 €"',
            ]
        ),
        encoding="utf-8",
    )

    payload = compute_economic_gain_report(
        arima_metrics_json=arima_metrics_json,
        baselines_json=baselines_json,
        output_json=output_json,
        predictions_csv=predictions_csv,
        test_csv=test_csv,
        raw_sales_csv=raw_sales_csv,
    )

    persisted_payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["product_count"] == 2
    assert persisted_payload["product_count"] == 2
    assert set(persisted_payload["per_product_gain"]) == {"BAGUETTE", "CROISSANT"}
    assert persisted_payload["per_product_gain"]["BAGUETTE"]["best_baseline_name"] == "blend_lag_1_lag_7_50_50"
    assert persisted_payload["per_product_gain"]["BAGUETTE"]["unit_sale_price_eur"] == pytest.approx(0.9)
    assert persisted_payload["per_product_gain"]["BAGUETTE"]["unit_production_cost_eur"] == pytest.approx(0.315)
    assert persisted_payload["per_product_gain"]["BAGUETTE"]["model_total_loss_eur"] == pytest.approx(1.215)
    assert persisted_payload["per_product_gain"]["BAGUETTE"]["baseline_total_loss_eur"] == pytest.approx(2.43)
    assert persisted_payload["per_product_gain"]["BAGUETTE"]["estimated_savings_eur_vs_best_baseline"] == pytest.approx(
        1.215
    )
    assert persisted_payload["per_product_gain"]["CROISSANT"]["estimated_savings_eur_vs_best_baseline"] == pytest.approx(
        1.62
    )
    assert persisted_payload["total_estimated_savings_eur_vs_best_baselines"] == pytest.approx(2.835)


def test_compute_economic_gain_report_uses_default_unit_cost_when_summary_is_missing(tmp_path: Path) -> None:
    arima_metrics_json = tmp_path / "arima_metrics.json"
    baselines_json = tmp_path / "baselines.json"
    predictions_csv = tmp_path / "predictions.csv"
    test_csv = tmp_path / "test.csv"
    output_json = tmp_path / "economic_gain.json"

    arima_metrics_json.write_text(
        json.dumps(
            {
                "per_product_metrics": {
                    "BAGUETTE": {"mae": 2.0, "test_rows": 3},
                }
            }
        ),
        encoding="utf-8",
    )
    baselines_json.write_text(
        json.dumps(
            {
                "field_baseline_name": "blend_lag_1_lag_7_50_50",
                "per_product_field_baselines": {
                    "BAGUETTE": {
                        "name": "blend_lag_1_lag_7_50_50",
                        "mae": 3.0,
                        "prediction_rows": [
                            {"product": "BAGUETTE", "target_date": "2021-03-01", "actual": 10.0, "prediction": 8.0},
                            {"product": "BAGUETTE", "target_date": "2021-03-02", "actual": 10.0, "prediction": 8.0},
                            {"product": "BAGUETTE", "target_date": "2021-03-03", "actual": 10.0, "prediction": 8.0},
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    predictions_csv.write_text(
        "\n".join(
            [
                "origin_date,target_date,actual,prediction_raw,prediction_rounded,lower_80,upper_80,lower_95,upper_95,train_rows_used,used_order,used_seasonal_order,used_trend,product",
                '2021-02-28,2021-03-01,10.0,9.0,9.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-03-01,2021-03-02,10.0,9.0,9.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-03-02,2021-03-03,10.0,9.0,9.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
            ]
        ),
        encoding="utf-8",
    )
    test_csv.write_text(
        "\n".join(
            [
                "date,product,quantity,is_missing_day",
                "2021-03-01,BAGUETTE,10.0,0",
                "2021-03-02,BAGUETTE,10.0,0",
                "2021-03-03,BAGUETTE,10.0,0",
            ]
        ),
        encoding="utf-8",
    )

    payload = compute_economic_gain_report(
        arima_metrics_json=arima_metrics_json,
        baselines_json=baselines_json,
        output_json=output_json,
        predictions_csv=predictions_csv,
        test_csv=test_csv,
        raw_sales_csv=None,
    )

    assert payload["per_product_gain"]["BAGUETTE"]["unit_sale_price_eur"] == pytest.approx(1.0)
    assert payload["per_product_gain"]["BAGUETTE"]["unit_production_cost_eur"] == pytest.approx(0.35)
    assert payload["per_product_gain"]["BAGUETTE"]["estimated_savings_eur_vs_best_baseline"] == pytest.approx(3.0)


def test_compute_economic_gain_report_logs_summary(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    arima_metrics_json = tmp_path / "arima_metrics.json"
    baselines_json = tmp_path / "baselines.json"
    predictions_csv = tmp_path / "predictions.csv"
    test_csv = tmp_path / "test.csv"
    raw_sales_csv = tmp_path / "raw_sales.csv"
    output_json = tmp_path / "economic_gain.json"

    arima_metrics_json.write_text(
        json.dumps(
            {
                "model_family": "ARIMA_BY_PRODUCT",
                "per_product_metrics": {
                    "BAGUETTE": {"mae": 2.0, "test_rows": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    baselines_json.write_text(
        json.dumps(
            {
                "field_baseline_name": "blend_lag_1_lag_7_50_50",
                "per_product_field_baselines": {
                    "BAGUETTE": {
                        "name": "blend_lag_1_lag_7_50_50",
                        "mae": 3.0,
                        "prediction_rows": [
                            {"product": "BAGUETTE", "target_date": "2021-03-01", "actual": 10.0, "prediction": 8.0},
                            {"product": "BAGUETTE", "target_date": "2021-03-02", "actual": 10.0, "prediction": 12.0},
                            {"product": "BAGUETTE", "target_date": "2021-03-03", "actual": 10.0, "prediction": 10.0},
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    predictions_csv.write_text(
        "\n".join(
            [
                "origin_date,target_date,actual,prediction_raw,prediction_rounded,lower_80,upper_80,lower_95,upper_95,train_rows_used,used_order,used_seasonal_order,used_trend,product",
                '2021-02-28,2021-03-01,10.0,9.0,9.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-03-01,2021-03-02,10.0,11.0,11.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
                '2021-03-02,2021-03-03,10.0,10.0,10.0,0,0,0,0,0,"(0, 0, 0)","(0, 0, 0, 0)",n,BAGUETTE',
            ]
        ),
        encoding="utf-8",
    )
    test_csv.write_text(
        "\n".join(
            [
                "date,product,quantity,is_missing_day",
                "2021-03-01,BAGUETTE,10.0,0",
                "2021-03-02,BAGUETTE,10.0,0",
                "2021-03-03,BAGUETTE,10.0,0",
            ]
        ),
        encoding="utf-8",
    )
    raw_sales_csv.write_text(
        "\n".join(
            [
                "index,date,time,ticket_number,article,Quantity,unit_price",
                '0,2021-03-01,08:00,1,BAGUETTE,2.0,"0,90 €"',
                '1,2021-03-02,08:00,2,BAGUETTE,2.0,"0,90 €"',
                '2,2021-03-03,08:00,3,BAGUETTE,2.0,"0,90 €"',
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO):
        compute_economic_gain_report(
            arima_metrics_json=arima_metrics_json,
            baselines_json=baselines_json,
            output_json=output_json,
            predictions_csv=predictions_csv,
            test_csv=test_csv,
            raw_sales_csv=raw_sales_csv,
        )

    assert "Computed test-window unit prices for 1 products" in caplog.text
    assert "Product BAGUETTE: ARIMA loss 1.2150 EUR vs baseline blend_lag_1_lag_7_50_50 loss 2.4300 EUR" in caplog.text
    assert "Computed economic gain against field baseline for 1 products" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
