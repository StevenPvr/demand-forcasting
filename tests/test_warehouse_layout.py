from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WarehouseLayoutTests(unittest.TestCase):
    def test_warehouse_project_contains_required_core_files(self) -> None:
        required_paths = [
            PROJECT_ROOT / "doc" / "data_engineering" / "medaillon" / "bronze.md",
            PROJECT_ROOT / "doc" / "data_engineering" / "medaillon" / "gold.md",
            PROJECT_ROOT / "doc" / "data_engineering" / "local-dev.md",
            PROJECT_ROOT / "warehouse" / "dbt_project.yml",
            PROJECT_ROOT / "warehouse" / "profiles.yml",
            PROJECT_ROOT / "warehouse" / "profiles.example.yml",
            PROJECT_ROOT / "warehouse" / "models" / "bronze_sources" / "sources.yml",
            PROJECT_ROOT / "warehouse" / "models" / "silver" / "silver_daily_product_demand.sql",
            PROJECT_ROOT / "warehouse" / "models" / "staging" / "stg_open_macro_timeseries.sql",
            PROJECT_ROOT / "warehouse" / "models" / "silver" / "schema.yml",
            PROJECT_ROOT / "warehouse" / "models" / "gold" / "gold_daily_product_forecast_panel_d1.sql",
            PROJECT_ROOT / "warehouse" / "models" / "gold" / "schema.yml",
            PROJECT_ROOT / "scripts" / "load_bronze_duckdb.py",
            PROJECT_ROOT / "scripts" / "fetch_open_exogenous.py",
            PROJECT_ROOT / "scripts" / "run_local_silver.py",
            PROJECT_ROOT / "scripts" / "run_local_gold.py",
        ]

        for path in required_paths:
            self.assertTrue(path.exists(), path)

    def test_silver_model_declares_commercial_sources_and_excludes_m5(self) -> None:
        model_path = PROJECT_ROOT / "warehouse" / "models" / "silver" / "silver_daily_product_demand.sql"
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("silver_freshretail_daily_product_demand", sql)
        self.assertIn("silver_commercial_external_daily_product_demand", sql)
        self.assertIn("silver_bakery_daily_product_demand", sql)
        self.assertNotIn("silver_m5_daily_product_demand", sql)

    def test_gold_model_references_feature_panel(self) -> None:
        model_path = PROJECT_ROOT / "warehouse" / "models" / "gold" / "gold_daily_product_forecast_panel_d1.sql"
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("gold_feature_panel_d1", sql)
        self.assertIn("split_bucket", sql)
        self.assertIn("target_demand_qty_d_plus_1", sql)
        self.assertIn("avg_selling_price_lag_1", sql)
        self.assertIn("promo_rate_7", sql)
        self.assertIn("weather_humidity_lag_1", sql)
        self.assertIn("inflation_cpi_latest_delta_28", sql)


if __name__ == "__main__":
    unittest.main()
