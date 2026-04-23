from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())


class WarehouseLayoutTests(unittest.TestCase):
    def test_warehouse_project_contains_required_core_files(self) -> None:
        required_paths = [
            PROJECT_ROOT / "pyrightconfig.json",
            PROJECT_ROOT / ".vscode" / "settings.json",
            PROJECT_ROOT / "docs" / "data_engineering" / "medaillon" / "bronze.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "medaillon" / "gold.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "local-dev.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "python_architecture.md",
            PROJECT_ROOT / "platform" / "warehouse" / "dbt_project.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "profiles.example.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "seeds" / "source_registry.csv",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "bronze_sources" / "sources.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_daily_product_demand.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_source_registry.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_allowed_training_dataset_sources.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_location_profile.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_product_profile.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_location_calendar.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_location_catchment.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_location_capacity_daily.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "staging" / "stg_source_registry.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "staging" / "stg_open_location_catchment.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "staging" / "stg_open_macro_timeseries.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "schema.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "gold" / "gold_daily_product_forecast_panel_d1.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "gold" / "schema.yml",
            PROJECT_ROOT / "apps" / "warehouse" / "main.py",
            PROJECT_ROOT / "apps" / "warehouse" / "load_bronze" / "main.py",
            PROJECT_ROOT / "apps" / "platform" / "fetch_open_exogenous" / "main.py",
            PROJECT_ROOT / "apps" / "warehouse" / "run_silver" / "main.py",
            PROJECT_ROOT / "apps" / "warehouse" / "run_gold" / "main.py",
        ]

        for path in required_paths:
            self.assertTrue(path.exists(), path)

    def test_silver_model_declares_commercial_sources_only(self) -> None:
        model_path = PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_daily_product_demand.sql"
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("silver_freshretail_daily_product_demand", sql)
        self.assertIn("silver_supplemental_corpus_daily_product_demand", sql)
        self.assertIn("silver_bakery_daily_product_demand", sql)
        self.assertIn("silver_allowed_training_dataset_sources", sql)

    def test_gold_model_references_feature_panel(self) -> None:
        model_path = PROJECT_ROOT / "platform" / "warehouse" / "models" / "gold" / "gold_daily_product_forecast_panel_d1.sql"
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("gold_feature_panel_d1", sql)
        self.assertIn("split_bucket", sql)
        self.assertIn("target_demand_qty_d_plus_1", sql)
        self.assertIn("avg_selling_price_lag_1", sql)
        self.assertIn("promo_rate_7", sql)
        self.assertIn("weather_humidity_lag_1", sql)
        self.assertIn("weather_temperature_max_lag_1", sql)
        self.assertIn("lending_interest_rate_latest", sql)
        self.assertIn("current_holiday_name", sql)
        self.assertIn("target_holiday_name", sql)
        self.assertIn("feature_status_bases", sql)
        self.assertIn("data_quality_metadata_bases", sql)
        self.assertIn("data_quality_pricing_promo_bases", sql)
        self.assertIn("data_quality_weather_bases", sql)
        self.assertIn("data_quality_macro_bases", sql)
        self.assertIn("data_quality_operations_bases", sql)
        self.assertIn("data_quality_history_bases", sql)
        self.assertIn('("metadata", data_quality_metadata_bases)', sql)
        self.assertIn('("weather", data_quality_weather_bases)', sql)
        self.assertIn('("history", data_quality_history_bases)', sql)
        self.assertIn("data_quality_{{ family_name }}_unavailable_count", sql)
        self.assertIn("data_quality_{{ family_name }}_missing_count", sql)
        self.assertNotIn("{{ base }}_status", sql)
        self.assertIn("cold_start_bucket", sql)
        self.assertIn("history_available_days", sql)

    def test_silver_location_calendar_keeps_holiday_labels(self) -> None:
        model_path = PROJECT_ROOT / "platform" / "warehouse" / "models" / "silver" / "silver_location_calendar.sql"
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("holiday_name", sql)
        self.assertIn("school_holiday_name", sql)


if __name__ == "__main__":
    unittest.main()
