from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)


class WarehouseLayoutTests(unittest.TestCase):
    def test_warehouse_project_contains_required_core_files(self) -> None:
        required_paths = [
            PROJECT_ROOT / "pyrightconfig.json",
            PROJECT_ROOT / ".vscode" / "settings.json",
            PROJECT_ROOT / "docs" / "data_engineering" / "medaillon" / "bronze.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "medaillon" / "gold.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "medaillon" / "runbook.md",
            PROJECT_ROOT
            / "docs"
            / "data_engineering"
            / "medaillon"
            / "how_to_add_source.md",
            PROJECT_ROOT
            / "docs"
            / "data_engineering"
            / "medaillon"
            / "how_to_add_feature.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "local-dev.md",
            PROJECT_ROOT / "docs" / "data_engineering" / "python_architecture.md",
            PROJECT_ROOT / "platform" / "warehouse" / "dbt_project.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "profiles.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "profiles.example.yml",
            PROJECT_ROOT / "platform" / "warehouse" / "seeds" / "source_registry.csv",
            PROJECT_ROOT / "platform" / "warehouse" / "seeds" / "feature_registry.csv",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "bronze_sources"
            / "sources.yml",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_synthetic_foodservice_daily_product_demand.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_source_registry.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_bronze_source_manifest.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_allowed_training_dataset_sources.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_allowed_provider_sources.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_quarantined_provider_sources.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_feature_registry.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_location_profile.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_product_profile.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_location_calendar.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_location_catchment.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_location_capacity_daily.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "staging"
            / "stg_source_registry.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "staging"
            / "stg_bronze_source_manifest.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "staging"
            / "stg_synthetic_foodservice_daily.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "staging"
            / "stg_open_location_catchment.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "staging"
            / "stg_open_macro_timeseries.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "schema.yml",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_daily_product_forecast_panel_d1.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_synthetic_foodservice_d1.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_supplemental_corpus_d1.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_model_training_panel_d1.sql",
            PROJECT_ROOT / "platform" / "warehouse" / "models" / "gold" / "schema.yml",
            PROJECT_ROOT / "apps" / "warehouse" / "main.py",
            PROJECT_ROOT / "apps" / "warehouse" / "load_bronze" / "main.py",
            PROJECT_ROOT / "apps" / "platform" / "fetch_open_exogenous" / "main.py",
            PROJECT_ROOT
            / "apps"
            / "platform"
            / "generate_synthetic_cold_start"
            / "main.py",
            PROJECT_ROOT / "apps" / "warehouse" / "run_silver" / "main.py",
            PROJECT_ROOT / "apps" / "warehouse" / "run_gold" / "main.py",
        ]

        for path in required_paths:
            self.assertTrue(path.exists(), path)

    def test_silver_model_declares_commercial_sources_only(self) -> None:
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand.sql"
        )
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("silver_freshretail_daily_product_demand", sql)
        self.assertIn("silver_supplemental_corpus_daily_product_demand", sql)
        self.assertIn("silver_synthetic_foodservice_daily_product_demand", sql)
        self.assertIn("silver_bakery_daily_product_demand", sql)
        self.assertIn("silver_allowed_training_dataset_sources", sql)
        self.assertIn("source_priority", sql)
        self.assertIn("partition by dataset_source, dt, location_id, product_id", sql)
        self.assertIn("where source_rank = 1", sql)

    def test_gold_model_references_feature_panel(self) -> None:
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_daily_product_forecast_panel_d1.sql"
        )
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("gold_feature_panel_d1", sql)
        self.assertIn("split_bucket", sql)
        self.assertIn("target_demand_qty_d_plus_1", sql)
        self.assertIn("promo_rate_7", sql)
        self.assertIn("weather_humidity_lag_1", sql)
        self.assertIn("weather_temperature_max_lag_1", sql)
        self.assertIn("lending_interest_rate_latest", sql)
        self.assertNotIn("avg_selling_price_lag_1", sql)
        self.assertNotIn("current_holiday_name", sql)
        self.assertNotIn("target_holiday_name", sql)
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

    def test_gold_model_training_panel_is_the_downstream_contract(self) -> None:
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_model_training_panel_d1.sql"
        )
        constants_path = (
            PROJECT_ROOT
            / "products"
            / "demand_forecast"
            / "src"
            / "praedixa"
            / "demand_forecast"
            / "training"
            / "config"
            / "constants.py"
        )
        sql = model_path.read_text(encoding="utf-8")
        constants = constants_path.read_text(encoding="utf-8")

        self.assertIn("gold_daily_product_forecast_panel_d1", sql)
        self.assertIn("forecast_horizon_days", sql)
        self.assertIn("target_true_zero_demand_flag", sql)
        self.assertNotIn("* exclude", sql.lower())
        self.assertIn(
            'DEFAULT_GOLD_TABLE = "gold.gold_model_training_panel_d1"', constants
        )

    def test_gold_source_split_contract_is_explicit(self) -> None:
        freshretail_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_freshretail_d1.sql"
        )
        bakery_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_bakery_d1.sql"
        )
        split_test_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "tests"
            / "gold_pilot_ready_split_by_source.sql"
        )
        synthetic_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_synthetic_foodservice_d1.sql"
        )
        supplemental_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_supplemental_corpus_d1.sql"
        )
        freshretail_sql = freshretail_path.read_text(encoding="utf-8")
        bakery_sql = bakery_path.read_text(encoding="utf-8")
        split_test_sql = split_test_path.read_text(encoding="utf-8")
        synthetic_sql = synthetic_path.read_text(encoding="utf-8")
        supplemental_sql = supplemental_path.read_text(encoding="utf-8")

        self.assertIn(
            "dataset_source in ('freshretail', 'freshretail_lt')", freshretail_sql
        )
        self.assertIn('"chrono_60_40"', freshretail_sql)
        self.assertIn("dataset_source = 'bakery'", bakery_sql)
        self.assertIn('"bakery_test_only"', bakery_sql)
        self.assertIn("synthetic_foodservice_qsr", synthetic_sql)
        self.assertIn('"train_only"', synthetic_sql)
        self.assertIn("m5_forecasting_accuracy", supplemental_sql)
        self.assertIn("uci_online_retail_ii", supplemental_sql)
        self.assertIn("restaurant_sales_report", supplemental_sql)
        self.assertIn('"chrono_60_40"', supplemental_sql)
        self.assertIn("freshretail_train_val_contract", split_test_sql)
        self.assertIn("bakery_test_holdout_contract", split_test_sql)
        self.assertIn("supplemental_corpus_train_val_contract", split_test_sql)
        self.assertIn("synthetic_foodservice_train_only_contract", split_test_sql)
        self.assertIn("or val_rows <> 0", split_test_sql)
        self.assertIn("or test_rows <> 0", split_test_sql)
        self.assertIn("PRAEDIXA_GOLD_BAKERY_TEST_MONTHS", split_test_sql)

    def test_gold_feature_slice_aligns_label_metadata_with_d_plus_1_target(
        self,
    ) -> None:
        macro_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "macros"
            / "praedixa_gold_feature_slice.sql"
        )
        sql = macro_path.read_text(encoding="utf-8")

        self.assertIn(
            "lead(target_semantics, 1) over series_window as target_semantics_d_plus_1",
            sql,
        )
        self.assertIn(
            "lead(censor_flag, 1) over series_window as censor_flag_d_plus_1", sql
        )
        self.assertIn(
            "lead(target_source, 1) over series_window as target_source_d_plus_1", sql
        )
        self.assertIn(
            "lead(label_quality_score, 1) over series_window as label_quality_score_d_plus_1",
            sql,
        )
        self.assertIn(
            "lead(usable_for_training_flag, 1) over series_window as usable_for_training_flag_d_plus_1",
            sql,
        )
        self.assertIn(
            "lead(true_zero_demand_flag, 1) over series_window as target_true_zero_demand_flag",
            sql,
        )
        self.assertIn("current_day_demand_qty,\n        true_zero_demand_flag,", sql)
        self.assertIn(
            "target_true_zero_demand_flag,\n        history_available_days,", sql
        )
        self.assertIn(
            "split_labeled.target_semantics_d_plus_1 as target_semantics", sql
        )
        self.assertIn("split_labeled.censor_flag_d_plus_1 as censor_flag", sql)
        self.assertIn("split_labeled.target_source_d_plus_1 as target_source", sql)
        self.assertIn(
            "split_labeled.label_quality_score_d_plus_1 as label_quality_score", sql
        )
        self.assertIn("dense_calendar_zero_fill", sql)

    def test_silver_location_calendar_drops_school_holiday_labels(self) -> None:
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_location_calendar.sql"
        )
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("holiday_name", sql)
        self.assertNotIn("school_holiday_name", sql)


if __name__ == "__main__":
    unittest.main()
