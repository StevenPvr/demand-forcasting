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
            / "macros"
            / "praedixa_canonical_source_id.sql",
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
            / "silver_daily_product_demand_all.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand_training_candidates.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand_dedup_audit.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_date_spine.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_location_date_spine.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_series_date_spine.sql",
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
            / "gold_feature_quality_panel_d1.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_training_matrix_d1.sql",
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
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand.sql"
        )
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("silver_daily_product_demand_training_candidates", sql)
        self.assertNotIn("select *", sql.lower())

        all_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand_all.sql"
        )
        candidates_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_daily_product_demand_training_candidates.sql"
        )
        all_sql = all_path.read_text(encoding="utf-8")
        candidates_sql = candidates_path.read_text(encoding="utf-8")

        self.assertIn("silver_synthetic_foodservice_daily_product_demand", all_sql)
        self.assertIn("silver_bakery_daily_product_demand", all_sql)
        self.assertNotIn("silver_freshretail_daily_product_demand", all_sql)
        self.assertNotIn("select\n        20 as source_priority,\n        *", all_sql)
        self.assertIn(
            "partition by dataset_source, dt, location_id, product_id", all_sql
        )
        self.assertIn("where source_rank = 1", all_sql)
        self.assertIn("silver_source_registry", all_sql)
        self.assertIn("source_policy.source_priority", all_sql)
        self.assertIn("silver_allowed_training_dataset_sources", candidates_sql)
        self.assertIn("training_scope in ('train', 'reference_eval')", candidates_sql)

    def test_gold_feature_panel_unions_bakery_and_synthetic_slices(self) -> None:
        panel_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_panel_d1.sql"
        )
        sql = panel_path.read_text(encoding="utf-8")
        self.assertIn("gold_feature_bakery_d1", sql)
        self.assertIn("gold_feature_synthetic_foodservice_d1", sql)
        self.assertIn("union all", sql.lower())

    def test_gold_feature_quality_panel_references_feature_panel(self) -> None:
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_quality_panel_d1.sql"
        )
        sql = model_path.read_text(encoding="utf-8")

        self.assertIn("gold_feature_panel_d1", sql)
        self.assertIn("split_bucket", sql)
        self.assertIn("target_demand_qty_d_plus_1", sql)
        self.assertIn("promo_rate_7", sql)
        self.assertIn("commerce_modality", sql)
        self.assertIn("operation_type", sql)
        self.assertIn("product_taxonomy_depth", sql)
        self.assertIn("weather_humidity_lag_1", sql)
        self.assertIn("weather_humidity_lag_14", sql)
        self.assertIn("weather_temperature_max_lag_1", sql)
        self.assertIn("weather_temperature_max_lag_14", sql)
        self.assertIn("lending_interest_rate_latest", sql)
        self.assertIn('"lag_6"', sql)
        self.assertIn('"observed_revenue_net_lag_14"', sql)
        self.assertIn('"promo_flag_lag_14"', sql)
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

    def test_gold_training_matrix_is_the_downstream_contract(self) -> None:
        model_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_training_matrix_d1.sql"
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

        self.assertIn("gold_feature_quality_panel_d1", sql)
        self.assertIn("forecast_horizon_days", sql)
        self.assertIn("target_true_zero_demand_flag", sql)
        self.assertIn('"source_role"', sql)
        self.assertIn('"commerce_modality"', sql)
        self.assertIn('"operation_type"', sql)
        self.assertIn('"product_taxonomy_depth"', sql)
        self.assertIn('"lag_2"', sql)
        self.assertIn('"lag_6"', sql)
        self.assertIn('"lag_14"', sql)
        self.assertIn('"weather_temperature_lag_14"', sql)
        self.assertIn('"observed_stockout_flag_lag_14"', sql)
        self.assertNotIn("* exclude", sql.lower())
        self.assertIn('DEFAULT_GOLD_TABLE = "gold.gold_training_matrix_d1"', constants)

    def test_gold_feature_client_id_is_dataset_scoped(self) -> None:
        macro_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "macros"
            / "praedixa_gold_feature_slice.sql"
        )
        sql = macro_path.read_text(encoding="utf-8")

        self.assertIn("cast(base_panel.dataset_source as varchar)", sql)
        self.assertIn("cast(base_panel.location_id as varchar)", sql)
        self.assertIn("cast(base_panel.product_id as varchar)", sql)
        self.assertNotIn("base_panel.series_id as client_id", sql)
        self.assertIn(
            "lag(current_day_demand_qty, 6) over series_window as lag_6",
            sql,
        )
        self.assertIn(
            "lag(observed_revenue_net, 14) over series_window as observed_revenue_net_lag_14",
            sql,
        )

    def test_gold_source_split_contract_is_explicit(self) -> None:
        synthetic_slice_path = (
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "gold"
            / "gold_feature_synthetic_foodservice_d1.sql"
        )
        bakery_slice_path = (
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
        synthetic_slice_sql = synthetic_slice_path.read_text(encoding="utf-8")
        bakery_slice_sql = bakery_slice_path.read_text(encoding="utf-8")
        split_test_sql = split_test_path.read_text(encoding="utf-8")

        self.assertIn(
            "dataset_source like 'synthetic_foodservice%'", synthetic_slice_sql
        )
        self.assertIn('"pilot_ready_chrono_60_20_20"', synthetic_slice_sql)
        self.assertIn('"bakery_chrono_75_25_test_3mo"', bakery_slice_sql)
        self.assertIn("synthetic_foodservice_pilot_ready_contract", split_test_sql)
        self.assertIn("bakery_train_val_test_holdout_contract", split_test_sql)
        self.assertIn("unexpected_dataset_source", split_test_sql)
        self.assertNotIn("supplemental_corpus_train_val_contract", split_test_sql)
        self.assertIn("or test_rows = 0", split_test_sql)
        self.assertIn("or train_rows = 0", split_test_sql)
        self.assertIn("or val_rows = 0", split_test_sql)
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

    def test_silver_provider_joins_use_canonical_source_id_without_or(self) -> None:
        model_paths = [
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_open_weather_daily.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_open_location_metadata.sql",
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
            / "silver_open_public_holiday_calendar_daily.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_open_school_holidays_daily.sql",
            PROJECT_ROOT
            / "platform"
            / "warehouse"
            / "models"
            / "silver"
            / "silver_open_macro_country_daily.sql",
        ]

        for model_path in model_paths:
            sql = model_path.read_text(encoding="utf-8")
            self.assertIn("praedixa_canonical_source_id", sql, model_path)
            self.assertNotIn(" or stg.source_name = allowed.source_id", sql.lower())
            self.assertNotIn(
                " or holidays.source_name = allowed.source_id", sql.lower()
            )
            self.assertNotIn(" or annual.source_name = allowed.source_id", sql.lower())
            self.assertNotIn(
                " or timeseries.source_name = allowed.source_id", sql.lower()
            )


if __name__ == "__main__":
    unittest.main()
