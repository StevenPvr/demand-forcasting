from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.load_bronze_duckdb import (  # noqa: E402
    BronzeTableSpec,
    _prepare_bronze_batch,
    backup_local_bronze_sources,
    build_cloud_duckdb_config_from_env,
    build_local_duckdb_config_from_env,
    build_runtime_config_from_env,
    default_active_bronze_specs,
    default_bronze_specs,
    load_all_bronze_tables,
)


class LoadBronzeDuckDBTests(unittest.TestCase):
    def _build_bronze_fixture_root(self, root: Path) -> None:
        bakery_root = root / "bakery_sales"
        global_dataset_root = root / "global_dataset"
        m5_root = root / "m5"
        bakery_root.mkdir(parents=True)
        global_dataset_root.mkdir(parents=True)
        m5_root.mkdir(parents=True)
        self._write_freshretail_parquets(bakery_root)
        self._write_bakery_csv(bakery_root)
        self._write_commercial_external_parquet(global_dataset_root)
        self._write_m5_csvs(m5_root)

    def _write_freshretail_parquets(self, bakery_root: Path) -> None:
        pd.DataFrame([self._freshretail_row(product_id=10000, dt="2024-01-01", sale_amount=12.0)]).to_parquet(
            bakery_root / "data_train.parquet",
            index=False,
        )
        pd.DataFrame(
            [
                self._freshretail_row(
                    product_id=10001,
                    dt="2024-01-02",
                    sale_amount=8.0,
                    stock_hour6_22_cnt=1,
                    discount=0.0,
                    holiday_flag=True,
                    precpt=1.2,
                    avg_temperature=10.0,
                    avg_humidity=55.0,
                    avg_wind_level=4.0,
                    is_censored=True,
                )
            ]
        ).to_parquet(bakery_root / "data_val.parquet", index=False)

    def _freshretail_row(
        self,
        *,
        product_id: int,
        dt: str,
        sale_amount: float,
        stock_hour6_22_cnt: int = 0,
        discount: float = 1.5,
        holiday_flag: bool = False,
        precpt: float = 0.0,
        avg_temperature: float = 12.4,
        avg_humidity: float = 60.0,
        avg_wind_level: float = 3.0,
        is_censored: bool = False,
    ) -> dict[str, object]:
        return {
            "city_id": 1,
            "store_id": 10,
            "management_group_id": 100,
            "first_category_id": 1000,
            "second_category_id": 1001,
            "third_category_id": 1002,
            "product_id": product_id,
            "dt": dt,
            "sale_amount": sale_amount,
            "stock_hour6_22_cnt": stock_hour6_22_cnt,
            "discount": discount,
            "holiday_flag": holiday_flag,
            "activity_flag": True,
            "precpt": precpt,
            "avg_temperature": avg_temperature,
            "avg_humidity": avg_humidity,
            "avg_wind_level": avg_wind_level,
            "is_censored": is_censored,
        }

    def _write_bakery_csv(self, bakery_root: Path) -> None:
        pd.DataFrame(
            [
                {
                    "Unnamed: 0": 1,
                    "date": "2021-01-02",
                    "time": "08:38",
                    "ticket_number": 150040,
                    "article": "BAGUETTE",
                    "Quantity": 1.0,
                    "unit_price": "0,90 €",
                }
            ]
        ).to_csv(bakery_root / "Bakery sales.csv", index=False)

    def _write_commercial_external_parquet(self, root: Path) -> None:
        pd.DataFrame(
            [
                {
                    "dataset_source": "uci_online_retail",
                    "source_partition": "historical",
                    "source_run_id": "manual",
                    "series_id": "uk_online_retail_1__SKU_1",
                    "dt": "2011-01-29",
                    "location_id": "uk_online_retail_1",
                    "product_id": "SKU_1",
                    "region_id": None,
                    "org_group_id": None,
                    "category_level_1": None,
                    "category_level_2": None,
                    "category_level_3": None,
                    "observed_demand_qty": 2.0,
                    "observed_revenue_net": 7.0,
                    "observed_discount_amount": None,
                    "avg_selling_price": 3.5,
                    "promo_flag": None,
                    "holiday_flag": None,
                    "activity_flag": None,
                    "observed_stockout_flag": None,
                    "observed_stockout_available": False,
                    "observed_stockout_intensity": None,
                    "location_open_flag": True,
                    "day_complete_flag": True,
                    "missing_sales_flag": False,
                    "calendar_weekday_name": "Saturday",
                    "calendar_day_of_week": 5,
                    "calendar_month": 1,
                    "calendar_year": 2011,
                    "calendar_week_key": 4,
                    "event_name_1": None,
                    "event_type_1": None,
                    "event_name_2": None,
                    "event_type_2": None,
                    "snap_flag": None,
                    "weather_precipitation": None,
                    "weather_temperature": None,
                    "weather_humidity": None,
                    "weather_wind_level": None,
                    "anomaly_flag": False,
                    "silver_run_id": "manual",
                }
            ]
        ).to_parquet(root / "commercial_external_daily.parquet", index=False)

    def _write_m5_csvs(self, m5_root: Path) -> None:
        pd.DataFrame(
            [
                {
                    "date": "2011-01-29",
                    "wm_yr_wk": 11101,
                    "weekday": "Saturday",
                    "wday": 1,
                    "month": 1,
                    "year": 2011,
                    "d": "d_1",
                    "event_name_1": None,
                    "event_type_1": None,
                    "event_name_2": None,
                    "event_type_2": None,
                    "snap_CA": False,
                    "snap_TX": False,
                    "snap_WI": False,
                }
            ]
        ).to_csv(m5_root / "calendar.csv", index=False)
        pd.DataFrame(
            [{"store_id": "CA_1", "item_id": "ITEM_1", "wm_yr_wk": 11101, "sell_price": 3.5}]
        ).to_csv(m5_root / "sell_prices.csv", index=False)
        pd.DataFrame(
            [
                {
                    "id": "ITEM_1_CA_1_validation",
                    "item_id": "ITEM_1",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "d_1": 5.0,
                }
            ]
        ).to_csv(m5_root / "sales_train_validation.csv", index=False)

    def _assert_local_duckdb_counts(self, local_db_path: Path) -> None:
        connection = duckdb.connect(str(local_db_path))
        try:
            freshretail_count = connection.execute(
                "SELECT COUNT(*) FROM bronze.bronze_freshretail_daily"
            ).fetchone()[0]
            bakery_count = connection.execute(
                "SELECT COUNT(*) FROM bronze.bronze_bakery_order_lines"
            ).fetchone()[0]
            m5_count = connection.execute(
                "SELECT COUNT(*) FROM bronze.bronze_m5_sales_long"
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(freshretail_count, 2)
        self.assertEqual(bakery_count, 1)
        self.assertEqual(m5_count, 1)

    def test_build_local_duckdb_config_from_env_reads_expected_settings(self) -> None:
        environ = {
            "PRAEDIXA_DUCKDB_LOCAL_PATH": "warehouse/local.duckdb",
            "PRAEDIXA_DUCKDB_BRONZE_SCHEMA": "bronze_stage",
        }
        with unittest.mock.patch.dict("os.environ", environ, clear=True):
            config = build_local_duckdb_config_from_env()

        self.assertEqual(config.database_path, "warehouse/local.duckdb")
        self.assertEqual(config.schema_name, "bronze_stage")
        self.assertFalse(config.is_cloud)

    def test_build_cloud_duckdb_config_from_env_reads_expected_settings(self) -> None:
        environ = {
            "PRAEDIXA_DUCKDB_CLOUD_PATH": "md:praedixa_cloud",
            "PRAEDIXA_DUCKDB_CLOUD_TOKEN": "secret-token",
            "PRAEDIXA_DUCKDB_BRONZE_SCHEMA": "bronze_cloud",
        }
        with unittest.mock.patch.dict("os.environ", environ, clear=True):
            config = build_cloud_duckdb_config_from_env()

        self.assertEqual(config.database_path, "md:praedixa_cloud")
        self.assertEqual(config.schema_name, "bronze_cloud")
        self.assertTrue(config.is_cloud)
        self.assertEqual(config.token, "secret-token")

    def test_build_runtime_config_from_env_defaults_to_local_duckdb_mode(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            config = build_runtime_config_from_env()

        self.assertTrue(config.local_warehouse_enabled)
        self.assertFalse(config.cloud_warehouse_enabled)
        self.assertTrue(config.local_backup_enabled)

    def test_default_bronze_specs_declares_all_v1_sources(self) -> None:
        specs = default_bronze_specs("data", schema_name="bronze")
        table_names = {spec.table_name for spec in specs}

        self.assertIn("bronze_freshretail_daily", table_names)
        self.assertIn("bronze_bakery_order_lines", table_names)
        self.assertIn("bronze_m5_sales_long", table_names)

    def test_default_active_bronze_specs_excludes_m5_sources(self) -> None:
        specs = default_active_bronze_specs("data", schema_name="bronze")
        source_names = {spec.source_name for spec in specs}
        table_names = {spec.table_name for spec in specs}

        self.assertIn("freshretail_train", source_names)
        self.assertIn("freshretail_val", source_names)
        self.assertIn("bakery", source_names)
        self.assertIn("commercial_external_daily", source_names)
        self.assertNotIn("m5_sales", source_names)
        self.assertNotIn("m5_calendar", source_names)
        self.assertNotIn("m5_prices", source_names)
        self.assertNotIn("bronze_m5_sales_long", table_names)
        self.assertIn("bronze_commercial_external_daily", table_names)

    def test_prepare_bronze_batch_renames_bakery_columns(self) -> None:
        spec = BronzeTableSpec(
            table_name="bronze_bakery_order_lines",
            source_path=Path("bakery.csv"),
            ddl="create table x",
            source_name="bakery",
        )
        batch = pd.DataFrame(
            {
                "Unnamed: 0": [1],
                "date": ["2021-01-02"],
                "time": ["08:38"],
                "ticket_number": [150040],
                "article": ["BAGUETTE"],
                "Quantity": [1.0],
                "unit_price": ["0,90 €"],
            }
        )

        prepared = _prepare_bronze_batch(spec, batch)

        self.assertIn("sale_date_raw", prepared.columns)
        self.assertIn("unit_price_raw", prepared.columns)
        self.assertEqual(prepared.loc[0, "source_partition"], "historical")

    def test_backup_local_bronze_sources_copies_files_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.csv"
            source_path.write_text("a,b\n1,2\n", encoding="utf-8")
            spec = BronzeTableSpec(
                table_name="bronze_bakery_order_lines",
                source_path=source_path,
                ddl="create table x",
                source_name="bakery",
            )

            result = backup_local_bronze_sources([spec], backup_root=root / "backup")

            self.assertEqual(result["source_count"], 1)
            self.assertTrue((root / "backup" / "bakery" / "source.csv").exists())
            self.assertTrue((root / "backup" / "bronze_backup_manifest.json").exists())

    def test_load_all_bronze_tables_loads_into_local_duckdb_and_keeps_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_bronze_fixture_root(root)
            local_db_path = root / "warehouse" / "praedixa.duckdb"
            env = {
                "PRAEDIXA_ENABLE_LOCAL_WAREHOUSE": "true",
                "PRAEDIXA_ENABLE_CLOUD_WAREHOUSE": "false",
                "PRAEDIXA_ENABLE_LOCAL_BACKUP": "true",
                "PRAEDIXA_LOCAL_BACKUP_DIR": str(root / "backup"),
                "PRAEDIXA_DUCKDB_LOCAL_PATH": str(local_db_path),
                "PRAEDIXA_DUCKDB_BRONZE_SCHEMA": "bronze",
            }
            with mock.patch.dict("os.environ", env, clear=True):
                result = load_all_bronze_tables(data_dir=root)

            self.assertTrue(result["local_warehouse_enabled"])
            self.assertFalse(result["cloud_warehouse_enabled"])
            self.assertEqual(result["local_backup"]["source_count"], 6)
            self.assertEqual(result["local_warehouse_row_counts"]["freshretail_train"], 1)
            self.assertEqual(result["local_warehouse_row_counts"]["freshretail_val"], 1)
            self.assertTrue(local_db_path.exists())
            self._assert_local_duckdb_counts(local_db_path)


if __name__ == "__main__":
    unittest.main()
