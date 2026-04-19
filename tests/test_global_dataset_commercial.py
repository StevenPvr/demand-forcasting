from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_praedixa.global_dataset.commercial_external import (  # noqa: E402
    standardize_freshretail_lt_lazy_frame,
    standardize_mendeley_bangladesh_frame,
    standardize_mendeley_ecommerce_frame,
    standardize_mendeley_pharmacy_sql_text,
    standardize_uci_online_retail_frame,
)


class GlobalDatasetCommercialTests(unittest.TestCase):
    def test_freshretail_lt_promo_flag_treats_discount_one_as_non_promo(self) -> None:
        frame = pl.DataFrame(
            {
                "city_id": ["city_a", "city_a"],
                "store_id": ["store_1", "store_1"],
                "management_group_id": ["mg_1", "mg_1"],
                "first_category_id": ["cat_1", "cat_1"],
                "second_category_id": ["dept_1", "dept_1"],
                "third_category_id": ["family_1", "family_1"],
                "product_id": ["sku_1", "sku_2"],
                "dt": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "sale_amount": [12.0, 9.0],
                "stock_hour6_22_cnt": [1, 0],
                "discount": [1.0, 0.8],
                "holiday_flag": [False, False],
                "activity_flag": [False, False],
                "precpt": [0.0, 0.0],
                "avg_temperature": [18.0, 18.0],
                "avg_humidity": [0.6, 0.6],
                "avg_wind_level": [3.0, 3.0],
                "is_censored": [False, False],
            }
        )

        records = standardize_freshretail_lt_lazy_frame(frame.lazy(), source_partition="historical_train").collect()
        self.assertFalse(records.filter(pl.col("product_id") == "sku_1").select("promo_flag").item())
        self.assertTrue(records.filter(pl.col("product_id") == "sku_2").select("promo_flag").item())

    def test_standardize_uci_online_retail_filters_cancellations_and_non_product_lines(self) -> None:
        frame = pd.DataFrame(
            {
                "InvoiceNo": ["10001", "C10002", "10003"],
                "StockCode": ["SKU_1", "SKU_1", "POST"],
                "Description": ["Tea Cup", "Tea Cup", "POSTAGE"],
                "Quantity": [2, -2, 1],
                "InvoiceDate": ["2011-01-01 10:00:00", "2011-01-01 11:00:00", "2011-01-01 12:00:00"],
                "UnitPrice": [3.5, 3.5, 5.0],
            }
        )

        standardized = standardize_uci_online_retail_frame(
            frame,
            dataset_source="uci_online_retail",
            location_id="uk_online_retail_1",
        )
        records = standardized.to_dicts()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["product_id"], "SKU_1")
        self.assertEqual(records[0]["observed_demand_qty"], 2.0)
        self.assertEqual(records[0]["observed_revenue_net"], 7.0)
        self.assertEqual(records[0]["avg_selling_price"], 3.5)

    def test_standardize_mendeley_ecommerce_aggregates_daily_sku_quantity(self) -> None:
        frame = pd.DataFrame(
            {
                "order_id": [1, 2],
                "order_date": ["2024-01-01", "2024-01-01"],
                "prod_sku": ["sku_1", "sku_1"],
                "prod_qty": [2, 3],
            }
        )

        record = standardize_mendeley_ecommerce_frame(frame).to_dicts()[0]

        self.assertEqual(record["dataset_source"], "mendeley_ecommerce")
        self.assertEqual(record["product_id"], "sku_1")
        self.assertEqual(record["observed_demand_qty"], 5.0)

    def test_standardize_mendeley_bangladesh_frame_creates_single_product_series(self) -> None:
        frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "sales": [5, 7]})

        records = standardize_mendeley_bangladesh_frame(frame).sort("dt").to_dicts()

        self.assertEqual(records[0]["dataset_source"], "mendeley_bangladesh_retail")
        self.assertEqual(records[0]["location_id"], "bangladesh_retail_1")
        self.assertEqual(records[0]["product_id"], "product_1")
        self.assertEqual(records[1]["observed_demand_qty"], 7.0)

    def test_standardize_mendeley_pharmacy_sql_text_parses_transaction_table(self) -> None:
        sql_text = """
        CREATE TABLE `transaction` (`NO_RESEP` varchar(20),`TGL` datetime,`KD_OBAT` varchar(20),`QTY` int,`HJ` double);
        INSERT INTO `transaction` (`NO_RESEP`,`TGL`,`KD_OBAT`,`QTY`,`HJ`) VALUES
        ('RX1','2024-01-01 10:00:00','OBAT_A',2,5.0),
        ('RX2','2024-01-01 11:00:00','OBAT_A',3,5.0),
        ('RX3','2024-01-02 09:00:00','OBAT_B',1,7.5);
        """

        records = standardize_mendeley_pharmacy_sql_text(sql_text).sort(["dt", "product_id"]).to_dicts()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["dataset_source"], "mendeley_pharmacy_id")
        self.assertEqual(records[0]["product_id"], "OBAT_A")
        self.assertEqual(records[0]["observed_demand_qty"], 5.0)
        self.assertEqual(records[0]["observed_revenue_net"], 25.0)


if __name__ == "__main__":
    unittest.main()
