from __future__ import annotations

import pandas as pd
import polars as pl

from praedixa.platform.datasets.standardization.schema import align_lazy_frame_to_canonical_schema, build_series_id_expr


DEFAULT_SOURCE_RUN_ID = "manual_local"
DEFAULT_SILVER_RUN_ID = "manual_local"
UCI_NON_PRODUCT_DESCRIPTION_PATTERNS = (
    "postage",
    "bank charges",
    "manual",
    "adjustment",
    "discount",
    "samples",
)


def freshretail_promo_flag_expr(discount_column: str = "discount") -> pl.Expr:
    """Infer a promo flag from the FreshRetail discount field without forcing one encoding."""

    discount = pl.col(discount_column).cast(pl.Float32)
    return (
        pl.when(discount.is_null())
        .then(None)
        .when(discount <= 0)
        .then(False)
        .when(discount == 1)
        .then(False)
        .when(discount < 1)
        .then(True)
        .otherwise(True)
    )


def _ensure_datetime_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _build_common_daily_frame(
    grouped: pd.DataFrame,
    *,
    dataset_source: str,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
    region_id: str | None = None,
    org_group_id: str | None = None,
) -> pl.DataFrame:
    enriched = grouped.copy()
    enriched["dataset_source"] = dataset_source
    enriched["source_partition"] = source_partition
    enriched["source_run_id"] = source_run_id
    enriched["silver_run_id"] = silver_run_id
    enriched["region_id"] = region_id
    enriched["org_group_id"] = org_group_id
    enriched["category_level_1"] = None
    enriched["category_level_2"] = None
    enriched["category_level_3"] = None
    enriched["observed_discount_amount"] = None
    enriched["promo_flag"] = None
    enriched["holiday_flag"] = None
    enriched["activity_flag"] = None
    enriched["observed_stockout_flag"] = None
    enriched["observed_stockout_available"] = False
    enriched["observed_stockout_intensity"] = None
    enriched["location_open_flag"] = True
    enriched["day_complete_flag"] = True
    enriched["missing_sales_flag"] = False
    dt_series = pd.to_datetime(enriched["dt"], errors="coerce")
    enriched["calendar_weekday_name"] = dt_series.dt.day_name()
    enriched["calendar_day_of_week"] = dt_series.dt.dayofweek.astype("Int8")
    enriched["calendar_month"] = dt_series.dt.month.astype("Int8")
    enriched["calendar_year"] = dt_series.dt.year.astype("Int16")
    enriched["calendar_week_key"] = dt_series.dt.isocalendar().week.astype("Int32")
    enriched["event_name_1"] = None
    enriched["event_type_1"] = None
    enriched["event_name_2"] = None
    enriched["event_type_2"] = None
    enriched["weather_precipitation"] = None
    enriched["weather_temperature"] = None
    enriched["weather_humidity"] = None
    enriched["weather_wind_level"] = None
    enriched["anomaly_flag"] = False
    enriched["series_id"] = enriched["location_id"].astype(str) + "__" + enriched["product_id"].astype(str)
    return align_lazy_frame_to_canonical_schema(pl.from_pandas(enriched).lazy()).collect()


def aggregate_transaction_lines(
    frame: pd.DataFrame,
    *,
    dataset_source: str,
    source_partition: str,
    location_id: str,
    dt_column: str,
    product_column: str,
    qty_column: str,
    revenue_column: str | None,
    unit_price_column: str | None,
    source_run_id: str,
    silver_run_id: str,
) -> pl.DataFrame:
    normalized, has_revenue_signal = _normalize_transaction_lines(
        frame=frame,
        location_id=location_id,
        dt_column=dt_column,
        product_column=product_column,
        qty_column=qty_column,
        revenue_column=revenue_column,
        unit_price_column=unit_price_column,
    )
    grouped = _aggregate_normalized_transaction_lines(normalized, has_revenue_signal=has_revenue_signal)
    return _build_common_daily_frame(
        grouped,
        dataset_source=dataset_source,
        source_partition=source_partition,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _normalize_transaction_lines(
    *,
    frame: pd.DataFrame,
    location_id: str,
    dt_column: str,
    product_column: str,
    qty_column: str,
    revenue_column: str | None,
    unit_price_column: str | None,
) -> tuple[pd.DataFrame, bool]:
    normalized = frame.copy()
    normalized["dt"] = _ensure_datetime_date(normalized[dt_column])
    normalized["product_id"] = normalized[product_column].astype(str).str.strip()
    normalized["observed_demand_qty"] = pd.to_numeric(normalized[qty_column], errors="coerce")
    has_revenue_signal = revenue_column is not None or unit_price_column is not None
    if revenue_column is not None:
        normalized["observed_revenue_net"] = pd.to_numeric(normalized[revenue_column], errors="coerce")
    elif unit_price_column is not None:
        normalized["unit_price"] = pd.to_numeric(normalized[unit_price_column], errors="coerce")
        normalized["observed_revenue_net"] = normalized["observed_demand_qty"] * normalized["unit_price"]
    else:
        normalized["observed_revenue_net"] = pd.NA
    normalized["location_id"] = location_id
    filtered = normalized.loc[
        normalized["dt"].notna()
        & normalized["product_id"].ne("")
        & normalized["product_id"].notna()
        & normalized["observed_demand_qty"].notna()
        & normalized["observed_demand_qty"].gt(0)
    ].copy()
    return filtered, has_revenue_signal


def _aggregate_normalized_transaction_lines(
    normalized: pd.DataFrame,
    *,
    has_revenue_signal: bool,
) -> pd.DataFrame:
    grouped = (
        normalized.groupby(["dt", "location_id", "product_id"], dropna=False)
        .agg(
            observed_demand_qty=("observed_demand_qty", "sum"),
            observed_revenue_net=("observed_revenue_net", "sum"),
        )
        .reset_index()
    )
    if has_revenue_signal:
        grouped["avg_selling_price"] = grouped["observed_revenue_net"] / grouped["observed_demand_qty"].replace(0, pd.NA)
        return grouped
    grouped["observed_revenue_net"] = pd.NA
    grouped["avg_selling_price"] = pd.NA
    return grouped


def standardize_freshretail_lt_lazy_frame(
    frame: pl.LazyFrame,
    *,
    source_partition: str,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.LazyFrame:
    """Map FreshRetail-LT daily data into the canonical demand contract."""

    available_columns = set(frame.collect_schema().names())
    if "is_censored" in available_columns:
        stockout_flag_expr = pl.col("is_censored").cast(pl.Boolean)
    else:
        stockout_flag_expr = (pl.col("stock_hour6_22_cnt").cast(pl.Float32) > 0)

    normalized = frame.with_columns(
        _freshretail_lt_projection_columns(
            stockout_flag_expr=stockout_flag_expr,
            source_partition=source_partition,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    ).with_columns(build_series_id_expr("location_id", "product_id").alias("series_id"))
    return align_lazy_frame_to_canonical_schema(normalized)


def _freshretail_lt_projection_columns(
    *,
    stockout_flag_expr: pl.Expr,
    source_partition: str,
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.Expr]:
    return [
        pl.col("dt").cast(pl.Date).alias("dt"),
        pl.col("store_id").cast(pl.Utf8).alias("location_id"),
        pl.col("product_id").cast(pl.Utf8).alias("product_id"),
        pl.col("city_id").cast(pl.Utf8).alias("region_id"),
        pl.col("management_group_id").cast(pl.Utf8).alias("org_group_id"),
        pl.col("first_category_id").cast(pl.Utf8).alias("category_level_1"),
        pl.col("second_category_id").cast(pl.Utf8).alias("category_level_2"),
        pl.col("third_category_id").cast(pl.Utf8).alias("category_level_3"),
        pl.col("sale_amount").cast(pl.Float32).alias("observed_demand_qty"),
        pl.lit(None).cast(pl.Float32).alias("observed_revenue_net"),
        pl.col("discount").cast(pl.Float32).alias("observed_discount_amount"),
        pl.lit(None).cast(pl.Float32).alias("avg_selling_price"),
        freshretail_promo_flag_expr().alias("promo_flag"),
        pl.col("holiday_flag").cast(pl.Boolean).alias("holiday_flag"),
        pl.col("activity_flag").cast(pl.Boolean).alias("activity_flag"),
        stockout_flag_expr.alias("observed_stockout_flag"),
        pl.lit(True).alias("observed_stockout_available"),
        pl.col("stock_hour6_22_cnt").cast(pl.Float32).alias("observed_stockout_intensity"),
        pl.lit(True).alias("location_open_flag"),
        pl.lit(True).alias("day_complete_flag"),
        pl.lit(False).alias("missing_sales_flag"),
        pl.col("dt").cast(pl.Date).dt.strftime("%A").alias("calendar_weekday_name"),
        (pl.col("dt").cast(pl.Date).dt.weekday() - 1).cast(pl.Int8).alias("calendar_day_of_week"),
        pl.col("dt").cast(pl.Date).dt.month().cast(pl.Int8).alias("calendar_month"),
        pl.col("dt").cast(pl.Date).dt.year().cast(pl.Int16).alias("calendar_year"),
        pl.col("dt").cast(pl.Date).dt.week().cast(pl.Int32).alias("calendar_week_key"),
        pl.lit(None).cast(pl.Utf8).alias("event_name_1"),
        pl.lit(None).cast(pl.Utf8).alias("event_type_1"),
        pl.lit(None).cast(pl.Utf8).alias("event_name_2"),
        pl.lit(None).cast(pl.Utf8).alias("event_type_2"),
        pl.col("precpt").cast(pl.Float32).alias("weather_precipitation"),
        pl.col("avg_temperature").cast(pl.Float32).alias("weather_temperature"),
        pl.col("avg_humidity").cast(pl.Float32).alias("weather_humidity"),
        pl.col("avg_wind_level").cast(pl.Float32).alias("weather_wind_level"),
        pl.lit(False).alias("anomaly_flag"),
        pl.lit("freshretail_lt").alias("dataset_source"),
        pl.lit(source_partition).alias("source_partition"),
        pl.lit(source_run_id).alias("source_run_id"),
        pl.lit(silver_run_id).alias("silver_run_id"),
    ]


def _is_non_product_retail_line(description: object) -> bool:
    if description is None:
        return False
    if isinstance(description, float) and pd.isna(description):
        return False
    description_text = str(description).strip().lower()
    return any(pattern in description_text for pattern in UCI_NON_PRODUCT_DESCRIPTION_PATTERNS)


def _uci_invoice_column(frame: pd.DataFrame) -> str:
    return "InvoiceNo" if "InvoiceNo" in frame.columns else "Invoice"


def _uci_unit_price_column(frame: pd.DataFrame) -> str:
    return "UnitPrice" if "UnitPrice" in frame.columns else "Price"


def _uci_description_series(frame: pd.DataFrame) -> pd.Series:
    description_series = frame.get("Description")
    if description_series is None:
        return pd.Series(index=frame.index, dtype="object")
    return description_series


def _uci_invoice_series(frame: pd.DataFrame, invoice_column: str) -> pd.Series:
    invoice_series = frame.get(invoice_column)
    if invoice_series is None:
        return pd.Series(index=frame.index, dtype="object")
    return invoice_series


def _filtered_uci_online_retail_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    normalized = frame.copy()
    invoice_column = _uci_invoice_column(normalized)
    unit_price_column = _uci_unit_price_column(normalized)
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce")
    normalized[unit_price_column] = pd.to_numeric(normalized[unit_price_column], errors="coerce")
    if invoice_column in normalized.columns:
        normalized[invoice_column] = normalized[invoice_column].astype(str)
    merchandise_mask = ~_uci_description_series(normalized).map(_is_non_product_retail_line).fillna(False)
    cancellation_mask = ~_uci_invoice_series(normalized, invoice_column).astype(str).str.startswith("C")
    filtered = normalized.loc[
        merchandise_mask
        & cancellation_mask
        & normalized["Quantity"].gt(0)
        & normalized[unit_price_column].gt(0)
    ].copy()
    return filtered, unit_price_column


def standardize_uci_online_retail_frame(
    frame: pd.DataFrame,
    *,
    dataset_source: str,
    location_id: str,
    source_partition: str = "historical",
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Standardize UCI online retail transaction lines into daily product demand."""

    normalized, unit_price_column = _filtered_uci_online_retail_frame(frame)
    return aggregate_transaction_lines(
        normalized,
        dataset_source=dataset_source,
        source_partition=source_partition,
        location_id=location_id,
        dt_column="InvoiceDate",
        product_column="StockCode",
        qty_column="Quantity",
        revenue_column=None,
        unit_price_column=unit_price_column,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def standardize_mendeley_ecommerce_frame(
    frame: pd.DataFrame,
    *,
    source_partition: str = "historical",
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Standardize the Mendeley e-commerce order-line dataset."""

    return aggregate_transaction_lines(
        frame,
        dataset_source="mendeley_ecommerce",
        source_partition=source_partition,
        location_id="south_asia_ecommerce_1",
        dt_column="order_date",
        product_column="prod_sku",
        qty_column="prod_qty",
        revenue_column=None,
        unit_price_column=None,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def standardize_mendeley_bangladesh_frame(
    frame: pd.DataFrame,
    *,
    source_partition: str = "historical",
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Standardize the single-product Bangladeshi retailer demand dataset."""

    normalized = frame.rename(columns={"date": "dt", "sales": "observed_demand_qty"}).copy()
    normalized["dt"] = _ensure_datetime_date(normalized["dt"])
    normalized["observed_demand_qty"] = pd.to_numeric(normalized["observed_demand_qty"], errors="coerce")
    normalized = normalized.loc[normalized["dt"].notna() & normalized["observed_demand_qty"].notna()].copy()
    normalized["location_id"] = "bangladesh_retail_1"
    normalized["product_id"] = "product_1"
    normalized["observed_revenue_net"] = pd.NA
    normalized["avg_selling_price"] = pd.NA
    grouped = normalized[["dt", "location_id", "product_id", "observed_demand_qty", "observed_revenue_net", "avg_selling_price"]]
    return _build_common_daily_frame(
        grouped,
        dataset_source="mendeley_bangladesh_retail",
        source_partition=source_partition,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )
