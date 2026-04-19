from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile, is_zipfile

import pandas as pd
import polars as pl

from research_praedixa.global_dataset.schema import align_lazy_frame_to_canonical_schema, build_series_id_expr


DEFAULT_RAW_DIR = Path("data/commercial_datasets/raw")
DEFAULT_OUTPUT_PATH = Path("data/global_dataset/commercial_external_daily.parquet")
DEFAULT_SOURCE_RUN_ID = "manual_local"
DEFAULT_SILVER_RUN_ID = "manual_local"

DEFAULT_FRESHRETAIL_LT_TRAIN_PATH = DEFAULT_RAW_DIR / "freshretail_lt_train.parquet"
DEFAULT_FRESHRETAIL_LT_EVAL_PATH = DEFAULT_RAW_DIR / "freshretail_lt_eval.parquet"
DEFAULT_UCI_ONLINE_RETAIL_PATH = DEFAULT_RAW_DIR / "uci_online_retail.xlsx"
DEFAULT_UCI_ONLINE_RETAIL_II_PATH = DEFAULT_RAW_DIR / "uci_online_retail_ii.xlsx"
DEFAULT_MENDELEY_ECOMMERCE_PATH = DEFAULT_RAW_DIR / "mendeley_ecommerce.xlsx"
DEFAULT_MENDELEY_PHARMACY_SQL_PATH = DEFAULT_RAW_DIR / "mendeley_pharmacy.sql"
DEFAULT_MENDELEY_PHARMACY_ZIP_PATH = DEFAULT_RAW_DIR / "mendeley_pharmacy.zip"
DEFAULT_MENDELEY_BANGLADESH_PATH = DEFAULT_RAW_DIR / "mendeley_bangladesh.xlsx"

UCI_NON_PRODUCT_DESCRIPTION_PATTERNS = (
    "postage",
    "bank charges",
    "manual",
    "adjustment",
    "discount",
    "samples",
)


@dataclass(frozen=True)
class CommercialDatasetCompatibility:
    """Compatibility summary for one commercially usable dataset."""

    dataset_source: str
    target_semantics: str
    compatible_with_pipeline: bool
    commercial_use_allowed: bool
    reason: str


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


def commercial_dataset_compatibility_matrix() -> list[CommercialDatasetCompatibility]:
    """Return the whitelist verdict for commercially usable datasets."""

    return [
        CommercialDatasetCompatibility(
            dataset_source="freshretail_lt",
            target_semantics="daily normalized product sales by store x product",
            compatible_with_pipeline=True,
            commercial_use_allowed=True,
            reason="FreshRetailNet-LT exposes sale_amount at daily store-product grain under CC BY 4.0.",
        ),
        CommercialDatasetCompatibility(
            dataset_source="uci_online_retail",
            target_semantics="daily sold quantity by product after filtering returns and non-merchandise charges",
            compatible_with_pipeline=True,
            commercial_use_allowed=True,
            reason="Transaction lines expose quantity and invoice timestamp; we aggregate to daily product demand.",
        ),
        CommercialDatasetCompatibility(
            dataset_source="uci_online_retail_ii",
            target_semantics="daily sold quantity by product after filtering returns and non-merchandise charges",
            compatible_with_pipeline=True,
            commercial_use_allowed=True,
            reason="Transaction lines expose quantity and invoice timestamp across two yearly sheets.",
        ),
        CommercialDatasetCompatibility(
            dataset_source="mendeley_ecommerce",
            target_semantics="daily sold quantity by SKU aggregated from order lines",
            compatible_with_pipeline=True,
            commercial_use_allowed=True,
            reason="Order lines expose order_date, prod_sku, and prod_qty under CC BY 4.0.",
        ),
        CommercialDatasetCompatibility(
            dataset_source="mendeley_pharmacy_id",
            target_semantics="daily sold quantity by medicine code aggregated from pharmacy transactions",
            compatible_with_pipeline=True,
            commercial_use_allowed=True,
            reason="SQL dump exposes transaction date, medicine code, quantity, and price from a real pharmacy DB.",
        ),
        CommercialDatasetCompatibility(
            dataset_source="mendeley_bangladesh_retail",
            target_semantics="daily sold quantity for one product",
            compatible_with_pipeline=True,
            commercial_use_allowed=True,
            reason="The dataset is a single-product daily demand series under CC BY 4.0.",
        ),
        CommercialDatasetCompatibility(
            dataset_source="uci_hierarchical_sales",
            target_semantics="daily SKU sales by brand hierarchy",
            compatible_with_pipeline=False,
            commercial_use_allowed=False,
            reason="Excluded for now because the surfaced licensing signals conflict between UCI and the upstream DOI record.",
        ),
    ]


def _resolve_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_excel_all_sheets(path: Path) -> pd.DataFrame:
    if path.is_file() and is_zipfile(path):
        try:
            with ZipFile(path) as archive:
                embedded_excel_names = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
                if embedded_excel_names:
                    workbook = pd.read_excel(
                        BytesIO(archive.read(embedded_excel_names[0])),
                        sheet_name=None,
                        engine="openpyxl",
                    )
                    return pd.concat(workbook.values(), ignore_index=True)
        except BadZipFile:
            pass
    workbook = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return pd.concat(workbook.values(), ignore_index=True)


def _read_mendeley_pharmacy_sql_text(raw_dir: Path) -> str | None:
    sql_path = _resolve_existing_path(DEFAULT_MENDELEY_PHARMACY_SQL_PATH, raw_dir / DEFAULT_MENDELEY_PHARMACY_SQL_PATH.name)
    if sql_path is not None:
        return sql_path.read_text(encoding="utf-8", errors="ignore")

    zip_path = _resolve_existing_path(DEFAULT_MENDELEY_PHARMACY_ZIP_PATH, raw_dir / DEFAULT_MENDELEY_PHARMACY_ZIP_PATH.name)
    if zip_path is None:
        return None
    with ZipFile(zip_path) as archive:
        for member_name in archive.namelist():
            if member_name.lower().endswith(".sql"):
                return archive.read(member_name).decode("utf-8", errors="ignore")
    return None


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
    enriched["snap_flag"] = None
    enriched["weather_precipitation"] = None
    enriched["weather_temperature"] = None
    enriched["weather_humidity"] = None
    enriched["weather_wind_level"] = None
    enriched["anomaly_flag"] = False
    enriched["series_id"] = enriched["location_id"].astype(str) + "__" + enriched["product_id"].astype(str)
    return align_lazy_frame_to_canonical_schema(pl.from_pandas(enriched).lazy()).collect()


def _aggregate_transaction_lines(
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
    normalized = normalized.loc[
        normalized["dt"].notna()
        & normalized["product_id"].ne("")
        & normalized["product_id"].notna()
        & normalized["observed_demand_qty"].notna()
        & normalized["observed_demand_qty"].gt(0)
    ].copy()
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
    else:
        grouped["observed_revenue_net"] = pd.NA
        grouped["avg_selling_price"] = pd.NA
    return _build_common_daily_frame(
        grouped,
        dataset_source=dataset_source,
        source_partition=source_partition,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


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
        [
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
            pl.lit(None).cast(pl.Boolean).alias("snap_flag"),
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
    ).with_columns(build_series_id_expr("location_id", "product_id").alias("series_id"))
    return align_lazy_frame_to_canonical_schema(normalized)


def _is_non_product_retail_line(description: object) -> bool:
    if description is None or pd.isna(description):
        return False
    description_text = str(description).strip().lower()
    return any(pattern in description_text for pattern in UCI_NON_PRODUCT_DESCRIPTION_PATTERNS)


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

    normalized = frame.copy()
    invoice_column = "InvoiceNo" if "InvoiceNo" in normalized.columns else "Invoice"
    unit_price_column = "UnitPrice" if "UnitPrice" in normalized.columns else "Price"
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce")
    normalized[unit_price_column] = pd.to_numeric(normalized[unit_price_column], errors="coerce")
    if invoice_column in normalized.columns:
        normalized[invoice_column] = normalized[invoice_column].astype(str)
    merchandise_mask = ~normalized.get("Description", pd.Series(index=normalized.index, dtype="object")).map(
        _is_non_product_retail_line
    ).fillna(False)
    cancellation_mask = ~normalized.get(invoice_column, pd.Series(index=normalized.index, dtype="object")).astype(str).str.startswith(
        "C"
    )
    normalized = normalized.loc[
        merchandise_mask
        & cancellation_mask
        & normalized["Quantity"].gt(0)
        & normalized[unit_price_column].gt(0)
    ].copy()
    return _aggregate_transaction_lines(
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

    return _aggregate_transaction_lines(
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


def _split_sql_tuples(values_blob: str) -> list[str]:
    tuples: list[str] = []
    start_index: int | None = None
    depth = 0
    in_quote = False
    escaped = False
    for index, char in enumerate(values_blob):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            if depth == 0:
                start_index = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start_index is not None:
                tuples.append(values_blob[start_index:index])
                start_index = None
    return tuples


def _split_sql_fields(tuple_blob: str) -> list[str]:
    fields: list[str] = []
    buffer: list[str] = []
    in_quote = False
    escaped = False
    for char in tuple_blob:
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\":
            buffer.append(char)
            escaped = True
            continue
        if char == "'":
            buffer.append(char)
            in_quote = not in_quote
            continue
        if char == "," and not in_quote:
            fields.append("".join(buffer).strip())
            buffer = []
            continue
        buffer.append(char)
    fields.append("".join(buffer).strip())
    return fields


def _parse_sql_scalar(raw_value: str) -> object:
    if raw_value.upper() == "NULL":
        return None
    if raw_value.startswith("'") and raw_value.endswith("'"):
        inner = raw_value[1:-1]
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")
        return inner
    return raw_value


def _iter_insert_rows(sql_text: str, *, table_name: str) -> list[dict[str, object]]:
    pattern = re.compile(
        rf"INSERT INTO\s+`{re.escape(table_name)}`\s*\((?P<columns>.*?)\)\s*VALUES\s*(?P<values>.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    rows: list[dict[str, object]] = []
    for match in pattern.finditer(sql_text):
        columns = [column.strip().strip("`") for column in match.group("columns").split(",")]
        for tuple_blob in _split_sql_tuples(match.group("values")):
            field_values = [_parse_sql_scalar(value) for value in _split_sql_fields(tuple_blob)]
            if len(field_values) != len(columns):
                continue
            rows.append(dict(zip(columns, field_values, strict=False)))
    return rows


def standardize_mendeley_pharmacy_sql_text(
    sql_text: str,
    *,
    source_partition: str = "historical",
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> pl.DataFrame:
    """Standardize the Indonesian pharmacy SQL dump into daily medicine demand."""

    transaction_rows = _iter_insert_rows(sql_text, table_name="transaction")
    if not transaction_rows:
        raise ValueError("No INSERT rows were found for the `transaction` table in the pharmacy SQL dump.")
    frame = pd.DataFrame(transaction_rows)
    qty_column = "QTY" if "QTY" in frame.columns else "qty"
    date_column = "TGL" if "TGL" in frame.columns else "tgl"
    product_column = "KD_OBAT" if "KD_OBAT" in frame.columns else "kd_obat"
    revenue_column = None
    unit_price_column = None
    if "HJ" in frame.columns:
        unit_price_column = "HJ"
    elif "hj" in frame.columns:
        unit_price_column = "hj"
    return _aggregate_transaction_lines(
        frame,
        dataset_source="mendeley_pharmacy_id",
        source_partition=source_partition,
        location_id="indonesia_pharmacy_1",
        dt_column=date_column,
        product_column=product_column,
        qty_column=qty_column,
        revenue_column=revenue_column,
        unit_price_column=unit_price_column,
        source_run_id=source_run_id,
        silver_run_id=silver_run_id,
    )


def _standardize_freshretail_lt_files(
    *,
    train_path: Path | None,
    eval_path: Path | None,
    source_run_id: str,
    silver_run_id: str,
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    if train_path is not None:
        frames.append(
            standardize_freshretail_lt_lazy_frame(
                pl.scan_parquet(train_path),
                source_partition="historical_train",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            ).collect()
        )
    if eval_path is not None:
        frames.append(
            standardize_freshretail_lt_lazy_frame(
                pl.scan_parquet(eval_path),
                source_partition="historical_eval",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            ).collect()
        )
    return frames


def build_commercial_external_standardized_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    silver_run_id: str = DEFAULT_SILVER_RUN_ID,
) -> Path | None:
    """Build one canonical dataset containing commercially usable public datasets only."""

    root = Path(raw_dir)
    frames: list[pl.DataFrame] = []

    freshretail_lt_train = _resolve_existing_path(root / DEFAULT_FRESHRETAIL_LT_TRAIN_PATH.name)
    freshretail_lt_eval = _resolve_existing_path(root / DEFAULT_FRESHRETAIL_LT_EVAL_PATH.name)
    frames.extend(
        _standardize_freshretail_lt_files(
            train_path=freshretail_lt_train,
            eval_path=freshretail_lt_eval,
            source_run_id=source_run_id,
            silver_run_id=silver_run_id,
        )
    )

    uci_online_retail_path = _resolve_existing_path(
        root / DEFAULT_UCI_ONLINE_RETAIL_PATH.name,
        root / "uci_online_retail.zip",
    )
    if uci_online_retail_path is not None:
        frames.append(
            standardize_uci_online_retail_frame(
                _read_excel_all_sheets(uci_online_retail_path),
                dataset_source="uci_online_retail",
                location_id="uk_online_retail_1",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )

    uci_online_retail_ii_path = _resolve_existing_path(
        root / DEFAULT_UCI_ONLINE_RETAIL_II_PATH.name,
        root / "uci_online_retail_ii.zip",
    )
    if uci_online_retail_ii_path is not None:
        frames.append(
            standardize_uci_online_retail_frame(
                _read_excel_all_sheets(uci_online_retail_ii_path),
                dataset_source="uci_online_retail_ii",
                location_id="uk_online_retail_ii_1",
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )

    mendeley_ecommerce_path = _resolve_existing_path(root / DEFAULT_MENDELEY_ECOMMERCE_PATH.name)
    if mendeley_ecommerce_path is not None:
        frames.append(
            standardize_mendeley_ecommerce_frame(
                pd.read_excel(mendeley_ecommerce_path, engine="openpyxl"),
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )

    pharmacy_sql_text = _read_mendeley_pharmacy_sql_text(root)
    if pharmacy_sql_text is not None:
        frames.append(
            standardize_mendeley_pharmacy_sql_text(
                pharmacy_sql_text,
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )

    bangladesh_path = _resolve_existing_path(root / DEFAULT_MENDELEY_BANGLADESH_PATH.name)
    if bangladesh_path is not None:
        frames.append(
            standardize_mendeley_bangladesh_frame(
                pd.read_excel(bangladesh_path, engine="openpyxl"),
                source_run_id=source_run_id,
                silver_run_id=silver_run_id,
            )
        )

    if not frames:
        return None

    combined = pl.concat(frames, how="vertical_relaxed").sort(["dataset_source", "dt", "location_id", "product_id"])
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(target_path)
    return target_path
