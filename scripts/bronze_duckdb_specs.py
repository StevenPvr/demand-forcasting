from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BronzeTableSpec:
    """Definition of one local bronze table to load into DuckDB."""

    table_name: str
    source_path: Path
    ddl: str
    source_name: str


def _open_location_metadata_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_location_metadata (
        dataset_source VARCHAR,
        location_id VARCHAR,
        country_code VARCHAR,
        region_code VARCHAR,
        city_name VARCHAR,
        latitude DOUBLE,
        longitude DOUBLE,
        school_zone VARCHAR,
        weather_location_label VARCHAR,
        assumption_source VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _open_public_holidays_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_public_holidays (
        country_code VARCHAR,
        dt DATE,
        holiday_name VARCHAR,
        holiday_local_name VARCHAR,
        global_flag BOOLEAN,
        counties_json VARCHAR,
        holiday_types_json VARCHAR,
        source_name VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _open_school_holidays_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_school_holidays (
        dataset_source VARCHAR,
        location_id VARCHAR,
        dt DATE,
        school_holiday_name VARCHAR,
        school_zone VARCHAR,
        source_name VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _open_weather_daily_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_weather_daily (
        dataset_source VARCHAR,
        location_id VARCHAR,
        dt DATE,
        latitude DOUBLE,
        longitude DOUBLE,
        weather_temperature_mean DOUBLE,
        weather_temperature_min DOUBLE,
        weather_temperature_max DOUBLE,
        weather_precipitation_sum DOUBLE,
        weather_relative_humidity_mean DOUBLE,
        weather_wind_speed_mean DOUBLE,
        source_name VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _open_macro_annual_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_macro_annual (
        country_code VARCHAR,
        indicator_code VARCHAR,
        metric_name VARCHAR,
        observation_year INTEGER,
        effective_from DATE,
        metric_value DOUBLE,
        source_name VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _open_macro_timeseries_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_macro_timeseries (
        country_code VARCHAR,
        metric_name VARCHAR,
        source_series_id VARCHAR,
        source_frequency VARCHAR,
        metric_units VARCHAR,
        observation_date DATE,
        period_end DATE,
        effective_from DATE,
        metric_value DOUBLE,
        source_name VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _freshretail_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_freshretail_daily (
        source_partition VARCHAR,
        city_id INTEGER,
        store_id INTEGER,
        management_group_id INTEGER,
        first_category_id INTEGER,
        second_category_id INTEGER,
        third_category_id INTEGER,
        product_id INTEGER,
        dt DATE,
        sale_amount DOUBLE,
        stock_hour6_22_cnt SMALLINT,
        discount DOUBLE,
        holiday_flag BOOLEAN,
        activity_flag BOOLEAN,
        precpt DOUBLE,
        avg_temperature DOUBLE,
        avg_humidity DOUBLE,
        avg_wind_level DOUBLE,
        is_censored BOOLEAN,
        hours_sale_00 DOUBLE,
        hours_sale_01 DOUBLE,
        hours_sale_02 DOUBLE,
        hours_sale_03 DOUBLE,
        hours_sale_04 DOUBLE,
        hours_sale_05 DOUBLE,
        hours_sale_06 DOUBLE,
        hours_sale_07 DOUBLE,
        hours_sale_08 DOUBLE,
        hours_sale_09 DOUBLE,
        hours_sale_10 DOUBLE,
        hours_sale_11 DOUBLE,
        hours_sale_12 DOUBLE,
        hours_sale_13 DOUBLE,
        hours_sale_14 DOUBLE,
        hours_sale_15 DOUBLE,
        hours_sale_16 DOUBLE,
        hours_sale_17 DOUBLE,
        hours_sale_18 DOUBLE,
        hours_sale_19 DOUBLE,
        hours_sale_20 DOUBLE,
        hours_sale_21 DOUBLE,
        hours_sale_22 DOUBLE,
        hours_sale_23 DOUBLE,
        hours_stock_status_00 SMALLINT,
        hours_stock_status_01 SMALLINT,
        hours_stock_status_02 SMALLINT,
        hours_stock_status_03 SMALLINT,
        hours_stock_status_04 SMALLINT,
        hours_stock_status_05 SMALLINT,
        hours_stock_status_06 SMALLINT,
        hours_stock_status_07 SMALLINT,
        hours_stock_status_08 SMALLINT,
        hours_stock_status_09 SMALLINT,
        hours_stock_status_10 SMALLINT,
        hours_stock_status_11 SMALLINT,
        hours_stock_status_12 SMALLINT,
        hours_stock_status_13 SMALLINT,
        hours_stock_status_14 SMALLINT,
        hours_stock_status_15 SMALLINT,
        hours_stock_status_16 SMALLINT,
        hours_stock_status_17 SMALLINT,
        hours_stock_status_18 SMALLINT,
        hours_stock_status_19 SMALLINT,
        hours_stock_status_20 SMALLINT,
        hours_stock_status_21 SMALLINT,
        hours_stock_status_22 SMALLINT,
        hours_stock_status_23 SMALLINT,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _commercial_external_daily_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_commercial_external_daily (
        dataset_source VARCHAR,
        source_partition VARCHAR,
        source_run_id VARCHAR,
        series_id VARCHAR,
        dt DATE,
        location_id VARCHAR,
        product_id VARCHAR,
        region_id VARCHAR,
        org_group_id VARCHAR,
        category_level_1 VARCHAR,
        category_level_2 VARCHAR,
        category_level_3 VARCHAR,
        observed_demand_qty DOUBLE,
        observed_revenue_net DOUBLE,
        observed_discount_amount DOUBLE,
        avg_selling_price DOUBLE,
        promo_flag BOOLEAN,
        holiday_flag BOOLEAN,
        activity_flag BOOLEAN,
        observed_stockout_flag BOOLEAN,
        observed_stockout_available BOOLEAN,
        observed_stockout_intensity DOUBLE,
        location_open_flag BOOLEAN,
        day_complete_flag BOOLEAN,
        missing_sales_flag BOOLEAN,
        calendar_weekday_name VARCHAR,
        calendar_day_of_week SMALLINT,
        calendar_month SMALLINT,
        calendar_year SMALLINT,
        calendar_week_key INTEGER,
        event_name_1 VARCHAR,
        event_type_1 VARCHAR,
        event_name_2 VARCHAR,
        event_type_2 VARCHAR,
        snap_flag BOOLEAN,
        weather_precipitation DOUBLE,
        weather_temperature DOUBLE,
        weather_humidity DOUBLE,
        weather_wind_level DOUBLE,
        anomaly_flag BOOLEAN,
        silver_run_id VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _bakery_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_bakery_order_lines (
        source_partition VARCHAR,
        row_index BIGINT,
        sale_date_raw VARCHAR,
        sale_time_raw VARCHAR,
        ticket_number_raw VARCHAR,
        article_raw VARCHAR,
        quantity_raw DOUBLE,
        unit_price_raw VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _m5_calendar_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_m5_calendar (
        date DATE,
        wm_yr_wk INTEGER,
        weekday VARCHAR,
        wday SMALLINT,
        month SMALLINT,
        year SMALLINT,
        d VARCHAR,
        event_name_1 VARCHAR,
        event_type_1 VARCHAR,
        event_name_2 VARCHAR,
        event_type_2 VARCHAR,
        snap_CA BOOLEAN,
        snap_TX BOOLEAN,
        snap_WI BOOLEAN,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _m5_sell_prices_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_m5_sell_prices (
        store_id VARCHAR,
        item_id VARCHAR,
        wm_yr_wk INTEGER,
        sell_price DOUBLE,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _m5_sales_long_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_m5_sales_long (
        id VARCHAR,
        item_id VARCHAR,
        dept_id VARCHAR,
        cat_id VARCHAR,
        store_id VARCHAR,
        state_id VARCHAR,
        d VARCHAR,
        sale_qty DOUBLE,
        source_partition VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def default_core_bronze_specs(data_dir: str | Path, schema_name: str) -> list[BronzeTableSpec]:
    """Return the local core bronze replacement sources used by the V1 silver pipeline."""

    root = Path(data_dir)
    return [
        BronzeTableSpec(
            table_name="bronze_freshretail_daily",
            source_path=root / "bakery_sales" / "data_train.parquet",
            ddl=_freshretail_ddl(schema_name),
            source_name="freshretail_train",
        ),
        BronzeTableSpec(
            table_name="bronze_freshretail_daily",
            source_path=root / "bakery_sales" / "data_val.parquet",
            ddl=_freshretail_ddl(schema_name),
            source_name="freshretail_val",
        ),
        BronzeTableSpec(
            table_name="bronze_bakery_order_lines",
            source_path=root / "bakery_sales" / "Bakery sales.csv",
            ddl=_bakery_ddl(schema_name),
            source_name="bakery",
        ),
        BronzeTableSpec(
            table_name="bronze_m5_calendar",
            source_path=root / "m5" / "calendar.csv",
            ddl=_m5_calendar_ddl(schema_name),
            source_name="m5_calendar",
        ),
        BronzeTableSpec(
            table_name="bronze_m5_sell_prices",
            source_path=root / "m5" / "sell_prices.csv",
            ddl=_m5_sell_prices_ddl(schema_name),
            source_name="m5_prices",
        ),
        BronzeTableSpec(
            table_name="bronze_m5_sales_long",
            source_path=root / "m5" / "sales_train_validation.csv",
            ddl=_m5_sales_long_ddl(schema_name),
            source_name="m5_sales",
        ),
    ]


def default_active_core_bronze_specs(data_dir: str | Path, schema_name: str) -> list[BronzeTableSpec]:
    """Return the local core bronze sources used by the active commercial workflow."""

    root = Path(data_dir)
    return [
        BronzeTableSpec(
            table_name="bronze_freshretail_daily",
            source_path=root / "bakery_sales" / "data_train.parquet",
            ddl=_freshretail_ddl(schema_name),
            source_name="freshretail_train",
        ),
        BronzeTableSpec(
            table_name="bronze_freshretail_daily",
            source_path=root / "bakery_sales" / "data_val.parquet",
            ddl=_freshretail_ddl(schema_name),
            source_name="freshretail_val",
        ),
        BronzeTableSpec(
            table_name="bronze_bakery_order_lines",
            source_path=root / "bakery_sales" / "Bakery sales.csv",
            ddl=_bakery_ddl(schema_name),
            source_name="bakery",
        ),
        BronzeTableSpec(
            table_name="bronze_commercial_external_daily",
            source_path=root / "global_dataset" / "commercial_external_daily.parquet",
            ddl=_commercial_external_daily_ddl(schema_name),
            source_name="commercial_external_daily",
        ),
    ]


def default_open_exogenous_bronze_specs(
    data_dir: str | Path,
    schema_name: str,
    open_exogenous_dir: str | Path | None = None,
) -> list[BronzeTableSpec]:
    """Return the local open-source exogenous bronze files used by the gold pipeline."""

    root = Path(open_exogenous_dir) if open_exogenous_dir is not None else Path(data_dir) / "external_open"
    return [
        BronzeTableSpec(
            table_name="bronze_open_location_metadata",
            source_path=root / "location_metadata.csv",
            ddl=_open_location_metadata_ddl(schema_name),
            source_name="open_location_metadata",
        ),
        BronzeTableSpec(
            table_name="bronze_open_public_holidays",
            source_path=root / "public_holidays.csv",
            ddl=_open_public_holidays_ddl(schema_name),
            source_name="open_public_holidays",
        ),
        BronzeTableSpec(
            table_name="bronze_open_school_holidays",
            source_path=root / "school_holidays.csv",
            ddl=_open_school_holidays_ddl(schema_name),
            source_name="open_school_holidays",
        ),
        BronzeTableSpec(
            table_name="bronze_open_weather_daily",
            source_path=root / "weather_daily.csv",
            ddl=_open_weather_daily_ddl(schema_name),
            source_name="open_weather_daily",
        ),
        BronzeTableSpec(
            table_name="bronze_open_macro_annual",
            source_path=root / "macro_annual.csv",
            ddl=_open_macro_annual_ddl(schema_name),
            source_name="open_macro_annual",
        ),
        BronzeTableSpec(
            table_name="bronze_open_macro_timeseries",
            source_path=root / "macro_timeseries.csv",
            ddl=_open_macro_timeseries_ddl(schema_name),
            source_name="open_macro_timeseries",
        ),
    ]


def default_bronze_specs(
    data_dir: str | Path,
    schema_name: str,
    open_exogenous_dir: str | Path | None = None,
) -> list[BronzeTableSpec]:
    """Return the local bronze replacement sources used by the silver and gold pipelines."""

    return [
        *default_core_bronze_specs(data_dir, schema_name=schema_name),
        *default_open_exogenous_bronze_specs(
            data_dir,
            schema_name=schema_name,
            open_exogenous_dir=open_exogenous_dir,
        ),
    ]


def default_active_bronze_specs(
    data_dir: str | Path,
    schema_name: str,
    open_exogenous_dir: str | Path | None = None,
) -> list[BronzeTableSpec]:
    """Return the local bronze sources used by the active commercial workflow."""

    return [
        *default_active_core_bronze_specs(data_dir, schema_name=schema_name),
        *default_open_exogenous_bronze_specs(
            data_dir,
            schema_name=schema_name,
            open_exogenous_dir=open_exogenous_dir,
        ),
    ]
