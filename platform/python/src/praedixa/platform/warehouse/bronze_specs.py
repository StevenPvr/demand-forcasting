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
        site_format VARCHAR,
        service_model VARCHAR,
        drive_through_flag BOOLEAN,
        delivery_flag BOOLEAN,
        pickup_flag BOOLEAN,
        late_night_flag BOOLEAN,
        trade_area_type VARCHAR,
        mall_flag BOOLEAN,
        transit_hub_flag BOOLEAN,
        tourism_flag BOOLEAN,
        office_density_bucket VARCHAR,
        residential_density_bucket VARCHAR,
        competition_intensity_bucket VARCHAR,
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


def _open_location_catchment_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_open_location_catchment (
        dataset_source VARCHAR,
        location_id VARCHAR,
        population_1km DOUBLE,
        population_3km DOUBLE,
        office_poi_count_1km INTEGER,
        school_poi_count_1km INTEGER,
        transit_station_count_1km INTEGER,
        mall_poi_count_1km INTEGER,
        tourism_poi_count_1km INTEGER,
        competitor_count_500m INTEGER,
        competitor_count_1km INTEGER,
        parking_score VARCHAR,
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
        indicator_code VARCHAR,
        metric_name VARCHAR,
        frequency_code VARCHAR,
        observation_period VARCHAR,
        effective_from DATE,
        metric_value DOUBLE,
        source_name VARCHAR,
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _freshretail_hour_column_defs(prefix: str, sql_type: str) -> str:
    return ",\n".join(f"        {prefix}_{hour:02d} {sql_type}" for hour in range(24))


def _freshretail_ddl(schema_name: str) -> str:
    hour_sale_columns = _freshretail_hour_column_defs("hours_sale", "DOUBLE")
    hour_stock_columns = _freshretail_hour_column_defs("hours_stock_status", "SMALLINT")
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
{hour_sale_columns},
{hour_stock_columns},
        source_file_path VARCHAR,
        loaded_at TIMESTAMP
    );
    """


def _supplemental_corpus_daily_ddl(schema_name: str) -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS {schema_name};
    CREATE TABLE IF NOT EXISTS {schema_name}.bronze_supplemental_corpus_daily (
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
        target_semantics VARCHAR,
        censor_flag BOOLEAN,
        target_source VARCHAR,
        label_quality_score DOUBLE,
        usable_for_training_flag BOOLEAN,
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


def supplemental_corpus_daily_ddl(schema_name: str) -> str:
    """Return the bronze DDL for the in-memory supplemental corpus runtime table."""

    return _supplemental_corpus_daily_ddl(schema_name)


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
    ]


def default_active_core_bronze_specs(
    data_dir: str | Path,
    schema_name: str,
) -> list[BronzeTableSpec]:
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
    ]


def default_open_exogenous_bronze_specs(
    data_dir: str | Path,
    schema_name: str,
    open_exogenous_dir: str | Path | None = None,
) -> list[BronzeTableSpec]:
    """Return the local open-source exogenous bronze files used by the gold pipeline."""

    root = Path(open_exogenous_dir) if open_exogenous_dir is not None else Path(data_dir) / "external_open"
    spec_definitions = [
        ("bronze_open_location_metadata", root / "location_metadata.csv", _open_location_metadata_ddl, "open_location_metadata"),
        ("bronze_open_location_catchment", root / "location_catchment.csv", _open_location_catchment_ddl, "open_location_catchment"),
        ("bronze_open_public_holidays", root / "public_holidays.csv", _open_public_holidays_ddl, "open_public_holidays"),
        ("bronze_open_school_holidays", root / "school_holidays.csv", _open_school_holidays_ddl, "open_school_holidays"),
        ("bronze_open_weather_daily", root / "weather_daily.csv", _open_weather_daily_ddl, "open_weather_daily"),
        ("bronze_open_macro_annual", root / "macro_annual.csv", _open_macro_annual_ddl, "open_macro_annual"),
        ("bronze_open_macro_timeseries", root / "macro_timeseries.csv", _open_macro_timeseries_ddl, "open_macro_timeseries"),
    ]
    return [
        BronzeTableSpec(
            table_name=table_name,
            source_path=source_path,
            ddl=ddl_builder(schema_name),
            source_name=source_name,
        )
        for table_name, source_path, ddl_builder, source_name in spec_definitions
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
        *default_active_core_bronze_specs(
            data_dir,
            schema_name=schema_name,
        ),
        *default_open_exogenous_bronze_specs(
            data_dir,
            schema_name=schema_name,
            open_exogenous_dir=open_exogenous_dir,
        ),
    ]
