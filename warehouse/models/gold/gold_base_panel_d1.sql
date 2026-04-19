{{ config(tags=["gold", "d1"], schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

with source_rows as (
    select *
    from {{ ref("silver_daily_product_demand") }}
),
location_metadata as (
    select *
    from {{ ref("silver_open_location_metadata") }}
),
country_calendar as (
    select *
    from {{ ref("silver_open_public_holiday_calendar_daily") }}
),
school_calendar as (
    select *
    from {{ ref("silver_open_school_holidays_daily") }}
),
open_weather as (
    select *
    from {{ ref("silver_open_weather_daily") }}
),
macro_country as (
    select *
    from {{ ref("silver_open_macro_country_daily") }}
),
series_bounds as (
    select
        dataset_source,
        location_id,
        product_id,
        max(series_id) as series_id,
        min(dt) as min_dt,
        max(dt) as max_dt,
        max(region_id) as region_id,
        max(org_group_id) as org_group_id,
        max(category_level_1) as category_level_1,
        max(category_level_2) as category_level_2,
        max(category_level_3) as category_level_3,
        max(source_partition) as source_partition,
        max(source_run_id) as source_run_id
    from source_rows
    group by
        dataset_source,
        location_id,
        product_id
),
location_day_context as (
    select
        dataset_source,
        location_id,
        dt,
        true as location_has_activity_flag,
        bool_or(day_complete_flag) as day_complete_flag,
        bool_or(missing_sales_flag) as missing_sales_flag,
        bool_or(holiday_flag) as holiday_flag,
        bool_or(activity_flag) as activity_flag,
        max(event_name_1) as event_name_1,
        max(event_type_1) as event_type_1,
        max(event_name_2) as event_name_2,
        max(event_type_2) as event_type_2,
        bool_or(snap_flag) as snap_flag,
        avg(weather_precipitation) as weather_precipitation,
        avg(weather_temperature) as weather_temperature,
        avg(weather_humidity) as weather_humidity,
        avg(weather_wind_level) as weather_wind_level,
        max(source_partition) as source_partition
    from source_rows
    group by
        dataset_source,
        location_id,
        dt
),
dense_dates as (
    select
        bounds.dataset_source,
        bounds.location_id,
        bounds.product_id,
        bounds.series_id,
        generated.dt
    from series_bounds as bounds
    cross join generate_series(bounds.min_dt, bounds.max_dt, interval 1 day) as generated(dt)
),
dense_panel as (
    select
        dense.dataset_source,
        coalesce(observed.source_partition, context.source_partition, bounds.source_partition) as source_partition,
        coalesce(observed.source_run_id, bounds.source_run_id, 'warehouse_run') as source_run_id,
        dense.series_id,
        dense.dt,
        dense.location_id,
        dense.product_id,
        coalesce(bounds.region_id, metadata.region_code) as region_id,
        bounds.org_group_id,
        bounds.category_level_1,
        bounds.category_level_2,
        bounds.category_level_3,
        observed.dt is not null as is_observed_row,
        coalesce(context.location_has_activity_flag, false) as location_open_flag,
        true as product_active_flag,
        coalesce(context.day_complete_flag, observed.dt is not null) as day_complete_flag,
        coalesce(context.missing_sales_flag, false) as missing_sales_flag,
        observed.observed_stockout_flag,
        coalesce(observed.observed_stockout_available, false) as observed_stockout_available,
        coalesce(observed.anomaly_flag, false) as anomaly_flag,
        case
            when observed.dt is not null then observed.observed_demand_qty
            when coalesce(context.location_has_activity_flag, false) then 0.0
            else null
        end as current_day_demand_qty,
        case
            when observed.dt is null and coalesce(context.location_has_activity_flag, false) then true
            else false
        end as true_zero_demand_flag,
        case
            when observed.dt is null and not coalesce(context.location_has_activity_flag, false) then true
            else false
        end as location_closed_flag,
        observed.observed_revenue_net,
        observed.avg_selling_price,
        observed.observed_discount_amount,
        observed.promo_flag,
        coalesce(observed.holiday_flag, context.holiday_flag, country_calendar.holiday_flag, false) as holiday_flag,
        coalesce(context.activity_flag, observed.activity_flag) as activity_flag,
        coalesce(
            context.event_name_1,
            observed.event_name_1,
            context.event_name_2,
            observed.event_name_2,
            country_calendar.holiday_name
        ) as holiday_name,
        coalesce(school_calendar.school_holiday_flag, false) as school_holiday_flag,
        coalesce(country_calendar.bridge_day_flag, false) as bridge_day_flag,
        coalesce(country_calendar.pre_holiday_flag, false) as pre_holiday_flag,
        coalesce(country_calendar.post_holiday_flag, false) as post_holiday_flag,
        coalesce(context.snap_flag, observed.snap_flag) as snap_flag,
        coalesce(observed.weather_precipitation, context.weather_precipitation, open_weather.weather_precipitation_sum) as weather_precipitation,
        coalesce(observed.weather_temperature, context.weather_temperature, open_weather.weather_temperature_mean) as weather_temperature,
        open_weather.weather_temperature_min as weather_temperature_min,
        open_weather.weather_temperature_max as weather_temperature_max,
        coalesce(observed.weather_humidity, context.weather_humidity, open_weather.weather_relative_humidity_mean) as weather_humidity,
        coalesce(observed.weather_wind_level, context.weather_wind_level, open_weather.weather_wind_speed_mean) as weather_wind_level,
        macro_country.inflation_cpi_latest,
        macro_country.food_cpi_latest,
        macro_country.policy_rate_latest,
        macro_country.gdp_growth_latest,
        macro_country.gdp_quarterly_level_latest,
        macro_country.gdp_current_usd_latest,
        macro_country.unemployment_rate_latest,
        macro_country.consumer_confidence_latest,
        macro_country.retail_sales_index_latest,
        macro_country.lending_interest_rate_latest,
        macro_country.government_debt_pct_gdp_latest,
        case
            when dense.dataset_source = 'freshretail' then 'public_freshretailnet'
            when dense.dataset_source = 'freshretail_lt' then 'public_freshretailnet_lt'
            when dense.dataset_source = 'uci_online_retail' then 'public_uci_online_retail'
            when dense.dataset_source = 'uci_online_retail_ii' then 'public_uci_online_retail_ii'
            when dense.dataset_source = 'mendeley_ecommerce' then 'public_mendeley_ecommerce'
            when dense.dataset_source = 'mendeley_pharmacy_id' then 'public_mendeley_pharmacy_id'
            when dense.dataset_source = 'mendeley_bangladesh_retail' then 'public_mendeley_bangladesh_retail'
            when dense.dataset_source = 'bakery' then 'public_bakery_sales'
            else 'unknown_public_dataset'
        end as client_id,
        case
            when dense.dataset_source = 'bakery' then 'bakery'
            else 'retail'
        end as vertical_level_1,
        case
            when dense.dataset_source = 'freshretail' then 'grocery_delivery'
            when dense.dataset_source = 'freshretail_lt' then 'grocery_delivery'
            when dense.dataset_source in ('uci_online_retail', 'uci_online_retail_ii', 'mendeley_ecommerce') then 'ecommerce'
            when dense.dataset_source = 'mendeley_pharmacy_id' then 'pharmacy_retail'
            when dense.dataset_source = 'mendeley_bangladesh_retail' then 'retail'
            when dense.dataset_source = 'bakery' then 'bakery_pastry'
            else null
        end as vertical_level_2,
        coalesce(
            metadata.country_code,
            case
                when dense.dataset_source = 'freshretail' then 'CN'
                when dense.dataset_source = 'freshretail_lt' then 'CN'
                when dense.dataset_source in ('uci_online_retail', 'uci_online_retail_ii') then 'GB'
                when dense.dataset_source = 'mendeley_pharmacy_id' then 'ID'
                when dense.dataset_source = 'mendeley_bangladesh_retail' then 'BD'
                when dense.dataset_source = 'bakery' then 'FR'
                else null
            end
        ) as country_code,
        coalesce(bounds.region_id, metadata.region_code) as region_code,
        metadata.city_name as city_name,
        false as freshretail_rescaled_flag,
        'native_observed' as target_scale_assumption,
        '{{ env_var("PRAEDIXA_GOLD_RUN_ID", "manual") }}' as gold_run_id
    from dense_dates as dense
    inner join series_bounds as bounds
      on dense.dataset_source = bounds.dataset_source
     and dense.location_id = bounds.location_id
     and dense.product_id = bounds.product_id
    left join source_rows as observed
      on dense.dataset_source = observed.dataset_source
     and dense.dt = observed.dt
     and dense.location_id = observed.location_id
     and dense.product_id = observed.product_id
    left join location_day_context as context
      on dense.dataset_source = context.dataset_source
     and dense.location_id = context.location_id
     and dense.dt = context.dt
    left join location_metadata as metadata
      on dense.dataset_source = metadata.dataset_source
     and dense.location_id = metadata.location_id
    left join country_calendar
      on coalesce(
            metadata.country_code,
            case
                when dense.dataset_source = 'freshretail' then 'CN'
                when dense.dataset_source = 'freshretail_lt' then 'CN'
                when dense.dataset_source in ('uci_online_retail', 'uci_online_retail_ii') then 'GB'
                when dense.dataset_source = 'mendeley_pharmacy_id' then 'ID'
                when dense.dataset_source = 'mendeley_bangladesh_retail' then 'BD'
                when dense.dataset_source = 'bakery' then 'FR'
                else null
            end
         ) = country_calendar.country_code
     and dense.dt = country_calendar.dt
    left join school_calendar
      on dense.dataset_source = school_calendar.dataset_source
     and dense.location_id = school_calendar.location_id
     and dense.dt = school_calendar.dt
    left join open_weather
      on dense.dataset_source = open_weather.dataset_source
     and dense.location_id = open_weather.location_id
     and dense.dt = open_weather.dt
    left join macro_country
      on coalesce(
            metadata.country_code,
            case
                when dense.dataset_source = 'freshretail' then 'CN'
                when dense.dataset_source = 'freshretail_lt' then 'CN'
                when dense.dataset_source in ('uci_online_retail', 'uci_online_retail_ii') then 'GB'
                when dense.dataset_source = 'mendeley_pharmacy_id' then 'ID'
                when dense.dataset_source = 'mendeley_bangladesh_retail' then 'BD'
                when dense.dataset_source = 'bakery' then 'FR'
                else null
            end
         ) = macro_country.country_code
     and dense.dt = macro_country.dt
),
imputation_windows as (
    select
        dense_panel.*,
        last_value(case when avg_selling_price is not null then avg_selling_price end ignore nulls) over price_window as price_ffill_value,
        last_value(case when avg_selling_price is not null then dt end ignore nulls) over price_window as price_ffill_dt,
        last_value(case when weather_temperature is not null then weather_temperature end ignore nulls) over location_window as weather_temperature_ffill_value,
        last_value(case when weather_temperature is not null then dt end ignore nulls) over location_window as weather_temperature_ffill_dt,
        last_value(case when weather_temperature_min is not null then weather_temperature_min end ignore nulls) over location_window as weather_temperature_min_ffill_value,
        last_value(case when weather_temperature_min is not null then dt end ignore nulls) over location_window as weather_temperature_min_ffill_dt,
        last_value(case when weather_temperature_max is not null then weather_temperature_max end ignore nulls) over location_window as weather_temperature_max_ffill_value,
        last_value(case when weather_temperature_max is not null then dt end ignore nulls) over location_window as weather_temperature_max_ffill_dt,
        last_value(case when weather_humidity is not null then weather_humidity end ignore nulls) over location_window as weather_humidity_ffill_value,
        last_value(case when weather_humidity is not null then dt end ignore nulls) over location_window as weather_humidity_ffill_dt,
        last_value(case when weather_wind_level is not null then weather_wind_level end ignore nulls) over location_window as weather_wind_level_ffill_value,
        last_value(case when weather_wind_level is not null then dt end ignore nulls) over location_window as weather_wind_level_ffill_dt
    from dense_panel
    window
        price_window as (
            partition by dataset_source, location_id, product_id
            order by dt
            rows between unbounded preceding and current row
        ),
        location_window as (
            partition by dataset_source, location_id
            order by dt
            rows between unbounded preceding and current row
        )
),
imputed_panel as (
    select
        dataset_source,
        source_partition,
        source_run_id,
        series_id,
        dt,
        location_id,
        product_id,
        region_id,
        org_group_id,
        category_level_1,
        category_level_2,
        category_level_3,
        is_observed_row,
        location_open_flag,
        product_active_flag,
        day_complete_flag,
        missing_sales_flag,
        observed_stockout_flag,
        observed_stockout_available,
        anomaly_flag,
        current_day_demand_qty,
        true_zero_demand_flag,
        location_closed_flag,
        observed_revenue_net,
        case
            when avg_selling_price is not null then avg_selling_price
            when price_ffill_value is not null
             and date_diff('day', price_ffill_dt, dt) between 1 and {{ env_var("PRAEDIXA_PRICE_FFILL_LIMIT_DAYS", "28") | int }}
            then price_ffill_value
            else avg_selling_price
        end as avg_selling_price,
        observed_discount_amount,
        promo_flag,
        holiday_flag,
        activity_flag,
        holiday_name,
        school_holiday_flag,
        bridge_day_flag,
        pre_holiday_flag,
        post_holiday_flag,
        snap_flag,
        case
            when weather_temperature is not null then weather_temperature
            when weather_temperature_ffill_value is not null
             and date_diff('day', weather_temperature_ffill_dt, dt) between 1 and {{ env_var("PRAEDIXA_WEATHER_FFILL_LIMIT_DAYS", "2") | int }}
            then weather_temperature_ffill_value
            else weather_temperature
        end as weather_temperature,
        case
            when weather_temperature_min is not null then weather_temperature_min
            when weather_temperature_min_ffill_value is not null
             and date_diff('day', weather_temperature_min_ffill_dt, dt) between 1 and {{ env_var("PRAEDIXA_WEATHER_FFILL_LIMIT_DAYS", "2") | int }}
            then weather_temperature_min_ffill_value
            else weather_temperature_min
        end as weather_temperature_min,
        case
            when weather_temperature_max is not null then weather_temperature_max
            when weather_temperature_max_ffill_value is not null
             and date_diff('day', weather_temperature_max_ffill_dt, dt) between 1 and {{ env_var("PRAEDIXA_WEATHER_FFILL_LIMIT_DAYS", "2") | int }}
            then weather_temperature_max_ffill_value
            else weather_temperature_max
        end as weather_temperature_max,
        weather_precipitation,
        case
            when weather_humidity is not null then weather_humidity
            when weather_humidity_ffill_value is not null
             and date_diff('day', weather_humidity_ffill_dt, dt) between 1 and {{ env_var("PRAEDIXA_WEATHER_FFILL_LIMIT_DAYS", "2") | int }}
            then weather_humidity_ffill_value
            else weather_humidity
        end as weather_humidity,
        case
            when weather_wind_level is not null then weather_wind_level
            when weather_wind_level_ffill_value is not null
             and date_diff('day', weather_wind_level_ffill_dt, dt) between 1 and {{ env_var("PRAEDIXA_WEATHER_FFILL_LIMIT_DAYS", "2") | int }}
            then weather_wind_level_ffill_value
            else weather_wind_level
        end as weather_wind_level,
        inflation_cpi_latest,
        food_cpi_latest,
        policy_rate_latest,
        gdp_growth_latest,
        gdp_quarterly_level_latest,
        gdp_current_usd_latest,
        unemployment_rate_latest,
        consumer_confidence_latest,
        retail_sales_index_latest,
        lending_interest_rate_latest,
        government_debt_pct_gdp_latest,
        client_id,
        vertical_level_1,
        vertical_level_2,
        country_code,
        region_code,
        city_name,
        freshretail_rescaled_flag,
        target_scale_assumption,
        gold_run_id
    from imputation_windows
)
select
    *,
    dt + interval 1 day as target_dt,
    avg_selling_price is not null as price_available,
    weather_temperature is not null
        or weather_precipitation is not null
        or weather_humidity is not null
        or weather_wind_level is not null as weather_available,
    (
        inflation_cpi_latest is not null
        or food_cpi_latest is not null
        or policy_rate_latest is not null
        or gdp_growth_latest is not null
        or gdp_quarterly_level_latest is not null
        or gdp_current_usd_latest is not null
        or unemployment_rate_latest is not null
        or consumer_confidence_latest is not null
        or retail_sales_index_latest is not null
        or lending_interest_rate_latest is not null
        or government_debt_pct_gdp_latest is not null
    ) as macro_available
from imputed_panel
