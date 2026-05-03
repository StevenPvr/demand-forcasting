{{ config(tags=["gold", "d1"], schema=env_var("PRAEDIXA_DUCKDB_GOLD_SCHEMA", "gold")) }}

with source_rows as (
    select *
    from {{ ref("silver_daily_product_demand") }}
),
location_metadata as (
    select *
    from {{ ref("silver_open_location_metadata") }}
),
location_profile as (
    select *
    from {{ ref("silver_location_profile") }}
),
product_profile as (
    select *
    from {{ ref("silver_product_profile") }}
),
location_calendar as (
    select *
    from {{ ref("silver_location_calendar") }}
),
source_policy as (
    select
        dataset_source,
        max(legal_basis) as source_legal_basis,
        max(license_type) as source_license_type,
        max(review_status) as source_review_status
    from {{ ref("silver_source_registry") }}
    where source_kind = 'dataset'
      and dataset_source is not null
    group by dataset_source
),
country_calendar as (
    select *
    from {{ ref("silver_open_public_holiday_calendar_daily") }}
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
        bool_or(coalesce(try_cast(day_complete_flag as boolean), false)) as day_complete_flag,
        bool_or(coalesce(try_cast(holiday_flag as boolean), false)) as holiday_flag,
        bool_or(coalesce(try_cast(activity_flag as boolean), false)) as activity_flag,
        max(event_name_1) as event_name_1,
        max(event_type_1) as event_type_1,
        max(event_name_2) as event_name_2,
        max(event_type_2) as event_type_2,
        avg(try_cast(weather_precipitation as double)) as weather_precipitation,
        avg(try_cast(weather_temperature as double)) as weather_temperature,
        avg(try_cast(weather_humidity as double)) as weather_humidity,
        avg(try_cast(weather_wind_level as double)) as weather_wind_level,
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
        profile.drive_through_flag,
        profile.delivery_flag,
        profile.pickup_flag,
        product_profile.product_family,
        product_profile.product_subfamily,
        observed.dt is not null as is_observed_row,
        coalesce(try_cast(context.day_complete_flag as boolean), observed.dt is not null) as day_complete_flag,
        try_cast(observed.observed_stockout_flag as boolean) as observed_stockout_flag,
        coalesce(try_cast(observed.observed_stockout_available as boolean), false) as observed_stockout_available,
        case
            when observed.dt is not null then try_cast(observed.observed_demand_qty as double)
            when coalesce(context.location_has_activity_flag, false) then 0.0
            else null
        end as current_day_demand_qty,
        case
            when observed.dt is null and coalesce(context.location_has_activity_flag, false) then true
            else false
        end as true_zero_demand_flag,
        coalesce(observed.target_semantics, 'observed_sales') as target_semantics,
        coalesce(
            try_cast(observed.censor_flag as boolean),
            coalesce(try_cast(observed.observed_stockout_flag as boolean), false),
            false
        ) as censor_flag,
        coalesce(
            observed.target_source,
            case
                when observed.dt is not null then 'observed_sales'
                when coalesce(context.location_has_activity_flag, false) then 'dense_calendar_zero_fill'
                else 'closed_or_missing_observation'
            end
        ) as target_source,
        coalesce(
            try_cast(observed.label_quality_score as double),
            case
                when coalesce(try_cast(observed.observed_stockout_flag as boolean), false) then 0.5
                when observed.dt is null and coalesce(context.location_has_activity_flag, false) then 0.8
                when observed.dt is null then 0.0
                else 1.0
            end
        ) as label_quality_score,
        coalesce(
            try_cast(observed.usable_for_training_flag as boolean),
            not coalesce(try_cast(observed.observed_stockout_flag as boolean), false),
            false
        ) as usable_for_training_flag,
        try_cast(observed.observed_revenue_net as double) as observed_revenue_net,
        try_cast(observed.observed_discount_amount as double) as observed_discount_amount,
        try_cast(observed.promo_flag as boolean) as promo_flag,
        coalesce(
            try_cast(observed.holiday_flag as boolean),
            try_cast(context.holiday_flag as boolean),
            try_cast(country_calendar.holiday_flag as boolean),
            false
        ) as holiday_flag,
        coalesce(
            try_cast(context.activity_flag as boolean),
            try_cast(observed.activity_flag as boolean),
            observed.dt is not null
        ) as activity_flag,
        coalesce(try_cast(country_calendar.bridge_day_flag as boolean), false) as bridge_day_flag,
        coalesce(try_cast(country_calendar.pre_holiday_flag as boolean), false) as pre_holiday_flag,
        coalesce(try_cast(country_calendar.post_holiday_flag as boolean), false) as post_holiday_flag,
        coalesce(try_cast(location_calendar.month_start_flag as boolean), false) as month_start_flag,
        coalesce(try_cast(location_calendar.month_end_flag as boolean), false) as month_end_flag,
        coalesce(
            try_cast(observed.weather_precipitation as double),
            try_cast(context.weather_precipitation as double),
            try_cast(open_weather.weather_precipitation_sum as double)
        ) as weather_precipitation,
        coalesce(
            try_cast(observed.weather_temperature as double),
            try_cast(context.weather_temperature as double),
            try_cast(open_weather.weather_temperature_mean as double)
        ) as weather_temperature,
        try_cast(open_weather.weather_temperature_min as double) as weather_temperature_min,
        try_cast(open_weather.weather_temperature_max as double) as weather_temperature_max,
        coalesce(
            try_cast(observed.weather_humidity as double),
            try_cast(context.weather_humidity as double),
            try_cast(open_weather.weather_relative_humidity_mean as double)
        ) as weather_humidity,
        coalesce(
            try_cast(observed.weather_wind_level as double),
            try_cast(context.weather_wind_level as double),
            try_cast(open_weather.weather_wind_speed_mean as double)
        ) as weather_wind_level,
        try_cast(macro_country.lending_interest_rate_latest as double) as lending_interest_rate_latest,
        'bakery' as vertical_level_1,
        'bakery_pastry' as vertical_level_2,
        coalesce(metadata.country_code, 'FR') as country_code,
        coalesce(bounds.region_id, metadata.region_code) as region_code,
        metadata.city_name as city_name,
        coalesce(source_policy.source_legal_basis, 'unknown') as source_legal_basis,
        coalesce(source_policy.source_license_type, 'unknown') as source_license_type,
        coalesce(source_policy.source_review_status, 'unknown') as source_review_status,
        coalesce(source_policy.source_review_status, 'unknown') as source_legal_status_snapshot,
        case
            when dense.dataset_source like 'synthetic_foodservice%' then 'synthetic'
            when dense.dataset_source = 'bakery' then 'benchmark'
            else 'supplemental'
        end as source_role,
        dense.dataset_source like 'synthetic_foodservice%' as is_synthetic_source,
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
    left join location_profile as profile
      on dense.dataset_source = profile.dataset_source
     and dense.location_id = profile.location_id
    left join product_profile
      on dense.dataset_source = product_profile.dataset_source
     and dense.product_id = product_profile.product_id
    left join location_calendar
      on dense.dataset_source = location_calendar.dataset_source
     and dense.location_id = location_calendar.location_id
     and dense.dt = location_calendar.dt
    left join source_policy
      on dense.dataset_source = source_policy.dataset_source
    left join country_calendar
      on coalesce(metadata.country_code, 'FR') = country_calendar.country_code
     and dense.dt = country_calendar.dt
    left join open_weather
      on dense.dataset_source = open_weather.dataset_source
     and dense.location_id = open_weather.location_id
     and dense.dt = open_weather.dt
    left join macro_country
      on coalesce(metadata.country_code, 'FR') = macro_country.country_code
     and dense.dt = macro_country.dt
),
imputation_windows as (
    select
        dense_panel.*,
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
        drive_through_flag,
        delivery_flag,
        pickup_flag,
        product_family,
        product_subfamily,
        is_observed_row,
        day_complete_flag,
        observed_stockout_flag,
        observed_stockout_available,
        current_day_demand_qty,
        true_zero_demand_flag,
        target_semantics,
        censor_flag,
        target_source,
        label_quality_score,
        usable_for_training_flag,
        observed_revenue_net,
        observed_discount_amount,
        promo_flag,
        holiday_flag,
        activity_flag,
        bridge_day_flag,
        pre_holiday_flag,
        post_holiday_flag,
        month_start_flag,
        month_end_flag,
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
        lending_interest_rate_latest,
        vertical_level_1,
        vertical_level_2,
        country_code,
        region_code,
        city_name,
        source_legal_basis,
        source_license_type,
        source_review_status,
        source_legal_status_snapshot,
        source_role,
        is_synthetic_source,
        gold_run_id
    from imputation_windows
)
select
    *,
    dt + interval 1 day as target_dt
from imputed_panel
