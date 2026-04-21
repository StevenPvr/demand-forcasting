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
location_catchment as (
    select *
    from {{ ref("silver_location_catchment") }}
),
location_capacity_daily as (
    select *
    from {{ ref("silver_location_capacity_daily") }}
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
        profile.site_format,
        profile.service_model,
        profile.drive_through_flag,
        profile.delivery_flag,
        profile.pickup_flag,
        profile.late_night_flag,
        profile.trade_area_type,
        profile.mall_flag,
        profile.transit_hub_flag,
        profile.tourism_flag,
        profile.office_density_bucket,
        profile.residential_density_bucket,
        profile.competition_intensity_bucket,
        product_profile.product_family,
        product_profile.product_subfamily,
        product_profile.menu_role,
        product_profile.price_band,
        product_profile.bundle_flag,
        product_profile.core_menu_flag,
        product_profile.add_on_flag,
        product_profile.beverage_flag,
        product_profile.dessert_flag,
        product_profile.breakfast_flag,
        product_profile.lunch_flag,
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
        coalesce(context.activity_flag, observed.activity_flag, observed.dt is not null) as activity_flag,
        country_calendar.holiday_name as holiday_name,
        coalesce(school_calendar.school_holiday_flag, false) as school_holiday_flag,
        school_calendar.school_holiday_name as school_holiday_name,
        coalesce(country_calendar.bridge_day_flag, false) as bridge_day_flag,
        coalesce(country_calendar.pre_holiday_flag, false) as pre_holiday_flag,
        coalesce(country_calendar.post_holiday_flag, false) as post_holiday_flag,
        coalesce(location_calendar.holiday_flag_local, false) as holiday_flag_local,
        location_calendar.holiday_name as holiday_name_local,
        coalesce(location_calendar.school_holiday_flag_local, false) as school_holiday_flag_local,
        location_calendar.school_holiday_name as school_holiday_name_local,
        coalesce(location_calendar.bridge_day_flag, false) as bridge_day_flag_local,
        coalesce(location_calendar.pre_holiday_flag, false) as pre_holiday_flag_local,
        coalesce(location_calendar.post_holiday_flag, false) as post_holiday_flag_local,
        coalesce(location_calendar.payday_flag, false) as payday_flag,
        coalesce(location_calendar.month_start_flag, false) as month_start_flag,
        coalesce(location_calendar.month_end_flag, false) as month_end_flag,
        coalesce(location_calendar.event_count_local, 0) as event_count_local,
        coalesce(location_calendar.event_intensity_score, 0.0) as event_intensity_score,
        coalesce(location_calendar.major_sports_event_flag, false) as major_sports_event_flag,
        coalesce(location_calendar.exceptional_closure_flag, false) as exceptional_closure_flag,
        coalesce(observed.weather_precipitation, context.weather_precipitation, open_weather.weather_precipitation_sum) as weather_precipitation,
        coalesce(observed.weather_temperature, context.weather_temperature, open_weather.weather_temperature_mean) as weather_temperature,
        open_weather.weather_temperature_min as weather_temperature_min,
        open_weather.weather_temperature_max as weather_temperature_max,
        coalesce(observed.weather_humidity, context.weather_humidity, open_weather.weather_relative_humidity_mean) as weather_humidity,
        coalesce(observed.weather_wind_level, context.weather_wind_level, open_weather.weather_wind_speed_mean) as weather_wind_level,
        catchment.population_1km,
        catchment.population_3km,
        catchment.office_poi_count_1km,
        catchment.school_poi_count_1km,
        catchment.transit_station_count_1km,
        catchment.mall_poi_count_1km,
        catchment.tourism_poi_count_1km,
        catchment.competitor_count_500m,
        catchment.competitor_count_1km,
        catchment.parking_score,
        capacity.closure_minutes,
        capacity.channel_disabled_flag,
        capacity.kitchen_saturation_flag,
        capacity.assortment_restriction_flag,
        capacity.order_cutoff_flag,
        macro_country.gdp_growth_latest,
        macro_country.gdp_current_usd_latest,
        macro_country.lending_interest_rate_latest,
        macro_country.government_debt_pct_gdp_latest,
        macro_country.fr_business_climate_latest,
        macro_country.fr_unemployment_rate_latest,
        macro_country.fr_cpi_yoy_latest,
        macro_country.fr_food_cpi_yoy_latest,
        macro_country.fr_retail_food_volume_index_latest,
        case
            when dense.dataset_source = 'freshretail' then 'public_freshretailnet'
            when dense.dataset_source = 'freshretail_lt' then 'public_freshretailnet_lt'
            when dense.dataset_source = 'uci_online_retail' then 'public_uci_online_retail'
            when dense.dataset_source = 'uci_online_retail_ii' then 'public_uci_online_retail_ii'
            when dense.dataset_source = 'mendeley_ecommerce' then 'public_mendeley_ecommerce'
            when dense.dataset_source = 'mendeley_pharmacy_id' then 'public_mendeley_pharmacy_id'
            when dense.dataset_source = 'mendeley_bangladesh_retail' then 'public_mendeley_bangladesh_retail'
            when dense.dataset_source = 'bakery' then 'public_bakery_sales'
            when dense.dataset_source = 'first_party_daily' then 'first_party_client'
            when dense.dataset_source = 'synthetic_v1' then 'synthetic_training_corpus'
            else 'unknown_public_dataset'
        end as client_id,
        case
            when dense.dataset_source = 'bakery' then 'bakery'
            when dense.dataset_source = 'first_party_daily' then 'food_service'
            when dense.dataset_source = 'synthetic_v1' then 'food_service'
            else 'retail'
        end as vertical_level_1,
        case
            when dense.dataset_source = 'freshretail' then 'grocery_delivery'
            when dense.dataset_source = 'freshretail_lt' then 'grocery_delivery'
            when dense.dataset_source in ('uci_online_retail', 'uci_online_retail_ii', 'mendeley_ecommerce') then 'ecommerce'
            when dense.dataset_source = 'mendeley_pharmacy_id' then 'pharmacy_retail'
            when dense.dataset_source = 'mendeley_bangladesh_retail' then 'retail'
            when dense.dataset_source = 'bakery' then 'bakery_pastry'
            when dense.dataset_source = 'first_party_daily' then 'client_operation'
            when dense.dataset_source = 'synthetic_v1' then 'synthetic_operation'
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
        coalesce(source_policy.source_legal_basis, 'unknown') as source_legal_basis,
        coalesce(source_policy.source_license_type, 'unknown') as source_license_type,
        coalesce(source_policy.source_review_status, 'unknown') as source_review_status,
        coalesce(source_policy.source_review_status, 'unknown') as source_legal_status_snapshot,
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
    left join location_catchment as catchment
      on dense.dataset_source = catchment.dataset_source
     and dense.location_id = catchment.location_id
    left join location_capacity_daily as capacity
      on dense.dataset_source = capacity.dataset_source
     and dense.location_id = capacity.location_id
     and dense.dt = capacity.dt
    left join source_policy
      on dense.dataset_source = source_policy.dataset_source
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
        site_format,
        service_model,
        drive_through_flag,
        delivery_flag,
        pickup_flag,
        late_night_flag,
        trade_area_type,
        mall_flag,
        transit_hub_flag,
        tourism_flag,
        office_density_bucket,
        residential_density_bucket,
        competition_intensity_bucket,
        product_family,
        product_subfamily,
        menu_role,
        price_band,
        bundle_flag,
        core_menu_flag,
        add_on_flag,
        beverage_flag,
        dessert_flag,
        breakfast_flag,
        lunch_flag,
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
        holiday_name,
        activity_flag,
        school_holiday_flag,
        school_holiday_name,
        bridge_day_flag,
        pre_holiday_flag,
        post_holiday_flag,
        holiday_flag_local,
        holiday_name_local,
        school_holiday_flag_local,
        school_holiday_name_local,
        bridge_day_flag_local,
        pre_holiday_flag_local,
        post_holiday_flag_local,
        payday_flag,
        month_start_flag,
        month_end_flag,
        event_count_local,
        event_intensity_score,
        major_sports_event_flag,
        exceptional_closure_flag,
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
        gdp_growth_latest,
        gdp_current_usd_latest,
        lending_interest_rate_latest,
        government_debt_pct_gdp_latest,
        fr_business_climate_latest,
        fr_unemployment_rate_latest,
        fr_cpi_yoy_latest,
        fr_food_cpi_yoy_latest,
        fr_retail_food_volume_index_latest,
        population_1km,
        population_3km,
        office_poi_count_1km,
        school_poi_count_1km,
        transit_station_count_1km,
        mall_poi_count_1km,
        tourism_poi_count_1km,
        competitor_count_500m,
        competitor_count_1km,
        parking_score,
        closure_minutes,
        channel_disabled_flag,
        kitchen_saturation_flag,
        assortment_restriction_flag,
        order_cutoff_flag,
        client_id,
        vertical_level_1,
        vertical_level_2,
        country_code,
        region_code,
        city_name,
        source_legal_basis,
        source_license_type,
        source_review_status,
        source_legal_status_snapshot,
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
        or weather_temperature_min is not null
        or weather_temperature_max is not null
        or weather_precipitation is not null
        or weather_humidity is not null
        or weather_wind_level is not null as weather_available,
    (
        gdp_growth_latest is not null
        or gdp_current_usd_latest is not null
        or lending_interest_rate_latest is not null
        or government_debt_pct_gdp_latest is not null
        or fr_business_climate_latest is not null
        or fr_unemployment_rate_latest is not null
        or fr_cpi_yoy_latest is not null
        or fr_food_cpi_yoy_latest is not null
        or fr_retail_food_volume_index_latest is not null
    ) as macro_available
from imputed_panel
