{% macro praedixa_gold_feature_slice(base_relation, slice_filter_sql, split_scope_filter_sql, split_strategy) %}
with base as (
    select *
    from {{ base_relation }}
    where {{ slice_filter_sql }}
),
base_panel as (
    select
        dataset_source,
        source_partition,
        source_run_id,
        series_id,
        dt,
        target_dt,
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
        avg_selling_price,
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
        weather_temperature,
        weather_temperature_min,
        weather_temperature_max,
        weather_precipitation,
        weather_humidity,
        weather_wind_level,
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
        gold_run_id,
        price_available,
        weather_available,
        macro_available
    from base
),
window_inputs as (
    select
        dataset_source,
        dt,
        target_dt,
        location_id,
        product_id,
        current_day_demand_qty,
        avg_selling_price,
        observed_discount_amount,
        promo_flag,
        activity_flag,
        observed_stockout_flag,
        weather_temperature,
        weather_temperature_min,
        weather_temperature_max,
        weather_precipitation,
        weather_humidity,
        weather_wind_level,
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
        holiday_flag,
        holiday_name,
        school_holiday_flag,
        bridge_day_flag,
        pre_holiday_flag,
        post_holiday_flag
    from base_panel
),
window_lagged as (
    select
        *,
        lead(current_day_demand_qty, 1) over series_window as target_demand_qty_d_plus_1,
        lag(current_day_demand_qty, 1) over series_window as lag_1,
        lag(current_day_demand_qty, 7) over series_window as lag_7,
        lag(current_day_demand_qty, 14) over series_window as lag_14,
        lag(current_day_demand_qty, 21) over series_window as lag_21_same_dow,
        lag(current_day_demand_qty, 28) over series_window as lag_28,
        lag(current_day_demand_qty, 6) over series_window as target_lag_7,
        lag(current_day_demand_qty, 13) over series_window as target_lag_14,
        lag(current_day_demand_qty, 20) over series_window as target_lag_21,
        lag(current_day_demand_qty, 27) over series_window as target_lag_28,
        avg(current_day_demand_qty) over rolling_7 as rolling_mean_7,
        avg(current_day_demand_qty) over rolling_14 as rolling_mean_14,
        avg(current_day_demand_qty) over rolling_28 as rolling_mean_28,
        stddev_samp(current_day_demand_qty) over rolling_28 as rolling_std_28,
        current_day_demand_qty as naive_last_value,
        lag(avg_selling_price, 1) over series_window as avg_selling_price_lag_1,
        lag(avg_selling_price, 7) over series_window as avg_selling_price_lag_7,
        lag(avg_selling_price, 28) over series_window as avg_selling_price_lag_28,
        avg(avg_selling_price) over rolling_7 as avg_selling_price_rolling_mean_7,
        avg(avg_selling_price) over rolling_28 as avg_selling_price_rolling_mean_28,
        lag(observed_discount_amount, 1) over series_window as observed_discount_amount_lag_1,
        lag(observed_discount_amount, 7) over series_window as observed_discount_amount_lag_7,
        lag(observed_discount_amount, 28) over series_window as observed_discount_amount_lag_28,
        avg(observed_discount_amount) over rolling_7 as observed_discount_amount_rolling_mean_7,
        avg(observed_discount_amount) over rolling_28 as observed_discount_amount_rolling_mean_28,
        lag(promo_flag, 1) over series_window as promo_flag_lag_1,
        lag(promo_flag, 7) over series_window as promo_flag_lag_7,
        lag(promo_flag, 28) over series_window as promo_flag_lag_28,
        avg(cast(promo_flag as double)) over rolling_7 as promo_rate_7,
        avg(cast(promo_flag as double)) over rolling_28 as promo_rate_28,
        lag(activity_flag, 1) over series_window as activity_flag_lag_1,
        lag(activity_flag, 7) over series_window as activity_flag_lag_7,
        lag(activity_flag, 28) over series_window as activity_flag_lag_28,
        avg(cast(activity_flag as double)) over rolling_7 as activity_rate_7,
        avg(cast(activity_flag as double)) over rolling_28 as activity_rate_28,
        lag(observed_stockout_flag, 1) over series_window as observed_stockout_flag_lag_1,
        lag(observed_stockout_flag, 7) over series_window as observed_stockout_flag_lag_7,
        lag(observed_stockout_flag, 28) over series_window as observed_stockout_flag_lag_28,
        avg(cast(observed_stockout_flag as double)) over rolling_7 as observed_stockout_rate_7,
        avg(cast(observed_stockout_flag as double)) over rolling_28 as observed_stockout_rate_28,
        lag(weather_temperature, 1) over series_window as weather_temperature_lag_1,
        lag(weather_temperature, 7) over series_window as weather_temperature_lag_7,
        avg(weather_temperature) over rolling_7 as weather_temperature_rolling_mean_7,
        avg(weather_temperature) over rolling_28 as weather_temperature_rolling_mean_28,
        lag(weather_temperature_min, 1) over series_window as weather_temperature_min_lag_1,
        lag(weather_temperature_min, 7) over series_window as weather_temperature_min_lag_7,
        avg(weather_temperature_min) over rolling_7 as weather_temperature_min_rolling_mean_7,
        lag(weather_temperature_max, 1) over series_window as weather_temperature_max_lag_1,
        lag(weather_temperature_max, 7) over series_window as weather_temperature_max_lag_7,
        avg(weather_temperature_max) over rolling_7 as weather_temperature_max_rolling_mean_7,
        lag(weather_precipitation, 1) over series_window as weather_precipitation_lag_1,
        lag(weather_precipitation, 7) over series_window as weather_precipitation_lag_7,
        avg(weather_precipitation) over rolling_7 as weather_precipitation_rolling_mean_7,
        avg(weather_precipitation) over rolling_28 as weather_precipitation_rolling_mean_28,
        lag(weather_humidity, 1) over series_window as weather_humidity_lag_1,
        lag(weather_humidity, 7) over series_window as weather_humidity_lag_7,
        avg(weather_humidity) over rolling_7 as weather_humidity_rolling_mean_7,
        lag(weather_wind_level, 1) over series_window as weather_wind_level_lag_1,
        lag(weather_wind_level, 7) over series_window as weather_wind_level_lag_7,
        avg(weather_wind_level) over rolling_7 as weather_wind_level_rolling_mean_7,
        lag(inflation_cpi_latest, 28) over series_window as inflation_cpi_latest_lag_28,
        lag(food_cpi_latest, 28) over series_window as food_cpi_latest_lag_28,
        lag(policy_rate_latest, 28) over series_window as policy_rate_latest_lag_28,
        lag(gdp_growth_latest, 28) over series_window as gdp_growth_latest_lag_28,
        lag(gdp_quarterly_level_latest, 28) over series_window as gdp_quarterly_level_latest_lag_28,
        lag(gdp_current_usd_latest, 28) over series_window as gdp_current_usd_latest_lag_28,
        lag(unemployment_rate_latest, 28) over series_window as unemployment_rate_latest_lag_28,
        lag(consumer_confidence_latest, 28) over series_window as consumer_confidence_latest_lag_28,
        lag(retail_sales_index_latest, 28) over series_window as retail_sales_index_latest_lag_28,
        lag(lending_interest_rate_latest, 28) over series_window as lending_interest_rate_latest_lag_28,
        lag(government_debt_pct_gdp_latest, 28) over series_window as government_debt_pct_gdp_latest_lag_28,
        lead(holiday_flag, 1) over series_window as target_holiday_flag,
        lead(holiday_name, 1) over series_window as target_holiday_name,
        lead(school_holiday_flag, 1) over series_window as target_school_holiday_flag,
        lead(bridge_day_flag, 1) over series_window as target_bridge_day_flag,
        lead(pre_holiday_flag, 1) over series_window as target_pre_holiday_flag,
        lead(post_holiday_flag, 1) over series_window as target_post_holiday_flag
    from window_inputs
    window
        series_window as (
            partition by dataset_source, location_id, product_id
            order by dt
        ),
        rolling_7 as (
            partition by dataset_source, location_id, product_id
            order by dt
            rows between 6 preceding and current row
        ),
        rolling_14 as (
            partition by dataset_source, location_id, product_id
            order by dt
            rows between 13 preceding and current row
        ),
        rolling_28 as (
            partition by dataset_source, location_id, product_id
            order by dt
            rows between 27 preceding and current row
        )
),
window_features as (
    select
        dataset_source,
        dt,
        target_dt,
        location_id,
        product_id,
        target_demand_qty_d_plus_1,
        lag_1,
        lag_7,
        lag_14,
        lag_28,
        target_lag_7,
        target_lag_14,
        target_lag_21,
        target_lag_28,
        rolling_mean_7,
        rolling_mean_14,
        rolling_mean_28,
        rolling_std_28,
        avg_selling_price_lag_1,
        avg_selling_price_lag_7,
        avg_selling_price_lag_28,
        avg_selling_price_rolling_mean_7,
        avg_selling_price_rolling_mean_28,
        observed_discount_amount_lag_1,
        observed_discount_amount_lag_7,
        observed_discount_amount_lag_28,
        observed_discount_amount_rolling_mean_7,
        observed_discount_amount_rolling_mean_28,
        promo_flag_lag_1,
        promo_flag_lag_7,
        promo_flag_lag_28,
        promo_rate_7,
        promo_rate_28,
        activity_flag_lag_1,
        activity_flag_lag_7,
        activity_flag_lag_28,
        activity_rate_7,
        activity_rate_28,
        observed_stockout_flag_lag_1,
        observed_stockout_flag_lag_7,
        observed_stockout_flag_lag_28,
        observed_stockout_rate_7,
        observed_stockout_rate_28,
        (
            coalesce(lag_7, 0.0)
            + coalesce(lag_14, 0.0)
            + coalesce(lag_21_same_dow, 0.0)
            + coalesce(lag_28, 0.0)
        ) / nullif(
            (case when lag_7 is not null then 1 else 0 end)
            + (case when lag_14 is not null then 1 else 0 end)
            + (case when lag_21_same_dow is not null then 1 else 0 end)
            + (case when lag_28 is not null then 1 else 0 end),
            0
        ) as same_dow_mean_4w,
        (
            coalesce(target_lag_7, 0.0)
            + coalesce(target_lag_14, 0.0)
            + coalesce(target_lag_21, 0.0)
            + coalesce(target_lag_28, 0.0)
        ) / nullif(
            (case when target_lag_7 is not null then 1 else 0 end)
            + (case when target_lag_14 is not null then 1 else 0 end)
            + (case when target_lag_21 is not null then 1 else 0 end)
            + (case when target_lag_28 is not null then 1 else 0 end),
            0
        ) as target_same_dow_mean_4w,
        naive_last_value,
        lag_7 as seasonal_naive_d7,
        target_lag_7 as target_seasonal_naive_d7,
        rolling_mean_7 as moving_average_7,
        rolling_mean_28 as moving_average_28,
        weather_temperature as weather_temperature_lag_0,
        weather_temperature_lag_1,
        weather_temperature_lag_7,
        weather_temperature_rolling_mean_7,
        weather_temperature_rolling_mean_28,
        weather_temperature_min_lag_1,
        weather_temperature_min_lag_7,
        weather_temperature_min_rolling_mean_7,
        weather_temperature_max_lag_1,
        weather_temperature_max_lag_7,
        weather_temperature_max_rolling_mean_7,
        weather_precipitation as weather_precipitation_lag_0,
        weather_precipitation_lag_1,
        weather_precipitation_lag_7,
        weather_precipitation_rolling_mean_7,
        weather_precipitation_rolling_mean_28,
        weather_humidity as weather_humidity_lag_0,
        weather_humidity_lag_1,
        weather_humidity_lag_7,
        weather_humidity_rolling_mean_7,
        weather_wind_level as weather_wind_level_lag_0,
        weather_wind_level_lag_1,
        weather_wind_level_lag_7,
        weather_wind_level_rolling_mean_7,
        inflation_cpi_latest_lag_28,
        inflation_cpi_latest - inflation_cpi_latest_lag_28 as inflation_cpi_latest_delta_28,
        food_cpi_latest_lag_28,
        food_cpi_latest - food_cpi_latest_lag_28 as food_cpi_latest_delta_28,
        policy_rate_latest_lag_28,
        policy_rate_latest - policy_rate_latest_lag_28 as policy_rate_latest_delta_28,
        gdp_growth_latest_lag_28,
        gdp_growth_latest - gdp_growth_latest_lag_28 as gdp_growth_latest_delta_28,
        gdp_quarterly_level_latest_lag_28,
        gdp_quarterly_level_latest - gdp_quarterly_level_latest_lag_28 as gdp_quarterly_level_latest_delta_28,
        gdp_current_usd_latest_lag_28,
        gdp_current_usd_latest - gdp_current_usd_latest_lag_28 as gdp_current_usd_latest_delta_28,
        unemployment_rate_latest_lag_28,
        unemployment_rate_latest - unemployment_rate_latest_lag_28 as unemployment_rate_latest_delta_28,
        consumer_confidence_latest_lag_28,
        consumer_confidence_latest - consumer_confidence_latest_lag_28 as consumer_confidence_latest_delta_28,
        retail_sales_index_latest_lag_28,
        retail_sales_index_latest - retail_sales_index_latest_lag_28 as retail_sales_index_latest_delta_28,
        lending_interest_rate_latest_lag_28,
        lending_interest_rate_latest - lending_interest_rate_latest_lag_28 as lending_interest_rate_latest_delta_28,
        government_debt_pct_gdp_latest_lag_28,
        government_debt_pct_gdp_latest - government_debt_pct_gdp_latest_lag_28
            as government_debt_pct_gdp_latest_delta_28,
        target_holiday_flag,
        target_holiday_name,
        target_school_holiday_flag,
        target_bridge_day_flag,
        target_pre_holiday_flag,
        target_post_holiday_flag
    from window_lagged
),
eligible as (
    select
        *,
        case
            when target_demand_qty_d_plus_1 >= 0
             and target_lag_7 >= 0
            then ln(1.0 + target_demand_qty_d_plus_1) - ln(1.0 + target_lag_7)
            else null
        end as target_delta_log_wow_d_plus_1,
        cast(strftime(target_dt, '%u') as integer) - 1 as target_day_of_week,
        cast(strftime(target_dt, '%d') as integer) as target_day_of_month,
        cast(strftime(target_dt, '%V') as integer) as target_week_of_year,
        cast(strftime(target_dt, '%m') as integer) as target_month,
        cast(ceil(cast(strftime(target_dt, '%m') as integer) / 3.0) as integer) as target_quarter,
        cast(strftime(target_dt, '%Y') as integer) as target_year,
        (cast(strftime(target_dt, '%u') as integer) in (6, 7)) as target_weekend_flag,
        sin(2 * pi() * (cast(strftime(target_dt, '%u') as double) - 1) / 7.0) as sin_target_day_of_week,
        cos(2 * pi() * (cast(strftime(target_dt, '%u') as double) - 1) / 7.0) as cos_target_day_of_week,
        sin(2 * pi() * cast(strftime(target_dt, '%V') as double) / 53.0) as sin_target_week_of_year,
        cos(2 * pi() * cast(strftime(target_dt, '%V') as double) / 53.0) as cos_target_week_of_year,
        sin(2 * pi() * cast(strftime(target_dt, '%m') as double) / 12.0) as sin_target_month,
        cos(2 * pi() * cast(strftime(target_dt, '%m') as double) / 12.0) as cos_target_month
    from window_features
    where target_demand_qty_d_plus_1 is not null
),
split_scope_dates as (
    select distinct
        dataset_source,
        dt
    from {{ base_relation }}
    where {{ split_scope_filter_sql }}
),
date_ranks as (
    select
        dataset_source,
        dt,
        row_number() over (partition by dataset_source order by dt) as date_rank,
        count(*) over (partition by dataset_source) as date_count
    from split_scope_dates
),
bakery_dataset_bounds as (
    select
        max(dt) as max_dt
    from {{ ref("silver_bakery_daily_product_demand") }}
),
split_labeled as (
    select
        eligible.*,
        {% if split_strategy == "bakery_test_only" %}
        case
            when eligible.dt > bakery_dataset_bounds.max_dt - interval {{ env_var("PRAEDIXA_GOLD_BAKERY_TEST_MONTHS", "3") | int }} month then 'test'
            else null
        end as split_bucket
        {% elif split_strategy == "chrono_60_40" %}
        case
            when ranks.date_rank <= cast(floor(ranks.date_count * 0.6) as bigint) then 'train'
            else 'val'
        end as split_bucket
        {% else %}
        cast(null as varchar) as split_bucket
        {% endif %}
    from eligible
    {% if split_strategy == "chrono_60_40" %}
    inner join date_ranks as ranks
      on eligible.dataset_source = ranks.dataset_source
     and eligible.dt = ranks.dt
    {% endif %}
    {% if split_strategy == "bakery_test_only" %}
    cross join bakery_dataset_bounds
    {% endif %}
),
final_panel as (
    select
        base_panel.dataset_source,
        base_panel.source_partition,
        base_panel.source_run_id,
        base_panel.series_id,
        split_labeled.dt,
        split_labeled.target_dt,
        base_panel.location_id,
        base_panel.product_id,
        base_panel.region_id,
        base_panel.org_group_id,
        base_panel.category_level_1,
        base_panel.category_level_2,
        base_panel.category_level_3,
        base_panel.is_observed_row,
        base_panel.location_open_flag,
        base_panel.product_active_flag,
        base_panel.day_complete_flag,
        base_panel.missing_sales_flag,
        base_panel.observed_stockout_flag,
        base_panel.observed_stockout_available,
        base_panel.anomaly_flag,
        base_panel.current_day_demand_qty,
        split_labeled.target_demand_qty_d_plus_1,
        split_labeled.target_delta_log_wow_d_plus_1,
        base_panel.true_zero_demand_flag,
        base_panel.location_closed_flag,
        base_panel.observed_revenue_net,
        base_panel.avg_selling_price,
        base_panel.observed_discount_amount,
        base_panel.promo_flag,
        base_panel.holiday_flag,
        base_panel.activity_flag,
        base_panel.holiday_name,
        base_panel.school_holiday_flag,
        base_panel.bridge_day_flag,
        base_panel.pre_holiday_flag,
        base_panel.post_holiday_flag,
        split_labeled.target_holiday_flag,
        split_labeled.target_holiday_name,
        split_labeled.target_school_holiday_flag,
        split_labeled.target_bridge_day_flag,
        split_labeled.target_pre_holiday_flag,
        split_labeled.target_post_holiday_flag,
        base_panel.snap_flag,
        base_panel.weather_temperature,
        base_panel.weather_temperature_min,
        base_panel.weather_temperature_max,
        base_panel.weather_precipitation,
        base_panel.weather_humidity,
        base_panel.weather_wind_level,
        base_panel.inflation_cpi_latest,
        base_panel.food_cpi_latest,
        base_panel.policy_rate_latest,
        base_panel.gdp_growth_latest,
        base_panel.gdp_quarterly_level_latest,
        base_panel.gdp_current_usd_latest,
        base_panel.unemployment_rate_latest,
        base_panel.consumer_confidence_latest,
        base_panel.retail_sales_index_latest,
        base_panel.lending_interest_rate_latest,
        base_panel.government_debt_pct_gdp_latest,
        base_panel.client_id,
        base_panel.vertical_level_1,
        base_panel.vertical_level_2,
        base_panel.country_code,
        base_panel.region_code,
        base_panel.city_name,
        base_panel.freshretail_rescaled_flag,
        base_panel.target_scale_assumption,
        base_panel.gold_run_id,
        base_panel.price_available,
        base_panel.weather_available,
        base_panel.macro_available,
        split_labeled.lag_1,
        split_labeled.lag_7,
        split_labeled.lag_14,
        split_labeled.lag_28,
        split_labeled.target_lag_7,
        split_labeled.target_lag_14,
        split_labeled.target_lag_21,
        split_labeled.target_lag_28,
        split_labeled.rolling_mean_7,
        split_labeled.rolling_mean_14,
        split_labeled.rolling_mean_28,
        split_labeled.rolling_std_28,
        split_labeled.avg_selling_price_lag_1,
        split_labeled.avg_selling_price_lag_7,
        split_labeled.avg_selling_price_lag_28,
        split_labeled.avg_selling_price_rolling_mean_7,
        split_labeled.avg_selling_price_rolling_mean_28,
        split_labeled.observed_discount_amount_lag_1,
        split_labeled.observed_discount_amount_lag_7,
        split_labeled.observed_discount_amount_lag_28,
        split_labeled.observed_discount_amount_rolling_mean_7,
        split_labeled.observed_discount_amount_rolling_mean_28,
        split_labeled.promo_flag_lag_1,
        split_labeled.promo_flag_lag_7,
        split_labeled.promo_flag_lag_28,
        split_labeled.promo_rate_7,
        split_labeled.promo_rate_28,
        split_labeled.activity_flag_lag_1,
        split_labeled.activity_flag_lag_7,
        split_labeled.activity_flag_lag_28,
        split_labeled.activity_rate_7,
        split_labeled.activity_rate_28,
        split_labeled.observed_stockout_flag_lag_1,
        split_labeled.observed_stockout_flag_lag_7,
        split_labeled.observed_stockout_flag_lag_28,
        split_labeled.observed_stockout_rate_7,
        split_labeled.observed_stockout_rate_28,
        split_labeled.same_dow_mean_4w,
        split_labeled.target_same_dow_mean_4w,
        split_labeled.naive_last_value,
        split_labeled.seasonal_naive_d7,
        split_labeled.target_seasonal_naive_d7,
        split_labeled.moving_average_7,
        split_labeled.moving_average_28,
        split_labeled.weather_temperature_lag_0,
        split_labeled.weather_temperature_lag_1,
        split_labeled.weather_temperature_lag_7,
        split_labeled.weather_temperature_rolling_mean_7,
        split_labeled.weather_temperature_rolling_mean_28,
        split_labeled.weather_temperature_min_lag_1,
        split_labeled.weather_temperature_min_lag_7,
        split_labeled.weather_temperature_min_rolling_mean_7,
        split_labeled.weather_temperature_max_lag_1,
        split_labeled.weather_temperature_max_lag_7,
        split_labeled.weather_temperature_max_rolling_mean_7,
        split_labeled.weather_precipitation_lag_0,
        split_labeled.weather_precipitation_lag_1,
        split_labeled.weather_precipitation_lag_7,
        split_labeled.weather_precipitation_rolling_mean_7,
        split_labeled.weather_precipitation_rolling_mean_28,
        split_labeled.weather_humidity_lag_0,
        split_labeled.weather_humidity_lag_1,
        split_labeled.weather_humidity_lag_7,
        split_labeled.weather_humidity_rolling_mean_7,
        split_labeled.weather_wind_level_lag_0,
        split_labeled.weather_wind_level_lag_1,
        split_labeled.weather_wind_level_lag_7,
        split_labeled.weather_wind_level_rolling_mean_7,
        split_labeled.inflation_cpi_latest_lag_28,
        split_labeled.inflation_cpi_latest_delta_28,
        split_labeled.food_cpi_latest_lag_28,
        split_labeled.food_cpi_latest_delta_28,
        split_labeled.policy_rate_latest_lag_28,
        split_labeled.policy_rate_latest_delta_28,
        split_labeled.gdp_growth_latest_lag_28,
        split_labeled.gdp_growth_latest_delta_28,
        split_labeled.gdp_quarterly_level_latest_lag_28,
        split_labeled.gdp_quarterly_level_latest_delta_28,
        split_labeled.gdp_current_usd_latest_lag_28,
        split_labeled.gdp_current_usd_latest_delta_28,
        split_labeled.unemployment_rate_latest_lag_28,
        split_labeled.unemployment_rate_latest_delta_28,
        split_labeled.consumer_confidence_latest_lag_28,
        split_labeled.consumer_confidence_latest_delta_28,
        split_labeled.retail_sales_index_latest_lag_28,
        split_labeled.retail_sales_index_latest_delta_28,
        split_labeled.lending_interest_rate_latest_lag_28,
        split_labeled.lending_interest_rate_latest_delta_28,
        split_labeled.government_debt_pct_gdp_latest_lag_28,
        split_labeled.government_debt_pct_gdp_latest_delta_28,
        split_labeled.target_day_of_week,
        split_labeled.target_day_of_month,
        split_labeled.target_week_of_year,
        split_labeled.target_month,
        split_labeled.target_quarter,
        split_labeled.target_year,
        split_labeled.target_weekend_flag,
        split_labeled.sin_target_day_of_week,
        split_labeled.cos_target_day_of_week,
        split_labeled.sin_target_week_of_year,
        split_labeled.cos_target_week_of_year,
        split_labeled.sin_target_month,
        split_labeled.cos_target_month,
        split_labeled.split_bucket
    from base_panel
    inner join split_labeled
      on base_panel.dataset_source = split_labeled.dataset_source
     and base_panel.dt = split_labeled.dt
     and base_panel.location_id = split_labeled.location_id
     and base_panel.product_id = split_labeled.product_id
)
select
    *
from final_panel
where split_bucket is not null
{% endmacro %}
