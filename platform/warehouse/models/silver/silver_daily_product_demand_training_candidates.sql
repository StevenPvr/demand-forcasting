{{ config(tags=["silver", "demand", "training"], materialized="table") }}

with allowed_training_sources as (
    select
        dataset_source,
        medallion_enabled,
        training_scope
    from {{ ref('silver_allowed_training_dataset_sources') }}
    where medallion_enabled
      and training_scope in ('train', 'reference_eval')
),
candidate_rows as (
    select
        demand.*
    from {{ ref('silver_daily_product_demand_all') }} as demand
    inner join allowed_training_sources
      on demand.dataset_source = allowed_training_sources.dataset_source
)
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
    observed_demand_qty,
    observed_demand_missing_flag,
    observed_demand_parse_error_flag,
    target_semantics,
    target_semantics_raw,
    target_semantics_parse_error_flag,
    censor_flag,
    target_source,
    label_quality_score,
    label_quality_missing_flag,
    label_quality_parse_error_flag,
    case
        when observed_demand_qty is null then false
        when observed_demand_parse_error_flag then false
        when target_semantics is null then false
        when target_semantics_parse_error_flag then false
        when coalesce(label_quality_score, 0.0) < 0.0 then false
        when coalesce(label_quality_score, 0.0) > 1.0 then false
        else coalesce(usable_for_training_flag, false)
    end as usable_for_training_flag,
    usable_for_training_missing_flag,
    censor_flag_missing_flag,
    observed_revenue_net,
    observed_discount_amount,
    promo_flag,
    holiday_flag,
    activity_flag,
    observed_stockout_flag,
    observed_stockout_available,
    observed_stockout_available_missing_flag,
    observed_stockout_intensity,
    day_complete_flag,
    calendar_weekday_name,
    calendar_day_of_week,
    calendar_month,
    calendar_year,
    calendar_week_key,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    weather_precipitation,
    weather_temperature,
    weather_humidity,
    weather_wind_level,
    silver_run_id,
    source_name,
    source_policy_id,
    source_file_path,
    source_loaded_at,
    data_origin
from candidate_rows
