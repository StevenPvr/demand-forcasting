{{ config(tags=["silver"], materialized="table") }}

with allowed_training_sources as (
    select dataset_source
    from {{ ref("silver_allowed_training_dataset_sources") }}
),
unioned as (
    select
        10 as source_priority,
        *
    from {{ ref("silver_freshretail_daily_product_demand") }}
    union all by name
    select
        20 as source_priority,
        *
    from {{ ref("silver_supplemental_corpus_daily_product_demand") }}
    union all by name
    select
        30 as source_priority,
        *
    from {{ ref("silver_synthetic_foodservice_daily_product_demand") }}
    union all by name
    select
        10 as source_priority,
        *
    from {{ ref("silver_bakery_daily_product_demand") }}
),
normalized as (
select
    cast(unioned.dataset_source as varchar) as dataset_source,
    cast(unioned.source_partition as varchar) as source_partition,
    cast(unioned.source_run_id as varchar) as source_run_id,
    cast(unioned.series_id as varchar) as series_id,
    cast(unioned.dt as date) as dt,
    cast(unioned.location_id as varchar) as location_id,
    cast(unioned.product_id as varchar) as product_id,
    cast(unioned.region_id as varchar) as region_id,
    cast(unioned.org_group_id as varchar) as org_group_id,
    cast(unioned.category_level_1 as varchar) as category_level_1,
    cast(unioned.category_level_2 as varchar) as category_level_2,
    cast(unioned.category_level_3 as varchar) as category_level_3,
    try_cast(unioned.observed_demand_qty as double) as observed_demand_qty,
    case
        when lower(trim(cast(unioned.target_semantics as varchar))) in ('observed_sales', 'latent_demand_estimated')
        then lower(trim(cast(unioned.target_semantics as varchar)))
        else 'observed_sales'
    end as target_semantics,
    coalesce(try_cast(unioned.censor_flag as boolean), false) as censor_flag,
    coalesce(cast(unioned.target_source as varchar), 'observed_sales') as target_source,
    coalesce(
        try_cast(unioned.label_quality_score as double),
        case when coalesce(try_cast(unioned.censor_flag as boolean), false) then 0.5 else 1.0 end
    ) as label_quality_score,
    coalesce(
        try_cast(unioned.usable_for_training_flag as boolean),
        not coalesce(try_cast(unioned.censor_flag as boolean), false)
    ) as usable_for_training_flag,
    try_cast(unioned.observed_revenue_net as double) as observed_revenue_net,
    try_cast(unioned.observed_discount_amount as double) as observed_discount_amount,
    try_cast(unioned.promo_flag as boolean) as promo_flag,
    try_cast(unioned.holiday_flag as boolean) as holiday_flag,
    try_cast(unioned.activity_flag as boolean) as activity_flag,
    try_cast(unioned.observed_stockout_flag as boolean) as observed_stockout_flag,
    coalesce(try_cast(unioned.observed_stockout_available as boolean), false) as observed_stockout_available,
    try_cast(unioned.observed_stockout_intensity as double) as observed_stockout_intensity,
    try_cast(unioned.day_complete_flag as boolean) as day_complete_flag,
    cast(unioned.calendar_weekday_name as varchar) as calendar_weekday_name,
    try_cast(unioned.calendar_day_of_week as integer) as calendar_day_of_week,
    try_cast(unioned.calendar_month as integer) as calendar_month,
    try_cast(unioned.calendar_year as integer) as calendar_year,
    try_cast(unioned.calendar_week_key as integer) as calendar_week_key,
    cast(unioned.event_name_1 as varchar) as event_name_1,
    cast(unioned.event_type_1 as varchar) as event_type_1,
    cast(unioned.event_name_2 as varchar) as event_name_2,
    cast(unioned.event_type_2 as varchar) as event_type_2,
    try_cast(unioned.weather_precipitation as double) as weather_precipitation,
    try_cast(unioned.weather_temperature as double) as weather_temperature,
    try_cast(unioned.weather_humidity as double) as weather_humidity,
    try_cast(unioned.weather_wind_level as double) as weather_wind_level,
    cast(unioned.silver_run_id as varchar) as silver_run_id,
    unioned.source_priority
from unioned
inner join allowed_training_sources
  on unioned.dataset_source = allowed_training_sources.dataset_source
),
ranked as (
    select
        *,
        row_number() over (
            partition by dataset_source, dt, location_id, product_id
            order by source_priority, source_partition, source_run_id, silver_run_id
        ) as source_rank
    from normalized
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
    observed_stockout_flag,
    observed_stockout_available,
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
    silver_run_id
from ranked
where source_rank = 1
