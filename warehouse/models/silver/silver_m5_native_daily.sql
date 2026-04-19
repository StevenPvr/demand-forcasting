{{ config(tags=["m5", "legacy"], unique_key=["dataset_source", "source_partition", "dt", "location_id", "product_id"]) }}

with enriched as (
    select
        sales.source_partition,
        sales.item_id,
        sales.dept_id,
        sales.cat_id,
        sales.store_id,
        sales.state_id,
        sales.d,
        sales.sale_qty,
        calendar.dt,
        calendar.weekday,
        calendar.wday,
        calendar.month,
        calendar.year,
        calendar.wm_yr_wk,
        calendar.event_name_1,
        calendar.event_type_1,
        calendar.event_name_2,
        calendar.event_type_2,
        calendar.snap_ca,
        calendar.snap_tx,
        calendar.snap_wi,
        prices.sell_price
    from {{ ref("stg_m5_sales_long") }} as sales
    left join {{ ref("stg_m5_calendar") }} as calendar using (d)
    left join {{ ref("stg_m5_sell_prices") }} as prices
      on sales.store_id = prices.store_id
     and sales.item_id = prices.item_id
     and calendar.wm_yr_wk = prices.wm_yr_wk
)
select
    'm5' as dataset_source,
    source_partition,
    'warehouse_run' as source_run_id,
    concat(store_id, '__', item_id) as series_id,
    dt,
    store_id as location_id,
    item_id as product_id,
    state_id as region_id,
    cast(null as varchar) as org_group_id,
    cat_id as category_level_1,
    dept_id as category_level_2,
    cast(null as varchar) as category_level_3,
    sale_qty as observed_demand_qty,
    sale_qty * sell_price as observed_revenue_net,
    cast(null as double) as observed_discount_amount,
    sell_price as avg_selling_price,
    cast(null as boolean) as promo_flag,
    (event_name_1 is not null or event_name_2 is not null) as holiday_flag,
    cast(null as boolean) as activity_flag,
    cast(null as boolean) as observed_stockout_flag,
    false as observed_stockout_available,
    cast(null as double) as observed_stockout_intensity,
    true as location_open_flag,
    true as day_complete_flag,
    false as missing_sales_flag,
    weekday as calendar_weekday_name,
    wday - 1 as calendar_day_of_week,
    month as calendar_month,
    year as calendar_year,
    wm_yr_wk as calendar_week_key,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    case
        when state_id = 'CA' then snap_ca
        when state_id = 'TX' then snap_tx
        when state_id = 'WI' then snap_wi
        else null
    end as snap_flag,
    cast(null as double) as weather_precipitation,
    cast(null as double) as weather_temperature,
    cast(null as double) as weather_humidity,
    cast(null as double) as weather_wind_level,
    false as anomaly_flag,
    '{{ env_var("PRAEDIXA_SILVER_RUN_ID", "manual") }}' as silver_run_id
from enriched
