{{ config(tags=["staging", "m5"]) }}

select
    store_id,
    item_id,
    cast(wm_yr_wk as integer) as wm_yr_wk,
    cast(sell_price as double) as sell_price
from {{ source("bronze", "bronze_m5_sell_prices") }}
