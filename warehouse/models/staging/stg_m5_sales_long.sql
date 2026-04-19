{{ config(tags=["staging", "m5"]) }}

select
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d,
    cast(sale_qty as double) as sale_qty,
    source_partition
from {{ source("bronze", "bronze_m5_sales_long") }}
