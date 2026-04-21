{{ config(tags=["silver", "quality"], materialized="view") }}

select
    '{{ env_var("PRAEDIXA_SILVER_RUN_ID", "manual") }}' as silver_run_id,
    dataset_source,
    min(dt) as start_date,
    max(dt) as end_date,
    count(*) as row_count,
    count(distinct series_id) as series_count,
    now() as computed_at
from {{ ref("silver_daily_product_demand") }}
group by dataset_source
