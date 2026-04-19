{{ config(tags=["staging", "bakery"]) }}

select
    source_partition,
    row_index,
    sale_date_raw,
    sale_time_raw,
    ticket_number_raw,
    article_raw,
    try_cast(quantity_raw as double) as quantity,
    try_cast(
        regexp_replace(replace(unit_price_raw, ',', '.'), '[^0-9\\.-]', '', 'g') as double
    ) as unit_price,
    coalesce(
        try_strptime(sale_date_raw || ' ' || sale_time_raw, '%Y-%m-%d %H:%M:%S'),
        try_strptime(sale_date_raw || ' ' || sale_time_raw, '%Y-%m-%d %H:%M')
    ) as sold_at_local_ts,
    cast(
        coalesce(
            try_strptime(sale_date_raw || ' ' || sale_time_raw, '%Y-%m-%d %H:%M:%S'),
            try_strptime(sale_date_raw || ' ' || sale_time_raw, '%Y-%m-%d %H:%M')
        ) as date
    ) as dt,
    upper(trim(article_raw)) as product_id
from {{ source("bronze", "bronze_bakery_order_lines") }}
