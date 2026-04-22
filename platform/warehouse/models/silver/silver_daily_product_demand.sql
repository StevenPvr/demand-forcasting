{{ config(tags=["silver"], materialized="table") }}

with allowed_training_sources as (
    select dataset_source
    from {{ ref("silver_allowed_training_dataset_sources") }}
),
unioned as (
    select * from {{ ref("silver_freshretail_daily_product_demand") }}
    union all
    select * from {{ ref("silver_supplemental_corpus_daily_product_demand") }}
    union all
    select * from {{ ref("silver_bakery_daily_product_demand") }}
)
select
    unioned.*
from unioned
inner join allowed_training_sources
  on unioned.dataset_source = allowed_training_sources.dataset_source
