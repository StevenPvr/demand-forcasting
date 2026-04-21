{{ config(tags=["gold"]) }}

select *
from {{ ref("gold_feature_panel_d1") }}
where split_bucket = 'test'
  and dataset_source <> 'bakery'
