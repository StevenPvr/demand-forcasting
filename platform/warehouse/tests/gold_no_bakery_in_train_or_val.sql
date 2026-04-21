{{ config(tags=["gold"]) }}

select *
from {{ ref("gold_feature_panel_d1") }}
where dataset_source = 'bakery'
  and split_bucket in ('train', 'val')
