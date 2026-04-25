{{ config(tags=["silver", "governance"], materialized="table") }}

select
    cast(feature_name as varchar) as feature_name,
    cast(feature_family as varchar) as feature_family,
    cast(tft_role as varchar) as tft_role,
    cast(available_at_prediction as boolean) as available_at_prediction,
    cast(source_system as varchar) as source_system,
    cast(missing_policy as varchar) as missing_policy,
    cast(default_value as varchar) as default_value,
    cast(include_in_training as boolean) as include_in_training,
    cast(allow_constant as boolean) as allow_constant,
    cast(notes as varchar) as notes
from {{ ref("feature_registry") }}
