{{ config(tags=["silver", "exogenous"], materialized="table", unique_key=["country_code", "dt"]) }}

with demand_dates as (
    select distinct
        country_code,
        dt
    from {{ ref("silver_location_date_spine") }}
    where country_code is not null
),
country_bounds as (
    select
        country_code,
        min(cast(dt as date)) as min_dt,
        max(cast(dt as date)) as max_dt
    from demand_dates
    group by
        country_code
),
dense_dates as (
    select
        bounds.country_code,
        cast(generated.dt as date) as dt
    from country_bounds as bounds
    cross join generate_series(bounds.min_dt, bounds.max_dt, interval 1 day) as generated(dt)
),
macro_series as (
    select
        country_code,
        metric_name,
        effective_from,
        available_from,
        available_from_assumption_flag,
        metric_value,
        2 as source_priority
    from {{ ref("stg_open_macro_annual") }} as annual
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on {{ praedixa_canonical_source_id("annual.source_policy_id", "annual.source_name") }} = allowed.source_id

    union all

    select
        country_code,
        metric_name,
        effective_from,
        available_from,
        available_from_assumption_flag,
        metric_value,
        1 as source_priority
    from {{ ref("stg_open_macro_timeseries") }} as timeseries
    inner join {{ ref("silver_allowed_provider_sources") }} as allowed
      on {{ praedixa_canonical_source_id("timeseries.source_policy_id", "timeseries.source_name") }} = allowed.source_id
),
candidate_values as (
    select
        dates.country_code,
        dates.dt,
        macro.metric_name,
        macro.metric_value,
        macro.effective_from,
        macro.available_from,
        macro.available_from_assumption_flag,
        row_number() over (
            partition by dates.country_code, dates.dt, macro.metric_name
            order by macro.effective_from desc, macro.source_priority asc
        ) as recency_rank
    from dense_dates as dates
    left join macro_series as macro
      on dates.country_code = macro.country_code
     and macro.effective_from <= dates.dt
     and macro.available_from <= dates.dt
),
latest_values as (
    select
        country_code,
        dt,
        metric_name,
        metric_value,
        available_from_assumption_flag
    from candidate_values
    where recency_rank = 1
)
select
    country_code,
    dt,
    max(case when metric_name = 'gdp_growth_latest' then metric_value end) as gdp_growth_latest,
    max(case when metric_name = 'gdp_current_usd_latest' then metric_value end) as gdp_current_usd_latest,
    max(case when metric_name = 'lending_interest_rate_latest' then metric_value end) as lending_interest_rate_latest,
    max(case when metric_name = 'government_debt_pct_gdp_latest' then metric_value end) as government_debt_pct_gdp_latest,
    max(case when metric_name = 'fr_business_climate_latest' then metric_value end) as fr_business_climate_latest,
    max(case when metric_name = 'fr_unemployment_rate_latest' then metric_value end) as fr_unemployment_rate_latest,
    max(case when metric_name = 'fr_cpi_yoy_latest' then metric_value end) as fr_cpi_yoy_latest,
    max(case when metric_name = 'fr_food_cpi_yoy_latest' then metric_value end) as fr_food_cpi_yoy_latest,
    max(case when metric_name = 'fr_retail_food_volume_index_latest' then metric_value end) as fr_retail_food_volume_index_latest,
    (
        max(case when metric_name = 'gdp_growth_latest' then metric_value end) is not null
        or max(case when metric_name = 'gdp_current_usd_latest' then metric_value end) is not null
        or max(case when metric_name = 'lending_interest_rate_latest' then metric_value end) is not null
        or max(case when metric_name = 'government_debt_pct_gdp_latest' then metric_value end) is not null
        or max(case when metric_name = 'fr_business_climate_latest' then metric_value end) is not null
        or max(case when metric_name = 'fr_unemployment_rate_latest' then metric_value end) is not null
        or max(case when metric_name = 'fr_cpi_yoy_latest' then metric_value end) is not null
        or max(case when metric_name = 'fr_food_cpi_yoy_latest' then metric_value end) is not null
        or max(case when metric_name = 'fr_retail_food_volume_index_latest' then metric_value end) is not null
    ) as macro_available,
    bool_or(coalesce(available_from_assumption_flag, false)) as macro_available_from_assumption_flag
from latest_values
group by
    country_code,
    dt
