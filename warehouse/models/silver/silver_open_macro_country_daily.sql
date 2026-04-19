{{ config(tags=["silver", "exogenous"], unique_key=["country_code", "dt"]) }}

with demand_dates as (
    select distinct
        metadata.country_code,
        demand.dt
    from {{ ref("silver_daily_product_demand") }} as demand
    inner join {{ ref("silver_open_location_metadata") }} as metadata
      on demand.dataset_source = metadata.dataset_source
     and demand.location_id = metadata.location_id

    union

    select distinct
        metadata.country_code,
        demand.dt + interval 1 day as dt
    from {{ ref("silver_daily_product_demand") }} as demand
    inner join {{ ref("silver_open_location_metadata") }} as metadata
      on demand.dataset_source = metadata.dataset_source
     and demand.location_id = metadata.location_id
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
        metric_value
    from {{ ref("stg_open_macro_timeseries") }}

    union all

    select
        country_code,
        metric_name,
        effective_from,
        metric_value
    from {{ ref("stg_open_macro_annual") }}
),
candidate_values as (
    select
        dates.country_code,
        dates.dt,
        macro.metric_name,
        macro.metric_value,
        macro.effective_from,
        row_number() over (
            partition by dates.country_code, dates.dt, macro.metric_name
            order by macro.effective_from desc
        ) as recency_rank
    from dense_dates as dates
    left join macro_series as macro
      on dates.country_code = macro.country_code
     and macro.effective_from <= dates.dt
),
latest_values as (
    select
        country_code,
        dt,
        metric_name,
        metric_value
    from candidate_values
    where recency_rank = 1
)
select
    country_code,
    dt,
    max(case when metric_name = 'inflation_cpi_latest' then metric_value end) as inflation_cpi_latest,
    max(case when metric_name = 'food_cpi_latest' then metric_value end) as food_cpi_latest,
    max(case when metric_name = 'policy_rate_latest' then metric_value end) as policy_rate_latest,
    max(case when metric_name = 'unemployment_rate_latest' then metric_value end) as unemployment_rate_latest,
    max(case when metric_name = 'consumer_confidence_latest' then metric_value end) as consumer_confidence_latest,
    max(case when metric_name = 'retail_sales_index_latest' then metric_value end) as retail_sales_index_latest,
    max(case when metric_name = 'gdp_quarterly_level_latest' then metric_value end) as gdp_quarterly_level_latest,
    max(case when metric_name = 'gdp_growth_latest' then metric_value end) as gdp_growth_latest,
    max(case when metric_name = 'gdp_current_usd_latest' then metric_value end) as gdp_current_usd_latest,
    max(case when metric_name = 'lending_interest_rate_latest' then metric_value end) as lending_interest_rate_latest,
    max(case when metric_name = 'government_debt_pct_gdp_latest' then metric_value end) as government_debt_pct_gdp_latest,
    (
        max(case when metric_name = 'inflation_cpi_latest' then metric_value end) is not null
        or max(case when metric_name = 'food_cpi_latest' then metric_value end) is not null
        or max(case when metric_name = 'policy_rate_latest' then metric_value end) is not null
        or max(case when metric_name = 'unemployment_rate_latest' then metric_value end) is not null
        or max(case when metric_name = 'consumer_confidence_latest' then metric_value end) is not null
        or max(case when metric_name = 'retail_sales_index_latest' then metric_value end) is not null
        or max(case when metric_name = 'gdp_quarterly_level_latest' then metric_value end) is not null
        or max(case when metric_name = 'gdp_growth_latest' then metric_value end) is not null
        or max(case when metric_name = 'gdp_current_usd_latest' then metric_value end) is not null
        or max(case when metric_name = 'lending_interest_rate_latest' then metric_value end) is not null
        or max(case when metric_name = 'government_debt_pct_gdp_latest' then metric_value end) is not null
    ) as macro_available
from latest_values
group by
    country_code,
    dt
