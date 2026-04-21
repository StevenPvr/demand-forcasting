from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import xml.etree.ElementTree as ET

import pandas as pd

from praedixa.platform.signals.open_data.http import HttpTextReader
from praedixa.platform.signals.open_data.http import http_text


COUNTRY_CODE_COL = "country_code"
INDICATOR_CODE_COL = "indicator_code"
METRIC_NAME_COL = "metric_name"
FREQUENCY_CODE_COL = "frequency_code"
OBSERVATION_PERIOD_COL = "observation_period"
EFFECTIVE_FROM_COL = "effective_from"
METRIC_VALUE_COL = "metric_value"
SOURCE_NAME_COL = "source_name"

INSEE_BDM_SOURCE_NAME = "insee_bdm"
DEFAULT_INSEE_BDM_BASE_URL = "https://bdm.insee.fr/series/sdmx/data/SERIES_BDM"


@dataclass(frozen=True)
class InseeBdmSeriesSpec:
    metric_name: str
    idbank: str
    frequency_code: str


INSEE_FRANCE_MACRO_SERIES: tuple[InseeBdmSeriesSpec, ...] = (
    InseeBdmSeriesSpec(
        metric_name="fr_business_climate_latest",
        idbank="001565530",
        frequency_code="M",
    ),
    InseeBdmSeriesSpec(
        metric_name="fr_unemployment_rate_latest",
        idbank="001515842",
        frequency_code="Q",
    ),
    InseeBdmSeriesSpec(
        metric_name="fr_cpi_yoy_latest",
        idbank="011814614",
        frequency_code="M",
    ),
    InseeBdmSeriesSpec(
        metric_name="fr_food_cpi_yoy_latest",
        idbank="011813722",
        frequency_code="M",
    ),
    InseeBdmSeriesSpec(
        metric_name="fr_retail_food_volume_index_latest",
        idbank="010770043",
        frequency_code="M",
    ),
    InseeBdmSeriesSpec(
        metric_name="government_debt_pct_gdp_latest",
        idbank="010777608",
        frequency_code="Q",
    ),
)


def _metric_url(series_spec: InseeBdmSeriesSpec, base_url: str) -> str:
    return f"{base_url}/{series_spec.idbank}"


def _next_month_start(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


def _next_quarter_start(year: int, quarter: int) -> date:
    quarter_end_month = quarter * 3
    return _next_month_start(year, quarter_end_month)


def _effective_from_for_period(observation_period: str, frequency_code: str) -> date:
    if frequency_code == "M":
        year_str, month_str = observation_period.split("-")
        return _next_month_start(int(year_str), int(month_str))
    if frequency_code == "Q":
        year_str, quarter_str = observation_period.split("-Q")
        return _next_quarter_start(int(year_str), int(quarter_str))
    raise ValueError(f"Unsupported INSEE BDM frequency: {frequency_code}")


def _iter_observation_rows(xml_payload: str) -> list[ET.Element]:
    root = ET.fromstring(xml_payload)
    return [element for element in root.iter() if element.tag.endswith("Obs")]


def _rows_from_xml_payload(
    *,
    xml_payload: str,
    series_spec: InseeBdmSeriesSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for observation in _iter_observation_rows(xml_payload):
        observation_period = observation.attrib.get("TIME_PERIOD")
        metric_value = observation.attrib.get("OBS_VALUE")
        if observation_period is None or metric_value is None:
            continue
        rows.append(
            {
                COUNTRY_CODE_COL: "FR",
                INDICATOR_CODE_COL: series_spec.idbank,
                METRIC_NAME_COL: series_spec.metric_name,
                FREQUENCY_CODE_COL: series_spec.frequency_code,
                OBSERVATION_PERIOD_COL: observation_period,
                EFFECTIVE_FROM_COL: _effective_from_for_period(
                    observation_period=observation_period,
                    frequency_code=series_spec.frequency_code,
                ).isoformat(),
                METRIC_VALUE_COL: float(metric_value),
                SOURCE_NAME_COL: INSEE_BDM_SOURCE_NAME,
            }
        )
    return rows


def fetch_insee_macro_timeseries_frame(
    *,
    country_codes: list[str],
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: HttpTextReader = http_text,
    base_url: str = DEFAULT_INSEE_BDM_BASE_URL,
) -> pd.DataFrame:
    """Fetch FR-only macro time series from the official INSEE BDM SDMX service.

    The effective date is intentionally conservative:
    - monthly indicators become available on the first day of the next month
    - quarterly indicators become available on the first day of the next quarter
    """

    if "FR" not in {country_code.upper() for country_code in country_codes}:
        return empty_insee_macro_timeseries_frame()

    rows: list[dict[str, object]] = []
    for series_spec in INSEE_FRANCE_MACRO_SERIES:
        xml_payload = http_text_reader(
            _metric_url(series_spec, base_url),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        rows.extend(
            _rows_from_xml_payload(
                xml_payload=xml_payload,
                series_spec=series_spec,
            )
        )

    if not rows:
        return empty_insee_macro_timeseries_frame()

    return pd.DataFrame(rows).sort_values(
        [COUNTRY_CODE_COL, METRIC_NAME_COL, EFFECTIVE_FROM_COL]
    ).reset_index(drop=True)


def empty_insee_macro_timeseries_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COUNTRY_CODE_COL,
            INDICATOR_CODE_COL,
            METRIC_NAME_COL,
            FREQUENCY_CODE_COL,
            OBSERVATION_PERIOD_COL,
            EFFECTIVE_FROM_COL,
            METRIC_VALUE_COL,
            SOURCE_NAME_COL,
        ]
    )
