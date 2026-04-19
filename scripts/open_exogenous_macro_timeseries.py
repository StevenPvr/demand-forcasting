from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import logging
from typing import Callable, Literal
from urllib.parse import urlencode

import pandas as pd


FRED_GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
logger = logging.getLogger(__name__)

SourceFrequency = Literal["daily", "monthly", "quarterly", "annual"]


@dataclass(frozen=True)
class MacroSeriesSpec:
    """One open macro series fetched from a public provider."""

    country_code: str
    metric_name: str
    source_series_id: str
    source_frequency: SourceFrequency
    availability_lag_days: int
    source_name: str
    metric_units: str


FRED_MACRO_SERIES: tuple[MacroSeriesSpec, ...] = (
    MacroSeriesSpec("US", "inflation_cpi_latest", "CPIAUCSL", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("US", "food_cpi_latest", "CP0100USM086NEST", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("US", "unemployment_rate_latest", "UNRATE", "monthly", 10, "fred_stlouis_fed", "percent"),
    MacroSeriesSpec("US", "policy_rate_latest", "FEDFUNDS", "monthly", 1, "fred_stlouis_fed", "percent"),
    MacroSeriesSpec("US", "retail_sales_index_latest", "RSAFS", "monthly", 20, "fred_stlouis_fed", "level"),
    MacroSeriesSpec("US", "consumer_confidence_latest", "USACSCICP02STSAM", "monthly", 10, "fred_stlouis_fed", "balance"),
    MacroSeriesSpec("US", "gdp_quarterly_level_latest", "GDPC1", "quarterly", 45, "fred_stlouis_fed", "level"),
    MacroSeriesSpec("FR", "inflation_cpi_latest", "CP0000FRM086NEST", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("FR", "food_cpi_latest", "CP0100FRM086NEST", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("FR", "unemployment_rate_latest", "LRHUTTTTFRM156S", "monthly", 12, "fred_stlouis_fed", "percent"),
    MacroSeriesSpec("FR", "policy_rate_latest", "ECBDFR", "daily", 0, "fred_stlouis_fed", "percent"),
    MacroSeriesSpec("FR", "retail_sales_index_latest", "FRASLRTTO01IXEBSAM", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("FR", "consumer_confidence_latest", "CSCICP02FRM460S", "monthly", 10, "fred_stlouis_fed", "balance"),
    MacroSeriesSpec("FR", "gdp_quarterly_level_latest", "CLVMNACSCAB1GQFR", "quarterly", 45, "fred_stlouis_fed", "level"),
    MacroSeriesSpec("CN", "inflation_cpi_latest", "CHNCPIALLMINMEI", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("CN", "food_cpi_latest", "CHNCP010000IXOBM", "monthly", 20, "fred_stlouis_fed", "index"),
    MacroSeriesSpec("CN", "policy_rate_latest", "INTDSRCNM193N", "monthly", 5, "fred_stlouis_fed", "percent"),
    MacroSeriesSpec("CN", "retail_sales_index_latest", "CHNSLRTTO02MLM", "monthly", 20, "fred_stlouis_fed", "level"),
    MacroSeriesSpec("CN", "consumer_confidence_latest", "CSCICP02CNM460S", "monthly", 10, "fred_stlouis_fed", "balance"),
    MacroSeriesSpec("CN", "gdp_quarterly_level_latest", "CHNGDPNQDSMEI", "quarterly", 45, "fred_stlouis_fed", "level"),
)


def _empty_macro_timeseries_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "country_code",
            "metric_name",
            "source_series_id",
            "source_frequency",
            "metric_units",
            "observation_date",
            "period_end",
            "effective_from",
            "metric_value",
            "source_name",
        ]
    )


def _period_end_from_observation_date(observation_date: pd.Timestamp, frequency: SourceFrequency) -> pd.Timestamp:
    if frequency == "daily":
        return observation_date
    if frequency == "monthly":
        return observation_date + pd.offsets.MonthEnd(0)
    if frequency == "quarterly":
        return observation_date + pd.offsets.QuarterEnd(0)
    if frequency == "annual":
        return observation_date + pd.offsets.YearEnd(0)
    raise ValueError(f"Unsupported frequency: {frequency}")


def build_effective_from(
    observation_date: pd.Timestamp,
    *,
    frequency: SourceFrequency,
    availability_lag_days: int,
) -> pd.Timestamp:
    """Return the first date when one observation may be used without future knowledge."""

    period_end = _period_end_from_observation_date(observation_date, frequency)
    return period_end + pd.Timedelta(days=availability_lag_days)


def _fred_series_url(
    series_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    query_params = {"id": series_id}
    if start_date is not None:
        query_params["cosd"] = start_date
    if end_date is not None:
        query_params["coed"] = end_date
    return f"{FRED_GRAPH_CSV_URL}?{urlencode(query_params)}"


def _parse_fred_csv_frame(csv_text: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(csv_text))
    value_column = raw.columns[-1]
    frame = raw.rename(columns={"observation_date": "observation_date_raw", value_column: "metric_value"}).copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date_raw"], errors="coerce")
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    return frame.loc[frame["observation_date"].notna() & frame["metric_value"].notna()].copy()


def _decorate_macro_frame(frame: pd.DataFrame, spec: MacroSeriesSpec) -> pd.DataFrame:
    frame["period_end"] = frame["observation_date"].apply(
        lambda value: _period_end_from_observation_date(value, spec.source_frequency)
    )
    frame["effective_from"] = frame["observation_date"].apply(
        lambda value: build_effective_from(
            value,
            frequency=spec.source_frequency,
            availability_lag_days=spec.availability_lag_days,
        )
    )
    frame["country_code"] = spec.country_code
    frame["metric_name"] = spec.metric_name
    frame["source_series_id"] = spec.source_series_id
    frame["source_frequency"] = spec.source_frequency
    frame["metric_units"] = spec.metric_units
    frame["source_name"] = spec.source_name
    return frame[
        [
            "country_code",
            "metric_name",
            "source_series_id",
            "source_frequency",
            "metric_units",
            "observation_date",
            "period_end",
            "effective_from",
            "metric_value",
            "source_name",
        ]
    ].sort_values(["country_code", "metric_name", "observation_date"]).reset_index(drop=True)


def fetch_fred_macro_series_frame(
    spec: MacroSeriesSpec,
    *,
    date_bounds: tuple[str, str] | None = None,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: Callable[..., str],
) -> pd.DataFrame:
    """Fetch one FRED CSV series and convert it to an as-of macro timeseries frame."""

    csv_text = http_text_reader(
        _fred_series_url(
            spec.source_series_id,
            start_date=date_bounds[0] if date_bounds is not None else None,
            end_date=date_bounds[1] if date_bounds is not None else None,
        ),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    frame = _parse_fred_csv_frame(csv_text)
    if frame.empty:
        return _empty_macro_timeseries_frame()
    return _decorate_macro_frame(frame, spec)


def fetch_macro_timeseries_frame(
    *,
    country_codes: list[str],
    country_date_bounds: dict[str, tuple[str, str]] | None = None,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: Callable[..., str],
) -> pd.DataFrame:
    """Fetch the configured open macro series for the requested countries."""

    requested = set(country_codes)
    frames: list[pd.DataFrame] = []
    for spec in FRED_MACRO_SERIES:
        if spec.country_code not in requested:
            continue
        try:
            frames.append(
                fetch_fred_macro_series_frame(
                    spec,
                    date_bounds=country_date_bounds.get(spec.country_code) if country_date_bounds is not None else None,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    http_text_reader=http_text_reader,
                )
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning(
                "Skipping macro series %s/%s after fetch failure: %s",
                spec.country_code,
                spec.source_series_id,
                exc,
            )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_macro_timeseries_frame()
    return pd.concat(frames, axis=0, ignore_index=True).sort_values(
        ["country_code", "metric_name", "observation_date"]
    ).reset_index(drop=True)
