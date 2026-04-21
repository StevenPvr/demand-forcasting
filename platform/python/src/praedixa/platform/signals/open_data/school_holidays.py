from __future__ import annotations

from typing import Any, cast

import pandas as pd

from praedixa.platform.signals.open_data.http import HttpTextReader

def _as_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def _update_school_event(event: dict[str, str], line: str) -> None:
    if line.startswith("DTSTART"):
        event["start"] = line.split(":")[-1]
    elif line.startswith("DTEND"):
        event["end"] = line.split(":")[-1]
    elif line.startswith("SUMMARY:"):
        event["summary"] = line.split(":", 1)[1]


def _parse_school_holiday_events(ics_text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current_event: dict[str, str] = {}
    for raw_line in ics_text.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            current_event = {}
            continue
        if line == "END:VEVENT":
            if {"start", "end", "summary"} <= set(current_event):
                events.append(current_event.copy())
            current_event = {}
            continue
        _update_school_event(current_event, line)
    return events


def _event_rows(
    *,
    event: dict[str, str],
    dataset_source: str,
    location_id: str,
    school_zone: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start_dt = _as_timestamp(event["start"]).normalize()
    end_dt = _as_timestamp(event["end"]).normalize()
    current_dt = start_dt
    while current_dt < end_dt:
        rows.append(
            {
                "dataset_source": dataset_source,
                "location_id": location_id,
                "dt": current_dt.date().isoformat(),
                "school_holiday_name": event["summary"],
                "school_zone": school_zone,
                "source_name": "fr_education_school_calendar",
            }
        )
        current_dt += pd.Timedelta(days=1)
    return rows


def parse_school_holiday_ics(ics_text: str, *, dataset_source: str, location_id: str, school_zone: str) -> pd.DataFrame:
    """Parse one zone ICS file from the French official school holiday open data."""

    rows: list[dict[str, object]] = []
    for event in _parse_school_holiday_events(ics_text):
        rows.extend(
            _event_rows(
                event=event,
                dataset_source=dataset_source,
                location_id=location_id,
                school_zone=school_zone,
            )
        )
    return pd.DataFrame(rows)


def _location_school_holiday_slice(
    school_frame: pd.DataFrame,
    *,
    dataset_source: str,
    location_id: str,
    school_zone: str,
    min_dt: object,
    max_dt: object,
) -> pd.DataFrame:
    location_days = school_frame.loc[
        school_frame["school_zone"].eq(school_zone)
        & school_frame["dt"].between(
            pd.Timestamp(cast(Any, min_dt)).date().isoformat(),
            pd.Timestamp(cast(Any, max_dt)).date().isoformat(),
        )
    ].copy()
    if location_days.empty:
        return location_days

    location_days["dataset_source"] = dataset_source
    location_days["location_id"] = location_id
    return location_days


def _empty_school_holidays_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["dataset_source", "location_id", "dt", "school_holiday_name", "school_zone", "source_name"]
    )


def _school_rows_with_bounds(location_metadata: pd.DataFrame, silver_locations: pd.DataFrame) -> pd.DataFrame:
    school_rows = location_metadata.loc[
        location_metadata["school_zone"].notna()
        & location_metadata["country_code"].eq("FR")
    ].copy()
    if school_rows.empty:
        return school_rows

    return school_rows.merge(
        silver_locations[["dataset_source", "location_id", "min_dt", "max_dt"]],
        on=["dataset_source", "location_id"],
        how="left",
    )


def _fetch_school_zone_frames(
    school_rows: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: HttpTextReader,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for school_zone in sorted(school_rows["school_zone"].dropna().unique()):
        ics_text = http_text_reader(
            f"https://fr.ftp.opendatasoft.com/openscol/fr-en-calendrier-scolaire/Zone-{school_zone}.ics",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        frames.append(
            parse_school_holiday_ics(
                ics_text,
                dataset_source="bakery",
                location_id="bakery_store_1",
                school_zone=school_zone,
            )
        )
    return frames
def _school_source_frames(
    school_rows: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: HttpTextReader,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    french_school_rows = school_rows.loc[school_rows["country_code"].eq("FR")].copy()
    if not french_school_rows.empty:
        frames.extend(
            _fetch_school_zone_frames(
                french_school_rows,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                http_text_reader=http_text_reader,
            )
        )

    return frames


def _filtered_school_frames(
    *,
    school_frame: pd.DataFrame,
    school_rows: pd.DataFrame,
) -> list[pd.DataFrame]:
    school_row_records = school_rows.to_dict(orient="records")
    filtered_frames = [
        _location_school_holiday_slice(
            school_frame,
            dataset_source=str(row["dataset_source"]),
            location_id=str(row["location_id"]),
            school_zone=str(row["school_zone"]),
            min_dt=row["min_dt"],
            max_dt=row["max_dt"],
        )
        for row in school_row_records
    ]
    return [frame for frame in filtered_frames if not frame.empty]


def fetch_school_holidays_frame(
    location_metadata: pd.DataFrame,
    silver_locations: pd.DataFrame,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    http_text_reader: HttpTextReader,
) -> pd.DataFrame:
    """Fetch French school holidays for any configured French school zones."""

    school_rows = _school_rows_with_bounds(location_metadata, silver_locations)
    if school_rows.empty:
        return _empty_school_holidays_frame()

    frames = _school_source_frames(
        school_rows,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        http_text_reader=http_text_reader,
    )
    school_frame = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if school_frame.empty:
        return _empty_school_holidays_frame()

    filtered_frames = _filtered_school_frames(
        school_frame=school_frame,
        school_rows=school_rows,
    )
    if not filtered_frames:
        return _empty_school_holidays_frame()

    return pd.concat(filtered_frames, axis=0, ignore_index=True).drop_duplicates().sort_values(
        ["dataset_source", "location_id", "dt"]
    ).reset_index(drop=True)
