from __future__ import annotations

from datetime import date
from datetime import timedelta

from dateutil.easter import easter


COUNTRY_CODE_COL = "country_code"
DT_COL = "dt"
HOLIDAY_NAME_COL = "holiday_name"
LOCAL_HOLIDAY_NAME_COL = "holiday_local_name"
SOURCE_NAME_COL = "source_name"
DETERMINISTIC_PUBLIC_HOLIDAY_SOURCE_NAME = "deterministic_public_holiday_rules"


def supported_public_holiday_country_codes() -> set[str]:
    """Return the country codes covered by the local deterministic holiday rules."""

    return {"FR", "GB"}


def _first_monday(year: int, month: int) -> date:
    day = date(year, month, 1)
    while day.weekday() != 0:
        day += timedelta(days=1)
    return day


def _last_monday(year: int, month: int) -> date:
    if month == 12:
        day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        day = date(year, month + 1, 1) - timedelta(days=1)
    while day.weekday() != 0:
        day -= timedelta(days=1)
    return day


def _observed_monday(day: date) -> date:
    if day.weekday() == 5:
        return day + timedelta(days=2)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _fr_holiday_specs(year: int) -> list[tuple[date, str]]:
    easter_sunday = easter(year)
    return [
        (date(year, 1, 1), "Jour de l'an"),
        (easter_sunday + timedelta(days=1), "Lundi de Pâques"),
        (date(year, 5, 1), "Fête du Travail"),
        (date(year, 5, 8), "Victoire 1945"),
        (easter_sunday + timedelta(days=39), "Ascension"),
        (easter_sunday + timedelta(days=50), "Lundi de Pentecôte"),
        (date(year, 7, 14), "Fête nationale"),
        (date(year, 8, 15), "Assomption"),
        (date(year, 11, 1), "Toussaint"),
        (date(year, 11, 11), "Armistice"),
        (date(year, 12, 25), "Noël"),
    ]


def _gb_christmas_specs(year: int) -> list[tuple[date, str]]:
    christmas_day = date(year, 12, 25)
    boxing_day = date(year, 12, 26)
    if christmas_day.weekday() == 5:
        return [
            (date(year, 12, 27), "Christmas Day"),
            (date(year, 12, 28), "Boxing Day"),
        ]
    if christmas_day.weekday() == 6:
        return [
            (date(year, 12, 27), "Christmas Day"),
            (date(year, 12, 26), "Boxing Day"),
        ]
    if boxing_day.weekday() == 5:
        return [
            (christmas_day, "Christmas Day"),
            (date(year, 12, 28), "Boxing Day"),
        ]
    if boxing_day.weekday() == 6:
        return [
            (christmas_day, "Christmas Day"),
            (date(year, 12, 27), "Boxing Day"),
        ]
    return [(christmas_day, "Christmas Day"), (boxing_day, "Boxing Day")]


def _gb_special_holiday_specs(year: int) -> list[tuple[date, str]]:
    if year == 2020:
        return [(date(2020, 5, 8), "Early May Bank Holiday")]
    if year == 2022:
        return [
            (date(2022, 6, 2), "Spring Bank Holiday"),
            (date(2022, 6, 3), "Platinum Jubilee Bank Holiday"),
            (date(2022, 9, 19), "State Funeral of Queen Elizabeth II"),
        ]
    if year == 2023:
        return [(date(2023, 5, 8), "Coronation of King Charles III")]
    return []


def _gb_holiday_specs(year: int) -> list[tuple[date, str]]:
    easter_sunday = easter(year)
    new_year = _observed_monday(date(year, 1, 1))
    early_may = _first_monday(year, 5)
    spring_bank = _last_monday(year, 5)
    if year == 2022:
        spring_bank = date(2022, 6, 2)
    summer_bank = _last_monday(year, 8)
    holidays = [
        (new_year, "New Year's Day"),
        (easter_sunday - timedelta(days=2), "Good Friday"),
        (easter_sunday + timedelta(days=1), "Easter Monday"),
        (early_may, "Early May Bank Holiday"),
        (spring_bank, "Spring Bank Holiday"),
        (summer_bank, "Summer Bank Holiday"),
    ]
    holidays.extend(_gb_special_holiday_specs(year))
    holidays.extend(_gb_christmas_specs(year))
    return holidays


def _country_holiday_specs(country_code: str, year: int) -> list[tuple[date, str]]:
    if country_code == "FR":
        return _fr_holiday_specs(year)
    if country_code == "GB":
        return _gb_holiday_specs(year)
    return []


def deterministic_public_holiday_rows(country_code: str, year: int) -> list[dict[str, object]]:
    """Return canonical public-holiday rows for one supported country/year pair."""

    rows: list[dict[str, object]] = []
    for holiday_date, holiday_name in _country_holiday_specs(country_code, year):
        rows.append(
            {
                COUNTRY_CODE_COL: country_code,
                DT_COL: holiday_date.isoformat(),
                HOLIDAY_NAME_COL: holiday_name,
                LOCAL_HOLIDAY_NAME_COL: holiday_name,
                "global_flag": True,
                "counties_json": None,
                "holiday_types_json": None,
                SOURCE_NAME_COL: DETERMINISTIC_PUBLIC_HOLIDAY_SOURCE_NAME,
            }
        )
    return rows
