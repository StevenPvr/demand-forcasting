from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.platform.signals.open_data.france_macro import (  # noqa: E402
    empty_insee_macro_timeseries_frame,
    fetch_insee_macro_timeseries_frame,
)


def _series_xml(*, idbank: str, title_fr: str, time_period: str, obs_value: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<message:StructureSpecificData xmlns:message="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message">
  <message:DataSet>
    <Series IDBANK="{idbank}" TITLE_FR="{title_fr}">
      <Obs TIME_PERIOD="{time_period}" OBS_VALUE="{obs_value}" />
    </Series>
  </message:DataSet>
</message:StructureSpecificData>
"""


def _empty_http_text_reader(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    del url, timeout_seconds, max_retries, retry_backoff_seconds
    return ""


def _build_insee_payloads() -> dict[str, str]:
    return {
        "001565530": _series_xml(idbank="001565530", title_fr="Climat des affaires", time_period="2026-03", obs_value="96.9"),
        "001515842": _series_xml(idbank="001515842", title_fr="Chômage", time_period="2025-Q4", obs_value="7.7"),
        "011814614": _series_xml(idbank="011814614", title_fr="IPC glissement annuel", time_period="2026-03", obs_value="1.7"),
        "011813722": _series_xml(idbank="011813722", title_fr="Inflation alimentaire", time_period="2026-03", obs_value="1.8"),
        "010770043": _series_xml(idbank="010770043", title_fr="Volume des ventes alimentaires", time_period="2026-01", obs_value="100.43"),
        "010777608": _series_xml(idbank="010777608", title_fr="Dette Maastricht", time_period="2025-Q4", obs_value="115.6"),
    }


def _build_http_text_reader(payloads: dict[str, str]):
    def fake_http_text_reader(
        url: str,
        *,
        timeout_seconds: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> str:
        del timeout_seconds, max_retries, retry_backoff_seconds
        return payloads[url.rsplit("/", 1)[-1]]

    return fake_http_text_reader


class InseeFranceMacroTests(unittest.TestCase):
    def test_empty_frame_has_expected_columns(self) -> None:
        frame = empty_insee_macro_timeseries_frame()

        self.assertEqual(
            frame.columns.tolist(),
            [
                "country_code",
                "indicator_code",
                "metric_name",
                "frequency_code",
                "observation_period",
                "effective_from",
                "metric_value",
                "source_name",
            ],
        )

    def test_fetch_frame_returns_empty_when_fr_not_requested(self) -> None:
        frame = fetch_insee_macro_timeseries_frame(
            country_codes=["GB"],
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            http_text_reader=_empty_http_text_reader,
        )

        self.assertTrue(frame.empty)

    def test_fetch_frame_builds_conservative_effective_dates(self) -> None:
        payloads = _build_insee_payloads()
        frame = fetch_insee_macro_timeseries_frame(
            country_codes=["FR"],
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0.0,
            http_text_reader=_build_http_text_reader(payloads),
        )

        self.assertEqual(len(frame), 6)

        monthly_row = frame.loc[frame["metric_name"].eq("fr_cpi_yoy_latest")].iloc[0]
        self.assertEqual(monthly_row["effective_from"], "2026-04-01")

        quarterly_row = frame.loc[frame["metric_name"].eq("government_debt_pct_gdp_latest")].iloc[0]
        self.assertEqual(quarterly_row["effective_from"], "2026-01-01")
        self.assertEqual(float(quarterly_row["metric_value"]), 115.6)


if __name__ == "__main__":
    unittest.main()
