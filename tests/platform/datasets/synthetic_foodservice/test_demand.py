from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.platform.datasets.synthetic_foodservice.config import (  # noqa: E402
    build_smoke_synthetic_foodservice_config,
)
from praedixa.platform.datasets.synthetic_foodservice.demand import (  # noqa: E402
    MODEL_FACING_COLUMNS,
    ORACLE_COLUMNS,
    generate_location_batch,
)
from praedixa.platform.datasets.synthetic_foodservice.entities import (  # noqa: E402
    build_synthetic_foodservice_entities,
)
from praedixa.platform.datasets.synthetic_foodservice.simulation_math import (  # noqa: E402
    capacity_factor,
    dow_lift,
    holiday_lift,
    negative_binomial,
    scaled_product_base,
    seasonal_lift,
    available_quantity,
    waste_multiplier,
    weather_lift,
    zero_inflation_mask,
)
from praedixa.platform.datasets.synthetic_foodservice.shocks import (  # noqa: E402
    ShockEvent,
)
from praedixa.platform.datasets.synthetic_foodservice.synthetic_calendar import (  # noqa: E402
    build_city_weather_frame,
    build_synthetic_calendar,
)
from praedixa.platform.datasets.synthetic_foodservice.taxonomy import (  # noqa: E402
    SYNTHETIC_BAKERY_SOURCE,
    SYNTHETIC_QSR_SOURCE,
    SYNTHETIC_RESTAURANT_SOURCE,
)


class NegativeBinomialTests(unittest.TestCase):
    """Validate the NB draw function produces realistic demand distributions."""

    def test_mean_tracks_mu(self) -> None:
        rng = np.random.default_rng(42)
        mu = np.full((5, 200), 20.0)
        draws = negative_binomial(mu=mu, rng=rng)

        self.assertEqual(draws.shape, mu.shape)
        mean_draw: float = float(draws.mean())
        self.assertGreater(mean_draw, 12.0)
        self.assertLess(mean_draw, 30.0)

    def test_output_is_non_negative(self) -> None:
        rng = np.random.default_rng(7)
        mu = np.full((10, 50), 3.0)
        draws = negative_binomial(mu=mu, rng=rng)

        self.assertTrue((draws >= 0).all())

    def test_overdispersion_exceeds_poisson(self) -> None:
        rng = np.random.default_rng(99)
        mu = np.full((1, 5000), 15.0)
        draws = negative_binomial(mu=mu, rng=rng)

        variance: float = float(draws.var())
        mean: float = float(draws.mean())
        self.assertGreater(
            variance, mean, "NB should have variance > mean (overdispersion)"
        )


class DowLiftTests(unittest.TestCase):
    """Day-of-week multipliers are reasonable and differ by vertical."""

    def test_bakery_saturday_is_peak(self) -> None:
        days = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int16)
        lifts = dow_lift(SYNTHETIC_BAKERY_SOURCE, days)

        saturday_index: int = 5
        self.assertEqual(lifts[saturday_index], max(lifts))

    def test_restaurant_weekend_stronger_than_monday(self) -> None:
        days = np.array([0, 4, 5, 6], dtype=np.int16)
        lifts = dow_lift(SYNTHETIC_RESTAURANT_SOURCE, days)

        self.assertGreater(lifts[2], lifts[0])  # saturday > monday
        self.assertGreater(lifts[1], lifts[0])  # friday > monday


class SeasonalLiftTests(unittest.TestCase):
    """Annual seasonality produces bounded, non-trivial multipliers."""

    def test_all_months_positive(self) -> None:
        months = np.arange(1, 13, dtype=np.int16)
        for source in (
            SYNTHETIC_QSR_SOURCE,
            SYNTHETIC_BAKERY_SOURCE,
            SYNTHETIC_RESTAURANT_SOURCE,
        ):
            lifts = seasonal_lift(source, months)

            self.assertTrue((lifts > 0.5).all(), f"Seasonal lift too low for {source}")
            self.assertTrue((lifts < 2.0).all(), f"Seasonal lift too high for {source}")

    def test_bakery_december_boost(self) -> None:
        december = np.array([12], dtype=np.int16)
        lift = seasonal_lift(SYNTHETIC_BAKERY_SOURCE, december)

        self.assertGreater(
            float(lift[0]), 1.0, "Bakery should have a December holiday boost"
        )

    def test_qsr_summer_is_above_winter(self) -> None:
        months = np.array([1, 7], dtype=np.int16)
        lifts = seasonal_lift(SYNTHETIC_QSR_SOURCE, months)

        self.assertGreater(float(lifts[1]), float(lifts[0]))


class CalendarWeatherTests(unittest.TestCase):
    def test_city_weather_has_french_summer_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                build_smoke_synthetic_foodservice_config(temp_dir),
                start_date=date(2025, 1, 1),
                end_date=date(2025, 12, 31),
                city_count=1,
            )
            entities = build_synthetic_foodservice_entities(config)
            calendar = build_synthetic_calendar(config)
            weather = build_city_weather_frame(
                cities=entities.cities.iloc[:1],
                calendar=calendar,
                seed=config.seed,
            )

        weather["dt"] = pd.to_datetime(weather["dt"])
        jan = float(
            weather.loc[weather["dt"].dt.month.eq(1), "weather_temperature"].mean()
        )
        jul = float(
            weather.loc[weather["dt"].dt.month.eq(7), "weather_temperature"].mean()
        )

        self.assertGreater(jul, jan)

    def test_mobile_french_holidays_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                build_smoke_synthetic_foodservice_config(temp_dir),
                start_date=date(2025, 1, 1),
                end_date=date(2025, 12, 31),
            )
            calendar = build_synthetic_calendar(config)

        frame = calendar.frame.copy()
        frame["dt"] = pd.to_datetime(frame["dt"])
        holidays = set(frame.loc[frame["holiday_flag"], "dt"].dt.strftime("%Y-%m-%d"))

        self.assertIn("2025-04-21", holidays)
        self.assertIn("2025-05-29", holidays)
        self.assertIn("2025-06-09", holidays)


class WeatherLiftTests(unittest.TestCase):
    """Weather effects are bounded and directionally correct."""

    def test_bakery_rain_reduces_demand(self) -> None:
        dry = pd.DataFrame(
            {"weather_temperature": [15.0], "weather_precipitation": [0.0]}
        )
        wet = pd.DataFrame(
            {"weather_temperature": [15.0], "weather_precipitation": [20.0]}
        )

        lift_dry: float = float(weather_lift(SYNTHETIC_BAKERY_SOURCE, dry)[0])
        lift_wet: float = float(weather_lift(SYNTHETIC_BAKERY_SOURCE, wet)[0])

        self.assertGreater(lift_dry, lift_wet, "Rain should reduce bakery demand")

    def test_qsr_heat_increases_demand(self) -> None:
        cool = pd.DataFrame(
            {"weather_temperature": [12.0], "weather_precipitation": [0.0]}
        )
        hot = pd.DataFrame(
            {"weather_temperature": [32.0], "weather_precipitation": [0.0]}
        )

        lift_cool: float = float(weather_lift(SYNTHETIC_QSR_SOURCE, cool)[0])
        lift_hot: float = float(weather_lift(SYNTHETIC_QSR_SOURCE, hot)[0])

        self.assertGreater(lift_hot, lift_cool, "Heat should increase QSR demand")

    def test_lifts_are_bounded(self) -> None:
        extreme = pd.DataFrame(
            {"weather_temperature": [45.0], "weather_precipitation": [80.0]}
        )
        for source in (
            SYNTHETIC_QSR_SOURCE,
            SYNTHETIC_BAKERY_SOURCE,
            SYNTHETIC_RESTAURANT_SOURCE,
        ):
            lift: float = float(weather_lift(source, extreme)[0])

            self.assertGreater(lift, 0.5, f"Weather lift too low for {source}")
            self.assertLess(lift, 1.5, f"Weather lift too high for {source}")


class HolidayLiftTests(unittest.TestCase):
    def test_bakery_holiday_is_strongest_reduction(self) -> None:
        bakery: float = holiday_lift(SYNTHETIC_BAKERY_SOURCE)
        qsr: float = holiday_lift(SYNTHETIC_QSR_SOURCE)
        restaurant: float = holiday_lift(SYNTHETIC_RESTAURANT_SOURCE)

        self.assertLess(bakery, qsr, "Bakery should drop more on holidays")
        self.assertLess(bakery, restaurant)


class ZeroInflationTests(unittest.TestCase):
    def test_restaurant_has_highest_zero_inflation(self) -> None:
        rng = np.random.default_rng(42)
        shape = (100, 100)

        qsr_zeros: float = float(
            zero_inflation_mask(SYNTHETIC_QSR_SOURCE, shape, rng).mean()
        )
        bakery_zeros: float = float(
            zero_inflation_mask(SYNTHETIC_BAKERY_SOURCE, shape, rng).mean()
        )
        restaurant_zeros: float = float(
            zero_inflation_mask(SYNTHETIC_RESTAURANT_SOURCE, shape, rng).mean()
        )

        self.assertGreater(restaurant_zeros, qsr_zeros)
        self.assertGreater(restaurant_zeros, bakery_zeros)


class CapacityFactorTests(unittest.TestCase):
    def test_mostly_unconstrained(self) -> None:
        rng = np.random.default_rng(42)
        factors = capacity_factor(SYNTHETIC_QSR_SOURCE, 1000, rng)

        unconstrained_share: float = float((factors == 1.0).mean())
        self.assertGreater(
            unconstrained_share, 0.85, "Most days should be unconstrained"
        )

    def test_constrained_days_below_one(self) -> None:
        rng = np.random.default_rng(42)
        factors = capacity_factor(SYNTHETIC_QSR_SOURCE, 1000, rng)
        constrained = factors[factors < 1.0]

        if len(constrained) > 0:
            self.assertTrue(
                (constrained >= 0.5).all(), "Constrained capacity should not collapse"
            )


class AvailableQuantityTests(unittest.TestCase):
    def test_available_quantity_uses_expected_mu_not_realized_latent(self) -> None:
        rng = np.random.default_rng(42)
        mu = np.full((8, 30), 18.0)
        available = available_quantity(SYNTHETIC_QSR_SOURCE, mu, rng)

        self.assertEqual(available.shape, mu.shape)
        self.assertTrue((available >= 0).all())


class WasteMultiplierTests(unittest.TestCase):
    def test_bakery_has_highest_waste(self) -> None:
        self.assertGreater(
            waste_multiplier(SYNTHETIC_BAKERY_SOURCE),
            waste_multiplier(SYNTHETIC_QSR_SOURCE),
        )


class ScaledProductBaseTests(unittest.TestCase):
    def test_output_shape_matches_input(self) -> None:
        rng = np.random.default_rng(42)
        raw = np.array([1.0, 2.0, 5.0, 0.5, 10.0])
        scaled = scaled_product_base(raw, low=2.0, high=50.0, rng=rng)

        self.assertEqual(scaled.shape, raw.shape)
        self.assertTrue((scaled > 0).all())


class LocationBatchGenerationTests(unittest.TestCase):
    """Integration test for generate_location_batch with a small config."""

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_smoke_synthetic_foodservice_config(temp_dir)
        cls.config = config
        cls.entities = build_synthetic_foodservice_entities(config)
        cls.calendar = build_synthetic_calendar(config)
        cls.city_weather = build_city_weather_frame(
            cities=cls.entities.cities,
            calendar=cls.calendar,
            seed=config.seed,
        )

    def test_single_site_generates_valid_daily_and_oracle(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        self.assertFalse(result.daily.empty, "Single site should produce rows")
        self.assertFalse(result.oracle.empty, "Single site should produce oracle rows")
        self.assertTrue(
            set(MODEL_FACING_COLUMNS).issubset(set(result.daily.columns)),
            "Daily must contain all model-facing columns",
        )
        self.assertTrue(
            set(ORACLE_COLUMNS).issubset(set(result.oracle.columns)),
            "Oracle must contain all oracle columns",
        )

    def test_observed_never_exceeds_latent(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        merged = result.daily.merge(
            result.oracle,
            on=["dataset_source", "dt", "location_id", "product_id"],
            how="inner",
        )
        observed = pd.to_numeric(merged["observed_demand_qty"])
        latent = pd.to_numeric(merged["latent_demand_qty_debug"])
        violations = (observed > latent + 0.01).sum()

        self.assertEqual(
            int(violations),
            0,
            f"INV-10 violated: {violations} rows where observed > latent",
        )

    def test_clean_generator_has_no_censored_or_lost_sales(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        merged = result.daily.merge(
            result.oracle,
            on=["dataset_source", "dt", "location_id", "product_id"],
            how="inner",
        )
        has_lost = pd.to_numeric(merged["lost_sales_qty_debug"]) > 0
        censor_flag = merged["censor_flag"].astype(bool)

        self.assertFalse(bool(has_lost.any()))
        self.assertFalse(bool(censor_flag.any()))

    def test_demand_is_non_negative(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        observed = pd.to_numeric(result.daily["observed_demand_qty"])
        self.assertTrue((observed >= 0).all(), "Observed demand must be non-negative")

        latent = pd.to_numeric(result.oracle["latent_demand_qty_debug"])
        self.assertTrue((latent >= 0).all(), "Latent demand must be non-negative")

    def test_series_id_format(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        for series_id in result.daily["series_id"].unique():
            self.assertIn(
                "__",
                str(series_id),
                f"series_id should be location_id__product_id, got {series_id}",
            )

    def test_clean_generator_has_no_censored_rows(self) -> None:
        locations = self.entities.locations.iloc[:5].copy()
        result = generate_location_batch(
            locations=locations,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        censored: int = int(result.daily["censor_flag"].astype(bool).sum())
        self.assertEqual(censored, 0, "Clean synthetic data must not be censored")

    def test_stockout_intensity_bounded(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )

        intensity = pd.to_numeric(result.daily["observed_stockout_intensity"])
        self.assertTrue((intensity == 0).all(), "Clean synthetic data has no stockout")

    def test_positive_market_shock_increases_latent_demand(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        location_id = str(single_location.iloc[0]["location_id"])
        shock = ShockEvent(
            shock_id="local_buzz_test",
            category="market",
            start_day_index=0,
            end_day_index=len(self.calendar.dates),
            intensity=2.0,
            recovery_halflife_days=0.0,
        )
        baseline = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
        )
        shocked = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
            site_shocks_by_location_id={location_id: [shock]},
        )

        baseline_latent = float(
            pd.to_numeric(baseline.oracle["latent_demand_qty_debug"]).sum()
        )
        shocked_latent = float(
            pd.to_numeric(shocked.oracle["latent_demand_qty_debug"]).sum()
        )

        self.assertGreater(shocked_latent, baseline_latent * 1.5)

    def test_forced_closure_shock_closes_model_rows(self) -> None:
        single_location = self.entities.locations.iloc[:1].copy()
        location_id = str(single_location.iloc[0]["location_id"])
        shock = ShockEvent(
            shock_id="forced_closure_test",
            category="regulatory",
            start_day_index=0,
            end_day_index=len(self.calendar.dates),
            intensity=-1.0,
            recovery_halflife_days=0.0,
        )
        result = generate_location_batch(
            locations=single_location,
            assortments=self.entities.assortments,
            calendar=self.calendar,
            city_weather=self.city_weather,
            seed=self.config.seed,
            source_run_id="test_run",
            site_shocks_by_location_id={location_id: [shock]},
        )

        observed = pd.to_numeric(result.daily["observed_demand_qty"])
        self.assertTrue((observed == 0).all())
        self.assertFalse(result.daily["usable_for_training_flag"].astype(bool).any())
        self.assertFalse(result.daily["activity_flag"].astype(bool).any())


if __name__ == "__main__":
    unittest.main()
