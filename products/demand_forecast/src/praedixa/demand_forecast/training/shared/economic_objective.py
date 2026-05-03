from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, cast

import numpy as np
import pandas as pd

from praedixa.demand_forecast.training.shared.metrics import (
    FLOAT_COMPARISON_EPSILON,
    compute_wape,
)

_GLOBAL_CALIBRATION_KEY = "__global__"
_PRODUCT_SEGMENT_PREFIX = "product:"
_DEFAULT_ECONOMIC_CALIBRATION_SLOPES: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05)
_DEFAULT_ECONOMIC_CALIBRATION_INTERCEPTS: tuple[float, ...] = (
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
)
_CALIBRATED_QUANTILE_COLS: tuple[str, ...] = (
    "prediction_p2_5",
    "prediction_p10",
    "prediction_p50",
    "prediction_p90",
    "prediction_p97_5",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
)
ECONOMIC_SEGMENT_METADATA_COLS: tuple[str, ...] = (
    "product_id",
    "product",
    "product_family",
    "product_subfamily",
    "category_level_1",
    "category_level_2",
    "category_level_3",
)
_SUBSTITUTION_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sandwich", ("SANDWICH", "SAND ", "FORMULE")),
    ("beverage", ("BOISSON", "CAFE", "EAU")),
    (
        "viennoiserie",
        (
            "BRIOCHE",
            "CHAUSSON",
            "CHOCOLAT",
            "CHOCO",
            "CROISSANT",
            "RAISINS",
        ),
    ),
    ("patisserie", ("COOKIE", "ECLAIR", "FINANCIER", "KOUIGN", "TARTELETTE")),
    (
        "bread",
        (
            "BAGUETTE",
            "BANETTE",
            "BANETTINE",
            "BOULE",
            "BREAD",
            "CAMPAGNE",
            "COMPLET",
            "FICELLE",
            "MOISSON",
            "PAIN",
            "SEIGLE",
        ),
    ),
)


@dataclass(frozen=True)
class EconomicCostProfile:
    overproduction_unit_cost: float = 1.0
    mild_underproduction_unit_cost: float = 0.50
    severe_underproduction_unit_cost: float = 1.50
    mild_underproduction_threshold_ratio: float = 0.10
    target_normalized_bias: float = -0.02
    max_allowed_negative_bias: float = -0.05
    positive_bias_penalty_weight: float = 2.0
    severe_negative_bias_penalty_weight: float = 1.5
    wape_guardrail_limit: float | None = None
    wape_guardrail_penalty_weight: float = 1.0


@dataclass(frozen=True)
class SegmentedEconomicObjectiveConfig:
    version: str = "segmented_asymmetric_economic_objective_final"
    default_profile: EconomicCostProfile = field(default_factory=EconomicCostProfile)
    family_profiles: dict[str, EconomicCostProfile] = field(
        default_factory=lambda: cast(dict[str, EconomicCostProfile], {})
    )
    product_profiles: dict[str, EconomicCostProfile] = field(
        default_factory=lambda: cast(dict[str, EconomicCostProfile], {})
    )
    product_id_columns: tuple[str, ...] = ("product_id", "product")
    family_columns: tuple[str, ...] = ("product_family", "category_level_1")
    product_name_columns: tuple[str, ...] = ("product", "product_id")


@dataclass(frozen=True)
class AsymmetricEconomicObjectiveConfig:
    version: str = "legacy_global_asymmetric_wrapper"
    overproduction_unit_cost: float = 1.0
    mild_underproduction_unit_cost: float = 0.35
    severe_underproduction_unit_cost: float = 1.5
    mild_underproduction_threshold_ratio: float = 0.10
    target_normalized_bias: float = -0.02
    max_allowed_negative_bias: float = -0.05
    positive_bias_penalty_weight: float = 2.0
    severe_negative_bias_penalty_weight: float = 1.5
    wape_guardrail_limit: float | None = None
    wape_guardrail_penalty_weight: float = 1.0


DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG = SegmentedEconomicObjectiveConfig(
    family_profiles={
        "bread": EconomicCostProfile(
            overproduction_unit_cost=1.0,
            mild_underproduction_unit_cost=0.25,
            severe_underproduction_unit_cost=1.35,
        ),
        "viennoiserie": EconomicCostProfile(
            overproduction_unit_cost=1.0,
            mild_underproduction_unit_cost=0.35,
            severe_underproduction_unit_cost=1.45,
        ),
        "patisserie": EconomicCostProfile(
            overproduction_unit_cost=1.0,
            mild_underproduction_unit_cost=0.55,
            severe_underproduction_unit_cost=1.65,
        ),
        "sandwich": EconomicCostProfile(
            overproduction_unit_cost=1.0,
            mild_underproduction_unit_cost=0.65,
            severe_underproduction_unit_cost=1.75,
        ),
        "beverage": EconomicCostProfile(
            overproduction_unit_cost=0.65,
            mild_underproduction_unit_cost=0.20,
            severe_underproduction_unit_cost=1.10,
        ),
        "default": EconomicCostProfile(),
    }
)
DEFAULT_ASYMMETRIC_ECONOMIC_OBJECTIVE_CONFIG = AsymmetricEconomicObjectiveConfig()


def economic_objective_config_payload(
    config: (
        SegmentedEconomicObjectiveConfig | AsymmetricEconomicObjectiveConfig
    ) = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
) -> dict[str, object]:
    return cast(dict[str, object], asdict(config))


def economic_scoring_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ECONOMIC_SEGMENT_METADATA_COLS if column in frame.columns]


def resolve_economic_segments(
    frame: pd.DataFrame,
    *,
    config: SegmentedEconomicObjectiveConfig = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
) -> pd.Series:
    segments = pd.Series("default", index=frame.index, dtype="string")
    product_override_mask = _apply_product_override_segments(
        frame,
        segments=segments,
        config=config,
    )
    _apply_family_segments(
        frame,
        segments=segments,
        protected_mask=product_override_mask,
        config=config,
    )
    _apply_bakery_fallback_segments(frame, segments=segments, config=config)
    return segments


def score_asymmetric_forecast_decision(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    *,
    config: AsymmetricEconomicObjectiveConfig = DEFAULT_ASYMMETRIC_ECONOMIC_OBJECTIVE_CONFIG,
) -> dict[str, float]:
    return _score_with_profile(
        actual_values=_finite_values(actual),
        prediction_values=_finite_values(prediction),
        profile=_profile_from_legacy_config(config),
    )


def score_segmented_asymmetric_forecast_decision(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    *,
    frame: pd.DataFrame | None = None,
    segments: pd.Series | np.ndarray | None = None,
    config: SegmentedEconomicObjectiveConfig = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
) -> dict[str, object]:
    actual_values = _finite_values(actual)
    prediction_values = _finite_values(prediction)
    if len(actual_values) != len(prediction_values):
        raise ValueError("Actual and prediction arrays must have the same length.")
    segment_values = _resolved_segment_values(
        frame=frame,
        segments=segments,
        row_count=len(actual_values),
        config=config,
    )
    segment_scores: dict[str, dict[str, float]] = {}
    for segment_name in sorted(set(segment_values.astype(str).tolist())):
        segment_mask = segment_values.astype(str) == segment_name
        segment_scores[segment_name] = _score_with_profile(
            actual_values=actual_values[segment_mask],
            prediction_values=prediction_values[segment_mask],
            profile=_profile_for_segment(segment_name, config=config),
        )
    return {
        **_aggregate_segment_scores(segment_scores),
        "segment_decision_loss": segment_scores,
    }


def decision_scores_by_group(
    frame: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    *,
    group_col: str,
    config: SegmentedEconomicObjectiveConfig = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
) -> dict[str, dict[str, float]]:
    if group_col not in frame.columns:
        return {}
    actual_values = _finite_values(actual)
    prediction_values = _finite_values(prediction)
    scores: dict[str, dict[str, float]] = {}
    groups = frame[group_col].astype("string").fillna("unknown").astype(str)
    for group_value in sorted(groups.unique().tolist()):
        group_mask = (groups == group_value).to_numpy(dtype=bool)
        group_frame: pd.DataFrame = frame.iloc[np.flatnonzero(group_mask)].reset_index(
            drop=True
        )
        scores[str(group_value)] = _compact_score(
            score_segmented_asymmetric_forecast_decision(
                actual_values[group_mask],
                prediction_values[group_mask],
                frame=group_frame,
                config=config,
            )
        )
    return scores


def fit_economic_decision_calibration(
    frame: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    *,
    config: SegmentedEconomicObjectiveConfig = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
    slope_grid: tuple[float, ...] = _DEFAULT_ECONOMIC_CALIBRATION_SLOPES,
    intercept_grid: tuple[float, ...] = _DEFAULT_ECONOMIC_CALIBRATION_INTERCEPTS,
    min_segment_rows: int = 60,
    min_product_rows: int = 60,
    wape_guardrail_multiplier: float = 1.05,
) -> dict[str, object]:
    actual_values = _finite_values(actual)
    prediction_values = _finite_values(prediction)
    if len(actual_values) != len(prediction_values):
        raise ValueError("Actual and prediction arrays must have the same length.")
    working_frame = frame.reset_index(drop=True).copy()
    segments = resolve_economic_segments(working_frame, config=config)
    calibration: dict[str, object] = {
        "version": "economic_decision_calibration_final",
        "enabled": True,
        "min_segment_rows": int(min_segment_rows),
        "min_product_rows": int(min_product_rows),
        "wape_guardrail_multiplier": float(wape_guardrail_multiplier),
        "global": _best_calibration_entry(
            working_frame,
            actual_values,
            prediction_values,
            config=config,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            wape_guardrail_multiplier=wape_guardrail_multiplier,
            scope="global",
            key=_GLOBAL_CALIBRATION_KEY,
        ),
        "families": _family_calibration_entries(
            working_frame=working_frame,
            segments=segments,
            actual_values=actual_values,
            prediction_values=prediction_values,
            config=config,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            min_segment_rows=min_segment_rows,
            wape_guardrail_multiplier=wape_guardrail_multiplier,
        ),
        "products": _product_calibration_entries(
            working_frame=working_frame,
            actual_values=actual_values,
            prediction_values=prediction_values,
            config=config,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            min_product_rows=min_product_rows,
            wape_guardrail_multiplier=wape_guardrail_multiplier,
        ),
    }
    calibration["applied"] = _calibration_has_applied_entry(calibration)
    return calibration


def apply_economic_decision_calibration_to_predictions(
    predictions_df: pd.DataFrame,
    *,
    calibration: dict[str, object] | None,
    config: SegmentedEconomicObjectiveConfig = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
) -> pd.DataFrame:
    if not _calibration_enabled(calibration):
        return predictions_df
    output = predictions_df.copy()
    slopes, intercepts, scopes, keys, applied = _row_calibration_arrays(
        output,
        calibration=cast(dict[str, object], calibration),
        config=config,
    )
    output["prediction_raw_before_economic_calibration"] = output[
        "prediction_raw"
    ].astype(float).to_numpy()
    output["economic_calibration_scope"] = scopes
    output["economic_calibration_key"] = keys
    output["economic_calibration_slope"] = slopes
    output["economic_calibration_intercept"] = intercepts
    output["economic_calibration_applied"] = applied
    if not bool(applied.any()):
        return output
    for column in ("prediction_raw", *_CALIBRATED_QUANTILE_COLS):
        if column not in output.columns:
            continue
        raw_values = output[column].astype(float).to_numpy()
        output[column] = np.clip(raw_values * slopes + intercepts, 0.0, None)
    output["prediction_rounded"] = np.round(
        output["prediction_raw"].astype(float).to_numpy(),
        0,
    )
    return output


def aggregate_asymmetric_decision_scores(
    fold_results: list[dict[str, object]],
    *,
    config: SegmentedEconomicObjectiveConfig = DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG,
) -> dict[str, object]:
    metric_keys = (
        "economic_loss",
        "normalized_economic_loss",
        "overproduction_units",
        "mild_underproduction_units",
        "severe_underproduction_units",
        "normalized_bias",
        "positive_bias_penalty",
        "severe_negative_bias_penalty",
        "wape_guardrail_penalty",
        "decision_loss",
    )
    config_payload = economic_objective_config_payload(config)
    return {
        **{
            f"mean_{metric_key}": _mean_fold_metric(fold_results, metric_key)
            for metric_key in metric_keys
        },
        "segment_decision_loss": _aggregate_nested_metric(
            fold_results,
            nested_key="segment_decision_loss",
            metric_key="decision_loss",
        ),
        "segment_bias": _aggregate_nested_metric(
            fold_results,
            nested_key="segment_decision_loss",
            metric_key="normalized_bias",
        ),
        "product_decision_loss": _aggregate_nested_metric(
            fold_results,
            nested_key="product_decision_loss",
            metric_key="decision_loss",
        ),
        "product_bias": _aggregate_nested_metric(
            fold_results,
            nested_key="product_decision_loss",
            metric_key="normalized_bias",
        ),
        "segmented_economic_objective_config": config_payload,
        "economic_objective_config": config_payload,
        "validation_monitor_metric": "segmented_asymmetric_decision_loss",
    }


def _profile_from_legacy_config(
    config: AsymmetricEconomicObjectiveConfig,
) -> EconomicCostProfile:
    return EconomicCostProfile(
        overproduction_unit_cost=config.overproduction_unit_cost,
        mild_underproduction_unit_cost=config.mild_underproduction_unit_cost,
        severe_underproduction_unit_cost=config.severe_underproduction_unit_cost,
        mild_underproduction_threshold_ratio=config.mild_underproduction_threshold_ratio,
        target_normalized_bias=config.target_normalized_bias,
        max_allowed_negative_bias=config.max_allowed_negative_bias,
        positive_bias_penalty_weight=config.positive_bias_penalty_weight,
        severe_negative_bias_penalty_weight=config.severe_negative_bias_penalty_weight,
        wape_guardrail_limit=config.wape_guardrail_limit,
        wape_guardrail_penalty_weight=config.wape_guardrail_penalty_weight,
    )


def _apply_product_override_segments(
    frame: pd.DataFrame,
    *,
    segments: pd.Series,
    config: SegmentedEconomicObjectiveConfig,
) -> np.ndarray:
    product_col = _first_present_column(frame, config.product_id_columns)
    product_override_mask = np.zeros(len(frame), dtype=bool)
    if product_col is None or not config.product_profiles:
        return product_override_mask
    product_values = frame[product_col].astype("string").fillna("").astype(str)
    for product_key in config.product_profiles:
        mask = product_values.to_numpy() == str(product_key)
        if not bool(mask.any()):
            continue
        segments.iloc[np.flatnonzero(mask)] = f"{_PRODUCT_SEGMENT_PREFIX}{product_key}"
        product_override_mask |= mask
    return product_override_mask


def _apply_family_segments(
    frame: pd.DataFrame,
    *,
    segments: pd.Series,
    protected_mask: np.ndarray,
    config: SegmentedEconomicObjectiveConfig,
) -> None:
    family_col = _first_present_column(frame, config.family_columns)
    if family_col is None:
        return
    family_values = frame[family_col].astype("string").fillna("").astype(str)
    matched_families = [
        _matched_family(value, config) or ""
        for value in family_values.to_numpy(dtype=str)
    ]
    for row_position, family in enumerate(matched_families):
        if protected_mask[row_position] or not family:
            continue
        segments.iloc[row_position] = family


def _apply_bakery_fallback_segments(
    frame: pd.DataFrame,
    *,
    segments: pd.Series,
    config: SegmentedEconomicObjectiveConfig,
) -> None:
    product_name_col = _first_present_column(frame, config.product_name_columns)
    if product_name_col is None:
        return
    unresolved = segments.astype(str).to_numpy() == "default"
    if not bool(unresolved.any()):
        return
    product_names = frame[product_name_col].astype("string").fillna("").astype(str)
    fallback_segments = [
        _substitution_family(product_name)
        for product_name in product_names.to_numpy(dtype=str)
    ]
    for row_position, family in enumerate(fallback_segments):
        if unresolved[row_position]:
            segments.iloc[row_position] = family


def _substitution_family(product_name: str) -> str:
    normalized_name = product_name.upper().strip()
    for family, keywords in _SUBSTITUTION_FAMILY_KEYWORDS:
        if any(keyword in normalized_name for keyword in keywords):
            return family
    return "default"


def _matched_family(
    value: str | None,
    config: SegmentedEconomicObjectiveConfig,
) -> str | None:
    if value is None:
        return None
    candidate = _canonical_segment_key(value)
    if candidate in config.family_profiles:
        return candidate
    for family in config.family_profiles:
        if family != "default" and family in candidate:
            return family
    return None


def _canonical_segment_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _profile_for_segment(
    segment_name: str,
    *,
    config: SegmentedEconomicObjectiveConfig,
) -> EconomicCostProfile:
    if segment_name.startswith(_PRODUCT_SEGMENT_PREFIX):
        product_key = segment_name.removeprefix(_PRODUCT_SEGMENT_PREFIX)
        return config.product_profiles.get(product_key, config.default_profile)
    return config.family_profiles.get(
        segment_name,
        config.family_profiles.get("default", config.default_profile),
    )


def _resolved_segment_values(
    *,
    frame: pd.DataFrame | None,
    segments: pd.Series | np.ndarray | None,
    row_count: int,
    config: SegmentedEconomicObjectiveConfig,
) -> np.ndarray:
    if segments is not None:
        segment_values = np.asarray(segments, dtype=str)
    elif frame is not None:
        segment_values = resolve_economic_segments(frame, config=config).to_numpy(
            dtype=str
        )
    else:
        segment_values = np.asarray(["default"] * row_count, dtype=str)
    if len(segment_values) != row_count:
        raise ValueError("Segment array must align with actual/prediction arrays.")
    return segment_values


def _score_with_profile(
    actual_values: np.ndarray,
    prediction_values: np.ndarray,
    *,
    profile: EconomicCostProfile,
) -> dict[str, float]:
    if len(actual_values) != len(prediction_values):
        raise ValueError("Actual and prediction arrays must have the same length.")
    error = prediction_values - actual_values
    over_units = np.maximum(error, 0.0)
    under_units = np.maximum(-error, 0.0)
    mild_under_limit = np.abs(actual_values) * profile.mild_underproduction_threshold_ratio
    mild_under_units = np.minimum(under_units, mild_under_limit)
    severe_under_units = np.maximum(under_units - mild_under_units, 0.0)
    economic_loss = _economic_loss(
        over_units=over_units,
        mild_under_units=mild_under_units,
        severe_under_units=severe_under_units,
        profile=profile,
    )
    denominator = _normalization_denominator(actual_values, prediction_values)
    normalized_bias = float(error.sum() / denominator)
    normalized_economic_loss = float(economic_loss / denominator)
    positive_bias_penalty = _positive_bias_penalty(normalized_bias, profile=profile)
    severe_negative_bias_penalty = _severe_negative_bias_penalty(
        normalized_bias,
        profile=profile,
    )
    wape_guardrail_penalty = _wape_guardrail_penalty(
        actual_values,
        prediction_values,
        profile=profile,
    )
    decision_loss = (
        normalized_economic_loss
        + positive_bias_penalty
        + severe_negative_bias_penalty
        + wape_guardrail_penalty
    )
    return {
        "economic_loss": economic_loss,
        "normalized_economic_loss": normalized_economic_loss,
        "overproduction_units": float(over_units.sum()),
        "mild_underproduction_units": float(mild_under_units.sum()),
        "severe_underproduction_units": float(severe_under_units.sum()),
        "normalized_bias": normalized_bias,
        "positive_bias_penalty": positive_bias_penalty,
        "severe_negative_bias_penalty": severe_negative_bias_penalty,
        "wape_guardrail_penalty": wape_guardrail_penalty,
        "decision_loss": decision_loss,
        "rows": float(len(actual_values)),
        "denominator": denominator,
    }


def _aggregate_segment_scores(
    segment_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not segment_scores:
        return _score_with_profile(
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            profile=EconomicCostProfile(),
        )
    denominator = max(
        sum(score["denominator"] for score in segment_scores.values()),
        FLOAT_COMPARISON_EPSILON,
    )
    economic_loss = sum(score["economic_loss"] for score in segment_scores.values())
    weighted_penalties = {
        key: sum(
            score[key] * score["denominator"] / denominator
            for score in segment_scores.values()
        )
        for key in (
            "positive_bias_penalty",
            "severe_negative_bias_penalty",
            "wape_guardrail_penalty",
        )
    }
    normalized_economic_loss = float(economic_loss / denominator)
    return {
        "economic_loss": float(economic_loss),
        "normalized_economic_loss": normalized_economic_loss,
        "overproduction_units": float(
            sum(score["overproduction_units"] for score in segment_scores.values())
        ),
        "mild_underproduction_units": float(
            sum(score["mild_underproduction_units"] for score in segment_scores.values())
        ),
        "severe_underproduction_units": float(
            sum(score["severe_underproduction_units"] for score in segment_scores.values())
        ),
        "normalized_bias": float(
            sum(
                score["normalized_bias"] * score["denominator"]
                for score in segment_scores.values()
            )
            / denominator
        ),
        **weighted_penalties,
        "decision_loss": float(normalized_economic_loss + sum(weighted_penalties.values())),
    }


def _family_calibration_entries(
    *,
    working_frame: pd.DataFrame,
    segments: pd.Series,
    actual_values: np.ndarray,
    prediction_values: np.ndarray,
    config: SegmentedEconomicObjectiveConfig,
    slope_grid: tuple[float, ...],
    intercept_grid: tuple[float, ...],
    min_segment_rows: int,
    wape_guardrail_multiplier: float,
) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for segment_name in sorted(segments.astype(str).unique().tolist()):
        if segment_name.startswith(_PRODUCT_SEGMENT_PREFIX):
            continue
        segment_mask = segments.astype(str).to_numpy() == segment_name
        if int(segment_mask.sum()) < min_segment_rows:
            continue
        entries[segment_name] = _best_calibration_entry(
            working_frame.iloc[np.flatnonzero(segment_mask)].reset_index(drop=True),
            actual_values[segment_mask],
            prediction_values[segment_mask],
            config=config,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            wape_guardrail_multiplier=wape_guardrail_multiplier,
            scope="family",
            key=segment_name,
        )
    return entries


def _product_calibration_entries(
    *,
    working_frame: pd.DataFrame,
    actual_values: np.ndarray,
    prediction_values: np.ndarray,
    config: SegmentedEconomicObjectiveConfig,
    slope_grid: tuple[float, ...],
    intercept_grid: tuple[float, ...],
    min_product_rows: int,
    wape_guardrail_multiplier: float,
) -> dict[str, dict[str, object]]:
    product_col = _first_present_column(working_frame, config.product_id_columns)
    if product_col is None or not config.product_profiles:
        return {}
    entries: dict[str, dict[str, object]] = {}
    products = working_frame[product_col].astype("string").fillna("").astype(str)
    for product_key in sorted(config.product_profiles):
        product_mask = products.to_numpy() == str(product_key)
        if int(product_mask.sum()) < min_product_rows:
            continue
        entries[product_key] = _best_calibration_entry(
            working_frame.iloc[np.flatnonzero(product_mask)].reset_index(drop=True),
            actual_values[product_mask],
            prediction_values[product_mask],
            config=config,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            wape_guardrail_multiplier=wape_guardrail_multiplier,
            scope="product",
            key=product_key,
        )
    return entries


def _best_calibration_entry(
    frame: pd.DataFrame,
    actual_values: np.ndarray,
    prediction_values: np.ndarray,
    *,
    config: SegmentedEconomicObjectiveConfig,
    slope_grid: tuple[float, ...],
    intercept_grid: tuple[float, ...],
    wape_guardrail_multiplier: float,
    scope: str,
    key: str,
) -> dict[str, object]:
    raw_score = score_segmented_asymmetric_forecast_decision(
        actual_values,
        prediction_values,
        frame=frame,
        config=config,
    )
    raw_decision_loss = float(cast(Any, raw_score["decision_loss"]))
    raw_wape = float(compute_wape(actual_values, prediction_values))
    best_entry = _calibration_entry(
        scope=scope,
        key=key,
        rows=len(actual_values),
        slope=1.0,
        intercept=0.0,
        raw_decision_loss=raw_decision_loss,
        calibrated_decision_loss=raw_decision_loss,
        raw_wape=raw_wape,
        calibrated_wape=raw_wape,
        applied=False,
    )
    for slope in slope_grid:
        for intercept in intercept_grid:
            calibrated = np.clip(prediction_values * float(slope) + float(intercept), 0.0, None)
            calibrated_wape = float(compute_wape(actual_values, calibrated))
            if calibrated_wape > raw_wape * wape_guardrail_multiplier:
                continue
            calibrated_score = score_segmented_asymmetric_forecast_decision(
                actual_values,
                calibrated,
                frame=frame,
                config=config,
            )
            calibrated_decision_loss = float(cast(Any, calibrated_score["decision_loss"]))
            best_decision_loss = float(
                cast(Any, best_entry["calibrated_decision_loss"])
            )
            if calibrated_decision_loss >= best_decision_loss:
                continue
            best_entry = _calibration_entry(
                scope=scope,
                key=key,
                rows=len(actual_values),
                slope=float(slope),
                intercept=float(intercept),
                raw_decision_loss=raw_decision_loss,
                calibrated_decision_loss=calibrated_decision_loss,
                raw_wape=raw_wape,
                calibrated_wape=calibrated_wape,
                applied=calibrated_decision_loss < raw_decision_loss,
            )
    return best_entry


def _calibration_entry(
    *,
    scope: str,
    key: str,
    rows: int,
    slope: float,
    intercept: float,
    raw_decision_loss: float,
    calibrated_decision_loss: float,
    raw_wape: float,
    calibrated_wape: float,
    applied: bool,
) -> dict[str, object]:
    return {
        "scope": scope,
        "key": key,
        "rows": int(rows),
        "slope": float(slope),
        "intercept": float(intercept),
        "raw_decision_loss": float(raw_decision_loss),
        "calibrated_decision_loss": float(calibrated_decision_loss),
        "raw_wape": float(raw_wape),
        "calibrated_wape": float(calibrated_wape),
        "applied": bool(applied),
    }


def _calibration_has_applied_entry(calibration: dict[str, object]) -> bool:
    global_entry = cast(dict[str, object], calibration.get("global", {}))
    if bool(global_entry.get("applied", False)):
        return True
    for nested_key in ("families", "products"):
        nested = cast(dict[str, dict[str, object]], calibration.get(nested_key, {}))
        if any(bool(entry.get("applied", False)) for entry in nested.values()):
            return True
    return False


def _calibration_enabled(calibration: dict[str, object] | None) -> bool:
    return bool(calibration and calibration.get("enabled") and calibration.get("applied"))


def _row_calibration_arrays(
    frame: pd.DataFrame,
    *,
    calibration: dict[str, object],
    config: SegmentedEconomicObjectiveConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_count = len(frame)
    slopes = np.ones(row_count, dtype=float)
    intercepts = np.zeros(row_count, dtype=float)
    scopes = np.asarray(["none"] * row_count, dtype=object)
    keys = np.asarray([""] * row_count, dtype=object)
    applied = np.zeros(row_count, dtype=bool)
    product_entries = cast(dict[str, dict[str, object]], calibration.get("products", {}))
    product_col = _first_present_column(frame, config.product_id_columns)
    if product_col is not None and product_entries:
        product_values = frame[product_col].astype("string").fillna("").astype(str)
        for product_key, entry in product_entries.items():
            mask = product_values.to_numpy() == str(product_key)
            _assign_calibration_entry(mask, entry, slopes, intercepts, scopes, keys, applied)
    family_entries = cast(dict[str, dict[str, object]], calibration.get("families", {}))
    if family_entries:
        segments = resolve_economic_segments(frame, config=config).astype(str).to_numpy()
        for family_key, entry in family_entries.items():
            mask = (segments == str(family_key)) & ~applied
            _assign_calibration_entry(mask, entry, slopes, intercepts, scopes, keys, applied)
    global_entry = cast(dict[str, object], calibration.get("global", {}))
    _assign_calibration_entry(~applied, global_entry, slopes, intercepts, scopes, keys, applied)
    return slopes, intercepts, scopes, keys, applied


def _assign_calibration_entry(
    mask: np.ndarray,
    entry: dict[str, object],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    scopes: np.ndarray,
    keys: np.ndarray,
    applied: np.ndarray,
) -> None:
    if not bool(entry.get("applied", False)) or not bool(mask.any()):
        return
    slopes[mask] = float(cast(Any, entry["slope"]))
    intercepts[mask] = float(cast(Any, entry["intercept"]))
    scopes[mask] = str(entry.get("scope", "global"))
    keys[mask] = str(entry.get("key", _GLOBAL_CALIBRATION_KEY))
    applied[mask] = True


def _first_present_column(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _economic_loss(
    *,
    over_units: np.ndarray,
    mild_under_units: np.ndarray,
    severe_under_units: np.ndarray,
    profile: EconomicCostProfile,
) -> float:
    return float(
        over_units.sum() * profile.overproduction_unit_cost
        + mild_under_units.sum() * profile.mild_underproduction_unit_cost
        + severe_under_units.sum() * profile.severe_underproduction_unit_cost
    )


def _normalization_denominator(
    actual_values: np.ndarray,
    prediction_values: np.ndarray,
) -> float:
    target_volume = float(np.abs(actual_values).sum())
    if target_volume > FLOAT_COMPARISON_EPSILON:
        return target_volume
    return max(float(len(actual_values)), float(np.abs(prediction_values).sum()), 1.0)


def _positive_bias_penalty(
    normalized_bias: float,
    *,
    profile: EconomicCostProfile,
) -> float:
    excess_positive_bias = max(
        normalized_bias - max(profile.target_normalized_bias, 0.0),
        0.0,
    )
    return excess_positive_bias * profile.positive_bias_penalty_weight


def _severe_negative_bias_penalty(
    normalized_bias: float,
    *,
    profile: EconomicCostProfile,
) -> float:
    excess_negative_bias = max(profile.max_allowed_negative_bias - normalized_bias, 0.0)
    return excess_negative_bias * profile.severe_negative_bias_penalty_weight


def _wape_guardrail_penalty(
    actual_values: np.ndarray,
    prediction_values: np.ndarray,
    *,
    profile: EconomicCostProfile,
) -> float:
    if profile.wape_guardrail_limit is None:
        return 0.0
    wape = compute_wape(actual_values, prediction_values)
    if not np.isfinite(wape):
        return profile.wape_guardrail_penalty_weight
    excess_wape = max(wape - profile.wape_guardrail_limit, 0.0)
    return excess_wape * profile.wape_guardrail_penalty_weight


def _mean_fold_metric(
    fold_results: list[dict[str, object]],
    metric_key: str,
) -> float | None:
    values = [
        float(cast(Any, row[metric_key]))
        for row in fold_results
        if row.get(metric_key) is not None
    ]
    return float(np.mean(values)) if values else None


def _aggregate_nested_metric(
    fold_results: list[dict[str, object]],
    *,
    nested_key: str,
    metric_key: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in fold_results:
        nested = row.get(nested_key)
        if not isinstance(nested, dict):
            continue
        nested_payload = cast(dict[object, object], nested)
        for segment_name, segment_payload in nested_payload.items():
            if not isinstance(segment_payload, dict):
                continue
            metric_payload = cast(dict[str, object], segment_payload)
            value = metric_payload.get(metric_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            grouped.setdefault(str(segment_name), []).append(float(value))
    return {
        key: float(np.mean(values))
        for key, values in sorted(grouped.items())
        if values
    }


def _compact_score(score: dict[str, object]) -> dict[str, float]:
    keep_keys = (
        "economic_loss",
        "normalized_economic_loss",
        "overproduction_units",
        "mild_underproduction_units",
        "severe_underproduction_units",
        "normalized_bias",
        "positive_bias_penalty",
        "severe_negative_bias_penalty",
        "wape_guardrail_penalty",
        "decision_loss",
    )
    return {key: float(cast(Any, score[key])) for key in keep_keys}


__all__ = [
    "AsymmetricEconomicObjectiveConfig",
    "DEFAULT_ASYMMETRIC_ECONOMIC_OBJECTIVE_CONFIG",
    "DEFAULT_SEGMENTED_ECONOMIC_OBJECTIVE_CONFIG",
    "ECONOMIC_SEGMENT_METADATA_COLS",
    "EconomicCostProfile",
    "SegmentedEconomicObjectiveConfig",
    "aggregate_asymmetric_decision_scores",
    "apply_economic_decision_calibration_to_predictions",
    "decision_scores_by_group",
    "economic_objective_config_payload",
    "economic_scoring_metadata_columns",
    "fit_economic_decision_calibration",
    "resolve_economic_segments",
    "score_asymmetric_forecast_decision",
    "score_segmented_asymmetric_forecast_decision",
]
