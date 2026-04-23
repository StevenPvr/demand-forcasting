from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
import unittest

import optuna


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
if str(PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC))
if str(PRODUCT_SRC) not in sys.path:
    sys.path.insert(0, str(PRODUCT_SRC))

import praedixa.demand_forecast.training.tuning_policy as tuning_policy_module  # noqa: E402
from praedixa.demand_forecast.training.constants import (  # noqa: E402
    DEFAULT_TUNING_BATCH_SIZE_CHOICES,
    DEFAULT_TUNING_GPU_BATCH_SIZE_CHOICES,
)
from praedixa.demand_forecast.training.tuning_policy import (  # noqa: E402
    resolve_stage_policy,
    sample_optuna_params,
)


class TuningPolicyTests(unittest.TestCase):
    def test_scaleway_l40s_uses_nvidia_aligned_batch_space(self) -> None:
        batch_size_choices = cast(Any, tuning_policy_module)._runtime_profile_batch_size_choices(
            "scaleway_l40s"
        )

        self.assertEqual(batch_size_choices, DEFAULT_TUNING_GPU_BATCH_SIZE_CHOICES)

    def test_local_cpu_uses_same_batch_space_as_nvidia_profile(self) -> None:
        batch_size_choices = cast(Any, tuning_policy_module)._runtime_profile_batch_size_choices("local_cpu")

        self.assertEqual(batch_size_choices, DEFAULT_TUNING_GPU_BATCH_SIZE_CHOICES)
        self.assertEqual(batch_size_choices, DEFAULT_TUNING_BATCH_SIZE_CHOICES)

    def test_gpu_learning_rate_bounds_scale_up_with_batch_size(self) -> None:
        low, high = cast(Any, tuning_policy_module)._batch_scaled_learning_rate_bounds(
            batch_size=512,
            runtime_profile_name="scaleway_l40s",
            low=1e-3,
            high=1e-2,
        )

        self.assertAlmostEqual(low, 0.0028284271247461905)
        self.assertAlmostEqual(high, 0.028284271247461905)

    def test_gpu_learning_rate_bounds_continue_scaling_at_2048_batch_size(self) -> None:
        low, high = cast(Any, tuning_policy_module)._batch_scaled_learning_rate_bounds(
            batch_size=2048,
            runtime_profile_name="scaleway_l40s",
            low=1e-3,
            high=1e-2,
        )

        self.assertAlmostEqual(low, 0.005656854249492381)
        self.assertAlmostEqual(high, 0.05656854249492381)

    def test_local_cpu_learning_rate_bounds_follow_same_batch_scaling(self) -> None:
        low, high = cast(Any, tuning_policy_module)._batch_scaled_learning_rate_bounds(
            batch_size=2048,
            runtime_profile_name="local_cpu",
            low=1e-3,
            high=1e-2,
        )

        self.assertAlmostEqual(low, 0.005656854249492381)
        self.assertAlmostEqual(high, 0.05656854249492381)

    def test_stage_a_gpu_sampling_accepts_large_batch_and_scaled_learning_rate(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "max_epochs": 2,
                "batch_size": 2048,
                "max_encoder_length": 14,
                "gradient_clip_val": 0.1,
                "learning_rate": 0.02,
                "hidden_size": 32,
                "hidden_continuous_size": 16,
                "attention_head_size": 2,
                "lstm_layers": 2,
                "dropout": 0.15,
                "weight_decay": 1e-4,
            }
        )

        sampled = sample_optuna_params(
            cast(Any, trial),
            random_seed=7,
            runtime_profile_name="scaleway_l40s",
            stage_name="stage_a",
        )

        self.assertEqual(sampled["batch_size"], 2048)
        self.assertEqual(sampled["learning_rate"], 0.02)
        self.assertEqual(sampled["max_epochs"], 2)

    def test_stage_policy_is_pinned_to_two_epochs_for_local_iteration_speed(self) -> None:
        stage_policy = resolve_stage_policy(4, stage_budget="standard")

        self.assertEqual(stage_policy["stage_a"]["epoch_range"], [2, 2])
        self.assertEqual(stage_policy["stage_b"]["epoch_range"], [2, 2])

    def test_stage_b_anchor_learning_rate_scales_with_batch_size(self) -> None:
        scaled_anchor = cast(Any, tuning_policy_module)._anchor_learning_rate_for_batch(
            anchor_params={"batch_size": 256, "learning_rate": 0.01},
            batch_size=512,
            runtime_profile_name="scaleway_l40s",
        )

        self.assertAlmostEqual(float(scaled_anchor), 0.014142135623730952)

    def test_stage_b_anchor_learning_rate_scales_with_batch_size_on_local_cpu_too(self) -> None:
        scaled_anchor = cast(Any, tuning_policy_module)._anchor_learning_rate_for_batch(
            anchor_params={"batch_size": 256, "learning_rate": 0.01},
            batch_size=512,
            runtime_profile_name="local_cpu",
        )

        self.assertAlmostEqual(float(scaled_anchor), 0.014142135623730952)


if __name__ == "__main__":
    unittest.main()
