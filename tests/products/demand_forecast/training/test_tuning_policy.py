from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
import unittest

import optuna


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
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
    DEFAULT_TUNING_LEARNING_RATE_HIGH,
    DEFAULT_TUNING_LEARNING_RATE_LOW,
    DEFAULT_TUNING_MAX_BATCH_SCALED_LEARNING_RATE_SCALE,
)
from praedixa.demand_forecast.training.tuning_policy import (  # noqa: E402
    resolve_stage_policy,
    sample_optuna_params,
)


class TuningPolicyTests(unittest.TestCase):
    def test_scaleway_l40s_uses_nvidia_aligned_batch_space(self) -> None:
        batch_size_choices = cast(
            Any, tuning_policy_module
        )._runtime_profile_batch_size_choices("scaleway_l40s")

        self.assertEqual(batch_size_choices, DEFAULT_TUNING_GPU_BATCH_SIZE_CHOICES)

    def test_local_cpu_uses_same_batch_space_as_nvidia_profile(self) -> None:
        batch_size_choices = cast(
            Any, tuning_policy_module
        )._runtime_profile_batch_size_choices("local_cpu")

        self.assertEqual(batch_size_choices, DEFAULT_TUNING_GPU_BATCH_SIZE_CHOICES)
        self.assertEqual(batch_size_choices, DEFAULT_TUNING_BATCH_SIZE_CHOICES)

    def test_stage_a_gpu_sampling_samples_learning_rate_for_large_batch(
        self,
    ) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "max_epochs": 6,
                "batch_size": 2048,
                "max_encoder_length": 14,
                "gradient_clip_val": 0.1,
                "hidden_size": 32,
                "hidden_continuous_size": 16,
                "attention_head_size": 2,
                "lstm_layers": 2,
                "learning_rate": 0.01,
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
        self.assertEqual(sampled["max_epochs"], 6)
        self.assertFalse(bool(sampled["use_learning_rate_finder"]))
        self.assertEqual(float(cast(Any, sampled["learning_rate"])), 0.01)

    def test_stage_policy_uses_real_training_budgets_for_standard_runs(self) -> None:
        stage_policy = resolve_stage_policy(4, stage_budget="standard")

        self.assertEqual(stage_policy["stage_a"]["epoch_range"], [6, 12])
        self.assertEqual(stage_policy["stage_b"]["epoch_range"], [18, 32])

    def test_stage_policy_uses_two_epochs_for_smoke_runs(self) -> None:
        stage_policy = resolve_stage_policy(2, stage_budget="smoke")

        self.assertEqual(stage_policy["stage_a"]["epoch_range"], [2, 2])
        self.assertEqual(stage_policy["stage_b"]["epoch_range"], [2, 2])

    def test_stage_b_sampling_keeps_explicit_learning_rate_and_narrows_regularization_around_anchor(
        self,
    ) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "max_epochs": 20,
                "batch_size": 4096,
                "max_encoder_length": 28,
                "gradient_clip_val": 0.1,
                "hidden_size": 64,
                "hidden_continuous_size": 32,
                "attention_head_size": 4,
                "lstm_layers": 2,
                "learning_rate": 0.02,
                "dropout": 0.18,
                "weight_decay": 1e-4,
            }
        )

        sampled = sample_optuna_params(
            cast(Any, trial),
            random_seed=7,
            runtime_profile_name="nvidia_h100",
            stage_name="stage_b",
            anchor_params={"dropout": 0.2, "weight_decay": 5e-4},
        )

        self.assertFalse(bool(sampled["use_learning_rate_finder"]))
        self.assertEqual(sampled["batch_size"], 4096)
        self.assertEqual(float(cast(Any, sampled["learning_rate"])), 0.02)

    def test_batch_scaled_learning_rate_bounds_expand_for_large_batches(self) -> None:
        low, high = cast(
            Any, tuning_policy_module
        )._batch_scaled_learning_rate_bounds(4096)

        expected_scale = DEFAULT_TUNING_MAX_BATCH_SCALED_LEARNING_RATE_SCALE
        self.assertEqual(low, DEFAULT_TUNING_LEARNING_RATE_LOW * expected_scale)
        self.assertEqual(high, DEFAULT_TUNING_LEARNING_RATE_HIGH * expected_scale)


if __name__ == "__main__":
    unittest.main()
