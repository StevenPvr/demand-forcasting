from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
PLATFORM_SRC = PROJECT_ROOT / "platform" / "python" / "src"
PRODUCT_SRC = PROJECT_ROOT / "products" / "demand_forecast" / "src"
for path in (PROJECT_ROOT, PLATFORM_SRC, PRODUCT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from praedixa.demand_forecast.training.pipeline import OptimisationBuildRequest  # noqa: E402
from praedixa.demand_forecast.training.main import (  # noqa: E402
    OFFICIAL_OPTIMISATION_MAIN_CONFIG,
    main as optimisation_main,
)
from praedixa.platform.runtime.paths import TRAINING_BUNDLE_DIR  # noqa: E402


def _write_bundle_fixture(bundle_dir: Path, *, include_optimisation_projection: bool = False) -> None:
    (bundle_dir / "train.parquet").write_text("placeholder", encoding="utf-8")
    (bundle_dir / "tuning.parquet").write_text("placeholder", encoding="utf-8")
    if include_optimisation_projection:
        (bundle_dir / "optimisation_train.parquet").write_text("placeholder", encoding="utf-8")
        (bundle_dir / "optimisation_tuning.parquet").write_text("placeholder", encoding="utf-8")
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"train_rows": 1, "tuning_rows": 1}),
        encoding="utf-8",
    )


def _assert_bundle_request(
    request: OptimisationBuildRequest,
    bundle_dir: Path,
    *,
    expected_runtime_profile: str,
) -> None:
    assert request.model_params is not None
    assert request.bundle_dir == bundle_dir
    assert request.train_input_path == bundle_dir / "train.parquet"
    assert request.tuning_input_path == bundle_dir / "tuning.parquet"
    assert request.n_folds == 2
    assert request.tuning_trials == 7
    assert request.train_sample_fraction == 0.02
    assert request.tuning_sample_fraction == 0.03
    assert request.model_params["runtime_profile"] == expected_runtime_profile
    assert request.model_params["stage_budget"] == "quick"
    assert request.model_params["tensorboard_logdir"] == str(bundle_dir / "tensorboard")


class OptimisationMainTests(unittest.TestCase):
    def test_official_main_config_uses_official_defaults(self) -> None:
        config = OFFICIAL_OPTIMISATION_MAIN_CONFIG

        self.assertEqual(config.runtime_profile, "local_cpu")
        self.assertEqual(config.n_folds, 2)
        self.assertEqual(config.max_trials, 2)
        self.assertEqual(config.stage_budget, "standard")
        self.assertEqual(config.train_sample_fraction, 0.001)
        self.assertEqual(config.tuning_sample_fraction, 0.001)
        self.assertEqual(config.bundle_dir, TRAINING_BUNDLE_DIR)

    def test_main_exposes_explicit_tft_not_ready_error(self) -> None:
        from praedixa.demand_forecast.backends.tft.backend import (
            TFTBackendNotReadyError,
        )

        with (
            mock.patch(
                "praedixa.demand_forecast.training.pipeline.build_optimisation_outputs",
                side_effect=TFTBackendNotReadyError("tft missing"),
            ),
            self.assertRaises(TFTBackendNotReadyError),
        ):
            optimisation_main()

    def test_main_uses_official_config_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir)
            patched_config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
                bundle_dir=bundle_dir,
                runtime_profile="local_cpu",
                output_dir=bundle_dir / "outputs",
                n_folds=2,
                max_trials=7,
                stage_budget="quick",
                train_sample_fraction=0.02,
                tuning_sample_fraction=0.03,
                tensorboard_logdir=bundle_dir / "tensorboard",
            )

            with (
                mock.patch(
                    "praedixa.demand_forecast.training.pipeline.build_optimisation_outputs",
                    return_value={"best_params": bundle_dir / "best_optuna_params.json"},
                ) as mocked_build,
                mock.patch(
                    "praedixa.demand_forecast.training.main.OFFICIAL_OPTIMISATION_MAIN_CONFIG",
                    patched_config,
                ),
            ):
                optimisation_main()

        args, kwargs = mocked_build.call_args
        self.assertEqual(kwargs, {})
        _assert_bundle_request(
            args[0],
            bundle_dir,
            expected_runtime_profile="local_cpu",
        )

    def test_main_prefers_optimisation_projection_when_bundle_exposes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir, include_optimisation_projection=True)
            patched_config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
                bundle_dir=bundle_dir,
                runtime_profile="local_cpu",
                output_dir=bundle_dir / "outputs",
                n_folds=2,
                max_trials=7,
                stage_budget="quick",
                train_sample_fraction=0.02,
                tuning_sample_fraction=0.03,
                tensorboard_logdir=bundle_dir / "tensorboard",
            )

            with (
                mock.patch(
                    "praedixa.demand_forecast.training.pipeline.build_optimisation_outputs",
                    return_value={"best_params": bundle_dir / "best_optuna_params.json"},
                ) as mocked_build,
                mock.patch(
                    "praedixa.demand_forecast.training.main.OFFICIAL_OPTIMISATION_MAIN_CONFIG",
                    patched_config,
                ),
            ):
                optimisation_main()

        args, _ = mocked_build.call_args
        request = args[0]
        self.assertEqual(request.train_input_path, bundle_dir / "optimisation_train.parquet")
        self.assertEqual(request.tuning_input_path, bundle_dir / "optimisation_tuning.parquet")

    def test_main_rejects_runtime_profile_mismatch_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir)
            patched_config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
                bundle_dir=bundle_dir,
                runtime_profile="scaleway_l40s",
                output_dir=bundle_dir / "outputs",
                n_folds=2,
                max_trials=7,
                stage_budget="quick",
                train_sample_fraction=0.02,
                tuning_sample_fraction=0.03,
                tensorboard_logdir=bundle_dir / "tensorboard",
            )

            with (
                mock.patch(
                    "praedixa.demand_forecast.training.pipeline.build_optimisation_outputs",
                    return_value={"best_params": bundle_dir / "best_optuna_params.json"},
                ),
                mock.patch(
                    "praedixa.demand_forecast.training.main.OFFICIAL_OPTIMISATION_MAIN_CONFIG",
                    patched_config,
                ),
                mock.patch(
                    "praedixa.demand_forecast.training.main._cuda_available",
                    return_value=False,
                ),
                mock.patch(
                    "praedixa.demand_forecast.training.main._mps_available",
                    return_value=False,
                ),
                self.assertRaisesRegex(RuntimeError, "Edit `OFFICIAL_OPTIMISATION_MAIN_CONFIG.runtime_profile` yourself."),
            ):
                optimisation_main()


if __name__ == "__main__":
    unittest.main()
