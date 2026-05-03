from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, cast
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

from praedixa.demand_forecast.training.orchestration.pipeline import (  # noqa: E402
    OptimisationBuildRequest,
)
from praedixa.demand_forecast.training.main import (  # noqa: E402
    OFFICIAL_OPTIMISATION_MAIN_CONFIG,
    build_default_optimisation_main_config,
    main as optimisation_main,
)
from praedixa.demand_forecast.training.config.main_config import (  # noqa: E402
    build_optimisation_model_params,
)
from praedixa.demand_forecast.backends.xgboost.runtime import (  # noqa: E402
    XGBoostRuntimeResolution,
)
from praedixa.demand_forecast.feature_selection.paths import (  # noqa: E402
    DEFAULT_FEATURE_SELECTION_DIR,
)


def _write_bundle_fixture(
    bundle_dir: Path, *, include_optimisation_projection: bool = False
) -> None:
    (bundle_dir / "train.parquet").write_text("placeholder", encoding="utf-8")
    (bundle_dir / "tuning.parquet").write_text("placeholder", encoding="utf-8")
    if include_optimisation_projection:
        (bundle_dir / "optimisation_train.parquet").write_text(
            "placeholder", encoding="utf-8"
        )
        (bundle_dir / "optimisation_tuning.parquet").write_text(
            "placeholder", encoding="utf-8"
        )
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"train_rows": 1, "tuning_rows": 1}),
        encoding="utf-8",
    )


def _assert_bundle_request(
    request: OptimisationBuildRequest,
    bundle_dir: Path,
    *,
    expected_model_backend: str,
    expected_runtime_profile: str,
) -> None:
    assert request.model_params is not None
    assert request.bundle_dir == bundle_dir
    assert request.model_backend == expected_model_backend
    assert request.train_input_path == bundle_dir / "train.parquet"
    assert request.tuning_input_path == bundle_dir / "tuning.parquet"
    assert request.n_folds == 2
    assert request.tuning_trials == 7
    assert request.train_sample_fraction == 0.02
    assert request.tuning_sample_fraction == 0.03
    assert request.model_params["model_backend"] == expected_model_backend
    assert request.model_params["runtime_profile"] == expected_runtime_profile
    assert request.model_params["stage_budget"] == "quick"
    assert request.model_params["tensorboard_logdir"] == str(bundle_dir / "tensorboard")


def _xgboost_runtime_resolution(
    *,
    requested_profile: str = "auto",
    runtime_profile: str = "local_cpu",
    device: str = "cpu",
) -> XGBoostRuntimeResolution:
    return XGBoostRuntimeResolution(
        requested_profile=requested_profile,
        runtime_profile=runtime_profile,
        device=device,
        tree_method="hist",
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        cuda_available=device == "cuda",
        cuda_device_name="NVIDIA L40S" if device == "cuda" else None,
        fallback_reason=None if device == "cuda" else "test fallback",
    )


class OptimisationMainTests(unittest.TestCase):
    def test_official_main_config_uses_official_defaults(self) -> None:
        config = OFFICIAL_OPTIMISATION_MAIN_CONFIG
        rebuilt = build_default_optimisation_main_config()

        self.assertEqual(config.model_backend, rebuilt.model_backend)
        self.assertEqual(config.runtime_profile, rebuilt.runtime_profile)
        self.assertEqual(config.n_folds, rebuilt.n_folds)
        self.assertEqual(config.max_trials, rebuilt.max_trials)
        self.assertEqual(config.stage_budget, rebuilt.stage_budget)
        self.assertEqual(config.train_sample_fraction, rebuilt.train_sample_fraction)
        self.assertEqual(config.tuning_sample_fraction, rebuilt.tuning_sample_fraction)
        self.assertEqual(config.bundle_dir, DEFAULT_FEATURE_SELECTION_DIR)

    def test_default_xgboost_config_uses_cuda_profile_when_preflight_passes(
        self,
    ) -> None:
        with mock.patch(
            "praedixa.demand_forecast.training.config.main_config.resolve_xgboost_runtime_profile",
            return_value=_xgboost_runtime_resolution(
                runtime_profile="scaleway_l40s",
                device="cuda",
            ),
        ):
            config = build_default_optimisation_main_config()

        self.assertEqual(config.model_backend, "xgboost")
        self.assertEqual(config.runtime_profile, "scaleway_l40s")
        self.assertEqual(config.n_folds, 5)
        self.assertEqual(config.max_trials, 200)
        self.assertEqual(config.stage_budget, "standard")

    def test_default_xgboost_config_keeps_standard_budget_when_h100_is_available(
        self,
    ) -> None:
        with mock.patch(
            "praedixa.demand_forecast.training.config.main_config.resolve_xgboost_runtime_profile",
            return_value=_xgboost_runtime_resolution(
                runtime_profile="nvidia_h100",
                device="cuda",
            ),
        ):
            config = build_default_optimisation_main_config()

        self.assertEqual(config.model_backend, "xgboost")
        self.assertEqual(config.runtime_profile, "nvidia_h100")
        self.assertEqual(config.n_folds, 5)
        self.assertEqual(config.max_trials, 200)
        self.assertEqual(config.stage_budget, "standard")
        self.assertEqual(config.train_sample_fraction, 1.0)
        self.assertEqual(config.tuning_sample_fraction, 1.0)

    def test_xgboost_model_params_use_two_threads_per_parallel_fold(self) -> None:
        config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
            model_backend="xgboost",
            runtime_profile="local_cpu",
        )

        params = build_optimisation_model_params(config)

        self.assertEqual(int(cast(Any, params["n_jobs"])), 2)
        self.assertEqual(int(cast(Any, params["max_parallel_fold_workers"])), 5)
        self.assertEqual(params["model_backend"], "xgboost")

    def test_tft_model_params_use_single_thread_on_local_cpu(self) -> None:
        config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
            model_backend="tft",
            runtime_profile="local_cpu",
        )

        params = build_optimisation_model_params(config)

        self.assertEqual(int(cast(Any, params["n_jobs"])), 1)
        self.assertEqual(params["model_backend"], "tft")

    def test_xgboost_model_params_use_single_fold_worker_on_cuda(self) -> None:
        config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
            model_backend="xgboost",
            runtime_profile="scaleway_l40s",
        )

        with mock.patch(
            "praedixa.demand_forecast.training.config.main_config.resolve_xgboost_runtime_profile",
            return_value=_xgboost_runtime_resolution(
                requested_profile="scaleway_l40s",
                runtime_profile="scaleway_l40s",
                device="cuda",
            ),
        ):
            params = build_optimisation_model_params(config)

        self.assertEqual(int(cast(Any, params["n_jobs"])), 1)
        self.assertEqual(int(cast(Any, params["max_parallel_fold_workers"])), 1)
        self.assertEqual(params["device"], "cuda")
        self.assertEqual(params["model_backend"], "xgboost")

    def test_default_xgboost_config_falls_back_to_cpu_when_preflight_fails(
        self,
    ) -> None:
        with mock.patch(
            "praedixa.demand_forecast.training.config.main_config.resolve_xgboost_runtime_profile",
            return_value=_xgboost_runtime_resolution(),
        ):
            config = build_default_optimisation_main_config()

        self.assertEqual(config.runtime_profile, "local_cpu")
        self.assertEqual(config.n_folds, 5)
        self.assertEqual(config.max_trials, 200)
        self.assertEqual(config.stage_budget, "standard")
        self.assertEqual(config.train_sample_fraction, 1.0)
        self.assertEqual(config.tuning_sample_fraction, 1.0)

    def test_main_uses_official_config_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir)
            patched_config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
                bundle_dir=bundle_dir,
                model_backend="xgboost",
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
                    "praedixa.demand_forecast.training.orchestration.pipeline.build_optimisation_outputs",
                    return_value={
                        "best_params": bundle_dir / "best_optuna_params.json"
                    },
                ) as mocked_build,
                mock.patch(
                    "praedixa.demand_forecast.training.entrypoints.main_runner.OFFICIAL_OPTIMISATION_MAIN_CONFIG",
                    patched_config,
                ),
            ):
                optimisation_main()

        args, kwargs = mocked_build.call_args
        self.assertEqual(kwargs, {})
        _assert_bundle_request(
            args[0],
            bundle_dir,
            expected_model_backend="xgboost",
            expected_runtime_profile="local_cpu",
        )

    def test_main_prefers_optimisation_projection_when_bundle_exposes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir, include_optimisation_projection=True)
            patched_config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
                bundle_dir=bundle_dir,
                model_backend="xgboost",
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
                    "praedixa.demand_forecast.training.orchestration.pipeline.build_optimisation_outputs",
                    return_value={
                        "best_params": bundle_dir / "best_optuna_params.json"
                    },
                ) as mocked_build,
                mock.patch(
                    "praedixa.demand_forecast.training.entrypoints.main_runner.OFFICIAL_OPTIMISATION_MAIN_CONFIG",
                    patched_config,
                ),
            ):
                optimisation_main()

        args, _ = mocked_build.call_args
        request = args[0]
        self.assertEqual(
            request.train_input_path, bundle_dir / "optimisation_train.parquet"
        )
        self.assertEqual(
            request.tuning_input_path, bundle_dir / "optimisation_tuning.parquet"
        )

    def test_main_rejects_runtime_profile_mismatch_instead_of_falling_back(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir)
            patched_config = OFFICIAL_OPTIMISATION_MAIN_CONFIG.__class__(
                bundle_dir=bundle_dir,
                model_backend="xgboost",
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
                    "praedixa.demand_forecast.training.orchestration.pipeline.build_optimisation_outputs",
                    return_value={
                        "best_params": bundle_dir / "best_optuna_params.json"
                    },
                ),
                mock.patch(
                    "praedixa.demand_forecast.training.entrypoints.main_runner.OFFICIAL_OPTIMISATION_MAIN_CONFIG",
                    patched_config,
                ),
                mock.patch(
                    "praedixa.demand_forecast.training.config.main_config.resolve_xgboost_runtime_profile",
                    side_effect=RuntimeError("requires a working XGBoost CUDA runtime"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "requires a working XGBoost CUDA runtime",
                ),
            ):
                optimisation_main()


if __name__ == "__main__":
    unittest.main()
