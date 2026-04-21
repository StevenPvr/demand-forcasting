from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from praedixa.demand_forecast.training.pipeline import OptimisationBuildRequest  # noqa: E402


def _write_bundle_fixture(bundle_dir: Path) -> None:
    (bundle_dir / "train.parquet").write_text("placeholder", encoding="utf-8")
    (bundle_dir / "tuning.parquet").write_text("placeholder", encoding="utf-8")
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"train_rows": 1, "tuning_rows": 1}),
        encoding="utf-8",
    )


def _assert_bundle_request(request: OptimisationBuildRequest, bundle_dir: Path) -> None:
    assert request.model_params is not None
    assert request.bundle_dir == bundle_dir
    assert request.train_input_path == bundle_dir / "train.parquet"
    assert request.tuning_input_path == bundle_dir / "tuning.parquet"
    assert request.n_folds == 2
    assert request.tuning_trials == 7
    assert request.train_sample_fraction == 0.02
    assert request.tuning_sample_fraction == 0.03
    assert request.model_params["runtime_profile"] == "scaleway_l40s"
    assert request.model_params["stage_budget"] == "quick"
    assert request.model_params["tensorboard_logdir"] == str(bundle_dir / "tensorboard")


class OptimisationMainTests(unittest.TestCase):
    def test_main_exposes_explicit_tft_not_ready_error(self) -> None:
        from praedixa.demand_forecast.backends.tft.backend import (
            TFTBackendNotReadyError,
        )

        with (
            mock.patch(
                "praedixa.demand_forecast.training.pipeline.build_optimisation_outputs",
                side_effect=TFTBackendNotReadyError("tft missing"),
            ),
            mock.patch.object(sys, "argv", ["optimisation.main"]),
            self.assertRaises(TFTBackendNotReadyError),
        ):
            runpy.run_module(
                "praedixa.demand_forecast.training.main", run_name="__main__"
            )

    def test_main_accepts_bundle_dir_and_runtime_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            _write_bundle_fixture(bundle_dir)

            with (
                mock.patch(
                    "praedixa.demand_forecast.training.pipeline.build_optimisation_outputs",
                    return_value={"best_params": bundle_dir / "best_optuna_params.json"},
                ) as mocked_build,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "optimisation.main",
                        "--bundle-dir",
                        str(bundle_dir),
                        "--runtime-profile",
                        "scaleway_l40s",
                        "--output-dir",
                        str(bundle_dir / "outputs"),
                        "--n-folds",
                        "2",
                        "--max-trials",
                        "7",
                        "--stage-budget",
                        "quick",
                        "--train-sample-fraction",
                        "0.02",
                        "--tuning-sample-fraction",
                        "0.03",
                        "--tensorboard-logdir",
                        str(bundle_dir / "tensorboard"),
                    ],
                ),
                ):
                runpy.run_module(
                    "praedixa.demand_forecast.training.main", run_name="__main__"
                )

        args, kwargs = mocked_build.call_args
        self.assertEqual(kwargs, {})
        _assert_bundle_request(args[0], bundle_dir)


if __name__ == "__main__":
    unittest.main()
