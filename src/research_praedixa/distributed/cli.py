from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from research_praedixa.distributed.backtest import run_distributed_backtest
from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.feature_selection import run_distributed_lag_selection
from research_praedixa.distributed.hpo import run_distributed_optuna_search
from research_praedixa.distributed.inference import run_distributed_batch_inference
from research_praedixa.distributed.logging import configure_logging
from research_praedixa.distributed.prepare import prepare_distributed_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed Praedixa orchestration CLI.")
    parser.add_argument("--config", default="config/distributed.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--runtime-dir", default="data/distributed_runtime")
    prepare_parser.add_argument("--run-local-silver", action="store_true")
    prepare_parser.add_argument("--run-local-gold", action="store_true")
    prepare_parser.add_argument("--rebuild-global-dataset", action="store_true")
    prepare_parser.add_argument("--sync-to-air", action="store_true")

    hpo_parser = subparsers.add_parser("hpo")
    hpo_parser.add_argument("--runtime-dir", default="data/distributed_runtime")
    hpo_parser.add_argument("--output-dir", default="data/optimisation_distributed")
    hpo_parser.add_argument("--study-name", default="praedixa-foundation-xgboost")

    feature_parser = subparsers.add_parser("feature-select")
    feature_parser.add_argument("--input-path", default="data/data_cleaning/data_train_cleaned.parquet")
    feature_parser.add_argument("--output-dir", default="data/features_selection_lag_distributed")

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--runtime-dir", default="data/distributed_runtime")
    backtest_parser.add_argument("--best-params-path", default="data/optimisation_distributed/best_optuna_params.json")
    backtest_parser.add_argument("--output-dir", default="data/evaluation_distributed")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input-path", required=True)
    predict_parser.add_argument("--model-path", default="data/evaluation_distributed/foundation_xgboost_final_model.json")
    predict_parser.add_argument("--model-card-path", default="data/evaluation_distributed/foundation_xgboost_model_card.json")
    predict_parser.add_argument("--output-dir", default="data/predictions_distributed")
    return parser


def main() -> None:
    """Dispatch one distributed Praedixa command from the command line."""

    parser = _build_parser()
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)

    if args.command == "prepare":
        outputs = prepare_distributed_runtime(
            config,
            runtime_dir=args.runtime_dir,
            run_local_silver_pipeline=args.run_local_silver,
            run_local_gold_pipeline=args.run_local_gold,
            rebuild_global_dataset=args.rebuild_global_dataset,
            sync_to_air=args.sync_to_air,
        )
        payload = outputs.__dict__
    elif args.command == "hpo":
        payload = run_distributed_optuna_search(
            config=config,
            study_name=args.study_name,
            runtime_dir=args.runtime_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "feature-select":
        payload = run_distributed_lag_selection(
            config=config,
            input_path=args.input_path,
            output_dir=args.output_dir,
        )
    elif args.command == "backtest":
        payload = run_distributed_backtest(
            config=config,
            runtime_dir=args.runtime_dir,
            best_params_path=args.best_params_path,
            output_dir=args.output_dir,
        )
    elif args.command == "predict":
        payload = run_distributed_batch_inference(
            config=config,
            input_path=args.input_path,
            model_path=args.model_path,
            model_card_path=args.model_card_path,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f"Unsupported command `{args.command}`.")
    sys.stdout.write(json.dumps({name: str(path) for name, path in payload.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
