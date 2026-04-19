from __future__ import annotations

import argparse

from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.logging import configure_logging
from research_praedixa.distributed.sync import sync_project_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the Praedixa repo and data to the Air.")
    parser.add_argument("--config", default="config/distributed.yaml")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)
    sync_project_tree(config, delete=args.delete)


if __name__ == "__main__":
    main()
