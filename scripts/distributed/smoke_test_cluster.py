from __future__ import annotations

import argparse
import json
import math
import socket

from distributed import as_completed

from research_praedixa.distributed.cluster import connect_client, wait_for_workers
from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.logging import configure_logging


def _smoke_task(iteration: int) -> dict[str, object]:
    total = 0.0
    for value in range(1, 200_000):
        total += math.sqrt(value)
    return {
        "iteration": iteration,
        "hostname": socket.gethostname(),
        "checksum": round(total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short distributed smoke test and show which host executed each task.")
    parser.add_argument("--config", default="config/distributed.yaml")
    parser.add_argument("--tasks", type=int, default=12)
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)
    client = connect_client(config)
    wait_for_workers(client, minimum_workers=2)
    futures = [client.submit(_smoke_task, index, pure=False) for index in range(args.tasks)]
    results = [future.result() for future in as_completed(futures)]
    summary: dict[str, int] = {}
    for result in results:
        hostname = str(result["hostname"])
        summary[hostname] = summary.get(hostname, 0) + 1
    print(json.dumps({"results": results, "by_host": summary}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
