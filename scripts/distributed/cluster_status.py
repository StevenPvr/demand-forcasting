from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from research_praedixa.distributed.cluster import connect_client
from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.logging import configure_logging


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the Praedixa distributed cluster status.")
    parser.add_argument("--config", default="config/distributed.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)
    pid_dir = config.resolve_pid_dir()
    payload: dict[str, object] = {
        "scheduler_pid": _read_pid(pid_dir / "scheduler.pid"),
        "pro_worker_pid": _read_pid(pid_dir / "worker_pro.pid"),
    }
    remote_pid_file = config.resolve_pid_dir() / "worker_air.pid"
    remote_command = f"test -f '{remote_pid_file}' && cat '{remote_pid_file}' || true"
    remote_pid = None
    for host in (config.air_workers.ssh_host, config.air_workers.fallback_host):
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, remote_command],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            remote_pid = int(probe.stdout.strip())
            payload["air_worker_host"] = host
            break
    payload["air_worker_pid"] = remote_pid
    try:
        client = connect_client(config, timeout_seconds=5.0)
        payload["scheduler_info"] = client.scheduler_info()
        client.close()
    except Exception as exc:
        payload["scheduler_info_error"] = str(exc)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
