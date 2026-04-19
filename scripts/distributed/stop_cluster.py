from __future__ import annotations

import argparse
from pathlib import Path
import signal
import subprocess

from research_praedixa.distributed.cluster import pid_path
from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.logging import configure_logging


def _stop_local_pid(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        Path(f"/proc/{pid}")
    except Exception:
        pass
    try:
        import os
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    pid_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop the Praedixa Dask scheduler and workers.")
    parser.add_argument("--config", default="config/distributed.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)
    pid_dir = config.resolve_pid_dir()
    _stop_local_pid(pid_path(pid_dir, "worker_air_ssh"))
    _stop_local_pid(pid_path(pid_dir, "worker_pro"))
    _stop_local_pid(pid_path(pid_dir, "scheduler"))
    remote_command = "pkill -f 'dask worker tcp://' || true"
    for host in (config.air_workers.ssh_host, config.air_workers.fallback_host):
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, remote_command], check=False)


if __name__ == "__main__":
    main()
