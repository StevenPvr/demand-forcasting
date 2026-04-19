from __future__ import annotations

import argparse
from pathlib import Path
import socket
import subprocess
import time

from research_praedixa.distributed.cluster import pid_path, wait_for_tcp_port
from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.logging import configure_logging


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")


def _start_local_process(command: list[str], *, log_path: Path, pid_file: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    _write_pid(pid_file, process.pid)


def _pick_remote_host(preferred_host: str, fallback_host: str) -> str:
    for host in (preferred_host, fallback_host):
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "echo", "ok"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return host
    raise RuntimeError("Unable to reach the Air over SSH. Check Remote Login, firewall, and hostname resolution.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Praedixa Dask scheduler and workers on the Pro and the Air.")
    parser.add_argument("--config", default="config/distributed.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)
    logs_dir = config.resolve_logs_dir()
    pid_dir = config.resolve_pid_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)

    scheduler_pid = pid_path(pid_dir, "scheduler")
    if not scheduler_pid.exists():
        _start_local_process(
            [
                str(config.project.resolve_dask_bin()),
                "scheduler",
                "--host",
                config.scheduler.bind_host,
                "--port",
                str(config.scheduler.port),
                "--dashboard-address",
                f":{config.scheduler.dashboard_port}",
            ],
            log_path=logs_dir / "scheduler.log",
            pid_file=scheduler_pid,
        )
        wait_for_tcp_port("127.0.0.1", config.scheduler.port, timeout_seconds=30.0)

    pro_worker_pid = pid_path(pid_dir, "worker_pro")
    if not pro_worker_pid.exists():
        _start_local_process(
            [
                str(config.project.resolve_dask_bin()),
                "worker",
                config.scheduler_address,
                "--nworkers",
                str(config.pro_workers.worker_count),
                "--nthreads",
                str(config.pro_workers.dask_threads_per_worker),
                "--memory-limit",
                config.pro_workers.memory_limit,
                "--local-directory",
                str(config.pro_workers.local_directory),
                "--name",
                "praedixa-pro",
            ],
            log_path=logs_dir / "worker_pro.log",
            pid_file=pro_worker_pid,
        )

    remote_host = _pick_remote_host(config.air_workers.ssh_host, config.air_workers.fallback_host)
    air_worker_pid = pid_path(pid_dir, "worker_air_ssh")
    if not air_worker_pid.exists():
        remote_command = (
            f"cd '{config.repo_path}' && "
            f"mkdir -p '{config.air_workers.local_directory}' && "
            f"for _ in $(seq 1 30); do "
            f"nc -z '{config.scheduler.host}' {config.scheduler.port} && break; "
            f"sleep 1; "
            f"done && "
            f"exec '{config.project.resolve_dask_bin()}' worker '{config.scheduler_address}' "
            f"--nworkers {config.air_workers.worker_count} "
            f"--nthreads {config.air_workers.dask_threads_per_worker} "
            f"--memory-limit '{config.air_workers.memory_limit}' "
            f"--local-directory '{config.air_workers.local_directory}' "
            f"--name praedixa-air"
        )
        _start_local_process(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                remote_host,
                remote_command,
            ],
            log_path=logs_dir / "worker_air_ssh.log",
            pid_file=air_worker_pid,
        )
    time.sleep(5.0)


if __name__ == "__main__":
    main()
