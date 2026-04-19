from __future__ import annotations

from pathlib import Path
import socket
import time

from distributed import Client

from research_praedixa.distributed.config import DistributedConfig


def wait_for_tcp_port(host: str, port: int, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for TCP port {host}:{port}.")


def connect_client(
    config: DistributedConfig,
    *,
    timeout_seconds: float = 30.0,
) -> Client:
    wait_for_tcp_port(config.scheduler.host, config.scheduler.port, timeout_seconds=timeout_seconds)
    return Client(config.scheduler_address, timeout=f"{int(timeout_seconds)}s")


def wait_for_workers(
    client: Client,
    *,
    minimum_workers: int,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        scheduler_info = client.scheduler_info()
        total_threads = int(scheduler_info.get("total_threads", 0))
        if total_threads >= minimum_workers:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {minimum_workers} Dask execution slots.")


def pid_path(pid_dir: Path, name: str) -> Path:
    return pid_dir / f"{name}.pid"
