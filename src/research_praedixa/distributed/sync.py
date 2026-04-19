from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from research_praedixa.distributed.config import DistributedConfig


def _pick_remote_host(config: DistributedConfig) -> str:
    for host in (config.air_workers.ssh_host, config.air_workers.fallback_host):
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "echo", "ok"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return host
    raise RuntimeError("Unable to reach the Air over SSH for rsync.")


def resolve_remote_host(config: DistributedConfig) -> str:
    return _pick_remote_host(config)


def sync_project_tree(
    config: DistributedConfig,
    *,
    delete: bool = False,
) -> None:
    rsync_bin = shutil.which("rsync")
    if rsync_bin is None:
        raise FileNotFoundError("rsync is required to synchronize the repo to the Air.")
    remote_host = resolve_remote_host(config)
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            remote_host,
            f"mkdir -p '{config.repo_path}'",
        ],
        check=True,
    )

    command = [
        rsync_bin,
        "-az",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=5",
    ]
    if delete:
        command.append("--delete")
    for pattern in config.runtime.sync_excludes:
        command.extend(["--exclude", pattern])
    source_root = config.repo_path
    remote_path = f"{remote_host}:{config.repo_path}/"
    subprocess.run(
        [*command, f"{source_root}/", remote_path],
        check=True,
        cwd=config.repo_path,
    )
