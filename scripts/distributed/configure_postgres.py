from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

from research_praedixa.distributed.config import load_distributed_config
from research_praedixa.distributed.logging import configure_logging


def _resolve_binary(name: str) -> str:
    bundled = Path("/opt/homebrew/opt/postgresql@16/bin") / name
    if bundled.exists():
        return str(bundled)
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve PostgreSQL binary `{name}`.")
    return resolved


def _ensure_conf_line(path: Path, prefix: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replacement = f"{prefix} = {value}"
    updated = False
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = replacement
            updated = True
            break
    if not updated:
        lines.append(replacement)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_pg_hba(path: Path, allow_cidrs: tuple[str, ...]) -> None:
    required_lines = [
        "host all all 127.0.0.1/32 scram-sha-256",
        "host all all ::1/128 scram-sha-256",
    ]
    required_lines.extend(f"host all all {cidr} scram-sha-256" for cidr in allow_cidrs)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for required in required_lines:
        if required not in lines:
            lines.append(required)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and start the local PostgreSQL 16 instance for Optuna.")
    parser.add_argument("--config", default="config/distributed.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_distributed_config(args.config)
    postgres = config.postgres

    data_dir = postgres.data_dir
    log_path = postgres.log_path
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    initdb = _resolve_binary("initdb")
    pg_ctl = _resolve_binary("pg_ctl")
    psql = _resolve_binary("psql")

    if not (data_dir / "PG_VERSION").exists():
        subprocess.run(
            [initdb, "-D", str(data_dir), "--encoding=UTF8", "--locale=C"],
            check=True,
        )

    _ensure_conf_line(data_dir / "postgresql.conf", "listen_addresses", f"'{postgres.bind_host}'")
    _ensure_conf_line(data_dir / "postgresql.conf", "port", str(postgres.port))
    _ensure_pg_hba(data_dir / "pg_hba.conf", postgres.allow_cidrs)

    status = subprocess.run([pg_ctl, "-D", str(data_dir), "status"], check=False)
    if status.returncode != 0:
        subprocess.run(
            [pg_ctl, "-D", str(data_dir), "-l", str(log_path), "start"],
            check=True,
        )
    else:
        subprocess.run([pg_ctl, "-D", str(data_dir), "reload"], check=True)

    role_sql = (
        "DO $$ BEGIN "
        f"IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{postgres.user}') THEN "
        f"CREATE ROLE {postgres.user} LOGIN PASSWORD '{postgres.password}'; "
        f"ELSE ALTER ROLE {postgres.user} WITH LOGIN PASSWORD '{postgres.password}'; "
        "END IF; END $$;"
    )
    database_sql = (
        "SELECT 'create database' "
        f"WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '{postgres.database}');"
    )
    subprocess.run(
        [psql, "-h", "127.0.0.1", "-p", str(postgres.port), "-d", "postgres", "-c", role_sql],
        check=True,
    )
    db_check = subprocess.run(
        [psql, "-h", "127.0.0.1", "-p", str(postgres.port), "-d", "postgres", "-tAc", database_sql],
        check=True,
        capture_output=True,
        text=True,
    )
    if db_check.stdout.strip():
        subprocess.run(
            [psql, "-h", "127.0.0.1", "-p", str(postgres.port), "-d", "postgres", "-c", f"CREATE DATABASE {postgres.database};"],
            check=True,
        )
    subprocess.run(
        [
            psql,
            "-h",
            "127.0.0.1",
            "-p",
            str(postgres.port),
            "-d",
            postgres.database,
            "-c",
            f"GRANT ALL PRIVILEGES ON DATABASE {postgres.database} TO {postgres.user};",
        ],
        check=True,
    )
    subprocess.run(
        [
            psql,
            "-h",
            "127.0.0.1",
            "-p",
            str(postgres.port),
            "-d",
            "postgres",
            "-c",
            f"ALTER DATABASE {postgres.database} OWNER TO {postgres.user};",
        ],
        check=True,
    )
    subprocess.run(
        [
            psql,
            "-h",
            "127.0.0.1",
            "-p",
            str(postgres.port),
            "-d",
            postgres.database,
            "-c",
            f"ALTER SCHEMA public OWNER TO {postgres.user}; GRANT ALL ON SCHEMA public TO {postgres.user};",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
