#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required on the Pro for PostgreSQL 16." >&2
  exit 1
fi

if [[ ! -x /opt/homebrew/opt/postgresql@16/bin/postgres ]]; then
  brew install postgresql@16
fi

echo "Pro bootstrap complete."
