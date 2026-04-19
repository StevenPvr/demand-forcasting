from __future__ import annotations

import logging
import os
import socket
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def execution_identity() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "pid": os.getpid(),
    }
