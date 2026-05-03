from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.util import find_spec


class TimesFMBackendNotReadyError(RuntimeError):
    """Erreur levee quand le backend TimesFM requis n'est pas disponible."""


@dataclass(frozen=True)
class BackendAvailability:
    backend_name: str
    is_ready: bool
    reason: str
    next_step: str


def get_timesfm_backend_availability() -> BackendAvailability:
    python_major, python_minor = _runtime_python_major_minor()
    if (python_major, python_minor) >= (3, 12):
        return BackendAvailability(
            backend_name="timesfm",
            is_ready=False,
            reason=(
                "Le package TimesFM officiel declare Python >=3.10,<3.12, "
                f"mais l'environnement courant est Python {python_major}.{python_minor}."
            ),
            next_step=(
                "Executer TimesFM dans un runtime dedie Python 3.11, ou attendre une "
                "release TimesFM compatible avec Python 3.12+."
            ),
        )
    missing = [
        package
        for package in ("timesfm", "numpy", "pandas", "jax")
        if find_spec(package) is None
    ]
    if missing:
        return BackendAvailability(
            backend_name="timesfm",
            is_ready=False,
            reason=(
                "Le backend TimesFM n'est pas disponible car des dependances "
                f"runtime manquent: {', '.join(missing)}."
            ),
            next_step="Installer TimesFM dans un environnement Python 3.11 avec `pip install timesfm[torch] jax`.",
        )
    return BackendAvailability(
        backend_name="timesfm",
        is_ready=True,
        reason="Le backend TimesFM zero-shot est disponible.",
        next_step="Executer l'evaluation avec model_backend='timesfm'.",
    )


def _runtime_python_major_minor() -> tuple[int, int]:
    version = platform.python_version_tuple()
    return int(version[0]), int(version[1])


def raise_if_timesfm_backend_required(stage: str) -> None:
    availability = get_timesfm_backend_availability()
    if availability.is_ready:
        return
    raise TimesFMBackendNotReadyError(
        f"Le backend {availability.backend_name} n'est pas disponible pour l'etape "
        f"`{stage}`. Raison: {availability.reason} Prochaine action: {availability.next_step}"
    )
