from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


class MoiraiBackendNotReadyError(RuntimeError):
    """Erreur levee quand le backend Moirai requis n'est pas disponible."""


@dataclass(frozen=True)
class BackendAvailability:
    backend_name: str
    is_ready: bool
    reason: str
    next_step: str


def get_moirai_backend_availability() -> BackendAvailability:
    missing = [
        package
        for package in ("uni2ts", "gluonts", "torch")
        if find_spec(package) is None
    ]
    if missing:
        return BackendAvailability(
            backend_name="moirai",
            is_ready=False,
            reason=(
                "Le backend Moirai n'est pas disponible car des dependances "
                f"runtime manquent: {', '.join(missing)}."
            ),
            next_step=(
                "Installer `uni2ts`/`gluonts` dans un runtime compatible Moirai. "
                "Attention: uni2ts 2.0.0 declare torch>=2.1,<2.5 alors que le repo "
                "principal utilise torch>=2.11."
            ),
        )
    return BackendAvailability(
        backend_name="moirai",
        is_ready=True,
        reason="Le backend Moirai zero-shot est disponible.",
        next_step="Executer l'evaluation avec model_backend='moirai'.",
    )


def raise_if_moirai_backend_required(stage: str) -> None:
    availability = get_moirai_backend_availability()
    if availability.is_ready:
        return
    raise MoiraiBackendNotReadyError(
        f"Le backend {availability.backend_name} n'est pas disponible pour l'etape "
        f"`{stage}`. Raison: {availability.reason} Prochaine action: {availability.next_step}"
    )
