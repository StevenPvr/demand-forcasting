from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


class XGBoostBackendNotReadyError(RuntimeError):
    """Erreur levee quand le backend XGBoost requis n'est pas disponible."""


@dataclass(frozen=True)
class BackendAvailability:
    backend_name: str
    is_ready: bool
    reason: str
    next_step: str


def get_xgboost_backend_availability() -> BackendAvailability:
    """Expose l'etat du backend XGBoost sans importer la librairie lourde."""

    if find_spec("xgboost") is None:
        return BackendAvailability(
            backend_name="xgboost",
            is_ready=False,
            reason="Le backend XGBoost n'est pas disponible car la dependance `xgboost` manque.",
            next_step="Synchroniser l'environnement projet puis relancer le main d'optimisation.",
        )
    return BackendAvailability(
        backend_name="xgboost",
        is_ready=True,
        reason="Le backend XGBoost est disponible et pret pour l'optimisation.",
        next_step="Executer l'orchestrateur d'optimisation avec model_backend='xgboost'.",
    )


def _not_ready_error(stage: str, availability: BackendAvailability) -> XGBoostBackendNotReadyError:
    return XGBoostBackendNotReadyError(
        f"Le backend {availability.backend_name} n'est pas disponible pour l'etape `{stage}`. "
        f"Raison: {availability.reason} Prochaine action: {availability.next_step}"
    )


def raise_if_xgboost_backend_required(stage: str) -> None:
    """Leve uniquement si le backend XGBoost est indisponible."""

    availability = get_xgboost_backend_availability()
    if not availability.is_ready:
        raise _not_ready_error(stage, availability)
