from __future__ import annotations

from dataclasses import dataclass
import importlib.util


class TFTBackendNotReadyError(NotImplementedError):
    """Erreur levee quand le backend TFT requis n'est pas disponible."""


@dataclass(frozen=True)
class BackendAvailability:
    """Etat publie du backend TFT sans importer de dependances lourdes."""

    backend_name: str
    is_ready: bool
    reason: str
    next_step: str


_REQUIRED_PACKAGES: tuple[str, ...] = (
    "torch",
    "lightning",
    "pytorch_forecasting",
    "torchmetrics",
)


def _find_missing_packages() -> list[str]:
    return [package for package in _REQUIRED_PACKAGES if importlib.util.find_spec(package) is None]


def get_tft_backend_availability() -> BackendAvailability:
    """Expose l'etat du backend TFT de facon legere et deterministic."""

    missing_packages = _find_missing_packages()
    if missing_packages:
        return BackendAvailability(
            backend_name="TFT",
            is_ready=False,
            reason=(
                "Le backend TFT n'est pas disponible car des dependances runtime manquent: "
                f"{', '.join(missing_packages)}."
            ),
            next_step="Installer les dependances torch/lightning/pytorch-forecasting requises.",
        )
    return BackendAvailability(
        backend_name="TFT",
        is_ready=True,
        reason="Le backend TFT est disponible et pret a etre utilise dans ce repo.",
        next_step="Executer les orchestrateurs optimisation/evaluation ou les tests backend TFT.",
    )


def _build_not_ready_message(stage: str) -> str:
    availability = get_tft_backend_availability()
    return (
        f"Le backend {availability.backend_name} n'est pas disponible pour l'etape `{stage}`. "
        f"{availability.reason} Prochaine action recommandee: {availability.next_step}"
    )


def raise_if_tft_backend_required(stage: str) -> None:
    """Leve uniquement si le backend TFT est indisponible."""

    availability = get_tft_backend_availability()
    if availability.is_ready:
        return
    raise TFTBackendNotReadyError(_build_not_ready_message(stage))
