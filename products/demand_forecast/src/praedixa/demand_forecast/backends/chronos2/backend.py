from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


class Chronos2BackendNotReadyError(RuntimeError):
    """Erreur levee quand le backend Chronos-2 requis n'est pas disponible."""


@dataclass(frozen=True)
class BackendAvailability:
    backend_name: str
    is_ready: bool
    reason: str
    next_step: str


def get_chronos2_backend_availability() -> BackendAvailability:
    """Expose l'etat du backend Chronos-2 sans charger le modele Hugging Face."""

    missing = [
        package
        for package in ("chronos", "torch", "transformers")
        if find_spec(package) is None
    ]
    if missing:
        return BackendAvailability(
            backend_name="chronos2",
            is_ready=False,
            reason=(
                "Le backend Chronos-2 n'est pas disponible car des dependances "
                f"runtime manquent: {', '.join(missing)}."
            ),
            next_step="Synchroniser l'environnement avec `uv sync --extra chronos2`.",
        )
    try:
        chronos2_pipeline_module = __import__(
            "chronos.chronos2.pipeline",
            fromlist=["Chronos2Pipeline"],
        )
        getattr(chronos2_pipeline_module, "Chronos2Pipeline")
    except (ImportError, AttributeError) as exc:
        return BackendAvailability(
            backend_name="chronos2",
            is_ready=False,
            reason=(
                "Le package `chronos-forecasting` est installe, mais la classe "
                f"`Chronos2Pipeline` est introuvable ({exc})."
            ),
            next_step="Verifier la version de `chronos-forecasting` dans `uv.lock`.",
        )
    return BackendAvailability(
        backend_name="chronos2",
        is_ready=True,
        reason="Le backend Chronos-2 zero-shot est disponible.",
        next_step="Executer l'orchestrateur d'evaluation avec model_backend='chronos2'.",
    )


def raise_if_chronos2_finetune_backend_required(stage: str) -> None:
    """Leve si Chronos-2 ou LoRA ne sont pas disponibles pour le fine-tuning."""

    raise_if_chronos2_backend_required(stage)
    if find_spec("peft") is None:
        raise Chronos2BackendNotReadyError(
            "Le backend chronos2_finetune requiert `peft` pour le fine-tuning LoRA. "
            "Synchroniser l'environnement apres ajout de la dependance `peft`."
        )


def raise_if_chronos2_backend_required(stage: str) -> None:
    """Leve uniquement si le backend Chronos-2 est indisponible."""

    availability = get_chronos2_backend_availability()
    if availability.is_ready:
        return
    raise Chronos2BackendNotReadyError(
        f"Le backend {availability.backend_name} n'est pas disponible pour l'etape "
        f"`{stage}`. Raison: {availability.reason} Prochaine action: {availability.next_step}"
    )
