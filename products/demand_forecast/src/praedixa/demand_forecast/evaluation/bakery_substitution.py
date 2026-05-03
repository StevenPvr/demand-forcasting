from __future__ import annotations

from typing import Final


DEFAULT_SUBSTITUTION_RECOVERY_RATE = 0.50
SUBSTITUTION_RECOVERY_RATE_BY_FAMILY: Final[dict[str, float]] = {
    "bread": 0.75,
    "viennoiserie": 0.65,
    "patisserie": 0.45,
    "sandwich": 0.35,
    "beverage": 0.80,
    "default": DEFAULT_SUBSTITUTION_RECOVERY_RATE,
}
SUBSTITUTION_FAMILY_KEYWORDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("sandwich", ("SANDWICH", "SAND ", "FORMULE")),
    ("beverage", ("BOISSON", "CAFE", "EAU")),
    (
        "viennoiserie",
        (
            "BRIOCHE",
            "CHAUSSON",
            "CHOCOLAT",
            "CHOCO",
            "CROISSANT",
            "RAISINS",
        ),
    ),
    ("patisserie", ("COOKIE", "ECLAIR", "FINANCIER", "KOUIGN", "TARTELETTE")),
    (
        "bread",
        (
            "BAGUETTE",
            "BANETTE",
            "BANETTINE",
            "BOULE",
            "BREAD",
            "CAMPAGNE",
            "COMPLET",
            "FICELLE",
            "MOISSON",
            "PAIN",
            "SEIGLE",
        ),
    ),
)


def substitution_family(product_name: str) -> str:
    normalized_name = product_name.upper().strip()
    for family, keywords in SUBSTITUTION_FAMILY_KEYWORDS:
        if any(keyword in normalized_name for keyword in keywords):
            return family
    return "default"


def substitution_recovery_rate(product_name: str) -> float:
    family = substitution_family(product_name)
    return float(SUBSTITUTION_RECOVERY_RATE_BY_FAMILY[family])
