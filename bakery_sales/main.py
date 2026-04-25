from __future__ import annotations

"""Entree principale du projet bakery_sales."""

from src.launch.main import main as run_launch_main


def main() -> None:
    """Delegue au pipeline canonique SARIMAX de bout en bout."""

    run_launch_main()

if __name__ == "__main__":
    main()
