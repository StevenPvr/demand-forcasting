from __future__ import annotations


def build_tft_dataloader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> dict[str, object]:
    resolved_num_workers = max(0, int(num_workers))
    return {
        "num_workers": resolved_num_workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) if resolved_num_workers > 0 else False,
    }

