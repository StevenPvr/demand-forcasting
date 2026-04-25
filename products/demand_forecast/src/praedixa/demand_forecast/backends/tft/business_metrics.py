from __future__ import annotations

from typing import Any, cast

import torch

from pytorch_forecasting.metrics.base_metrics import Metric

TORCH = cast(Any, torch)


class WAPEMetric(Metric):
    """Weighted absolute percentage error on point forecasts."""

    full_state_update = False
    higher_is_better = False
    is_differentiable = False
    absolute_error_sum: torch.Tensor
    absolute_target_sum: torch.Tensor

    def __init__(self, name: str = "wape") -> None:
        metric_super = cast(Any, super())
        metric_super.__init__(name=name)
        metric_self = cast(Any, self)
        metric_self.add_state(
            "absolute_error_sum",
            default=TORCH.tensor(0.0, dtype=TORCH.float32),
            dist_reduce_fx="sum",
        )
        metric_self.add_state(
            "absolute_target_sum",
            default=TORCH.tensor(0.0, dtype=TORCH.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        y_pred: torch.Tensor,
        y_actual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        target: torch.Tensor
        weights: torch.Tensor | None
        if isinstance(y_actual, tuple):
            target, weights = y_actual
        else:
            target = y_actual
            weights = None
        prediction = self.to_prediction(y_pred).detach().float()
        resolved_target = target.detach().float()
        absolute_error = (prediction - resolved_target).abs()
        absolute_target = resolved_target.abs()
        if weights is not None:
            resolved_weights = weights.detach().float()
            while resolved_weights.ndim < absolute_error.ndim:
                resolved_weights = resolved_weights.unsqueeze(-1)
            absolute_error = absolute_error * resolved_weights
            absolute_target = absolute_target * resolved_weights
        self.absolute_error_sum += absolute_error.sum()
        self.absolute_target_sum += absolute_target.sum()

    def compute(self) -> torch.Tensor:
        denominator = TORCH.clamp(self.absolute_target_sum, min=1e-8)
        return self.absolute_error_sum / denominator
