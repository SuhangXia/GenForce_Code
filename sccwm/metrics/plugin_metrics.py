from __future__ import annotations

import torch

from .regression_metrics import compute_regression_metrics


def compute_plugin_metrics(pred: torch.Tensor, target: torch.Tensor, baseline_pred: torch.Tensor | None = None) -> dict[str, float]:
    metrics = compute_regression_metrics(pred, target)
    if baseline_pred is not None:
        baseline = compute_regression_metrics(baseline_pred, target)
        metrics["plugin_gain_mae_mean"] = baseline["mae_mean"] - metrics["mae_mean"]
        metrics["plugin_gain_mse_mean"] = baseline["mse_mean"] - metrics["mse_mean"]
    return metrics
