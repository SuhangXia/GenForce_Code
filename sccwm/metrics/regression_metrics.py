from __future__ import annotations

import torch


def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    diff = pred - target
    mae = diff.abs()
    mse = diff.square()
    return {
        "mae_x": float(mae[:, 0].mean().item()),
        "mae_y": float(mae[:, 1].mean().item()),
        "mae_depth": float(mae[:, 2].mean().item()),
        "mae_mean": float(mae.mean().item()),
        "mse_x": float(mse[:, 0].mean().item()),
        "mse_y": float(mse[:, 1].mean().item()),
        "mse_depth": float(mse[:, 2].mean().item()),
        "mse_mean": float(mse.mean().item()),
    }
