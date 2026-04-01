from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def per_axis_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f'pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}')
    sq = (pred - target).pow(2)
    return {
        'task_x': sq[:, 0].mean(),
        'task_y': sq[:, 1].mean(),
        'task_depth': sq[:, 2].mean(),
    }


def compute_identity_loss(
    adapted_features: torch.Tensor,
    source_features: torch.Tensor,
    source_scale_mm: torch.Tensor,
    target_scale_mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_flat = source_scale_mm.reshape(-1)
    target_flat = target_scale_mm.reshape(-1)
    identity_mask = torch.isclose(source_flat, target_flat, atol=1e-6, rtol=1e-6)
    if identity_mask.any():
        loss = F.mse_loss(adapted_features[identity_mask], source_features[identity_mask])
    else:
        loss = adapted_features.new_zeros(())
    return loss, identity_mask.to(dtype=adapted_features.dtype).mean()


def compute_stea_losses(
    *,
    adapted_features: torch.Tensor,
    target_features: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    source_features: torch.Tensor,
    source_scale_mm: torch.Tensor,
    target_scale_mm: torch.Tensor,
    aux: dict[str, torch.Tensor],
    lambda_latent: float,
    lambda_task: float,
    lambda_id: float,
    lambda_mod: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latent_loss = F.mse_loss(adapted_features, target_features)
    task_loss = F.mse_loss(student_pred, teacher_pred)
    task_components = per_axis_mse(student_pred, teacher_pred)
    id_loss, identity_fraction = compute_identity_loss(
        adapted_features,
        source_features,
        source_scale_mm,
        target_scale_mm,
    )

    gamma = aux['gamma']
    beta = aux['beta']
    delta = aux['delta']
    mod_loss = gamma.abs().mean() + beta.abs().mean() + delta.pow(2).mean()

    total_loss = (
        float(lambda_latent) * latent_loss
        + float(lambda_task) * task_loss
        + float(lambda_id) * id_loss
        + float(lambda_mod) * mod_loss
    )

    metrics = {
        'loss': total_loss.detach(),
        'latent_loss': latent_loss.detach(),
        'task_loss': task_loss.detach(),
        'id_loss': id_loss.detach(),
        'mod_loss': mod_loss.detach(),
        'task_x': task_components['task_x'].detach(),
        'task_y': task_components['task_y'].detach(),
        'task_depth': task_components['task_depth'].detach(),
        'valid_ratio': aux['valid_ratio'].detach(),
        'ratio_mean': aux['ratio_mean'].detach(),
        'gamma_abs': aux['gamma_abs_mean'].detach(),
        'beta_abs': aux['beta_abs_mean'].detach(),
        'residual_norm': aux['residual_norm'].detach(),
        'identity_fraction': identity_fraction.detach(),
    }
    return total_loss, metrics


def aggregate_epoch_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {
        key: float(sum(row[key] for row in metric_rows) / max(len(metric_rows), 1))
        for key in keys
    }
