from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sccwm_force_grounded_v21a.losses.force_grounded_losses_v21a import (
    ForceGroundedV21ALossOutput,
    _penetration_ranking_loss,
    compute_force_grounded_v21a_losses,
    phase_load_progress_target_v21a,
    summarize_force_grounded_v21a_batch,
)


@dataclass
class ForceGroundedV22LossOutput:
    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _phase_name_from_window(batch: dict[str, Any], batch_index: int, time_index: int) -> str:
    phase_names = batch.get("phase_names")
    if phase_names is None:
        return "unknown"
    if isinstance(phase_names, (list, tuple)) and len(phase_names) > 0:
        first = phase_names[0]
        if isinstance(first, (list, tuple)) and len(first) == int(batch["phase_progress"].shape[0]):
            return str(phase_names[time_index][batch_index])
        if isinstance(first, str):
            return str(phase_names[time_index])
        return str(phase_names[batch_index][time_index])
    return "unknown"


def window_load_targets_v22(batch: dict[str, Any]) -> torch.Tensor | None:
    if "phase_progress" not in batch or not torch.is_tensor(batch["phase_progress"]):
        return None
    phase_progress = batch["phase_progress"].to(torch.float32)
    batch_size, steps = phase_progress.shape
    targets = []
    for batch_idx in range(batch_size):
        row = []
        for time_idx in range(steps):
            phase_name = _phase_name_from_window(batch, batch_idx, time_idx)
            progress = phase_progress[batch_idx, time_idx]
            if phase_name == "precontact":
                target = progress.new_zeros(())
            elif phase_name == "press":
                target = progress.clamp(0.0, 1.0)
            elif phase_name == "dwell":
                target = progress.new_ones(())
            elif phase_name == "release":
                target = (1.0 - progress).clamp(0.0, 1.0)
            else:
                target = progress.clamp(0.0, 1.0)
            row.append(target)
        targets.append(torch.stack(row, dim=0))
    return torch.stack(targets, dim=0)


def _paired_window_regression_loss(source_seq: torch.Tensor, target_seq: torch.Tensor, target_seq_value: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(source_seq, target_seq_value) + F.smooth_l1_loss(target_seq, target_seq_value)


def _window_order_loss(pred_seq: torch.Tensor, target_seq: torch.Tensor, *, margin: float, target_gap: float) -> torch.Tensor:
    if pred_seq.shape[1] <= 1:
        return pred_seq.new_zeros(())
    pred_delta = pred_seq[:, 1:] - pred_seq[:, :-1]
    target_delta = target_seq[:, 1:] - target_seq[:, :-1]
    increase_mask = target_delta > float(target_gap)
    decrease_mask = target_delta < -float(target_gap)
    loss_terms: list[torch.Tensor] = []
    if bool(increase_mask.any()):
        loss_terms.append(F.relu(float(margin) - pred_delta[increase_mask]).mean())
    if bool(decrease_mask.any()):
        loss_terms.append(F.relu(float(margin) + pred_delta[decrease_mask]).mean())
    if not loss_terms:
        return pred_seq.new_zeros(())
    return sum(loss_terms)


def _window_dwell_flatness_loss(pred_seq: torch.Tensor, target_seq: torch.Tensor, *, target_gap: float) -> torch.Tensor:
    if pred_seq.shape[1] <= 1:
        return pred_seq.new_zeros(())
    pred_delta = pred_seq[:, 1:] - pred_seq[:, :-1]
    target_delta = target_seq[:, 1:] - target_seq[:, :-1]
    flat_mask = target_delta.abs() <= float(target_gap)
    if not bool(flat_mask.any()):
        return pred_seq.new_zeros(())
    return pred_delta[flat_mask].abs().mean()


def summarize_force_grounded_v22_batch(batch: dict[str, Any], loss_cfg: dict[str, Any], *, max_examples: int = 4) -> dict[str, Any]:
    summary = summarize_force_grounded_v21a_batch(batch, loss_cfg, max_examples=max_examples)
    window_targets = window_load_targets_v22(batch)
    if window_targets is not None:
        summary["trajectory_window_target_stats"] = {
            "mean": float(window_targets.mean().item()),
            "std": float(window_targets.std(unbiased=False).item()),
            "min": float(window_targets.min().item()),
            "max": float(window_targets.max().item()),
        }
    return summary


def compute_force_grounded_v22_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> ForceGroundedV22LossOutput:
    base: ForceGroundedV21ALossOutput = compute_force_grounded_v21a_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
    total = base.total
    metrics = dict(base.metrics)
    source = outputs["source"]
    target = outputs["target"]

    window_targets = window_load_targets_v22(batch)
    if window_targets is not None and "pred_load_progress_seq" in source and "pred_load_progress_seq" in target:
        trajectory_shape = _paired_window_regression_loss(source["pred_load_progress_seq"], target["pred_load_progress_seq"], window_targets)
        trajectory_order = _window_order_loss(
            source["pred_load_progress_seq"],
            window_targets,
            margin=float(loss_cfg.get("trajectory_order_margin", 0.0)),
            target_gap=float(loss_cfg.get("trajectory_order_target_gap", 0.05)),
        ) + _window_order_loss(
            target["pred_load_progress_seq"],
            window_targets,
            margin=float(loss_cfg.get("trajectory_order_margin", 0.0)),
            target_gap=float(loss_cfg.get("trajectory_order_target_gap", 0.05)),
        )
        trajectory_flat = _window_dwell_flatness_loss(
            source["pred_load_progress_seq"],
            window_targets,
            target_gap=float(loss_cfg.get("trajectory_dwell_target_gap", 0.02)),
        ) + _window_dwell_flatness_loss(
            target["pred_load_progress_seq"],
            window_targets,
            target_gap=float(loss_cfg.get("trajectory_dwell_target_gap", 0.02)),
        )
        anchor_load_target = phase_load_progress_target_v21a(batch)
        trajectory_anchor_ranking = _penetration_ranking_loss(
            0.5 * (source["pred_load_progress_proxy"] + target["pred_load_progress_proxy"]),
            anchor_load_target,
            batch["episode_id"].to(torch.long),
            margin=float(loss_cfg.get("trajectory_anchor_ranking_margin", 0.02)),
            min_target_gap=float(loss_cfg.get("trajectory_anchor_ranking_min_gap", 0.02)),
        )
    else:
        zero = total.detach().new_zeros(())
        trajectory_shape = zero
        trajectory_order = zero
        trajectory_flat = zero
        trajectory_anchor_ranking = zero

    if "contact_support_entropy" in source and "contact_support_entropy" in target:
        contact_support_entropy = source["contact_support_entropy"].mean() + target["contact_support_entropy"].mean()
        contact_support_confidence = 0.5 * (source["contact_support_confidence"].mean() + target["contact_support_confidence"].mean())
        contact_support_spread = 0.5 * (source["contact_support_spread"].mean() + target["contact_support_spread"].mean())
    else:
        zero = total.detach().new_zeros(())
        contact_support_entropy = zero
        contact_support_confidence = zero
        contact_support_spread = zero

    total = total + float(loss_cfg.get("trajectory_window_shape", 0.0)) * trajectory_shape
    total = total + float(loss_cfg.get("trajectory_window_order", 0.0)) * trajectory_order
    total = total + float(loss_cfg.get("trajectory_dwell_flat", 0.0)) * trajectory_flat
    total = total + float(loss_cfg.get("trajectory_anchor_ranking", 0.0)) * trajectory_anchor_ranking
    total = total + float(loss_cfg.get("contact_support_entropy", 0.0)) * contact_support_entropy

    metrics.update(
        {
            "loss": total.detach(),
            "trajectory_window_shape": trajectory_shape.detach(),
            "trajectory_window_order": trajectory_order.detach(),
            "trajectory_dwell_flat": trajectory_flat.detach(),
            "trajectory_anchor_ranking": trajectory_anchor_ranking.detach(),
            "contact_support_entropy": contact_support_entropy.detach(),
            "contact_support_confidence": contact_support_confidence.detach(),
            "contact_support_spread": contact_support_spread.detach(),
        }
    )
    if window_targets is not None:
        metrics["trajectory_window_target_mean"] = window_targets.detach().mean()
        metrics["trajectory_window_target_std"] = window_targets.detach().std(unbiased=False)
    return ForceGroundedV22LossOutput(total=total, metrics=metrics)
