from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from sccwm_force_grounded_v21a.losses.force_grounded_losses_v21a import (
    ForceGroundedV21ALossOutput,
    _cross_covariance_penalty,
    _scale_bucket_labels,
    _stopgrad_pairwise_mse,
    compute_force_grounded_v21a_losses,
    summarize_force_grounded_v21a_batch,
)


@dataclass
class ForceGroundedV24LossOutput:
    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _marker_names_from_batch(batch: dict[str, Any], branch: str) -> list[str | None]:
    key = f"{branch}_marker_name"
    if key in batch:
        values = batch[key]
    else:
        metadata = batch.get("metadata")
        values = metadata.get(key) if isinstance(metadata, dict) else None
    if values is None:
        batch_size = int(batch["x_norm"].shape[0]) if "x_norm" in batch and torch.is_tensor(batch["x_norm"]) else 0
        return [None] * batch_size
    if isinstance(values, (list, tuple)):
        return [None if value is None else str(value) for value in values]
    return [str(values)]


def _marker_labels(names: list[str | None], vocab: Iterable[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    label_map = {str(name): idx for idx, name in enumerate(vocab)}
    labels = []
    valid = []
    for name in names:
        idx = label_map.get(str(name)) if name is not None else None
        labels.append(-1 if idx is None else int(idx))
        valid.append(idx is not None)
    return torch.tensor(labels, dtype=torch.long, device=device), torch.tensor(valid, dtype=torch.bool, device=device)


def _ce_with_mask(logits: torch.Tensor | None, labels: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits is None or logits.numel() == 0 or not bool(valid_mask.any()):
        zero = labels.new_zeros((), dtype=torch.float32)
        return zero, zero
    masked_logits = logits[valid_mask]
    masked_labels = labels[valid_mask]
    loss = F.cross_entropy(masked_logits, masked_labels)
    acc = (masked_logits.argmax(dim=1) == masked_labels).to(torch.float32).mean()
    return loss, acc


def _cosine_alignment_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(a, b, dim=1)).mean()


def _pairwise_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    return x @ x.transpose(0, 1)


def summarize_force_grounded_v24_batch(batch: dict[str, Any], loss_cfg: dict[str, Any], *, max_examples: int = 4) -> dict[str, Any]:
    summary = summarize_force_grounded_v21a_batch(batch, loss_cfg, max_examples=max_examples)
    marker_vocab = [str(v) for v in loss_cfg.get("marker_vocab", [])]
    if marker_vocab:
        source_names = _marker_names_from_batch(batch, "source")
        target_names = _marker_names_from_batch(batch, "target")
        summary["marker_metadata"] = {
            "marker_vocab": marker_vocab,
            "source_available": int(sum(name in set(marker_vocab) for name in source_names if name is not None)),
            "target_available": int(sum(name in set(marker_vocab) for name in target_names if name is not None)),
        }
    summary["force_guidance_weight"] = float(loss_cfg.get("force_guidance_weight", 1.0))
    return summary


def _v24_factorization_losses(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = source["geometry_latent_canonical"].device
    scale_edges = [float(v) for v in loss_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])]
    source_scale_labels = _scale_bucket_labels(batch["source_scale_mm"], scale_edges).to(torch.long)
    target_scale_labels = _scale_bucket_labels(batch["target_scale_mm"], scale_edges).to(torch.long)
    scale_labels = torch.cat([source_scale_labels, target_scale_labels], dim=0)

    branch_labels = torch.cat(
        [
            torch.zeros(source["canonical_branch_adv_logits"].shape[0], device=device, dtype=torch.long),
            torch.ones(target["canonical_branch_adv_logits"].shape[0], device=device, dtype=torch.long),
        ],
        dim=0,
    )

    canonical_scale_logits = torch.cat([source["canonical_scale_adv_logits"], target["canonical_scale_adv_logits"]], dim=0)
    canonical_branch_logits = torch.cat([source["canonical_branch_adv_logits"], target["canonical_branch_adv_logits"]], dim=0)
    canonical_scale_loss = F.cross_entropy(canonical_scale_logits, scale_labels)
    canonical_branch_loss = F.cross_entropy(canonical_branch_logits, branch_labels)
    canonical_scale_acc = (canonical_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32).mean()
    canonical_branch_acc = (canonical_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32).mean()

    operator_scale_logits = torch.cat([source["operator_scale_logits"], target["operator_scale_logits"]], dim=0)
    operator_branch_logits = torch.cat([source["operator_branch_logits"], target["operator_branch_logits"]], dim=0)
    operator_scale_loss = F.cross_entropy(operator_scale_logits, scale_labels)
    operator_branch_loss = F.cross_entropy(operator_branch_logits, branch_labels)
    operator_scale_acc = (operator_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32).mean()
    operator_branch_acc = (operator_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32).mean()

    marker_vocab = [str(v) for v in loss_cfg.get("marker_vocab", [])]
    source_marker_labels, source_marker_valid = _marker_labels(_marker_names_from_batch(batch, "source"), marker_vocab, device)
    target_marker_labels, target_marker_valid = _marker_labels(_marker_names_from_batch(batch, "target"), marker_vocab, device)
    marker_labels = torch.cat([source_marker_labels, target_marker_labels], dim=0)
    marker_valid = torch.cat([source_marker_valid, target_marker_valid], dim=0)
    canonical_marker_logits = None
    operator_marker_logits = None
    if "canonical_marker_adv_logits" in source and "canonical_marker_adv_logits" in target:
        canonical_marker_logits = torch.cat([source["canonical_marker_adv_logits"], target["canonical_marker_adv_logits"]], dim=0)
    if "operator_marker_logits" in source and "operator_marker_logits" in target:
        operator_marker_logits = torch.cat([source["operator_marker_logits"], target["operator_marker_logits"]], dim=0)
    canonical_marker_loss, canonical_marker_acc = _ce_with_mask(canonical_marker_logits, marker_labels, marker_valid)
    operator_marker_loss, operator_marker_acc = _ce_with_mask(operator_marker_logits, marker_labels, marker_valid)

    canonical_consistency = _stopgrad_pairwise_mse(source["geometry_latent_canonical"], target["geometry_latent_canonical"])
    canonical_operator_independence = _cross_covariance_penalty(source["geometry_latent_canonical"], source["operator_code"])
    canonical_operator_independence = canonical_operator_independence + _cross_covariance_penalty(
        target["geometry_latent_canonical"],
        target["operator_code"],
    )

    total = (
        float(loss_cfg.get("canonical_scale_confusion", 0.0)) * canonical_scale_loss
        + float(loss_cfg.get("canonical_branch_confusion", 0.0)) * canonical_branch_loss
        + float(loss_cfg.get("canonical_marker_confusion", 0.0)) * canonical_marker_loss
        + float(loss_cfg.get("operator_scale_supervision", 0.0)) * operator_scale_loss
        + float(loss_cfg.get("operator_branch_supervision", 0.0)) * operator_branch_loss
        + float(loss_cfg.get("operator_marker_supervision", 0.0)) * operator_marker_loss
        + float(loss_cfg.get("canonical_consistency", 0.0)) * canonical_consistency
        + float(loss_cfg.get("canonical_operator_independence", 0.0)) * canonical_operator_independence
    )
    metrics = {
        "canonical_scale_confusion": canonical_scale_loss.detach(),
        "canonical_branch_confusion": canonical_branch_loss.detach(),
        "canonical_marker_confusion": canonical_marker_loss.detach(),
        "operator_scale_supervision": operator_scale_loss.detach(),
        "operator_branch_supervision": operator_branch_loss.detach(),
        "operator_marker_supervision": operator_marker_loss.detach(),
        "canonical_consistency": canonical_consistency.detach(),
        "canonical_operator_independence": canonical_operator_independence.detach(),
        "canonical_scale_adv_acc": canonical_scale_acc.detach(),
        "canonical_branch_adv_acc": canonical_branch_acc.detach(),
        "canonical_marker_adv_acc": canonical_marker_acc.detach(),
        "operator_scale_acc": operator_scale_acc.detach(),
        "operator_branch_acc": operator_branch_acc.detach(),
        "operator_marker_acc": operator_marker_acc.detach(),
        "geometry_marker_available": marker_valid.to(torch.float32).mean().detach()
        if marker_valid.numel() > 0
        else source["geometry_latent_canonical"].new_zeros(()),
    }
    return total, metrics


def _v24_force_guidance_losses(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    source_canonical_proj = source["canonical_force_projected"]
    target_canonical_proj = target["canonical_force_projected"]
    source_force_teacher = source["force_teacher_latent"].detach()
    target_force_teacher = target["force_teacher_latent"].detach()

    canonical_force_alignment = 0.5 * (
        _cosine_alignment_loss(source_canonical_proj, source_force_teacher)
        + _cosine_alignment_loss(target_canonical_proj, target_force_teacher)
    )

    teacher_pair_cosine = F.cosine_similarity(source_force_teacher, target_force_teacher, dim=1)
    pair_weights = ((teacher_pair_cosine + 1.0) * 0.5).detach()
    canonical_pair_mse = (source["geometry_latent_canonical"] - target["geometry_latent_canonical"]).pow(2).mean(dim=1)
    force_weighted_pair_consistency = (pair_weights * canonical_pair_mse).mean()

    canonical_all = torch.cat([source_canonical_proj, target_canonical_proj], dim=0)
    force_all = torch.cat([source_force_teacher, target_force_teacher], dim=0)
    canonical_sim = _pairwise_cosine_matrix(canonical_all)
    force_sim = _pairwise_cosine_matrix(force_all)
    canonical_force_similarity_distill = F.mse_loss(canonical_sim, force_sim)

    force_guidance_weight = float(loss_cfg.get("force_guidance_weight", 1.0))
    total = force_guidance_weight * (
        float(loss_cfg.get("canonical_force_alignment", 1.0)) * canonical_force_alignment
        + float(loss_cfg.get("canonical_force_similarity_distill", 0.25)) * canonical_force_similarity_distill
        + float(loss_cfg.get("force_weighted_pair_consistency", 0.5)) * force_weighted_pair_consistency
    )
    metrics = {
        "canonical_force_alignment": canonical_force_alignment.detach(),
        "canonical_force_similarity_distill": canonical_force_similarity_distill.detach(),
        "force_weighted_pair_consistency": force_weighted_pair_consistency.detach(),
        "canonical_force_teacher_pair_cosine": teacher_pair_cosine.mean().detach(),
        "canonical_force_projected_pair_cosine": F.cosine_similarity(source_canonical_proj, target_canonical_proj, dim=1).mean().detach(),
        "force_guidance_weight": source_canonical_proj.new_tensor(force_guidance_weight),
    }
    return total, metrics


def compute_force_grounded_v24_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> ForceGroundedV24LossOutput:
    base: ForceGroundedV21ALossOutput = compute_force_grounded_v21a_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
    total = base.total
    metrics = dict(base.metrics)
    source = outputs["source"]
    target = outputs["target"]

    factor_total, factor_metrics = _v24_factorization_losses(source, target, batch, loss_cfg)
    guidance_total, guidance_metrics = _v24_force_guidance_losses(source, target, loss_cfg)

    total = total + factor_total + guidance_total
    metrics["loss"] = total.detach()
    metrics.update(factor_metrics)
    metrics.update(guidance_metrics)
    return ForceGroundedV24LossOutput(total=total, metrics=metrics)
