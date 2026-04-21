from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from sccwm_force_grounded_v24.losses.force_grounded_losses_v24 import (
    _ce_with_mask,
    _cosine_alignment_loss,
    _marker_labels,
    _marker_names_from_batch,
    _pairwise_cosine_matrix,
)


@dataclass
class ForceGroundedV25LossOutput:
    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def summarize_force_grounded_v25_batch(batch: dict[str, Any], loss_cfg: dict[str, Any], *, max_examples: int = 4) -> dict[str, Any]:
    summary = summarize_force_grounded_v21a_batch(batch, loss_cfg, max_examples=max_examples)
    summary["force_guidance_weight"] = float(loss_cfg.get("force_guidance_weight", 1.0))
    summary["enable_split_canonical"] = bool(loss_cfg.get("enable_split_canonical", True))
    return summary


def _marker_terms(
    batch: dict[str, Any],
    vocab: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_marker_labels, source_marker_valid = _marker_labels(_marker_names_from_batch(batch, "source"), vocab, device)
    target_marker_labels, target_marker_valid = _marker_labels(_marker_names_from_batch(batch, "target"), vocab, device)
    marker_labels = torch.cat([source_marker_labels, target_marker_labels], dim=0)
    marker_valid = torch.cat([source_marker_valid, target_marker_valid], dim=0)
    return marker_labels, marker_valid


def _v25_factorization_losses(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = source["z_canon_load"].device
    scale_edges = [float(v) for v in loss_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])]
    source_scale_labels = _scale_bucket_labels(batch["source_scale_mm"], scale_edges).to(torch.long)
    target_scale_labels = _scale_bucket_labels(batch["target_scale_mm"], scale_edges).to(torch.long)
    scale_labels = torch.cat([source_scale_labels, target_scale_labels], dim=0)
    branch_labels = torch.cat(
        [
            torch.zeros(source["canonical_load_branch_adv_logits"].shape[0], device=device, dtype=torch.long),
            torch.ones(target["canonical_load_branch_adv_logits"].shape[0], device=device, dtype=torch.long),
        ],
        dim=0,
    )

    load_scale_logits = torch.cat([source["canonical_load_scale_adv_logits"], target["canonical_load_scale_adv_logits"]], dim=0)
    load_branch_logits = torch.cat([source["canonical_load_branch_adv_logits"], target["canonical_load_branch_adv_logits"]], dim=0)
    pose_scale_logits = torch.cat([source["canonical_pose_scale_adv_logits"], target["canonical_pose_scale_adv_logits"]], dim=0)
    pose_branch_logits = torch.cat([source["canonical_pose_branch_adv_logits"], target["canonical_pose_branch_adv_logits"]], dim=0)
    operator_scale_logits = torch.cat([source["operator_scale_logits"], target["operator_scale_logits"]], dim=0)
    operator_branch_logits = torch.cat([source["operator_branch_logits"], target["operator_branch_logits"]], dim=0)

    load_scale_loss = F.cross_entropy(load_scale_logits, scale_labels)
    load_branch_loss = F.cross_entropy(load_branch_logits, branch_labels)
    pose_scale_loss = F.cross_entropy(pose_scale_logits, scale_labels)
    pose_branch_loss = F.cross_entropy(pose_branch_logits, branch_labels)
    operator_scale_loss = F.cross_entropy(operator_scale_logits, scale_labels)
    operator_branch_loss = F.cross_entropy(operator_branch_logits, branch_labels)

    load_scale_acc = (load_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32).mean()
    load_branch_acc = (load_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32).mean()
    pose_scale_acc = (pose_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32).mean()
    pose_branch_acc = (pose_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32).mean()
    operator_scale_acc = (operator_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32).mean()
    operator_branch_acc = (operator_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32).mean()

    marker_vocab = [str(v) for v in loss_cfg.get("marker_vocab", [])]
    marker_labels, marker_valid = _marker_terms(batch, marker_vocab, device)
    load_marker_logits = None
    pose_marker_logits = None
    operator_marker_logits = None
    if "canonical_load_marker_adv_logits" in source and "canonical_load_marker_adv_logits" in target:
        load_marker_logits = torch.cat([source["canonical_load_marker_adv_logits"], target["canonical_load_marker_adv_logits"]], dim=0)
    if "canonical_pose_marker_adv_logits" in source and "canonical_pose_marker_adv_logits" in target:
        pose_marker_logits = torch.cat([source["canonical_pose_marker_adv_logits"], target["canonical_pose_marker_adv_logits"]], dim=0)
    if "operator_marker_logits" in source and "operator_marker_logits" in target:
        operator_marker_logits = torch.cat([source["operator_marker_logits"], target["operator_marker_logits"]], dim=0)
    load_marker_loss, load_marker_acc = _ce_with_mask(load_marker_logits, marker_labels, marker_valid)
    pose_marker_loss, pose_marker_acc = _ce_with_mask(pose_marker_logits, marker_labels, marker_valid)
    operator_marker_loss, operator_marker_acc = _ce_with_mask(operator_marker_logits, marker_labels, marker_valid)

    pose_consistency = _stopgrad_pairwise_mse(source["z_canon_pose"], target["z_canon_pose"])
    load_consistency = _stopgrad_pairwise_mse(source["z_canon_load"], target["z_canon_load"])

    load_operator_independence = _cross_covariance_penalty(source["z_canon_load"], source["operator_code"])
    load_operator_independence = load_operator_independence + _cross_covariance_penalty(target["z_canon_load"], target["operator_code"])
    pose_operator_independence = _cross_covariance_penalty(source["z_canon_pose"], source["operator_code"])
    pose_operator_independence = pose_operator_independence + _cross_covariance_penalty(target["z_canon_pose"], target["operator_code"])

    split_enabled = bool(source.get("split_canonical_enabled", True))
    if split_enabled:
        load_pose_independence = _cross_covariance_penalty(source["z_canon_load"], source["z_canon_pose"])
        load_pose_independence = load_pose_independence + _cross_covariance_penalty(target["z_canon_load"], target["z_canon_pose"])
    else:
        load_pose_independence = source["z_canon_load"].new_zeros(())

    total = (
        float(loss_cfg.get("load_scale_confusion", 0.0)) * load_scale_loss
        + float(loss_cfg.get("load_branch_confusion", 0.0)) * load_branch_loss
        + float(loss_cfg.get("load_marker_confusion", 0.0)) * load_marker_loss
        + float(loss_cfg.get("pose_scale_confusion", 0.0)) * pose_scale_loss
        + float(loss_cfg.get("pose_branch_confusion", 0.0)) * pose_branch_loss
        + float(loss_cfg.get("pose_marker_confusion", 0.0)) * pose_marker_loss
        + float(loss_cfg.get("operator_scale_supervision", 0.0)) * operator_scale_loss
        + float(loss_cfg.get("operator_branch_supervision", 0.0)) * operator_branch_loss
        + float(loss_cfg.get("operator_marker_supervision", 0.0)) * operator_marker_loss
        + float(loss_cfg.get("pose_consistency", 0.0)) * pose_consistency
        + float(loss_cfg.get("load_consistency", 0.0)) * load_consistency
        + float(loss_cfg.get("load_pose_independence", 0.0)) * load_pose_independence
        + float(loss_cfg.get("load_operator_independence", 0.0)) * load_operator_independence
        + float(loss_cfg.get("pose_operator_independence", 0.0)) * pose_operator_independence
    )
    metrics = {
        "load_scale_confusion": load_scale_loss.detach(),
        "load_branch_confusion": load_branch_loss.detach(),
        "load_marker_confusion": load_marker_loss.detach(),
        "pose_scale_confusion": pose_scale_loss.detach(),
        "pose_branch_confusion": pose_branch_loss.detach(),
        "pose_marker_confusion": pose_marker_loss.detach(),
        "operator_scale_supervision": operator_scale_loss.detach(),
        "operator_branch_supervision": operator_branch_loss.detach(),
        "operator_marker_supervision": operator_marker_loss.detach(),
        "pose_consistency": pose_consistency.detach(),
        "load_consistency": load_consistency.detach(),
        "load_pose_independence": load_pose_independence.detach(),
        "load_operator_independence": load_operator_independence.detach(),
        "pose_operator_independence": pose_operator_independence.detach(),
        "load_scale_adv_acc": load_scale_acc.detach(),
        "load_branch_adv_acc": load_branch_acc.detach(),
        "load_marker_adv_acc": load_marker_acc.detach(),
        "pose_scale_adv_acc": pose_scale_acc.detach(),
        "pose_branch_adv_acc": pose_branch_acc.detach(),
        "pose_marker_adv_acc": pose_marker_acc.detach(),
        "operator_scale_acc": operator_scale_acc.detach(),
        "operator_branch_acc": operator_branch_acc.detach(),
        "operator_marker_acc": operator_marker_acc.detach(),
        "marker_metadata_available": marker_valid.to(torch.float32).mean().detach()
        if marker_valid.numel() > 0
        else source["z_canon_load"].new_zeros(()),
        "split_canonical_enabled": source["z_canon_load"].new_tensor(1.0 if split_enabled else 0.0),
    }
    return total, metrics


def _v25_force_guidance_losses(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    source_load_proj = source["canonical_force_projected"]
    target_load_proj = target["canonical_force_projected"]
    source_force_teacher = source["force_teacher_latent"].detach()
    target_force_teacher = target["force_teacher_latent"].detach()

    canonical_force_alignment = 0.5 * (
        _cosine_alignment_loss(source_load_proj, source_force_teacher)
        + _cosine_alignment_loss(target_load_proj, target_force_teacher)
    )
    teacher_pair_cosine = F.cosine_similarity(source_force_teacher, target_force_teacher, dim=1)
    pair_weights = ((teacher_pair_cosine + 1.0) * 0.5).detach()
    load_pair_mse = (source["z_canon_load"] - target["z_canon_load"]).pow(2).mean(dim=1)
    force_weighted_load_pair_consistency = (pair_weights * load_pair_mse).mean()

    load_all = torch.cat([source_load_proj, target_load_proj], dim=0)
    force_all = torch.cat([source_force_teacher, target_force_teacher], dim=0)
    load_sim = _pairwise_cosine_matrix(load_all)
    force_sim = _pairwise_cosine_matrix(force_all)
    canonical_force_similarity_distill = F.mse_loss(load_sim, force_sim)

    force_guidance_weight = float(loss_cfg.get("force_guidance_weight", 1.0))
    total = force_guidance_weight * (
        float(loss_cfg.get("canonical_force_alignment", 1.0)) * canonical_force_alignment
        + float(loss_cfg.get("canonical_force_similarity_distill", 0.25)) * canonical_force_similarity_distill
        + float(loss_cfg.get("force_weighted_load_pair_consistency", 0.5)) * force_weighted_load_pair_consistency
    )
    metrics = {
        "canonical_force_alignment": canonical_force_alignment.detach(),
        "canonical_force_similarity_distill": canonical_force_similarity_distill.detach(),
        "force_weighted_load_pair_consistency": force_weighted_load_pair_consistency.detach(),
        "canonical_force_teacher_pair_cosine": teacher_pair_cosine.mean().detach(),
        "canonical_force_projected_pair_cosine": F.cosine_similarity(source_load_proj, target_load_proj, dim=1).mean().detach(),
        "force_guidance_weight": source_load_proj.new_tensor(force_guidance_weight),
    }
    return total, metrics


def compute_force_grounded_v25_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> ForceGroundedV25LossOutput:
    base: ForceGroundedV21ALossOutput = compute_force_grounded_v21a_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
    total = base.total
    metrics = dict(base.metrics)
    source = outputs["source"]
    target = outputs["target"]

    factor_total, factor_metrics = _v25_factorization_losses(source, target, batch, loss_cfg)
    guidance_total, guidance_metrics = _v25_force_guidance_losses(source, target, loss_cfg)
    total = total + factor_total + guidance_total
    metrics["loss"] = total.detach()
    metrics.update(factor_metrics)
    metrics.update(guidance_metrics)
    return ForceGroundedV25LossOutput(total=total, metrics=metrics)
