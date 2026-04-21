from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sccwm.losses.sccwm_losses import (
    auxiliary_observation_loss,
    feature_reconstruction_loss,
    geometry_consistency_loss,
    rest_state_lock_loss,
    same_event_state_consistency_loss,
    temporal_consistency_loss,
)


@dataclass
class ForceGroundedV21ALossOutput:
    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _weighted_state_supervision_loss(
    branch: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    weight_x: float,
    weight_y: float,
    weight_depth: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_x = F.smooth_l1_loss(branch["pred_x_norm"], batch["x_norm"])
    loss_y = F.smooth_l1_loss(branch["pred_y_norm"], batch["y_norm"])
    loss_depth = F.smooth_l1_loss(branch["pred_depth_mm"], batch["depth_mm"])
    total = float(weight_x) * loss_x + float(weight_y) * loss_y + float(weight_depth) * loss_depth
    return total, {
        "state_x": loss_x.detach(),
        "state_y": loss_y.detach(),
        "state_depth": loss_depth.detach(),
    }


def resolve_anchor_indices_v21a(batch: dict[str, Any]) -> tuple[torch.Tensor, str]:
    """Resolve the anchor frame inside the local temporal window.

    Priority:
    1. Explicit dataset field `anchor_index_in_window`
    2. Window-center fallback for fixed SCCWM windows
    3. Last-valid fallback when the batch only exposes a generic validity mask

    We intentionally do *not* treat `seq_valid_mask.argmax()` as a generic rule.
    In the SCCWM paired dataset this tensor is a one-hot anchor marker today, but
    using `argmax()` on a normal valid-prefix mask would silently return the first
    valid frame, which is the wrong semantic anchor for load-progress supervision.
    """

    if "anchor_index_in_window" in batch:
        return batch["anchor_index_in_window"].to(torch.long), "explicit_window_anchor"

    if "source_obs" in batch and torch.is_tensor(batch["source_obs"]):
        seq_len = int(batch["source_obs"].shape[1])
    elif "seq_valid_mask" in batch and torch.is_tensor(batch["seq_valid_mask"]):
        seq_len = int(batch["seq_valid_mask"].shape[1])
    else:
        seq_len = 1

    if "seq_valid_mask" in batch and torch.is_tensor(batch["seq_valid_mask"]):
        seq_valid_mask = batch["seq_valid_mask"].to(torch.float32)
        is_binary = bool(torch.all((seq_valid_mask == 0.0) | (seq_valid_mask == 1.0)).item())
        has_single_hot = bool(torch.all(seq_valid_mask.sum(dim=1) == 1.0).item())
        if is_binary and has_single_hot:
            return seq_valid_mask.argmax(dim=1).to(torch.long), "seq_valid_mask_one_hot_anchor"

    if seq_len > 0 and seq_len % 2 == 1:
        batch_size = int(batch["x_norm"].shape[0]) if "x_norm" in batch and torch.is_tensor(batch["x_norm"]) else 1
        device = batch["x_norm"].device if "x_norm" in batch and torch.is_tensor(batch["x_norm"]) else None
        return torch.full((batch_size,), seq_len // 2, dtype=torch.long, device=device), "window_center_fallback"

    if "seq_valid_mask" in batch and torch.is_tensor(batch["seq_valid_mask"]):
        seq_valid_mask = batch["seq_valid_mask"].to(torch.float32)
        last_valid = seq_valid_mask.shape[1] - 1 - seq_valid_mask.flip(dims=[1]).argmax(dim=1)
        return last_valid.to(torch.long), "last_valid_fallback"

    batch_size = int(batch["x_norm"].shape[0]) if "x_norm" in batch and torch.is_tensor(batch["x_norm"]) else 1
    device = batch["x_norm"].device if "x_norm" in batch and torch.is_tensor(batch["x_norm"]) else None
    return torch.zeros(batch_size, dtype=torch.long, device=device), "zero_fallback"


def penetration_target_v21a(batch: dict[str, Any], *, proxy_scale: float) -> torch.Tensor:
    mean_scale = 0.5 * (batch["source_scale_mm"] + batch["target_scale_mm"])
    normalized = batch["depth_mm"] / mean_scale.clamp_min(1e-6)
    return (normalized * float(proxy_scale)).clamp(0.0, 1.0)


def _gather_anchor_channel_frames(obs_seq: torch.Tensor, anchor_indices: torch.Tensor, *, channel_index: int) -> torch.Tensor:
    if obs_seq.dim() != 5:
        raise ValueError(f"Expected obs_seq shape (B,T,C,H,W), got {tuple(obs_seq.shape)}")
    batch_size = int(obs_seq.shape[0])
    time_dim = int(obs_seq.shape[1])
    if anchor_indices.shape[0] != batch_size:
        raise ValueError(f"Anchor batch shape mismatch: obs batch={batch_size}, anchors={tuple(anchor_indices.shape)}")
    anchor = anchor_indices.to(device=obs_seq.device, dtype=torch.long).clamp(min=0, max=max(time_dim - 1, 0))
    batch_indices = torch.arange(batch_size, device=obs_seq.device, dtype=torch.long)
    return obs_seq[batch_indices, anchor, int(channel_index)]


def _contact_activation_ratio_from_frames(frames: torch.Tensor, *, threshold: float) -> torch.Tensor:
    return (frames.abs() > float(threshold)).to(torch.float32).mean(dim=(1, 2))


def _legacy_last_frame_contact_target_v21a(batch: dict[str, Any], *, threshold: float) -> torch.Tensor:
    source_ratio = _contact_activation_ratio_from_frames(batch["source_obs"][:, -1, 1], threshold=threshold)
    target_ratio = _contact_activation_ratio_from_frames(batch["target_obs"][:, -1, 1], threshold=threshold)
    return 0.5 * (source_ratio + target_ratio)


def contact_intensity_target_v21a(batch: dict[str, Any], *, threshold: float = 0.03) -> torch.Tensor:
    anchor_idx, _ = resolve_anchor_indices_v21a(batch)
    source_anchor = _gather_anchor_channel_frames(batch["source_obs"], anchor_idx, channel_index=1)
    target_anchor = _gather_anchor_channel_frames(batch["target_obs"], anchor_idx, channel_index=1)
    source_ratio = _contact_activation_ratio_from_frames(source_anchor, threshold=threshold)
    target_ratio = _contact_activation_ratio_from_frames(target_anchor, threshold=threshold)
    target = 0.5 * (source_ratio + target_ratio)
    if __debug__:
        last_index = int(batch["source_obs"].shape[1] - 1)
        if bool((anchor_idx == last_index).all().item()):
            legacy = _legacy_last_frame_contact_target_v21a(batch, threshold=threshold)
            if not torch.allclose(target.detach(), legacy.detach(), atol=1e-6, rtol=1e-5):
                raise AssertionError("Anchor-aware contact target diverged from legacy last-frame target on last-frame anchors.")
    return target


def phase_load_progress_target_v21a(batch: dict[str, Any]) -> torch.Tensor:
    anchor_idx, _ = resolve_anchor_indices_v21a(batch)
    phase_progress = batch["phase_progress"].to(torch.float32)
    targets: list[torch.Tensor] = []
    for batch_idx, anchor in enumerate(anchor_idx.tolist()):
        if "anchor_phase_name" in batch:
            phase_name = str(batch["anchor_phase_name"][batch_idx])
            progress = batch["anchor_phase_progress"][batch_idx].to(torch.float32)
        elif "phase_names" in batch:
            phase_names = batch["phase_names"]
            if isinstance(phase_names, (list, tuple)) and len(phase_names) > 0:
                first = phase_names[0]
                if isinstance(first, (list, tuple)) and len(first) == phase_progress.shape[0]:
                    phase_name = str(phase_names[anchor][batch_idx])
                else:
                    phase_name = str(phase_names[batch_idx][anchor])
            else:
                phase_name = "unknown"
            progress = phase_progress[batch_idx, anchor]
        else:
            phase_name = str(batch.get("anchor_phase_name", ["unknown"])[batch_idx])
            progress = phase_progress[batch_idx, anchor]
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
        targets.append(target)
    return torch.stack(targets, dim=0)


def summarize_force_grounded_v21a_batch(
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
    *,
    max_examples: int = 4,
) -> dict[str, Any]:
    anchor_idx, anchor_mode = resolve_anchor_indices_v21a(batch)
    penetration_target = penetration_target_v21a(batch, proxy_scale=float(loss_cfg.get("penetration_proxy_scale", 20.0)))
    contact_target = contact_intensity_target_v21a(batch, threshold=float(loss_cfg.get("contact_intensity_threshold", 0.03)))
    load_target = phase_load_progress_target_v21a(batch)
    examples: list[dict[str, Any]] = []
    batch_size = int(anchor_idx.shape[0])
    for batch_idx in range(min(batch_size, max_examples)):
        anchor = int(anchor_idx[batch_idx].item())
        if "anchor_phase_name" in batch:
            phase_name = str(batch["anchor_phase_name"][batch_idx])
            progress = float(batch["anchor_phase_progress"][batch_idx].item())
        elif "phase_names" in batch:
            phase_names = batch["phase_names"]
            if isinstance(phase_names, (list, tuple)) and len(phase_names) > 0:
                first = phase_names[0]
                if isinstance(first, (list, tuple)) and len(first) == batch_size:
                    phase_name = str(phase_names[anchor][batch_idx])
                else:
                    phase_name = str(phase_names[batch_idx][anchor])
            else:
                phase_name = "unknown"
            progress = float(batch["phase_progress"][batch_idx, anchor].item()) if "phase_progress" in batch else float("nan")
        else:
            phase_name = str(batch.get("anchor_phase_name", ["unknown"])[batch_idx])
            progress = float("nan")
        examples.append(
            {
                "batch_index": batch_idx,
                "episode_id": int(batch["episode_id"][batch_idx].item()) if "episode_id" in batch else None,
                "anchor_index_in_window": anchor,
                "window_length": int(batch["source_obs"].shape[1]) if "source_obs" in batch else None,
                "window_center_index": int(batch["window_center_index"][batch_idx].item()) if "window_center_index" in batch else None,
                "phase_name": phase_name,
                "phase_progress": progress,
                "load_progress_target": float(load_target[batch_idx].item()),
                "penetration_target": float(penetration_target[batch_idx].item()),
                "contact_intensity_target": float(contact_target[batch_idx].item()),
            }
        )
    return {
        "anchor_selection_mode": anchor_mode,
        "anchor_index_stats": {
            "mean": float(anchor_idx.to(torch.float32).mean().item()),
            "min": int(anchor_idx.min().item()),
            "max": int(anchor_idx.max().item()),
        },
        "penetration_proxy_target_stats": {
            "mean": float(penetration_target.mean().item()),
            "std": float(penetration_target.std(unbiased=False).item()),
            "min": float(penetration_target.min().item()),
            "max": float(penetration_target.max().item()),
        },
        "contact_intensity_target_stats": {
            "mean": float(contact_target.mean().item()),
            "std": float(contact_target.std(unbiased=False).item()),
            "min": float(contact_target.min().item()),
            "max": float(contact_target.max().item()),
        },
        "load_progress_target_stats": {
            "mean": float(load_target.mean().item()),
            "std": float(load_target.std(unbiased=False).item()),
            "min": float(load_target.min().item()),
            "max": float(load_target.max().item()),
        },
        "examples": examples,
    }


def _paired_proxy_regression_loss(source_pred: torch.Tensor, target_pred: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(source_pred, target_value) + F.smooth_l1_loss(target_pred, target_value)


def _stopgrad_pairwise_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.mse_loss(a, b.detach()) + F.mse_loss(a.detach(), b))


def _force_latent_consistency_loss(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    map_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_cons = _stopgrad_pairwise_mse(source["z_force_global"], target["z_force_global"])
    map_cons = _stopgrad_pairwise_mse(source["z_force_map"], target["z_force_map"])
    total = global_cons + float(map_weight) * map_cons
    return total, global_cons.detach(), map_cons.detach()


def _force_variance_regularization(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor], *, floor: float) -> torch.Tensor:
    z = torch.cat([source["z_force_global"], target["z_force_global"]], dim=0)
    if z.shape[0] <= 1:
        return z.new_zeros(())
    per_dim_std = z.std(dim=0, unbiased=False)
    return F.relu(float(floor) - per_dim_std).mean()


def _cross_covariance_penalty(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if a.shape[0] <= 1:
        return a.new_zeros(())
    a_center = a - a.mean(dim=0, keepdim=True)
    b_center = b - b.mean(dim=0, keepdim=True)
    a_norm = a_center / (a_center.std(dim=0, keepdim=True, unbiased=False) + float(eps))
    b_norm = b_center / (b_center.std(dim=0, keepdim=True, unbiased=False) + float(eps))
    cross_cov = a_norm.transpose(0, 1) @ b_norm / float(a.shape[0])
    return cross_cov.pow(2).mean()


def _scale_bucket_labels(scale_mm: torch.Tensor, edges_mm: list[float]) -> torch.Tensor:
    edges = scale_mm.new_tensor(edges_mm, dtype=torch.float32)
    return torch.bucketize(scale_mm.to(torch.float32), edges)


def _adversarial_shortcut_losses(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    scale_bucket_edges_mm: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_logits = torch.cat([source["scale_adv_logits"], target["scale_adv_logits"]], dim=0)
    scale_labels = torch.cat(
        [
            _scale_bucket_labels(batch["source_scale_mm"], scale_bucket_edges_mm),
            _scale_bucket_labels(batch["target_scale_mm"], scale_bucket_edges_mm),
        ],
        dim=0,
    ).to(torch.long)
    branch_logits = torch.cat([source["branch_adv_logits"], target["branch_adv_logits"]], dim=0)
    branch_labels = torch.cat(
        [
            torch.zeros(source["branch_adv_logits"].shape[0], device=branch_logits.device, dtype=torch.long),
            torch.ones(target["branch_adv_logits"].shape[0], device=branch_logits.device, dtype=torch.long),
        ],
        dim=0,
    )
    scale_loss = F.cross_entropy(scale_logits, scale_labels)
    branch_loss = F.cross_entropy(branch_logits, branch_labels)
    scale_acc = (scale_logits.argmax(dim=1) == scale_labels).to(torch.float32).mean()
    branch_acc = (branch_logits.argmax(dim=1) == branch_labels).to(torch.float32).mean()
    return scale_loss, branch_loss, scale_acc.detach(), branch_acc.detach()


def _penetration_ranking_loss(
    pair_score: torch.Tensor,
    pair_target: torch.Tensor,
    episode_id: torch.Tensor,
    *,
    margin: float,
    min_target_gap: float,
) -> torch.Tensor:
    if pair_score.shape[0] <= 1:
        return pair_score.new_zeros(())
    same_episode = episode_id[:, None] == episode_id[None, :]
    target_gap = pair_target[None, :] - pair_target[:, None]
    mask = same_episode & (target_gap > float(min_target_gap))
    if not bool(mask.any()):
        return pair_score.new_zeros(())
    score_gap = pair_score[None, :] - pair_score[:, None]
    violations = F.relu(float(margin) - score_gap[mask])
    return violations.mean()


def compute_force_grounded_v21a_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> ForceGroundedV21ALossOutput:
    source = outputs["source"]
    target = outputs["target"]
    source_to_target = outputs["source_to_target"]
    target_to_source = outputs["target_to_source"]
    source_target_features = target["patch_features"].detach()
    target_source_features = source["patch_features"].detach()
    feature_loss = feature_reconstruction_loss(source_to_target["decoded_target_features"], source_target_features)
    feature_loss = feature_loss + feature_reconstruction_loss(target_to_source["decoded_target_features"], target_source_features)
    obs_loss = auxiliary_observation_loss(source_to_target["decoded_target_observation"], batch["target_obs"])
    obs_loss = obs_loss + auxiliary_observation_loss(target_to_source["decoded_target_observation"], batch["source_obs"])
    temporal_loss, temporal_parts = temporal_consistency_loss(
        source_to_target["decoded_target_features"],
        source_target_features,
        source["pred_depth_mm_seq"],
        batch["phase_names"],
    )
    rest_loss, rest_parts = rest_state_lock_loss(
        source["pred_depth_mm_seq"],
        source_to_target["decoded_target_features"],
        source_target_features,
        source_to_target["decoded_target_observation"],
        batch["target_obs"],
        batch["phase_names"],
    )
    state_loss_src, state_src_parts = _weighted_state_supervision_loss(
        source,
        batch,
        weight_x=float(loss_cfg.get("state_weight_x", 0.5)),
        weight_y=float(loss_cfg.get("state_weight_y", 0.5)),
        weight_depth=float(loss_cfg.get("state_weight_depth", 2.0)),
    )
    state_loss_tgt, state_tgt_parts = _weighted_state_supervision_loss(
        target,
        batch,
        weight_x=float(loss_cfg.get("state_weight_x", 0.5)),
        weight_y=float(loss_cfg.get("state_weight_y", 0.5)),
        weight_depth=float(loss_cfg.get("state_weight_depth", 2.0)),
    )
    state_total = state_loss_src + state_loss_tgt
    state_consistency = same_event_state_consistency_loss(source, target)
    geometry_consistency = geometry_consistency_loss(source["geometry_latent"], target["geometry_latent"])

    anchor_idx, anchor_mode = resolve_anchor_indices_v21a(batch)
    penetration_target = penetration_target_v21a(batch, proxy_scale=float(loss_cfg.get("penetration_proxy_scale", 20.0)))
    contact_target = contact_intensity_target_v21a(batch, threshold=float(loss_cfg.get("contact_intensity_threshold", 0.03)))
    load_progress_target = phase_load_progress_target_v21a(batch)

    penetration_loss = _paired_proxy_regression_loss(source["pred_penetration_proxy"], target["pred_penetration_proxy"], penetration_target)
    contact_loss = _paired_proxy_regression_loss(source["pred_contact_intensity_proxy"], target["pred_contact_intensity_proxy"], contact_target)
    load_progress_loss = _paired_proxy_regression_loss(source["pred_load_progress_proxy"], target["pred_load_progress_proxy"], load_progress_target)

    pair_penetration_score = 0.5 * (source["pred_penetration_proxy"] + target["pred_penetration_proxy"])
    ranking_loss = _penetration_ranking_loss(
        pair_penetration_score,
        penetration_target,
        batch["episode_id"].to(torch.long),
        margin=float(loss_cfg.get("penetration_ranking_margin", 0.02)),
        min_target_gap=float(loss_cfg.get("penetration_ranking_min_target_gap", 0.02)),
    )

    force_latent_consistency, force_latent_consistency_global, force_latent_consistency_map = _force_latent_consistency_loss(
        source,
        target,
        map_weight=float(loss_cfg.get("force_map_consistency_weight", 0.5)),
    )
    force_variance = _force_variance_regularization(
        source,
        target,
        floor=float(loss_cfg.get("force_variance_floor", 0.5)),
    )
    force_orth_sensor = _cross_covariance_penalty(source["z_force_global"], source["sensor_embedding"]) + _cross_covariance_penalty(
        target["z_force_global"],
        target["sensor_embedding"],
    )
    force_orth_visibility = _cross_covariance_penalty(source["z_force_global"], source["anchor_visibility_latent"]) + _cross_covariance_penalty(
        target["z_force_global"],
        target["anchor_visibility_latent"],
    )
    scale_adv_loss, branch_adv_loss, scale_adv_acc, branch_adv_acc = _adversarial_shortcut_losses(
        source,
        target,
        batch,
        scale_bucket_edges_mm=[float(v) for v in loss_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])],
    )

    total = loss_cfg.get("feature_recon", 1.0) * feature_loss
    total = total + loss_cfg.get("aux_observation_recon", 0.0) * obs_loss
    total = total + loss_cfg.get("temporal_consistency", 0.0) * temporal_loss
    total = total + loss_cfg.get("rest_state_lock", 0.0) * rest_loss
    total = total + loss_cfg.get("state_supervision", 1.0) * state_total
    total = total + loss_cfg.get("state_consistency", 0.0) * state_consistency
    total = total + loss_cfg.get("geometry_consistency", 0.0) * geometry_consistency
    total = total + loss_cfg.get("force_penetration_proxy", 0.0) * penetration_loss
    total = total + loss_cfg.get("force_contact_intensity", 0.0) * contact_loss
    total = total + loss_cfg.get("force_load_progress", 0.0) * load_progress_loss
    total = total + loss_cfg.get("force_penetration_ranking", 0.0) * ranking_loss
    total = total + loss_cfg.get("force_latent_consistency", 0.0) * force_latent_consistency
    total = total + loss_cfg.get("force_variance", 0.0) * force_variance
    total = total + loss_cfg.get("force_orth_sensor", 0.0) * force_orth_sensor
    total = total + loss_cfg.get("force_orth_visibility", 0.0) * force_orth_visibility
    total = total + loss_cfg.get("force_scale_confusion", 0.0) * scale_adv_loss
    total = total + loss_cfg.get("force_branch_confusion", 0.0) * branch_adv_loss

    metrics: dict[str, torch.Tensor] = {
        "loss": total.detach(),
        "feature_recon": feature_loss.detach(),
        "aux_observation_recon": obs_loss.detach(),
        "state_supervision_source": state_loss_src.detach(),
        "state_supervision_target": state_loss_tgt.detach(),
        "state_consistency": state_consistency.detach(),
        "geometry_consistency": geometry_consistency.detach(),
        "force_penetration_proxy": penetration_loss.detach(),
        "force_contact_intensity": contact_loss.detach(),
        "force_load_progress": load_progress_loss.detach(),
        "force_penetration_ranking": ranking_loss.detach(),
        "force_latent_consistency": force_latent_consistency.detach(),
        "force_latent_consistency_global": force_latent_consistency_global,
        "force_latent_consistency_map": force_latent_consistency_map,
        "force_variance": force_variance.detach(),
        "force_orth_sensor": force_orth_sensor.detach(),
        "force_orth_visibility": force_orth_visibility.detach(),
        "force_scale_confusion": scale_adv_loss.detach(),
        "force_branch_confusion": branch_adv_loss.detach(),
        "force_scale_adv_acc": scale_adv_acc,
        "force_branch_adv_acc": branch_adv_acc,
        "penetration_proxy_target_mean": penetration_target.detach().mean(),
        "penetration_proxy_target_std": penetration_target.detach().std(unbiased=False),
        "contact_target_mean": contact_target.detach().mean(),
        "contact_target_std": contact_target.detach().std(unbiased=False),
        "load_progress_target_mean": load_progress_target.detach().mean(),
        "load_progress_target_std": load_progress_target.detach().std(unbiased=False),
        "anchor_index_mean": anchor_idx.to(torch.float32).mean().detach(),
        "anchor_index_min": anchor_idx.to(torch.float32).min().detach(),
        "anchor_index_max": anchor_idx.to(torch.float32).max().detach(),
        "penetration_proxy_mean_source": source["pred_penetration_proxy"].detach().mean(),
        "penetration_proxy_mean_target": target["pred_penetration_proxy"].detach().mean(),
        **temporal_parts,
        **rest_parts,
        "state_x_source": state_src_parts["state_x"],
        "state_y_source": state_src_parts["state_y"],
        "state_depth_source": state_src_parts["state_depth"],
        "state_x_target": state_tgt_parts["state_x"],
        "state_y_target": state_tgt_parts["state_y"],
        "state_depth_target": state_tgt_parts["state_depth"],
    }
    if anchor_mode == "explicit_window_anchor":
        metrics["anchor_mode_explicit"] = total.detach().new_ones(())
    else:
        metrics["anchor_mode_explicit"] = total.detach().new_zeros(())
    return ForceGroundedV21ALossOutput(total=total, metrics=metrics)
