from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sccwm.models.common import charbonnier_loss


@dataclass
class SCCWMLossOutput:
    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def feature_reconstruction_loss(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(decoded, target)


def _resize_decoded_observation_to_target(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if decoded.shape[-2:] == target.shape[-2:]:
        return decoded
    if decoded.dim() != 5 or target.dim() != 5:
        raise ValueError(f"Expected decoded/target observation tensors with shape (B,T,C,H,W), got {tuple(decoded.shape)} and {tuple(target.shape)}")
    b, t, c, _, _ = decoded.shape
    resized = F.interpolate(
        decoded.reshape(b * t, c, decoded.shape[-2], decoded.shape[-1]),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(b, t, c, target.shape[-2], target.shape[-1])


def auxiliary_observation_loss(decoded: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if decoded is None:
        return target.new_zeros(())
    decoded = _resize_decoded_observation_to_target(decoded, target)
    return charbonnier_loss(decoded, target)


def state_supervision_loss(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    pred_depth: torch.Tensor,
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    depth_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_x = _smooth_l1(pred_x, x_target)
    loss_y = _smooth_l1(pred_y, y_target)
    loss_depth = _smooth_l1(pred_depth, depth_target)
    return loss_x + loss_y + loss_depth, {
        "state_x": loss_x.detach(),
        "state_y": loss_y.detach(),
        "state_depth": loss_depth.detach(),
    }


def same_event_state_consistency_loss(source_branch: dict[str, torch.Tensor], target_branch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        F.mse_loss(source_branch["pred_x_mm"], target_branch["pred_x_mm"])
        + F.mse_loss(source_branch["pred_y_mm"], target_branch["pred_y_mm"])
        + F.mse_loss(source_branch["pred_depth_mm"], target_branch["pred_depth_mm"])
        + F.mse_loss(source_branch["geometry_latent"], target_branch["geometry_latent"])
    )


def geometry_consistency_loss(source_geometry: torch.Tensor, target_geometry: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(source_geometry, target_geometry)


def temporal_consistency_loss(
    decoded_features: torch.Tensor,
    target_features: torch.Tensor,
    pred_depth_seq: torch.Tensor,
    phase_names: list[list[str]] | list[tuple[str, ...]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if decoded_features.shape[1] <= 1:
        zero = decoded_features.new_zeros(())
        return zero, {"temporal_feature": zero.detach(), "temporal_depth_smooth": zero.detach(), "temporal_monotonic": zero.detach()}
    feat_delta_loss = F.smooth_l1_loss(decoded_features[:, 1:] - decoded_features[:, :-1], target_features[:, 1:] - target_features[:, :-1])
    depth_smooth = (pred_depth_seq[:, 1:] - pred_depth_seq[:, :-1]).abs().mean()
    monotonic_terms: list[torch.Tensor] = []
    for batch_idx, phases in enumerate(phase_names):
        for step in range(1, len(phases)):
            prev_depth = pred_depth_seq[batch_idx, step - 1]
            cur_depth = pred_depth_seq[batch_idx, step]
            phase = str(phases[step])
            if phase == "press":
                monotonic_terms.append(F.relu(prev_depth - cur_depth))
            elif phase == "release":
                monotonic_terms.append(F.relu(cur_depth - prev_depth))
    if monotonic_terms:
        monotonic = torch.stack(monotonic_terms).mean()
    else:
        monotonic = decoded_features.new_zeros(())
    total = feat_delta_loss + depth_smooth + monotonic
    return total, {
        "temporal_feature": feat_delta_loss.detach(),
        "temporal_depth_smooth": depth_smooth.detach(),
        "temporal_monotonic": monotonic.detach(),
    }


def rest_state_lock_loss(
    pred_depth_seq: torch.Tensor,
    decoded_features: torch.Tensor,
    target_features: torch.Tensor,
    decoded_obs: torch.Tensor | None,
    target_obs: torch.Tensor,
    phase_names: list[list[str]] | list[tuple[str, ...]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    indices: list[tuple[int, int]] = []
    for batch_idx, phases in enumerate(phase_names):
        for step, phase in enumerate(phases):
            if phase == "precontact":
                indices.append((batch_idx, step))
    if not indices:
        zero = pred_depth_seq.new_zeros(())
        return zero, {"rest_depth": zero.detach(), "rest_feature": zero.detach(), "rest_observation": zero.detach()}
    batch_idx = torch.tensor([pair[0] for pair in indices], device=pred_depth_seq.device, dtype=torch.long)
    time_idx = torch.tensor([pair[1] for pair in indices], device=pred_depth_seq.device, dtype=torch.long)
    depth_lock = pred_depth_seq[batch_idx, time_idx].abs().mean()
    feat_lock = F.smooth_l1_loss(decoded_features[batch_idx, time_idx], target_features[batch_idx, time_idx])
    obs_lock = target_obs.new_zeros(())
    if decoded_obs is not None:
        decoded_obs = _resize_decoded_observation_to_target(decoded_obs, target_obs)
        obs_lock = charbonnier_loss(decoded_obs[batch_idx, time_idx], target_obs[batch_idx, time_idx])
    return depth_lock + feat_lock + obs_lock, {
        "rest_depth": depth_lock.detach(),
        "rest_feature": feat_lock.detach(),
        "rest_observation": obs_lock.detach(),
    }


def occupancy_proxy(observation: torch.Tensor) -> torch.Tensor:
    if observation.dim() != 5:
        raise ValueError(f"Expected observation shape (B,T,C,H,W), got {tuple(observation.shape)}")
    gray_t = observation[:, -1, 0]
    gray_0 = observation[:, 0, 0]
    return (gray_t - gray_0).abs().mean(dim=(1, 2))


def build_negative_labels(
    occupancy_score: torch.Tensor,
    depth_mm: torch.Tensor,
    indenter_ids: torch.Tensor | None = None,
    margin_depth_mm: float = 0.25,
) -> torch.Tensor:
    b = occupancy_score.shape[0]
    dist = (occupancy_score[:, None] - occupancy_score[None, :]).abs()
    depth_sep = (depth_mm[:, None] - depth_mm[None, :]).abs() >= float(margin_depth_mm)
    if indenter_ids is not None:
        indenter_sep = indenter_ids[:, None] != indenter_ids[None, :]
    else:
        indenter_sep = torch.ones_like(depth_sep, dtype=torch.bool)
    non_self = ~torch.eye(b, device=dist.device, dtype=torch.bool)
    mask = (depth_sep | indenter_sep) & non_self
    fallback_mask = non_self
    no_valid = ~mask.any(dim=1)
    if bool(no_valid.any()):
        mask = mask.clone()
        mask[no_valid] = fallback_mask[no_valid]
    dist = dist + (~mask).to(dist.dtype) * 1e6
    neg_idx = dist.argmin(dim=1)
    return neg_idx


def build_state_embedding(branch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            branch["geometry_latent"],
            branch["pred_depth_mm"].unsqueeze(1),
            branch["pred_x_mm"].unsqueeze(1),
            branch["pred_y_mm"].unsqueeze(1),
        ],
        dim=1,
    )


def compute_counterfactual_ranking_loss(
    source_state_embedding: torch.Tensor,
    positive_state_embedding: torch.Tensor,
    negative_state_embedding: torch.Tensor,
    margin: float = 0.25,
) -> torch.Tensor:
    if source_state_embedding.dim() != 2 or positive_state_embedding.dim() != 2 or negative_state_embedding.dim() != 2:
        raise ValueError(
            "Expected source/positive/negative state embeddings to have shape (B,D), "
            f"got {tuple(source_state_embedding.shape)}, {tuple(positive_state_embedding.shape)}, {tuple(negative_state_embedding.shape)}"
        )
    positive_score = F.cosine_similarity(source_state_embedding, positive_state_embedding, dim=-1)
    negative_score = F.cosine_similarity(source_state_embedding, negative_state_embedding, dim=-1)
    target = torch.ones_like(positive_score)
    return F.margin_ranking_loss(positive_score, negative_score, target, margin=float(margin))


def plugin_distillation_loss(
    plugin_pred: torch.Tensor | None,
    plugin_target_pred: torch.Tensor | None,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if plugin_pred is None:
        zero = ground_truth.new_zeros(())
        return zero, {"plugin_distill": zero.detach(), "plugin_gt": zero.detach()}
    distill = ground_truth.new_zeros(())
    if plugin_target_pred is not None:
        distill = F.smooth_l1_loss(plugin_pred, plugin_target_pred)
    gt = F.smooth_l1_loss(plugin_pred, ground_truth)
    return distill + gt, {"plugin_distill": distill.detach(), "plugin_gt": gt.detach()}


def _stage_enabled(stage: str, at_least: str) -> bool:
    order = {"stage1": 1, "stage2": 2, "stage3": 3, "stage4": 4}
    return order[stage] >= order[at_least]


def compute_sccwm_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    stage: str,
    loss_cfg: dict[str, Any],
    plugin_pred: torch.Tensor | None = None,
    plugin_target_pred: torch.Tensor | None = None,
    indenter_ids: torch.Tensor | None = None,
) -> SCCWMLossOutput:
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

    total = loss_cfg.get("feature_recon", 1.0) * feature_loss
    total = total + loss_cfg.get("aux_observation_recon", 0.0) * obs_loss
    total = total + loss_cfg.get("temporal_consistency", 0.0) * temporal_loss
    total = total + loss_cfg.get("rest_state_lock", 0.0) * rest_loss

    metrics: dict[str, torch.Tensor] = {
        "loss": total.detach(),
        "feature_recon": feature_loss.detach(),
        "aux_observation_recon": obs_loss.detach(),
        **temporal_parts,
        **rest_parts,
    }

    if _stage_enabled(stage, "stage2"):
        state_loss_src, state_src_parts = state_supervision_loss(
            source["pred_x_norm"], source["pred_y_norm"], source["pred_depth_mm"], batch["x_norm"], batch["y_norm"], batch["depth_mm"]
        )
        state_loss_tgt, state_tgt_parts = state_supervision_loss(
            target["pred_x_norm"], target["pred_y_norm"], target["pred_depth_mm"], batch["x_norm"], batch["y_norm"], batch["depth_mm"]
        )
        state_consistency = same_event_state_consistency_loss(source, target)
        geometry_consistency = geometry_consistency_loss(source["geometry_latent"], target["geometry_latent"])
        state_total = state_loss_src + state_loss_tgt
        total = total + loss_cfg.get("state_supervision", 1.0) * state_total
        total = total + loss_cfg.get("state_consistency", 0.0) * state_consistency
        total = total + loss_cfg.get("geometry_consistency", 0.0) * geometry_consistency
        metrics.update(
            {
                "state_supervision_source": state_loss_src.detach(),
                "state_supervision_target": state_loss_tgt.detach(),
                "state_consistency": state_consistency.detach(),
                "geometry_consistency": geometry_consistency.detach(),
                "state_x_source": state_src_parts["state_x"],
                "state_y_source": state_src_parts["state_y"],
                "state_depth_source": state_src_parts["state_depth"],
                "state_x_target": state_tgt_parts["state_x"],
                "state_y_target": state_tgt_parts["state_y"],
                "state_depth_target": state_tgt_parts["state_depth"],
            }
        )

    if _stage_enabled(stage, "stage3"):
        occ = occupancy_proxy(batch["source_obs"])
        source_state = build_state_embedding(source)
        positive_state = build_state_embedding(target)
        neg_idx = build_negative_labels(
            occ,
            batch["depth_mm"],
            indenter_ids=indenter_ids,
            margin_depth_mm=float(loss_cfg.get("counterfactual_margin", 0.25)),
        )
        negative_state = positive_state[neg_idx]
        counterfactual = compute_counterfactual_ranking_loss(
            source_state,
            positive_state,
            negative_state,
            margin=float(loss_cfg.get("counterfactual_margin", 0.25)),
        )
        positive_score = F.cosine_similarity(source_state, positive_state, dim=-1).mean()
        negative_score = F.cosine_similarity(source_state, negative_state, dim=-1).mean()
        total = total + loss_cfg.get("counterfactual_ranking", 0.0) * counterfactual
        metrics["counterfactual_ranking"] = counterfactual.detach()
        metrics["counterfactual_positive_score"] = positive_score.detach()
        metrics["counterfactual_negative_score"] = negative_score.detach()

    if _stage_enabled(stage, "stage4"):
        ground_truth = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
        plugin_loss, plugin_parts = plugin_distillation_loss(plugin_pred, plugin_target_pred, ground_truth)
        total = total + loss_cfg.get("plugin_distillation", 0.0) * plugin_loss
        metrics.update(plugin_parts)

    metrics["loss"] = total.detach()
    return SCCWMLossOutput(total=total, metrics=metrics)
