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
class ForceGroundedLossOutput:
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


def _contact_intensity_target(batch: dict[str, Any], *, threshold: float = 0.03) -> torch.Tensor:
    source_delta = batch["source_obs"][:, -1, 1].abs()
    target_delta = batch["target_obs"][:, -1, 1].abs()
    source_ratio = (source_delta > float(threshold)).to(torch.float32).mean(dim=(1, 2))
    target_ratio = (target_delta > float(threshold)).to(torch.float32).mean(dim=(1, 2))
    return 0.5 * (source_ratio + target_ratio)


def _force_depth_proxy_loss(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    depth_target = batch["depth_mm"]
    return F.smooth_l1_loss(source["pred_normal_force_proxy"], depth_target) + F.smooth_l1_loss(
        target["pred_normal_force_proxy"],
        depth_target,
    )


def _force_contact_proxy_loss(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    contact_target = _contact_intensity_target(batch)
    loss = F.smooth_l1_loss(source["pred_contact_intensity_proxy"], contact_target) + F.smooth_l1_loss(
        target["pred_contact_intensity_proxy"],
        contact_target,
    )
    return loss, contact_target.detach().mean()


def _force_latent_consistency_loss(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    z_s = source["z_force"]
    z_t = target["z_force"]
    return 0.5 * (F.mse_loss(z_s, z_t.detach()) + F.mse_loss(z_s.detach(), z_t))


def _force_variance_regularization(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor], *, floor: float) -> torch.Tensor:
    z = torch.cat([source["z_force"], target["z_force"]], dim=0)
    if z.shape[0] <= 1:
        return z.new_zeros(())
    per_dim_std = z.std(dim=0, unbiased=False)
    return F.relu(float(floor) - per_dim_std).mean()


def compute_force_grounded_losses(
    *,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> ForceGroundedLossOutput:
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
    force_depth_loss = _force_depth_proxy_loss(source, target, batch)
    force_contact_loss, force_contact_target_mean = _force_contact_proxy_loss(source, target, batch)
    force_latent_consistency = _force_latent_consistency_loss(source, target)
    force_variance = _force_variance_regularization(
        source,
        target,
        floor=float(loss_cfg.get("force_variance_floor", 0.5)),
    )
    total = loss_cfg.get("feature_recon", 1.0) * feature_loss
    total = total + loss_cfg.get("aux_observation_recon", 0.0) * obs_loss
    total = total + loss_cfg.get("temporal_consistency", 0.0) * temporal_loss
    total = total + loss_cfg.get("rest_state_lock", 0.0) * rest_loss
    total = total + loss_cfg.get("state_supervision", 1.0) * state_total
    total = total + loss_cfg.get("state_consistency", 0.0) * state_consistency
    total = total + loss_cfg.get("geometry_consistency", 0.0) * geometry_consistency
    total = total + loss_cfg.get("force_proxy_depth", 0.0) * force_depth_loss
    total = total + loss_cfg.get("force_proxy_contact_intensity", 0.0) * force_contact_loss
    total = total + loss_cfg.get("force_latent_consistency", 0.0) * force_latent_consistency
    total = total + loss_cfg.get("force_variance", 0.0) * force_variance
    metrics: dict[str, torch.Tensor] = {
        "loss": total.detach(),
        "feature_recon": feature_loss.detach(),
        "aux_observation_recon": obs_loss.detach(),
        "state_supervision_source": state_loss_src.detach(),
        "state_supervision_target": state_loss_tgt.detach(),
        "state_consistency": state_consistency.detach(),
        "geometry_consistency": geometry_consistency.detach(),
        "force_proxy_depth": force_depth_loss.detach(),
        "force_proxy_contact_intensity": force_contact_loss.detach(),
        "force_latent_consistency": force_latent_consistency.detach(),
        "force_variance": force_variance.detach(),
        "force_contact_target_mean": force_contact_target_mean.detach(),
        "force_proxy_mean_source": source["pred_normal_force_proxy"].detach().mean(),
        "force_proxy_mean_target": target["pred_normal_force_proxy"].detach().mean(),
        **temporal_parts,
        **rest_parts,
        "state_x_source": state_src_parts["state_x"],
        "state_y_source": state_src_parts["state_y"],
        "state_depth_source": state_src_parts["state_depth"],
        "state_x_target": state_tgt_parts["state_x"],
        "state_y_target": state_tgt_parts["state_y"],
        "state_depth_target": state_tgt_parts["state_depth"],
    }
    return ForceGroundedLossOutput(total=total, metrics=metrics)

