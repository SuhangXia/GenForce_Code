from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from sccwm.models.common import ConvGNAct, MLP, ResidualConvBlock, repeat_time, spatial_softargmax2d
from sccwm_force_grounded_v21a.models import EMBEDDING_VIEW_TO_KEY as EMBEDDING_VIEW_TO_KEY_V21A
from sccwm_force_grounded_v21a.models import SCCWMForceGroundedV21A
from sccwm_force_grounded_v21a.models import select_embedding_view_v21a


EMBEDDING_VIEW_TO_KEY = dict(EMBEDDING_VIEW_TO_KEY_V21A)


def select_embedding_view_v22(encoding: dict[str, torch.Tensor], embedding_view: str) -> torch.Tensor:
    return select_embedding_view_v21a(encoding, embedding_view)


def _support_spread(prob: torch.Tensor, x_center: torch.Tensor, y_center: torch.Tensor) -> torch.Tensor:
    if prob.dim() != 4 or prob.shape[1] != 1:
        raise ValueError(f"Expected support probability map with shape (B,1,H,W), got {tuple(prob.shape)}")
    _, _, h, w = prob.shape
    grid_y = torch.linspace(-1.0, 1.0, h, device=prob.device, dtype=prob.dtype)
    grid_x = torch.linspace(-1.0, 1.0, w, device=prob.device, dtype=prob.dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    dx2 = (xx.unsqueeze(0) - x_center.view(-1, 1, 1)).pow(2)
    dy2 = (yy.unsqueeze(0) - y_center.view(-1, 1, 1)).pow(2)
    return (prob[:, 0] * (dx2 + dy2)).sum(dim=(1, 2))


class SCCWMForceGroundedV22Base(SCCWMForceGroundedV21A):
    """Conservative v22 ablation base on top of the cleaned v21a baseline.

    This remains a world model. It does not replace the backbone/projector/decoder.
    The only additions are:
    - optional sequence-level trajectory/load-order heads and losses
    - optional canonical spatial contact-support head for x/y
    """

    variant_name = "v22_base"

    def __init__(
        self,
        *,
        enable_trajectory_constraints: bool = False,
        enable_spatial_contact_head: bool = False,
        temporal_load_hidden_dim: int = 64,
        contact_support_hidden_dim: int = 32,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_trajectory_constraints = bool(enable_trajectory_constraints)
        self.enable_spatial_contact_head = bool(enable_spatial_contact_head)
        if self.enable_trajectory_constraints:
            temporal_in_dim = self.world_hidden_dim + self.geometry_dim + self.visibility_dim
            self.temporal_load_head = MLP([temporal_in_dim, int(temporal_load_hidden_dim), 1])
        else:
            self.temporal_load_head = None
        if self.enable_spatial_contact_head:
            hidden_dim = int(contact_support_hidden_dim)
            self.contact_support_head = nn.Sequential(
                ConvGNAct(self.z_force_dim, hidden_dim, kernel_size=3, stride=1),
                ResidualConvBlock(hidden_dim),
                nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0),
            )
        else:
            self.contact_support_head = None

    def _apply_spatial_contact_head(
        self,
        encoding: dict[str, torch.Tensor],
        scale_mm: torch.Tensor,
    ) -> None:
        if self.contact_support_head is None:
            return
        logits = self.contact_support_head(encoding["z_force_map"])
        lattice_x_norm, lattice_y_norm, prob = spatial_softargmax2d(logits)
        pred_x_mm, pred_y_mm = self.projector.lattice_normalized_to_mm(lattice_x_norm, lattice_y_norm)
        pred_x_norm, pred_y_norm = self._metric_to_sensor_norm(pred_x_mm, pred_y_mm, scale_mm)
        entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=(1, 2, 3))
        confidence = prob.flatten(1).amax(dim=1)
        spread = _support_spread(prob, lattice_x_norm, lattice_y_norm)
        encoding.update(
            {
                "contact_support_logits": logits,
                "contact_support_heatmap": prob,
                "contact_support_confidence": confidence,
                "contact_support_entropy": entropy,
                "contact_support_spread": spread,
                "pred_x_mm": pred_x_mm,
                "pred_y_mm": pred_y_mm,
                "pred_x_norm": pred_x_norm,
                "pred_y_norm": pred_y_norm,
            }
        )

    def _apply_trajectory_head(self, encoding: dict[str, torch.Tensor]) -> None:
        if self.temporal_load_head is None:
            return
        world_latent_seq = encoding["world_latent_seq"]
        b, t = world_latent_seq.shape[:2]
        pooled_seq = world_latent_seq.mean(dim=(3, 4))
        temporal_in = torch.cat([pooled_seq, repeat_time(encoding["geometry_latent"], t), encoding["visibility_latent"]], dim=-1)
        pred_seq = torch.sigmoid(self.temporal_load_head(temporal_in.reshape(b * t, -1))).reshape(b, t)
        encoding["pred_load_progress_seq"] = pred_seq

    def _refresh_state_embeddings(self, encoding: dict[str, torch.Tensor]) -> None:
        state_embedding_full = torch.cat(
            [
                encoding["geometry_latent"],
                encoding["z_force_global"],
                encoding["pred_depth_mm"].unsqueeze(1),
                encoding["pred_x_mm"].unsqueeze(1),
                encoding["pred_y_mm"].unsqueeze(1),
            ],
            dim=1,
        )
        encoding["state_embedding"] = state_embedding_full
        encoding["state_embedding_full"] = state_embedding_full

    def encode_sequence(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor,
        scale_mm: torch.Tensor | float,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoding = super().encode_sequence(
            obs_seq,
            coord_map,
            scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=valid_mask,
        )
        scale = encoding["scale_mm"]
        self._apply_trajectory_head(encoding)
        self._apply_spatial_contact_head(encoding, scale)
        self._refresh_state_embeddings(encoding)
        return encoding


class SCCWMForceGroundedV22T(SCCWMForceGroundedV22Base):
    variant_name = "v22t"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(enable_trajectory_constraints=True, enable_spatial_contact_head=False, **kwargs)


class SCCWMForceGroundedV22S(SCCWMForceGroundedV22Base):
    variant_name = "v22s"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(enable_trajectory_constraints=False, enable_spatial_contact_head=True, **kwargs)


class SCCWMForceGroundedV22TS(SCCWMForceGroundedV22Base):
    variant_name = "v22ts"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(enable_trajectory_constraints=True, enable_spatial_contact_head=True, **kwargs)


def build_force_grounded_v22_model(model_cfg: dict[str, Any]) -> SCCWMForceGroundedV22Base:
    variant = str(model_cfg.get("variant", "v22ts")).strip().lower()
    common_kwargs = dict(
        input_channels=int(model_cfg.get("input_channels", 3)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        sensor_dim=int(model_cfg.get("sensor_dim", 64)),
        world_hidden_dim=int(model_cfg.get("world_hidden_dim", 128)),
        geometry_dim=int(model_cfg.get("geometry_dim", 64)),
        visibility_dim=int(model_cfg.get("visibility_dim", 32)),
        lattice_size=int(model_cfg.get("lattice_size", 32)),
        reconstruct_observation=bool(model_cfg.get("reconstruct_observation", False)),
        enable_contact_mask=bool(model_cfg.get("enable_contact_mask", False)),
        z_force_dim=int(model_cfg.get("z_force_dim", 6)),
        adv_grl_lambda=float(model_cfg.get("adv_grl_lambda", 0.25)),
        scale_bucket_edges_mm=[float(v) for v in model_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])],
        temporal_load_hidden_dim=int(model_cfg.get("temporal_load_hidden_dim", 64)),
        contact_support_hidden_dim=int(model_cfg.get("contact_support_hidden_dim", 32)),
    )
    if variant == "v22t":
        return SCCWMForceGroundedV22T(**common_kwargs)
    if variant == "v22s":
        return SCCWMForceGroundedV22S(**common_kwargs)
    if variant == "v22ts":
        return SCCWMForceGroundedV22TS(**common_kwargs)
    raise ValueError(f"Unsupported v22 variant: {variant}")
