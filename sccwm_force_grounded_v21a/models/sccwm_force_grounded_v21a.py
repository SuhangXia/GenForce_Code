from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sccwm.models.common import ConvGNAct, MLP, ResidualConvBlock, sequence_last_valid
from sccwm.models.sccwm import SCCWM

from .force_conditioned_decoder_v21a import ForceConditionedCounterfactualDecoderV21A


EMBEDDING_VIEW_TO_KEY = {
    "full_state": "state_embedding_full",
    "latent_only": "state_embedding_latent_only",
    "force_only": "state_embedding_force_only",
    "force_map_pooled": "state_embedding_force_map_pooled",
    "geometry_only": "state_embedding_geometry_only",
}


def select_embedding_view_v21a(encoding: dict[str, torch.Tensor], embedding_view: str) -> torch.Tensor:
    key = EMBEDDING_VIEW_TO_KEY.get(str(embedding_view))
    if key is None:
        raise ValueError(f"Unsupported embedding view: {embedding_view}")
    if key not in encoding:
        raise KeyError(f"Encoding does not contain embedding view {embedding_view!r} ({key!r})")
    return encoding[key]


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReverseFn.apply(x, float(lambd))


class SCCWMForceGroundedV21A(SCCWM):
    """Force-grounded SCCWM v2.1a.

    This is still a world model. It is not explicitly force supervised.
    It adds a spatial force map, a global force summary, decode-side force
    conditioning, and anti-shortcut regularization heads.

    v2.1a intentionally keeps the v2.1 architecture and only serves as a
    correctness/stability fork for anchor semantics and training preflight.
    """

    def __init__(
        self,
        *,
        z_force_dim: int = 6,
        adv_grl_lambda: float = 0.25,
        scale_bucket_edges_mm: list[float] | tuple[float, ...] = (14.0, 18.0, 22.0),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.z_force_dim = int(z_force_dim)
        self.adv_grl_lambda = float(adv_grl_lambda)
        self.scale_bucket_edges_mm = tuple(float(v) for v in scale_bucket_edges_mm)
        force_map_in_dim = self.world_hidden_dim + self.geometry_dim + self.visibility_dim + self.sensor_dim
        self.force_map_encoder = nn.Sequential(
            ConvGNAct(force_map_in_dim, self.world_hidden_dim, kernel_size=3, stride=1),
            ResidualConvBlock(self.world_hidden_dim),
            nn.Conv2d(self.world_hidden_dim, self.z_force_dim, kernel_size=1, padding=0),
            nn.GroupNorm(1, self.z_force_dim),
        )
        self.force_global_norm = nn.LayerNorm(self.z_force_dim)
        self.penetration_proxy_head = MLP([self.z_force_dim, 64, 1])
        self.contact_intensity_proxy_head = MLP([self.z_force_dim, 64, 1])
        self.load_progress_proxy_head = MLP([self.z_force_dim, 64, 1])
        self.scale_adv_classifier = MLP([self.z_force_dim, 64, len(self.scale_bucket_edges_mm) + 1])
        self.branch_adv_classifier = MLP([self.z_force_dim, 32, 2])
        self.decoder = ForceConditionedCounterfactualDecoderV21A(
            world_hidden_dim=self.world_hidden_dim,
            feature_dim=self.feature_dim,
            sensor_dim=self.sensor_dim,
            geometry_dim=self.geometry_dim,
            visibility_dim=self.visibility_dim,
            z_force_dim=self.z_force_dim,
            reconstruct_observation=self.reconstruct_observation,
        )

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
        base = super().encode_sequence(
            obs_seq,
            coord_map,
            scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=valid_mask,
        )
        anchor_visibility = sequence_last_valid(base["visibility_latent"], valid_mask)
        anchor_hidden = base["last_hidden"]
        h, w = anchor_hidden.shape[-2:]
        geometry_map = base["geometry_latent"].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        visibility_map = anchor_visibility.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        sensor_map = base["sensor_embedding"].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        force_map_input = torch.cat([anchor_hidden, geometry_map, visibility_map, sensor_map], dim=1)
        z_force_map = self.force_map_encoder(force_map_input)
        z_force_map_pooled = F.adaptive_avg_pool2d(z_force_map, output_size=1).flatten(1)
        z_force_global = self.force_global_norm(z_force_map_pooled)
        pred_penetration_proxy = torch.sigmoid(self.penetration_proxy_head(z_force_global)).squeeze(1)
        pred_contact_intensity_proxy = torch.sigmoid(self.contact_intensity_proxy_head(z_force_global)).squeeze(1)
        pred_load_progress_proxy = torch.sigmoid(self.load_progress_proxy_head(z_force_global)).squeeze(1)
        z_force_adv = grad_reverse(z_force_global, self.adv_grl_lambda)
        scale_adv_logits = self.scale_adv_classifier(z_force_adv)
        branch_adv_logits = self.branch_adv_classifier(z_force_adv)
        appearance_pooled = sequence_last_valid(base["patch_features"].mean(dim=(3, 4)), valid_mask)
        state_embedding_full = torch.cat(
            [
                base["geometry_latent"],
                z_force_global,
                base["pred_depth_mm"].unsqueeze(1),
                base["pred_x_mm"].unsqueeze(1),
                base["pred_y_mm"].unsqueeze(1),
            ],
            dim=1,
        )
        state_embedding_latent_only = torch.cat([base["geometry_latent"], z_force_global], dim=1)
        base.update(
            {
                "z_force": z_force_global,
                "z_force_global": z_force_global,
                "z_force_map": z_force_map,
                "z_force_map_pooled": z_force_map_pooled,
                "pred_penetration_proxy": pred_penetration_proxy,
                "pred_contact_intensity_proxy": pred_contact_intensity_proxy,
                "pred_load_progress_proxy": pred_load_progress_proxy,
                "scale_adv_logits": scale_adv_logits,
                "branch_adv_logits": branch_adv_logits,
                "state_embedding": state_embedding_full,
                "state_embedding_full": state_embedding_full,
                "state_embedding_latent_only": state_embedding_latent_only,
                "state_embedding_force_only": z_force_global,
                "state_embedding_force_map_pooled": z_force_map_pooled,
                "state_embedding_geometry_only": base["geometry_latent"],
                "anchor_visibility_latent": anchor_visibility,
                "appearance_pooled": appearance_pooled,
            }
        )
        return base

    def decode_to_target(
        self,
        source_encoding: dict[str, torch.Tensor],
        target_coord_map: torch.Tensor,
        target_scale_mm: torch.Tensor | float,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        b = source_encoding["world_latent_seq"].shape[0]
        device = source_encoding["world_latent_seq"].device
        dtype = source_encoding["world_latent_seq"].dtype
        target_scale = self._prepare_scale(target_scale_mm, b, device, dtype)
        target_coord_map_t = target_coord_map.to(device=device, dtype=dtype)
        target_sensor_embedding = self.sensor_encoder(target_coord_map_t, target_scale)
        gathered_world_base_features, gathered_world_base_occ = self.projector.gather_from_world_lattice(
            source_encoding["world_features"],
            target_coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        world_feature_seq = self.world_to_feature(
            source_encoding["world_latent_seq"].reshape(
                -1,
                self.world_hidden_dim,
                source_encoding["world_latent_seq"].shape[-2],
                source_encoding["world_latent_seq"].shape[-1],
            )
        ).reshape(
            b,
            source_encoding["world_latent_seq"].shape[1],
            self.feature_dim,
            source_encoding["world_latent_seq"].shape[-2],
            source_encoding["world_latent_seq"].shape[-1],
        )
        gathered_world_features, gathered_world_occ = self.projector.gather_from_world_lattice(
            world_feature_seq,
            target_coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        force_map_world = source_encoding["z_force_map"].unsqueeze(1).expand(-1, source_encoding["world_latent_seq"].shape[1], -1, -1, -1)
        gathered_force_map, _ = self.projector.gather_from_world_lattice(
            force_map_world,
            target_coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        decoded_features, decoded_obs = self.decoder(
            gathered_world_features=gathered_world_features,
            gathered_world_base_features=gathered_world_base_features,
            gathered_force_map=gathered_force_map,
            target_sensor_embedding=target_sensor_embedding,
            geometry_latent=source_encoding["geometry_latent"],
            visibility_latent=source_encoding["visibility_latent"],
            z_force_global=source_encoding["z_force_global"],
        )
        return {
            "target_sensor_embedding": target_sensor_embedding,
            "gathered_world_features": gathered_world_features,
            "gathered_world_base_features": gathered_world_base_features,
            "gathered_force_map": gathered_force_map,
            "decoded_target_features": decoded_features,
            "decoded_target_observation": decoded_obs,
            "decoded_target_images": decoded_obs,
            "decoded_target_occupancy": gathered_world_occ,
            "gathered_world_base_occupancy": gathered_world_base_occ,
            "decoded_world_occupancy": gathered_world_occ,
        }
