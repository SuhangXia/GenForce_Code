from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sccwm.models.common import MLP, sequence_last_valid
from sccwm.models.sccwm import SCCWM


class SCCWMForceGrounded(SCCWM):
    """Force-grounded SCCWM v2.

    This model is not explicitly force supervised. It adds a small force-like latent
    subspace and proxy heads on top of the original SCCWM stage-2 backbone.
    """

    def __init__(self, *, z_force_dim: int = 6, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.z_force_dim = int(z_force_dim)
        force_in_dim = self.world_hidden_dim + self.geometry_dim + self.visibility_dim + self.sensor_dim
        self.force_latent_head = MLP([force_in_dim, self.world_hidden_dim, self.z_force_dim])
        self.force_latent_norm = nn.LayerNorm(self.z_force_dim)
        self.normal_force_proxy_head = MLP([self.z_force_dim, 64, 1])
        self.contact_intensity_proxy_head = MLP([self.z_force_dim, 64, 1])

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
        last_hidden = base["last_hidden"].mean(dim=(2, 3))
        anchor_visibility = sequence_last_valid(base["visibility_latent"], valid_mask)
        force_input = torch.cat(
            [
                last_hidden,
                base["geometry_latent"],
                anchor_visibility,
                base["sensor_embedding"],
            ],
            dim=1,
        )
        z_force = self.force_latent_norm(self.force_latent_head(force_input))
        pred_normal_force_proxy = F.softplus(self.normal_force_proxy_head(z_force)).squeeze(1)
        pred_contact_intensity_proxy = torch.sigmoid(self.contact_intensity_proxy_head(z_force)).squeeze(1)
        state_embedding = torch.cat(
            [
                base["geometry_latent"],
                z_force,
                base["pred_depth_mm"].unsqueeze(1),
                base["pred_x_mm"].unsqueeze(1),
                base["pred_y_mm"].unsqueeze(1),
            ],
            dim=1,
        )
        base.update(
            {
                "z_force": z_force,
                "pred_normal_force_proxy": pred_normal_force_proxy,
                "pred_contact_intensity_proxy": pred_contact_intensity_proxy,
                "state_embedding": state_embedding,
                "anchor_visibility_latent": anchor_visibility,
            }
        )
        return base

