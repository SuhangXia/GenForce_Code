from __future__ import annotations

import torch
import torch.nn as nn

from sccwm.models.common import ConvGNAct, MLP, ResidualConvBlock


class ForceConditionedCounterfactualDecoderV21A(nn.Module):
    """Counterfactual decoder with explicit force-latent conditioning.

    This is still a feature-space decoder. It does not add an image-heavy branch,
    but it makes force latents part of the decode path through both spatial
    injection and global FiLM modulation.
    """

    def __init__(
        self,
        *,
        world_hidden_dim: int = 128,
        feature_dim: int = 128,
        sensor_dim: int = 64,
        geometry_dim: int = 64,
        visibility_dim: int = 32,
        z_force_dim: int = 6,
        reconstruct_observation: bool = False,
    ) -> None:
        super().__init__()
        self.reconstruct_observation = bool(reconstruct_observation)
        cond_dim = sensor_dim + geometry_dim + visibility_dim
        self.base_film = MLP([cond_dim, world_hidden_dim, feature_dim * 2])
        self.force_film = MLP([z_force_dim, world_hidden_dim, feature_dim * 2])
        self.force_map_proj = nn.Sequential(
            ConvGNAct(z_force_dim, feature_dim, kernel_size=1, stride=1, padding=0),
            ResidualConvBlock(feature_dim),
        )
        self.pre_refine = nn.Sequential(
            ConvGNAct(feature_dim * 3, feature_dim, kernel_size=3, stride=1),
            ResidualConvBlock(feature_dim),
            ResidualConvBlock(feature_dim),
        )
        if self.reconstruct_observation:
            self.obs_decoder = nn.Sequential(
                ConvGNAct(feature_dim, feature_dim, kernel_size=3, stride=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(feature_dim, feature_dim // 2, kernel_size=3, stride=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(feature_dim // 2, feature_dim // 4, kernel_size=3, stride=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvGNAct(feature_dim // 4, feature_dim // 8, kernel_size=3, stride=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(feature_dim // 8, 3, kernel_size=3, padding=1),
            )
        else:
            self.obs_decoder = None

    def forward(
        self,
        *,
        gathered_world_features: torch.Tensor,
        gathered_world_base_features: torch.Tensor,
        gathered_force_map: torch.Tensor,
        target_sensor_embedding: torch.Tensor,
        geometry_latent: torch.Tensor,
        visibility_latent: torch.Tensor,
        z_force_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if gathered_world_features.dim() != 5:
            raise ValueError(f"Expected gathered_world_features shape (B,T,C,H,W), got {tuple(gathered_world_features.shape)}")
        b, t, _, hp, wp = gathered_world_features.shape
        cond = torch.cat(
            [
                target_sensor_embedding.unsqueeze(1).expand(-1, t, -1),
                geometry_latent.unsqueeze(1).expand(-1, t, -1),
                visibility_latent,
            ],
            dim=-1,
        )
        base_gamma_beta = self.base_film(cond.reshape(b * t, -1)).reshape(b * t, -1, 1, 1)
        force_gamma_beta = self.force_film(z_force_global.unsqueeze(1).expand(-1, t, -1).reshape(b * t, -1)).reshape(b * t, -1, 1, 1)
        base_gamma, base_beta = base_gamma_beta.chunk(2, dim=1)
        force_gamma, force_beta = force_gamma_beta.chunk(2, dim=1)
        gathered_world_bt = gathered_world_features.reshape(b * t, gathered_world_features.shape[2], hp, wp)
        gathered_world_base_bt = gathered_world_base_features.reshape(b * t, gathered_world_base_features.shape[2], hp, wp)
        gathered_force_bt = gathered_force_map.reshape(b * t, gathered_force_map.shape[2], hp, wp)
        force_feat = self.force_map_proj(gathered_force_bt)
        fused = torch.cat([gathered_world_bt, gathered_world_base_bt, force_feat], dim=1)
        refined = self.pre_refine(fused)
        decoded_features = refined * (1.0 + base_gamma + 0.5 * force_gamma) + base_beta + force_beta
        decoded_features = decoded_features.reshape(b, t, decoded_features.shape[1], hp, wp)
        if self.obs_decoder is None:
            return decoded_features, None
        decoded_obs = self.obs_decoder(decoded_features.reshape(b * t, decoded_features.shape[2], hp, wp))
        return decoded_features, decoded_obs.reshape(b, t, 3, decoded_obs.shape[-2], decoded_obs.shape[-1])
