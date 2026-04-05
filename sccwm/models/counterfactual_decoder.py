from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvGNAct, MLP, ResidualConvBlock


class CounterfactualDecoder(nn.Module):
    """Decode world latents into target-sensor feature grids and auxiliary observations."""

    def __init__(
        self,
        world_hidden_dim: int = 128,
        feature_dim: int = 128,
        sensor_dim: int = 64,
        geometry_dim: int = 64,
        visibility_dim: int = 32,
        reconstruct_observation: bool = True,
    ) -> None:
        super().__init__()
        self.reconstruct_observation = bool(reconstruct_observation)
        cond_dim = sensor_dim + geometry_dim + visibility_dim
        self.film = MLP([cond_dim, world_hidden_dim, feature_dim * 2])
        self.pre_refine = nn.Sequential(
            ConvGNAct(feature_dim * 2, feature_dim, kernel_size=3, stride=1),
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
        gathered_world_features: torch.Tensor,
        gathered_world_base_features: torch.Tensor,
        target_sensor_embedding: torch.Tensor,
        geometry_latent: torch.Tensor,
        visibility_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if gathered_world_features.dim() != 5:
            raise ValueError(f"Expected gathered_world_features shape (B,T,C,H,W), got {tuple(gathered_world_features.shape)}")
        b, t, _, hp, wp = gathered_world_features.shape
        cond = torch.cat([target_sensor_embedding.unsqueeze(1).expand(-1, t, -1), geometry_latent.unsqueeze(1).expand(-1, t, -1), visibility_latent], dim=-1)
        gamma_beta = self.film(cond.reshape(b * t, -1)).reshape(b * t, -1, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gathered_world_bt = gathered_world_features.reshape(b * t, gathered_world_features.shape[2], hp, wp)
        gathered_world_base_bt = gathered_world_base_features.reshape(b * t, gathered_world_base_features.shape[2], hp, wp)
        fused = torch.cat([gathered_world_bt, gathered_world_base_bt], dim=1)
        refined = self.pre_refine(fused)
        decoded_features = refined * (1.0 + gamma) + beta
        decoded_features = decoded_features.reshape(b, t, decoded_features.shape[1], hp, wp)
        if self.obs_decoder is None:
            return decoded_features, None
        decoded_obs = self.obs_decoder(decoded_features.reshape(b * t, decoded_features.shape[2], hp, wp))
        return decoded_features, decoded_obs.reshape(b, t, 3, decoded_obs.shape[-2], decoded_obs.shape[-1])
