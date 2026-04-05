from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .common import ConvGNAct, MLP, ResidualConvBlock, sequence_last_valid, spatial_softargmax2d
from .patch_feature_encoder import PatchFeatureEncoder
from .sensor_context_encoder import SensorContextEncoder
from .temporal_state_encoder import ConvGRU
from .world_lattice import WorldLatticeProjector


class _FeatureStateHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pos_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.depth_head = MLP([in_channels, in_channels, 1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_last = x
        if x_last.dim() == 5:
            x_last = x_last[:, -1]
        logits = self.pos_head(x_last)
        x_norm, y_norm, heatmap = spatial_softargmax2d(logits)
        pooled = x_last.mean(dim=(2, 3))
        depth = self.depth_head(pooled).squeeze(-1)
        pred = torch.stack([x_norm, y_norm, depth], dim=1)
        return pred, logits, heatmap


class LegacyStaticRegressor(nn.Module):
    def __init__(self, observation_channels: int = 3, feature_dim: int = 128, output_dim: int = 3) -> None:
        super().__init__()
        self.encoder = PatchFeatureEncoder(in_channels=observation_channels, feature_dim=feature_dim)
        self.head = _FeatureStateHead(feature_dim)
        self.output_dim = int(output_dim)

    def forward(
        self,
        clip_obs: torch.Tensor | None = None,
        *,
        feature_grids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_grids is None:
            if clip_obs is None:
                raise ValueError("LegacyStaticRegressor needs either clip_obs or feature_grids")
            feature_grids = self.encoder(clip_obs[:, -1:].contiguous())
        last_features = sequence_last_valid(feature_grids, valid_mask)
        pred, logits, heatmap = self.head(last_features)
        return {"pred": pred, "heatmap_logits": logits, "heatmap": heatmap, "feature_grids": feature_grids}


class LegacyTemporalRegressor(nn.Module):
    def __init__(self, observation_channels: int = 3, feature_dim: int = 128, hidden_dim: int = 128, output_dim: int = 3) -> None:
        super().__init__()
        self.frame_encoder = PatchFeatureEncoder(in_channels=observation_channels, feature_dim=feature_dim)
        self.temporal = ConvGRU(feature_dim, hidden_dim=hidden_dim, num_layers=2)
        self.head = _FeatureStateHead(hidden_dim)
        self.output_dim = int(output_dim)

    def forward(
        self,
        clip_obs: torch.Tensor | None = None,
        *,
        feature_grids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_grids is None:
            if clip_obs is None:
                raise ValueError("LegacyTemporalRegressor needs either clip_obs or feature_grids")
            feature_grids = self.frame_encoder(clip_obs)
        hidden_seq, hidden_states = self.temporal(feature_grids)
        pred, logits, heatmap = self.head(hidden_seq)
        last_hidden = sequence_last_valid(hidden_seq, valid_mask)
        return {
            "pred": pred,
            "heatmap_logits": logits,
            "heatmap": heatmap,
            "feature_grids": feature_grids,
            "hidden_seq": hidden_seq,
            "last_hidden": last_hidden,
            "hidden_states": hidden_states,
        }


class MultiScalePooledRegressor(LegacyTemporalRegressor):
    pass


class DeterministicTransportTemporalPredictor(nn.Module):
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.patch_encoder = PatchFeatureEncoder(in_channels=3, feature_dim=feature_dim)
        self.projector = WorldLatticeProjector()
        self.temporal = ConvGRU(feature_dim, hidden_dim=hidden_dim, num_layers=2)
        self.head = _FeatureStateHead(hidden_dim)

    def forward(self, obs: torch.Tensor, coord_map: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.patch_encoder(obs)
        world, occupancy = self.projector.splat_to_world_lattice(features, coord_map)
        hidden_seq, _ = self.temporal(world)
        pred, logits, heatmap = self.head(hidden_seq)
        return {
            "pred": pred,
            "hidden_seq": hidden_seq,
            "world": world,
            "occupancy": occupancy,
            "heatmap_logits": logits,
            "heatmap": heatmap,
        }


class FeatureSpaceSCTABaseline(nn.Module):
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128, sensor_dim: int = 64) -> None:
        super().__init__()
        self.patch_encoder = PatchFeatureEncoder(in_channels=3, feature_dim=feature_dim)
        self.sensor_encoder = SensorContextEncoder(embedding_dim=sensor_dim)
        self.projector = WorldLatticeProjector()
        self.temporal = ConvGRU(feature_dim, hidden_dim=hidden_dim, num_layers=2)
        self.head = _FeatureStateHead(hidden_dim)
        self.modulation = MLP([sensor_dim * 2, hidden_dim, hidden_dim * 2])

    def forward(
        self,
        source_obs: torch.Tensor,
        source_coord_map: torch.Tensor,
        source_scale_mm: torch.Tensor,
        target_coord_map: torch.Tensor,
        target_scale_mm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source_features = self.patch_encoder(source_obs)
        world, occupancy = self.projector.splat_to_world_lattice(source_features, source_coord_map)
        source_emb = self.sensor_encoder(source_coord_map, source_scale_mm)
        target_emb = self.sensor_encoder(target_coord_map, target_scale_mm)
        hidden_seq, _ = self.temporal(world)
        gathered, _ = self.projector.gather_from_world_lattice(hidden_seq, target_coord_map)
        cond = torch.cat([source_emb, target_emb], dim=1)
        gamma, beta = self.modulation(cond).chunk(2, dim=1)
        decoded = gathered * (1.0 + gamma[:, None, :, None, None]) + beta[:, None, :, None, None]
        pred, logits, heatmap = self.head(decoded)
        return {
            "pred": pred,
            "decoded_features": decoded,
            "world": world,
            "occupancy": occupancy,
            "heatmap_logits": logits,
            "heatmap": heatmap,
        }
