from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, sequence_last_valid, spatial_softargmax2d
from .patch_feature_encoder import PatchFeatureEncoder
from .temporal_state_encoder import ConvGRU
from .world_lattice import WorldLatticeConfig, WorldLatticeProjector


class DWLTR(nn.Module):
    """Deterministic World-Lattice Transport + Temporal Regressor baseline."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        feature_dim: int = 128,
        world_hidden_dim: int = 128,
        lattice_size: int = 32,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.world_hidden_dim = int(world_hidden_dim)

        self.patch_encoder = PatchFeatureEncoder(in_channels=input_channels, feature_dim=feature_dim)
        self.projector = WorldLatticeProjector(WorldLatticeConfig(lattice_size=lattice_size))
        self.temporal = ConvGRU(feature_dim, hidden_dim=world_hidden_dim, num_layers=2)
        self.position_head = nn.Conv2d(world_hidden_dim, 1, kernel_size=3, padding=1)
        self.depth_head = MLP([world_hidden_dim, world_hidden_dim, 1])

    def _prepare_scale(self, scale_mm: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(scale_mm, torch.Tensor):
            if scale_mm.dim() == 0:
                return scale_mm.to(device=device, dtype=dtype).expand(batch_size)
            return scale_mm.to(device=device, dtype=dtype).reshape(batch_size)
        return torch.full((batch_size,), float(scale_mm), device=device, dtype=dtype)

    def _metric_to_sensor_norm(self, x_mm: torch.Tensor, y_mm: torch.Tensor, scale_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        view_shape = [scale_mm.shape[0]] + [1] * max(x_mm.dim() - 1, 0)
        half_scale = (scale_mm.reshape(view_shape) / 2.0).clamp_min(1e-6)
        return x_mm / half_scale, y_mm / half_scale

    def _align_patch_features_to_coord_map(self, patch_features: torch.Tensor, coord_map: torch.Tensor) -> torch.Tensor:
        if patch_features.dim() != 5:
            raise ValueError(f"Expected patch_features shape (B,T,D,H,W), got {tuple(patch_features.shape)}")
        if coord_map.dim() != 4 or coord_map.shape[-1] != 2:
            raise ValueError(f"Expected coord_map shape (B,H,W,2), got {tuple(coord_map.shape)}")
        target_hw = (int(coord_map.shape[1]), int(coord_map.shape[2]))
        if patch_features.shape[-2:] == target_hw:
            return patch_features
        b, t, d, _, _ = patch_features.shape
        pooled = F.adaptive_avg_pool2d(
            patch_features.reshape(b * t, d, patch_features.shape[-2], patch_features.shape[-1]),
            target_hw,
        )
        return pooled.reshape(b, t, d, target_hw[0], target_hw[1])

    def _predict_state(
        self,
        hidden_seq: torch.Tensor,
        scale_mm: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t, c, h, w = hidden_seq.shape
        logits = self.position_head(hidden_seq.reshape(b * t, c, h, w))
        lattice_x_norm, lattice_y_norm, heatmap = spatial_softargmax2d(logits)
        lattice_x_norm = lattice_x_norm.reshape(b, t)
        lattice_y_norm = lattice_y_norm.reshape(b, t)
        x_mm, y_mm = self.projector.lattice_normalized_to_mm(lattice_x_norm, lattice_y_norm)
        x_norm, y_norm = self._metric_to_sensor_norm(x_mm, y_mm, scale_mm)
        heatmap = heatmap.reshape(b, t, 1, h, w)
        pooled = hidden_seq.mean(dim=(3, 4))
        depth = self.depth_head(pooled.reshape(b * t, -1)).reshape(b, t)
        pred_x_mm = sequence_last_valid(x_mm.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_y_mm = sequence_last_valid(y_mm.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_x_norm = sequence_last_valid(x_norm.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_y_norm = sequence_last_valid(y_norm.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_depth_mm = sequence_last_valid(depth.unsqueeze(-1), valid_mask).squeeze(-1)
        return {
            "pred_x_mm_seq": x_mm,
            "pred_y_mm_seq": y_mm,
            "pred_x_norm_seq": x_norm,
            "pred_y_norm_seq": y_norm,
            "pred_depth_mm_seq": depth,
            "pred_x_mm": pred_x_mm,
            "pred_y_mm": pred_y_mm,
            "pred_x_norm": pred_x_norm,
            "pred_y_norm": pred_y_norm,
            "pred_depth_mm": pred_depth_mm,
            "world_position_heatmap": heatmap,
            "position_heatmap": heatmap,
        }

    def forward(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor,
        scale_mm: torch.Tensor | float,
        *,
        valid_mask: torch.Tensor | None = None,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = obs_seq.shape[0]
        device = obs_seq.device
        dtype = obs_seq.dtype
        scale = self._prepare_scale(scale_mm, b, device, dtype)
        coord_map_t = coord_map.to(device=device, dtype=dtype)
        patch_features = self._align_patch_features_to_coord_map(self.patch_encoder(obs_seq), coord_map_t)
        world_features, occupancy = self.projector.splat_to_world_lattice(
            patch_features,
            coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        hidden_seq, hidden_states = self.temporal(world_features)
        state = self._predict_state(hidden_seq, scale, valid_mask=valid_mask)
        last_hidden = sequence_last_valid(hidden_seq, valid_mask)
        pooled_last_hidden = last_hidden.mean(dim=(2, 3))
        state_embedding = torch.cat(
            [
                pooled_last_hidden,
                state["pred_depth_mm"].unsqueeze(1),
                state["pred_x_mm"].unsqueeze(1),
                state["pred_y_mm"].unsqueeze(1),
            ],
            dim=1,
        )
        return {
            "patch_features": patch_features,
            "scale_mm": scale,
            "world_features": world_features,
            "world_occupancy": occupancy,
            "world_latent_seq": hidden_seq,
            "hidden_states": hidden_states,
            "last_hidden": last_hidden,
            "state_embedding": state_embedding,
            **state,
        }

    def forward_single(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor,
        scale_mm: torch.Tensor | float,
        *,
        valid_mask: torch.Tensor | None = None,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward(
            obs_seq,
            coord_map,
            scale_mm,
            valid_mask=valid_mask,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
