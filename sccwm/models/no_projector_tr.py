from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .common import MLP, sequence_last_valid, spatial_softargmax2d
from .patch_feature_encoder import PatchFeatureEncoder
from .temporal_state_encoder import ConvGRU


class NoProjectorTR(nn.Module):
    """Image-space temporal regressor baseline without any world-lattice projection."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        feature_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.patch_encoder = PatchFeatureEncoder(in_channels=input_channels, feature_dim=feature_dim)
        self.temporal = ConvGRU(feature_dim, hidden_dim=hidden_dim, num_layers=2)
        self.position_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.depth_head = MLP([hidden_dim, hidden_dim, 1])

    def _prepare_scale(
        self,
        scale_mm: torch.Tensor | float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(scale_mm, torch.Tensor):
            if scale_mm.dim() == 0:
                return scale_mm.to(device=device, dtype=dtype).expand(batch_size)
            return scale_mm.to(device=device, dtype=dtype).reshape(batch_size)
        return torch.full((batch_size,), float(scale_mm), device=device, dtype=dtype)

    def _sensor_norm_to_mm(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
        scale_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_shape = [scale_mm.shape[0]] + [1] * max(x_norm.dim() - 1, 0)
        half_scale = (scale_mm.reshape(view_shape) / 2.0).clamp_min(1e-6)
        return x_norm * half_scale, y_norm * half_scale

    def _predict_state(
        self,
        hidden_seq: torch.Tensor,
        scale_mm: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t, c, h, w = hidden_seq.shape
        logits = self.position_head(hidden_seq.reshape(b * t, c, h, w))
        x_norm_img, y_norm_img, heatmap = spatial_softargmax2d(logits)
        x_norm_img = x_norm_img.reshape(b, t)
        y_norm_img = y_norm_img.reshape(b, t)
        x_mm, y_mm = self._sensor_norm_to_mm(x_norm_img, y_norm_img, scale_mm)
        heatmap = heatmap.reshape(b, t, 1, h, w)

        pooled = hidden_seq.mean(dim=(3, 4))
        depth = self.depth_head(pooled.reshape(b * t, -1)).reshape(b, t)

        pred_x_mm = sequence_last_valid(x_mm.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_y_mm = sequence_last_valid(y_mm.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_x_norm = sequence_last_valid(x_norm_img.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_y_norm = sequence_last_valid(y_norm_img.unsqueeze(-1), valid_mask).squeeze(-1)
        pred_depth_mm = sequence_last_valid(depth.unsqueeze(-1), valid_mask).squeeze(-1)

        return {
            "pred_x_mm_seq": x_mm,
            "pred_y_mm_seq": y_mm,
            "pred_x_norm_seq": x_norm_img,
            "pred_y_norm_seq": y_norm_img,
            "pred_depth_mm_seq": depth,
            "pred_x_mm": pred_x_mm,
            "pred_y_mm": pred_y_mm,
            "pred_x_norm": pred_x_norm,
            "pred_y_norm": pred_y_norm,
            "pred_depth_mm": pred_depth_mm,
            "position_heatmap": heatmap,
            # Kept only for compatibility with downstream visualization/eval code.
            # This is still an image-space heatmap, not a world-lattice projection.
            "world_position_heatmap": heatmap,
        }

    def forward(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor | None,
        scale_mm: torch.Tensor | float,
        *,
        valid_mask: torch.Tensor | None = None,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del coord_map, absolute_contact_xy_mm, world_origin_xy_mm
        b = obs_seq.shape[0]
        device = obs_seq.device
        dtype = obs_seq.dtype
        scale = self._prepare_scale(scale_mm, b, device, dtype)
        patch_features = self.patch_encoder(obs_seq)
        hidden_seq, hidden_states = self.temporal(patch_features)
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
            # Explicitly image-space latent, not a world-lattice feature volume.
            "image_latent_seq": hidden_seq,
            "last_hidden": last_hidden,
            "hidden_states": hidden_states,
            "state_embedding": state_embedding,
            **state,
        }

    def forward_single(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor | None,
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
