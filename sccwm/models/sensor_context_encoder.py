from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvGNAct, MLP, ResidualConvBlock


class SensorContextEncoder(nn.Module):
    """Encode coord-map geometry and scale into a compact sensor embedding."""

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.coord_encoder = nn.Sequential(
            ConvGNAct(2, hidden_dim, kernel_size=3, stride=1),
            ResidualConvBlock(hidden_dim),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=2),
            ResidualConvBlock(hidden_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        self.stats_mlp = MLP([10, hidden_dim, hidden_dim])
        self.fuse = MLP([hidden_dim * 2, hidden_dim, embedding_dim])

    @staticmethod
    def _summary_stats(coord_map: torch.Tensor, scale_mm: torch.Tensor) -> torch.Tensor:
        if coord_map.dim() != 4 or coord_map.shape[-1] != 2:
            raise ValueError(f"Expected coord_map shape (B,H,W,2), got {tuple(coord_map.shape)}")
        b, h, w, _ = coord_map.shape
        x = coord_map[..., 0]
        y = coord_map[..., 1]
        dx = x[:, :, 1:] - x[:, :, :-1]
        dy = y[:, 1:, :] - y[:, :-1, :]
        stats = torch.stack(
            [
                x.amin(dim=(1, 2)),
                x.amax(dim=(1, 2)),
                y.amin(dim=(1, 2)),
                y.amax(dim=(1, 2)),
                dx.abs().mean(dim=(1, 2)),
                dy.abs().mean(dim=(1, 2)),
                x.mean(dim=(1, 2)),
                y.mean(dim=(1, 2)),
                torch.full((b,), float(h), device=coord_map.device, dtype=coord_map.dtype),
                scale_mm.reshape(-1).to(device=coord_map.device, dtype=coord_map.dtype),
            ],
            dim=1,
        )
        return stats

    def forward(self, coord_map: torch.Tensor, scale_mm: torch.Tensor) -> torch.Tensor:
        coord_chw = coord_map.permute(0, 3, 1, 2).contiguous()
        spatial = self.coord_encoder(coord_chw).flatten(1)
        stats = self.stats_mlp(self._summary_stats(coord_map, scale_mm))
        return self.fuse(torch.cat([spatial, stats], dim=1))
