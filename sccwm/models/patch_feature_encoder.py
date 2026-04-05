from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvGNAct, ResidualConvBlock


class PatchFeatureEncoder(nn.Module):
    """Lightweight conv encoder mapping 256x256 3ch observations to 16x16 feature grids."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        widths = [32, 64, 96, feature_dim]
        layers: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            layers.append(ConvGNAct(prev, width, kernel_size=3, stride=2))
            layers.append(ResidualConvBlock(width))
            prev = width
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 5:
            raise ValueError(f"Expected obs shape (B,T,C,H,W), got {tuple(obs.shape)}")
        b, t, c, h, w = obs.shape
        x = obs.reshape(b * t, c, h, w)
        y = self.net(x)
        return y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3])
