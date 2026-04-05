from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, stride: int = 1, padding: int | None = None) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, kernel_size=3, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


def spatial_softargmax2d(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dim() != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected logits shape (B,1,H,W), got {tuple(logits.shape)}")
    b, _, h, w = logits.shape
    flat = logits.reshape(b, -1)
    weights = torch.softmax(flat, dim=-1).reshape(b, 1, h, w)
    grid_y = torch.linspace(-1.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    grid_x = torch.linspace(-1.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    x = (weights[:, 0] * xx).sum(dim=(1, 2))
    y = (weights[:, 0] * yy).sum(dim=(1, 2))
    return x, y, weights


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + float(eps) ** 2).mean()


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None, dim: tuple[int, ...]) -> torch.Tensor:
    if mask is None:
        return value.mean(dim=dim)
    masked = value * mask
    denom = mask.sum(dim=dim).clamp_min(1e-6)
    return masked.sum(dim=dim) / denom


class MLP(nn.Module):
    def __init__(self, dims: list[int], *, dropout: float = 0.0, final_activation: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_dim, out_dim = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if (not is_last) or final_activation:
                layers.append(nn.SiLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sequence_last_valid(x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_mask is None:
        return x[:, -1]
    idx = valid_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    gather = idx.view(-1, 1, *([1] * (x.dim() - 2))).expand(-1, 1, *x.shape[2:])
    return x.gather(dim=1, index=gather).squeeze(1)


def repeat_time(x: torch.Tensor, steps: int) -> torch.Tensor:
    return x.unsqueeze(1).expand(-1, steps, -1)
