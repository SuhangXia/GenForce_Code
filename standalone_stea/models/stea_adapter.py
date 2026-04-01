from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EMBED_DIM = 768
DEFAULT_GRID_SIZE = 14
DEFAULT_NUM_TOKENS = DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim <= 0 or embed_dim % 2 != 0:
        raise ValueError(f'embed_dim must be a positive even integer, got {embed_dim}')
    if positions.dim() != 1:
        raise ValueError(f'positions must be 1D, got shape {tuple(positions.shape)}')

    half_dim = embed_dim // 2
    omega = torch.arange(half_dim, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half_dim, 1)))
    out = positions.to(dtype=torch.float32).unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError(f'embed_dim must be a positive multiple of 4, got {embed_dim}')
    if grid_size <= 0:
        raise ValueError(f'grid_size must be positive, got {grid_size}')

    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    pos_y = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y.reshape(-1))
    pos_x = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x.reshape(-1))
    return torch.cat([pos_y, pos_x], dim=1)


def _standardize_background_latent_map(
    background_latent_map: torch.Tensor,
    *,
    embed_dim: int,
    grid_size: int,
) -> torch.Tensor:
    tensor = background_latent_map.detach().to(dtype=torch.float32)
    if tensor.dim() == 3 and tensor.shape == (embed_dim, grid_size, grid_size):
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape == (grid_size, grid_size, embed_dim):
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    elif tensor.dim() == 4 and tensor.shape == (1, embed_dim, grid_size, grid_size):
        tensor = tensor.contiguous()
    elif tensor.dim() == 4 and tensor.shape == (1, grid_size, grid_size, embed_dim):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            'background_latent_map must be shaped like '
            f'(1,{embed_dim},{grid_size},{grid_size}), ({embed_dim},{grid_size},{grid_size}), '
            f'or the channel-last equivalents, got {tuple(tensor.shape)}'
        )
    return tensor


def load_background_latent_map(
    background_reference: str | Path | torch.Tensor,
    *,
    embed_dim: int = DEFAULT_EMBED_DIM,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> torch.Tensor:
    if isinstance(background_reference, torch.Tensor):
        return _standardize_background_latent_map(
            background_reference,
            embed_dim=embed_dim,
            grid_size=grid_size,
        )

    path = Path(background_reference).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            'A canonical 20mm background latent map is required. '
            f'Could not find: {path}'
        )

    if path.suffix.lower() == '.npy':
        arr = np.load(path)
        tensor = torch.from_numpy(arr)
    elif path.suffix.lower() == '.pt':
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(payload, dict):
            if 'background_latent_map' in payload:
                tensor = payload['background_latent_map']
            elif 'tensor' in payload:
                tensor = payload['tensor']
            else:
                raise KeyError(
                    f'Expected background_latent_map or tensor in {path}, got keys={list(payload.keys())}'
                )
        elif isinstance(payload, torch.Tensor):
            tensor = payload
        else:
            raise TypeError(f'Unsupported .pt payload type for background latent map: {type(payload)}')
    else:
        raise ValueError(
            'background_latent_map must be provided as a tensor, .pt, or .npy file. '
            f'Got suffix {path.suffix!r}'
        )

    return _standardize_background_latent_map(tensor, embed_dim=embed_dim, grid_size=grid_size)


class STEAAdapter(nn.Module):
    """Spatial-Topology & Energy-Aligned adapter on the fixed 14x14 token grid.

    STEA is intentionally conservative: it first performs explicit geometric
    scale correction by resampling in token space, then fills any out-of-bounds
    regions with a canonical no-contact background latent map, and only then
    applies a small scale/position-conditioned residual energy calibration.
    """

    def __init__(
        self,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        grid_size: int = DEFAULT_GRID_SIZE,
        background_latent_map: str | Path | torch.Tensor | None,
        condition_hidden_dim: int = 512,
        use_boundary_smoothing: bool = False,
        residual_gate_init: float = -7.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f'embed_dim must be positive, got {embed_dim}')
        if grid_size <= 0:
            raise ValueError(f'grid_size must be positive, got {grid_size}')
        if background_latent_map is None:
            raise ValueError(
                'STEAAdapter requires a canonical 20mm background latent map. '
                'Provide background_latent_map as a tensor, .pt, or .npy reference.'
            )

        self.embed_dim = int(embed_dim)
        self.grid_size = int(grid_size)
        self.num_tokens = self.grid_size * self.grid_size
        self.use_boundary_smoothing = bool(use_boundary_smoothing)

        bg_map = load_background_latent_map(
            background_latent_map,
            embed_dim=self.embed_dim,
            grid_size=self.grid_size,
        )
        self.register_buffer('background_latent_map', bg_map)

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.register_buffer('pos_embed_2d', pos_embed.unsqueeze(0))

        self.token_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)
        self.modulation_mlp = nn.Sequential(
            nn.Linear(self.embed_dim + 4, int(condition_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(condition_hidden_dim), self.embed_dim * 2),
        )
        nn.init.zeros_(self.modulation_mlp[-1].weight)
        nn.init.zeros_(self.modulation_mlp[-1].bias)

        if self.use_boundary_smoothing:
            self.boundary_smoother = nn.Conv2d(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                padding=1,
                groups=self.embed_dim,
                bias=True,
            )
            with torch.no_grad():
                self.boundary_smoother.weight.zero_()
                self.boundary_smoother.weight[:, 0, 1, 1] = 1.0
                self.boundary_smoother.bias.zero_()
        else:
            self.boundary_smoother = None

        self.residual_logit = nn.Parameter(torch.tensor(float(residual_gate_init), dtype=torch.float32))

    @staticmethod
    def _prepare_scale_tensor(
        scale_mm: float | int | torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if isinstance(scale_mm, (float, int)):
            return torch.full((batch_size, 1), float(scale_mm), device=device, dtype=dtype)
        if not isinstance(scale_mm, torch.Tensor):
            raise TypeError(f'{name} must be float, int, or tensor, got {type(scale_mm)}')
        out = scale_mm.to(device=device, dtype=dtype)
        if out.dim() == 0:
            return out.expand(batch_size).reshape(batch_size, 1)
        if out.dim() == 1:
            if out.shape[0] != batch_size:
                raise ValueError(f'{name} must have length {batch_size}, got {out.shape[0]}')
            return out.reshape(batch_size, 1)
        if out.shape == (batch_size, 1):
            return out
        raise ValueError(f'{name} must have shape (), (B,), or (B,1), got {tuple(out.shape)}')

    def _reshape_source_valid_mask(
        self,
        source_valid_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if source_valid_mask is None:
            return None
        mask = source_valid_mask.to(device=device, dtype=dtype)
        if mask.dim() == 2:
            if mask.shape != (batch_size, self.num_tokens):
                raise ValueError(
                    f'source_valid_mask must be {(batch_size, self.num_tokens)}, got {tuple(mask.shape)}'
                )
            return mask.reshape(batch_size, 1, self.grid_size, self.grid_size)
        if mask.dim() == 3:
            if mask.shape != (batch_size, self.grid_size, self.grid_size):
                raise ValueError(
                    f'source_valid_mask must be {(batch_size, self.grid_size, self.grid_size)}, got {tuple(mask.shape)}'
                )
            return mask.unsqueeze(1)
        if mask.dim() == 4 and mask.shape == (batch_size, 1, self.grid_size, self.grid_size):
            return mask
        raise ValueError(
            'source_valid_mask must be (B,N), (B,H,W), or (B,1,H,W), '
            f'got {tuple(mask.shape)}'
        )

    def _ratio_features(self, ratio: torch.Tensor) -> torch.Tensor:
        safe_ratio = ratio.clamp_min(1e-6)
        inv_ratio = safe_ratio.reciprocal().clamp(max=1e6)
        log_ratio = torch.log(safe_ratio)
        delta_ratio = safe_ratio - 1.0
        return torch.cat([safe_ratio, inv_ratio, log_ratio, delta_ratio], dim=-1)

    def _build_sampling_grid(self, ratio: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # To canonicalize the apparent geometry, we sample with scale 1/ratio.
        # ratio < 1 (smaller source sensor) shrinks the inflated contact pattern.
        # ratio > 1 (larger source sensor) expands the compressed contact pattern.
        sample_scale = ratio.clamp_min(1e-6).reciprocal().reshape(batch_size)
        theta = torch.zeros((batch_size, 2, 3), device=device, dtype=dtype)
        theta[:, 0, 0] = sample_scale
        theta[:, 1, 1] = sample_scale
        return F.affine_grid(
            theta,
            size=(batch_size, self.embed_dim, self.grid_size, self.grid_size),
            align_corners=False,
        )

    def forward(
        self,
        source_vit_features: torch.Tensor,
        source_scale_mm: float | int | torch.Tensor,
        target_scale_mm: float | int | torch.Tensor = 20.0,
        source_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if source_vit_features.dim() != 3 or source_vit_features.shape[1:] != (self.num_tokens, self.embed_dim):
            raise ValueError(
                f'source_vit_features must be (B, {self.num_tokens}, {self.embed_dim}), '
                f'got {tuple(source_vit_features.shape)}'
            )

        batch_size = source_vit_features.shape[0]
        device = source_vit_features.device
        dtype = source_vit_features.dtype
        source_scale = self._prepare_scale_tensor(
            source_scale_mm,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name='source_scale_mm',
        )
        target_scale = self._prepare_scale_tensor(
            target_scale_mm,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name='target_scale_mm',
        )
        ratio = source_scale / target_scale.clamp_min(1e-6)

        source_map = source_vit_features.reshape(batch_size, self.grid_size, self.grid_size, self.embed_dim)
        source_map = source_map.permute(0, 3, 1, 2).contiguous()
        grid = self._build_sampling_grid(ratio, batch_size, device, dtype)
        sampled = F.grid_sample(
            source_map,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        in_bounds = (
            (grid[..., 0] >= -1.0)
            & (grid[..., 0] <= 1.0)
            & (grid[..., 1] >= -1.0)
            & (grid[..., 1] <= 1.0)
        ).unsqueeze(1)
        sampled_valid = in_bounds
        source_valid_map = self._reshape_source_valid_mask(
            source_valid_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if source_valid_map is not None:
            sampled_source_valid = F.grid_sample(
                source_valid_map,
                grid,
                mode='nearest',
                padding_mode='zeros',
                align_corners=False,
            )
            sampled_valid = sampled_valid & (sampled_source_valid > 0.5)

        valid_float = sampled_valid.to(dtype=dtype)
        background = self.background_latent_map.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        base_map = valid_float * sampled + (1.0 - valid_float) * background
        base_tokens = base_map.permute(0, 2, 3, 1).reshape(batch_size, self.num_tokens, self.embed_dim)

        pos_embed = self.pos_embed_2d.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        ratio_features = self._ratio_features(ratio).unsqueeze(1).expand(-1, self.num_tokens, -1)
        gamma_beta = self.modulation_mlp(torch.cat([pos_embed, ratio_features], dim=-1))
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # AdaLN-Zero residual path: when gamma=0 and beta=0, delta_tokens is exactly zero.
        base_norm = self.token_norm(base_tokens)
        delta_tokens = gamma * base_norm + beta

        if self.boundary_smoother is not None:
            delta_map = delta_tokens.reshape(batch_size, self.grid_size, self.grid_size, self.embed_dim)
            delta_map = delta_map.permute(0, 3, 1, 2).contiguous()
            delta_map = self.boundary_smoother(delta_map)
            delta_tokens = delta_map.permute(0, 2, 3, 1).reshape(batch_size, self.num_tokens, self.embed_dim)

        gate = torch.sigmoid(self.residual_logit).to(device=device, dtype=dtype)
        gated_delta = gate * delta_tokens
        adapted_tokens = base_tokens + gated_delta

        aux = {
            'valid_ratio': sampled_valid.to(dtype=dtype).mean(),
            'ratio_mean': ratio.mean(),
            'gamma_abs_mean': gamma.abs().mean(),
            'beta_abs_mean': beta.abs().mean(),
            'residual_norm': torch.linalg.vector_norm(gated_delta.reshape(batch_size, -1), dim=1).mean(),
            'delta_abs_mean': delta_tokens.abs().mean(),
            'base_norm_mean': base_norm.abs().mean(),
            'smoothing_enabled': adapted_tokens.new_tensor(1.0 if self.use_boundary_smoothing else 0.0),
            'sampled_mask': sampled_valid.squeeze(1),
            'gamma': gamma,
            'beta': beta,
            'delta': delta_tokens,
            'residual_gate': gate.detach().clone(),
        }
        return adapted_tokens.contiguous(), aux
