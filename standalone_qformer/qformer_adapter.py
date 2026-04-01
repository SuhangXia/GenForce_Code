from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn


DEFAULT_EMBED_DIM: Final[int] = 768
DEFAULT_NUM_QUERIES: Final[int] = 196


def _get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Generate 1D sine-cosine embeddings for a vector of positions.

    This is the standard fixed transformer positional encoding written in a form
    that is easy to lift to 2D grids. The output is non-trainable and therefore
    acts as a stable spatial reference frame.
    """

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
    """Return fixed 2D sine-cosine positional encodings for a square grid.

    Args:
        embed_dim: Embedding dimension of each token. Must be divisible by 4 so
            half can be assigned to Y and half to X, each with sine/cosine pairs.
        grid_size: Height/width of the square grid.

    Returns:
        Tensor of shape (grid_size * grid_size, embed_dim).
    """

    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError(f'embed_dim must be a positive multiple of 4, got {embed_dim}')
    if grid_size <= 0:
        raise ValueError(f'grid_size must be positive, got {grid_size}')

    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    half_dim = embed_dim // 2
    pos_y = _get_1d_sincos_pos_embed_from_grid(half_dim, grid_y)
    pos_x = _get_1d_sincos_pos_embed_from_grid(half_dim, grid_x)
    return torch.cat([pos_y, pos_x], dim=1)


class ContinuousScaleEmbedding(nn.Module):
    """Embed a continuous source/target scale ratio into the ViT token space.

    The physics prior here is simple: the resize operation changes the perceived
    energy/magnitude of tactile patterns. We therefore encode several equivalent
    views of the scale ratio so the Q-Former can condition its latent correction
    on absolute ratio, inverse ratio, log-ratio, and deviation from identity.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        hidden_dim = int(hidden_dim or embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )

    def forward(self, scale_ratio: torch.Tensor) -> torch.Tensor:
        if scale_ratio.dim() != 2 or scale_ratio.shape[-1] != 1:
            raise ValueError(
                f'scale_ratio must have shape (B, 1), got {tuple(scale_ratio.shape)}'
            )

        safe_ratio = scale_ratio.clamp_min(1e-6)
        inv_ratio = safe_ratio.reciprocal().clamp(max=1e6)
        log_ratio = torch.log(safe_ratio)
        delta_ratio = safe_ratio - 1.0
        ratio_features = torch.cat(
            [safe_ratio, inv_ratio, log_ratio, delta_ratio],
            dim=-1,
        )
        return self.mlp(ratio_features).unsqueeze(1)


class ScaleConditionedQFormer(nn.Module):
    """Spatially-aware scale-conditioned Q-Former for tactile feature alignment.

    The core fix here is to anchor every query to a strict 14x14 lattice using a
    fixed 2D sine-cosine positional encoding. This converts each query from a
    free-floating latent slot into a stable spatial "GPS coordinate", preserving
    the topology expected by downstream pose heads while still allowing scale
    conditioning to re-calibrate feature energy.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_queries: int = DEFAULT_NUM_QUERIES,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        scale_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f'embed_dim must be positive, got {embed_dim}')
        if num_queries <= 0:
            raise ValueError(f'num_queries must be positive, got {num_queries}')
        if num_layers <= 0:
            raise ValueError(f'num_layers must be positive, got {num_layers}')
        if num_heads <= 0:
            raise ValueError(f'num_heads must be positive, got {num_heads}')
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}'
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f'mlp_ratio must be > 0, got {mlp_ratio}')

        grid_size = int(round(math.sqrt(num_queries)))
        if grid_size * grid_size != int(num_queries):
            raise ValueError(
                f'num_queries must form a square grid for 2D positional encoding, got {num_queries}'
            )

        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.grid_size = grid_size

        self.scale_embedding = ContinuousScaleEmbedding(
            embed_dim=self.embed_dim,
            hidden_dim=scale_hidden_dim,
        )
        self.learnable_queries = nn.Parameter(
            torch.empty(1, self.num_queries, self.embed_dim)
        )

        pe_tensor = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.register_buffer('pos_embed_2d', pe_tensor.unsqueeze(0))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.embed_dim),
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.learnable_queries, std=0.02)

    @staticmethod
    def _prepare_scale_tensor(
        scale_mm: float | int | torch.Tensor,
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
        raise ValueError(
            f'{name} must have shape (), (B,), or (B,1), got {tuple(out.shape)}'
        )

    def forward(
        self,
        source_vit_features: torch.Tensor,
        source_scale_mm: float | int | torch.Tensor,
        target_scale_mm: float | int | torch.Tensor = 20.0,
    ) -> torch.Tensor:
        if source_vit_features.dim() != 3:
            raise ValueError(
                'source_vit_features must have shape (B, N, D), '
                f'got {tuple(source_vit_features.shape)}'
            )

        batch_size, num_tokens, embed_dim = source_vit_features.shape
        if num_tokens != self.num_queries or embed_dim != self.embed_dim:
            raise ValueError(
                f'source_vit_features must have shape (B, {self.num_queries}, {self.embed_dim}), '
                f'got {tuple(source_vit_features.shape)}'
            )

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
        scale_ratio = source_scale / target_scale.clamp_min(1e-6)
        scale_embed = self.scale_embedding(scale_ratio)

        pos_embed = self.pos_embed_2d.to(device=device, dtype=dtype)
        queries = self.learnable_queries.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

        # First inject the fixed spatial lattice so each query corresponds to a
        # deterministic 14x14 location. Then add the global scale intent.
        spatial_queries = queries + pos_embed
        conditioned_queries = spatial_queries + scale_embed

        # Mirroring the same 2D reference frame onto the memory tokens helps the
        # decoder align target-grid queries with source-grid tokens in a strictly
        # spatial way, rather than treating the patch set as permutation-free.
        spatial_memory = source_vit_features + pos_embed

        aligned_features = self.decoder(
            tgt=conditioned_queries,
            memory=spatial_memory,
        )
        aligned_features = self.output_norm(aligned_features)
        if aligned_features.shape != (batch_size, self.num_queries, self.embed_dim):
            raise RuntimeError(
                'Q-Former produced an unexpected shape: '
                f'{tuple(aligned_features.shape)}'
            )
        return aligned_features
