from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierCoordinateEncoder(nn.Module):
    """Encode 2D coordinates with Fourier features followed by an MLP."""

    def __init__(
        self,
        embed_dim: int = 768,
        coord_dim: int = 2,
        num_frequencies: int = 4,
        include_input: bool = True,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if coord_dim != 2:
            raise ValueError(f'coord_dim must be 2, got {coord_dim}')
        if num_frequencies <= 0:
            raise ValueError(f'num_frequencies must be > 0, got {num_frequencies}')

        self.coord_dim = int(coord_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)

        self.register_buffer(
            'freq_bands',
            (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)),
            persistent=False,
        )

        raw_dim = self.coord_dim if self.include_input else 0
        fourier_dim = self.coord_dim * self.num_frequencies * 2
        in_dim = raw_dim + fourier_dim
        hidden_dim = hidden_dim or max(embed_dim // 2, 64)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f'coords last dim must be {self.coord_dim}, got shape {tuple(coords.shape)}'
            )

        freq = self.freq_bands.to(device=coords.device, dtype=coords.dtype)
        phase = coords.unsqueeze(-1) * freq
        phase = phase * (2.0 * math.pi)

        pieces: list[torch.Tensor] = []
        if self.include_input:
            pieces.append(coords)
        pieces.append(torch.sin(phase).reshape(*coords.shape[:-1], -1))
        pieces.append(torch.cos(phase).reshape(*coords.shape[:-1], -1))
        enc = torch.cat(pieces, dim=-1)
        return self.proj(enc)


class ScaleRatioEncoder(nn.Module):
    """Encode source/target scale ratio using a richer feature set than a single scalar."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 64)
        self.proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        source_scale_mm: torch.Tensor,
        target_scale_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_scale_mm.shape != target_scale_mm.shape:
            raise ValueError(
                f'source_scale_mm and target_scale_mm shape mismatch: '
                f'{tuple(source_scale_mm.shape)} vs {tuple(target_scale_mm.shape)}'
            )
        ratio = source_scale_mm / target_scale_mm.clamp_min(1e-6)
        inv_ratio = ratio.reciprocal().clamp(max=1e6)
        log_ratio = torch.log(ratio.clamp_min(1e-6))
        delta_ratio = ratio - 1.0
        ratio_features = torch.cat([ratio, inv_ratio, log_ratio, delta_ratio], dim=-1)
        return self.proj(ratio_features), ratio


class RelativeCoordBias(nn.Module):
    """Map relative coordinate geometry into per-head cross-attention bias."""

    def __init__(self, num_heads: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, delta_coords: torch.Tensor) -> torch.Tensor:
        dx = delta_coords[..., 0:1]
        dy = delta_coords[..., 1:2]
        r2 = dx.square() + dy.square()
        r = torch.sqrt(r2 + 1e-8)
        geom = torch.cat([dx, dy, r, r2], dim=-1)
        bias = self.mlp(geom)
        return bias.permute(0, 3, 1, 2).contiguous()


class CrossAttentionWithBias(nn.Module):
    """Batch-first cross-attention with injectable relative-coordinate bias."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}'
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, nq, d = query.shape
        _, nk, _ = key.shape

        q = self.q_proj(query).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias

        if key_padding_mask is not None:
            if key_padding_mask.shape != (b, nk):
                raise ValueError(
                    f'key_padding_mask shape must be {(b, nk)}, got {tuple(key_padding_mask.shape)}'
                )
            safe_mask = key_padding_mask
            all_masked = safe_mask.all(dim=1)
            if all_masked.any():
                safe_mask = safe_mask.clone()
                safe_mask[all_masked, 0] = False
            logits = logits.masked_fill(safe_mask[:, None, None, :], float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, nq, d)
        return self.out_proj(out)


class ConvFFN(nn.Module):
    """Token MLP with a depthwise 3x3 convolution in the hidden 2D feature space."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f'tokens must be (B,N,D), got {tuple(tokens.shape)}')
        h, w = hw
        b, n, _ = tokens.shape
        if h * w != n:
            raise ValueError(f'hw={hw} is incompatible with token count {n}')

        hidden = self.fc1(tokens)
        hidden = self.act(hidden)
        hidden_2d = hidden.view(b, h, w, hidden.shape[-1]).permute(0, 3, 1, 2).contiguous()
        hidden_2d = self.dwconv(hidden_2d)
        hidden = hidden_2d.permute(0, 2, 3, 1).contiguous().view(b, n, -1)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return hidden


class DDUSAAdapterBlock(nn.Module):
    """Pre-norm decoder block: self-attn -> cross-attn -> ConvFFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        convffn_ratio: float = 4.0,
        rel_bias_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(embed_dim)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = CrossAttentionWithBias(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.rel_bias = RelativeCoordBias(num_heads=num_heads, hidden_dim=rel_bias_hidden_dim)
        self.conv_ffn = ConvFFN(
            embed_dim=embed_dim,
            hidden_dim=int(embed_dim * convffn_ratio),
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_state: torch.Tensor,
        source_key: torch.Tensor,
        source_value: torch.Tensor,
        query_coords_norm: torch.Tensor,
        source_coords_norm: torch.Tensor,
        target_hw: tuple[int, int],
        source_key_padding_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        safe_query_padding = None
        if query_valid_mask is not None:
            safe_query_padding = ~query_valid_mask
            all_masked = safe_query_padding.all(dim=1)
            if all_masked.any():
                safe_query_padding = safe_query_padding.clone()
                safe_query_padding[all_masked, 0] = False

        q_norm = self.self_norm(query_state)
        self_out, _ = self.self_attn(
            query=q_norm,
            key=q_norm,
            value=q_norm,
            key_padding_mask=safe_query_padding,
            need_weights=False,
        )
        query_state = query_state + self.dropout(self_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        delta = query_coords_norm[:, :, None, :] - source_coords_norm[:, None, :, :]
        rel_bias = self.rel_bias(delta)

        q_norm = self.cross_norm(query_state)
        cross_out = self.cross_attn(
            query=q_norm,
            key=source_key,
            value=source_value,
            attn_bias=rel_bias,
            key_padding_mask=source_key_padding_mask,
        )
        query_state = query_state + self.dropout(cross_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        ffn_out = self.conv_ffn(self.ffn_norm(query_state), target_hw)
        query_state = query_state + self.dropout(ffn_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)
        return query_state


class DirectDriveUniversalScaleAdapter(nn.Module):
    """Direct-drive latent interface for mapping arbitrary sensor scales into a fixed canonical token grid."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        coord_num_frequencies: int = 4,
        coord_include_input: bool = True,
        convffn_ratio: float = 4.0,
        rel_bias_hidden_dim: int = 64,
        use_final_norm: bool = True,
        learnable_mix: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f'embed_dim must be > 0, got {embed_dim}')
        if num_heads <= 0:
            raise ValueError(f'num_heads must be > 0, got {num_heads}')
        if num_layers <= 0:
            raise ValueError(f'num_layers must be > 0, got {num_layers}')

        self.embed_dim = int(embed_dim)
        self.coord_encoder = FourierCoordinateEncoder(
            embed_dim=self.embed_dim,
            coord_dim=2,
            num_frequencies=coord_num_frequencies,
            include_input=coord_include_input,
        )
        self.ratio_encoder = ScaleRatioEncoder(embed_dim=self.embed_dim)
        self.query_bias = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.source_key_norm = nn.LayerNorm(self.embed_dim)
        self.use_final_norm = bool(use_final_norm)
        self.final_norm = nn.LayerNorm(self.embed_dim) if self.use_final_norm else None

        self.layers = nn.ModuleList(
            [
                DDUSAAdapterBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    convffn_ratio=convffn_ratio,
                    rel_bias_hidden_dim=rel_bias_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.modulation_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
        )

        mix_init = torch.tensor([2.0, -2.0], dtype=torch.float32)
        if learnable_mix:
            self.mix_logits = nn.Parameter(mix_init.clone())
        else:
            self.register_buffer('mix_logits', mix_init, persistent=False)

    @staticmethod
    def _flatten_feat(name: str, feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if feat.dim() == 4:
            b, h, w, d = feat.shape
            return feat.reshape(b, h * w, d), (h, w)
        if feat.dim() == 3:
            return feat, None
        raise ValueError(f'{name} must be 3D (B,N,D) or 4D (B,H,W,D), got {tuple(feat.shape)}')

    @staticmethod
    def _flatten_coords(name: str, coords: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if coords.shape[-1] != 2:
            raise ValueError(f'{name} last dim must be 2, got {tuple(coords.shape)}')
        if coords.dim() == 4:
            b, h, w, _ = coords.shape
            return coords.reshape(b, h * w, 2), (h, w)
        if coords.dim() == 3:
            return coords, None
        raise ValueError(f'{name} must be 3D (B,N,2) or 4D (B,H,W,2), got {tuple(coords.shape)}')

    @staticmethod
    def _flatten_mask(
        name: str,
        mask: torch.Tensor | None,
        batch: int,
        tokens: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 3:
            b, h, w = mask.shape
            mask = mask.reshape(b, h * w)
        elif mask.dim() != 2:
            raise ValueError(f'{name} must be 2D (B,N) or 3D (B,H,W), got {tuple(mask.shape)}')
        if mask.shape != (batch, tokens):
            raise ValueError(f'{name} shape mismatch: expected {(batch, tokens)}, got {tuple(mask.shape)}')
        return mask.to(dtype=torch.bool)

    @staticmethod
    def _prepare_scale_tensor(
        scale_mm: float | int | torch.Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if isinstance(scale_mm, (float, int)):
            return torch.full((batch, 1), float(scale_mm), device=device, dtype=dtype)
        if not isinstance(scale_mm, torch.Tensor):
            raise TypeError(f'{name} must be float/int/tensor, got {type(scale_mm)}')
        out = scale_mm.to(device=device, dtype=dtype)
        if out.dim() == 0:
            return out.expand(batch).reshape(batch, 1)
        if out.dim() == 1:
            if out.shape[0] != batch:
                raise ValueError(f'{name} length must be {batch}, got {out.shape[0]}')
            return out.reshape(batch, 1)
        if out.shape == (batch, 1):
            return out
        raise ValueError(f'{name} must have shape (), (B,), or (B,1), got {tuple(out.shape)}')

    @staticmethod
    def _infer_hw(tokens: int, name: str) -> tuple[int, int]:
        side = int(round(math.sqrt(tokens)))
        if side * side != tokens:
            raise ValueError(
                f'{name} token count {tokens} is not square; please provide a 4D tensor or explicit grid coordinates'
            )
        return side, side

    def _resolve_hw(
        self,
        feat_hw: tuple[int, int] | None,
        coord_hw: tuple[int, int] | None,
        tokens: int,
        name: str,
    ) -> tuple[int, int]:
        if feat_hw is not None and coord_hw is not None and feat_hw != coord_hw:
            raise ValueError(f'{name} feature/coord grid mismatch: {feat_hw} vs {coord_hw}')
        if feat_hw is not None:
            return feat_hw
        if coord_hw is not None:
            return coord_hw
        return self._infer_hw(tokens, name)

    @staticmethod
    def _normalize_coords_with_scale(coords_mm: torch.Tensor, scale_mm: torch.Tensor) -> torch.Tensor:
        denom = (scale_mm / 2.0).clamp_min(1e-6)
        return coords_mm / denom.unsqueeze(-1)

    @staticmethod
    def _build_uniform_coord_map(
        target_grid_hw: tuple[int, int],
        target_scale_mm: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch = int(target_scale_mm.shape[0])
        h, w = target_grid_hw
        ys = ((torch.arange(h, device=device, dtype=dtype) + 0.5) / max(h, 1)) * 2.0 - 1.0
        xs = ((torch.arange(w, device=device, dtype=dtype) + 0.5) / max(w, 1)) * 2.0 - 1.0
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(batch, h, w, 2)
        half_scale = (target_scale_mm / 2.0).reshape(batch, 1, 1, 1)
        return base * half_scale

    @staticmethod
    def _metric_bbox_support(
        source_coords_flat: torch.Tensor,
        target_coords_flat: torch.Tensor,
        source_valid_flat: torch.Tensor,
    ) -> torch.Tensor:
        valid_source = source_valid_flat.unsqueeze(-1)
        large_pos = torch.full_like(source_coords_flat, torch.finfo(source_coords_flat.dtype).max)
        large_neg = torch.full_like(source_coords_flat, torch.finfo(source_coords_flat.dtype).min)
        min_bounds = torch.where(valid_source, source_coords_flat, large_pos).min(dim=1, keepdim=True).values
        max_bounds = torch.where(valid_source, source_coords_flat, large_neg).max(dim=1, keepdim=True).values
        has_source = source_valid_flat.any(dim=1, keepdim=True)
        inside = (target_coords_flat >= (min_bounds - 1e-6)) & (target_coords_flat <= (max_bounds + 1e-6))
        return inside[..., 0] & inside[..., 1] & has_source

    @staticmethod
    def _metric_knn_anchor(
        source_feat_flat: torch.Tensor,
        source_coords_norm: torch.Tensor,
        target_coords_norm: torch.Tensor,
        source_valid_flat: torch.Tensor,
        k: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, ns, d = source_feat_flat.shape
        nt = target_coords_norm.shape[1]
        if ns == 0 or not bool(source_valid_flat.any()):
            zeros_feat = source_feat_flat.new_zeros((b, nt, d))
            zeros_conf = source_feat_flat.new_zeros((b, nt))
            return zeros_feat, zeros_conf

        k_eff = max(1, min(int(k), ns))
        dist = torch.cdist(target_coords_norm, source_coords_norm, p=2)
        large_val = torch.full_like(dist, torch.finfo(dist.dtype).max / 1024.0)
        dist = torch.where(source_valid_flat[:, None, :], dist, large_val)

        knn_dist, knn_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)
        feat_index = knn_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        gathered_feat = torch.gather(
            source_feat_flat[:, None, :, :].expand(-1, nt, -1, -1),
            2,
            feat_index,
        )
        gathered_valid = torch.gather(
            source_valid_flat[:, None, :].expand(-1, nt, -1),
            2,
            knn_idx,
        )

        weights_raw = gathered_valid.to(dtype=source_feat_flat.dtype) / knn_dist.clamp_min(1e-6)
        weight_mass = weights_raw.sum(dim=-1, keepdim=True)
        weights = weights_raw / weight_mass.clamp_min(1e-6)
        anchor_flat = (weights.unsqueeze(-1) * gathered_feat).sum(dim=-2)

        nearest_dist = knn_dist[..., 0]
        dist_conf = torch.exp(-nearest_dist / 0.35)
        support_conf = torch.where(weight_mass.squeeze(-1) > 0.0, dist_conf, dist_conf.new_zeros(()).expand_as(dist_conf))
        return anchor_flat, support_conf

    @staticmethod
    def _weighted_mean_pool(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f'tokens must be (B,N,D), got {tuple(tokens.shape)}')
        if weights.shape != tokens.shape[:2]:
            raise ValueError(
                f'weights shape mismatch: expected {tuple(tokens.shape[:2])}, got {tuple(weights.shape)}'
            )
        weights = weights.clamp_min(0.0)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1) / weight_sum

    @staticmethod
    def _weighted_mean_token_norm(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f'tokens must be (B,N,D), got {tuple(tokens.shape)}')
        weights = weights.clamp_min(0.0)
        if not bool((weights > 0).any()):
            return tokens.new_zeros(())
        norms = torch.linalg.vector_norm(tokens, dim=-1)
        return (norms * weights).sum() / weights.sum().clamp_min(1e-6)

    def forward(
        self,
        source_feat: torch.Tensor,
        source_coord_map_mm: torch.Tensor,
        source_scale_mm: float | int | torch.Tensor,
        target_scale_mm: float | int | torch.Tensor,
        target_grid_hw: tuple[int, int] | None = None,
        target_coord_map_mm: torch.Tensor | None = None,
        source_valid_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_feat_flat, source_feat_hw = self._flatten_feat('source_feat', source_feat)
        source_coords_flat, source_coord_hw = self._flatten_coords('source_coord_map_mm', source_coord_map_mm)
        b, ns, d = source_feat_flat.shape
        if d != self.embed_dim:
            raise ValueError(f'source_feat embed dim mismatch: expected {self.embed_dim}, got {d}')
        if source_coords_flat.shape[:2] != (b, ns):
            raise ValueError(
                f'source_coord_map_mm must align with source_feat tokens: '
                f'{tuple(source_coords_flat.shape[:2])} vs {(b, ns)}'
            )

        source_hw = self._resolve_hw(source_feat_hw, source_coord_hw, ns, 'source')
        hs, ws = source_hw

        source_valid_flat = self._flatten_mask('source_valid_mask', source_valid_mask, b, ns)
        if source_valid_flat is None:
            source_valid_flat = torch.ones((b, ns), device=source_feat_flat.device, dtype=torch.bool)

        device = source_feat_flat.device
        dtype = source_feat_flat.dtype
        source_coords_flat = source_coords_flat.to(device=device, dtype=dtype)
        source_valid_flat = source_valid_flat.to(device=device)
        source_scale_mm_t = self._prepare_scale_tensor(source_scale_mm, b, device, dtype, 'source_scale_mm')
        target_scale_mm_t = self._prepare_scale_tensor(target_scale_mm, b, device, dtype, 'target_scale_mm')

        if target_coord_map_mm is not None:
            target_coords_flat, target_coord_hw = self._flatten_coords('target_coord_map_mm', target_coord_map_mm)
            if target_coords_flat.shape[0] != b:
                raise ValueError(
                    f'target_coord_map_mm batch mismatch: expected {b}, got {target_coords_flat.shape[0]}'
                )
            target_hw = target_coord_hw or target_grid_hw or self._infer_hw(target_coords_flat.shape[1], 'target')
        else:
            if target_grid_hw is None:
                raise ValueError('Either target_grid_hw or target_coord_map_mm must be provided')
            target_hw = target_grid_hw
            target_coords_mm = self._build_uniform_coord_map(target_hw, target_scale_mm_t, device, dtype)
            target_coords_flat = target_coords_mm.reshape(b, target_hw[0] * target_hw[1], 2)

        ht, wt = target_hw
        nt = ht * wt
        if target_coords_flat.shape[1] != nt:
            raise ValueError(
                f'target token count mismatch: expected {nt}, got {target_coords_flat.shape[1]}'
            )
        target_coords_flat = target_coords_flat.to(device=device, dtype=dtype)

        source_coords_norm = self._normalize_coords_with_scale(source_coords_flat, source_scale_mm_t)
        target_coords_norm = self._normalize_coords_with_scale(target_coords_flat, target_scale_mm_t)

        anchor_flat, anchor_conf = self._metric_knn_anchor(
            source_feat_flat=source_feat_flat,
            source_coords_norm=source_coords_norm,
            target_coords_norm=target_coords_norm,
            source_valid_flat=source_valid_flat,
            k=4,
        )
        anchor_feat = anchor_flat.view(b, ht, wt, d)

        bbox_support = self._metric_bbox_support(
            source_coords_flat=source_coords_flat,
            target_coords_flat=target_coords_flat,
            source_valid_flat=source_valid_flat,
        )
        density_support = torch.clamp(source_scale_mm_t / target_scale_mm_t.clamp_min(1e-6), max=1.0)
        support_flat = bbox_support.to(dtype=dtype) * anchor_conf * density_support.view(b, 1)
        support_flat = support_flat.clamp(0.0, 1.0)
        support_map = support_flat.view(b, ht, wt, 1)
        support_valid_mask = support_flat > 1e-6

        ratio_embed, ratio = self.ratio_encoder(source_scale_mm_t, target_scale_mm_t)
        source_pos = self.coord_encoder(source_coords_norm)
        target_pos = self.coord_encoder(target_coords_norm)

        source_key = self.source_key_norm(source_feat_flat + source_pos)
        source_value = source_feat_flat
        query_state = anchor_flat + target_pos + ratio_embed.unsqueeze(1) + self.query_bias
        query_state = query_state.masked_fill(~support_valid_mask.unsqueeze(-1), 0.0)

        source_key_padding_mask = ~source_valid_flat
        for layer in self.layers:
            query_state = layer(
                query_state=query_state,
                source_key=source_key,
                source_value=source_value,
                query_coords_norm=target_coords_norm,
                source_coords_norm=source_coords_norm,
                target_hw=target_hw,
                source_key_padding_mask=source_key_padding_mask,
                query_valid_mask=support_valid_mask,
            )

        if self.final_norm is not None:
            query_state = self.final_norm(query_state)
        query_state = query_state.masked_fill(~support_valid_mask.unsqueeze(-1), 0.0)
        residual_feat = query_state

        corrected_flat = anchor_flat + support_flat.unsqueeze(-1) * residual_feat

        pooled_source = self._weighted_mean_pool(
            source_feat_flat,
            source_valid_flat.to(dtype=dtype),
        )
        film_in = torch.cat([pooled_source, ratio_embed], dim=-1)
        gamma, beta = self.modulation_mlp(film_in).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        corrected_mod = (1.0 + gamma.unsqueeze(1)) * corrected_flat + beta.unsqueeze(1)

        mix = torch.softmax(self.mix_logits.to(device=device, dtype=dtype), dim=0)
        alpha_anchor = mix[0]
        alpha_corrected = mix[1]
        output_flat = alpha_anchor * anchor_flat + alpha_corrected * corrected_mod
        output_flat = output_flat.masked_fill(~support_valid_mask.unsqueeze(-1), 0.0)
        output = output_flat.view(b, ht, wt, d)

        if not return_aux:
            return output

        aux: dict[str, torch.Tensor] = {
            'anchor_feat': anchor_feat,
            'corrected_feat': corrected_mod.view(b, ht, wt, d),
            'residual_feat': residual_feat.view(b, ht, wt, d),
            'support_map': support_map.contiguous(),
            'support_valid_mask': support_valid_mask.view(b, ht, wt, 1),
            'support_mean': support_flat.mean() if support_flat.numel() > 0 else output_flat.new_zeros(()),
            'alpha_anchor': alpha_anchor,
            'alpha_corrected': alpha_corrected,
            'gamma_mean': gamma.mean() if gamma.numel() > 0 else output_flat.new_zeros(()),
            'beta_mean': beta.mean() if beta.numel() > 0 else output_flat.new_zeros(()),
            'source_scale_mm': source_scale_mm_t.mean(),
            'target_scale_mm': target_scale_mm_t.mean(),
            'ratio_mean': ratio.mean(),
            'anchor_norm': self._weighted_mean_token_norm(anchor_flat, support_flat),
            'residual_norm': self._weighted_mean_token_norm(residual_feat, support_flat),
            'output_norm': self._weighted_mean_token_norm(output_flat, support_flat),
        }
        return output, aux
