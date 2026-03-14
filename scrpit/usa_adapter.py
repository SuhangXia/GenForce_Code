import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierCoordinateEncoder(nn.Module):
    """Encode 2D coordinates with Fourier features + MLP."""

    def __init__(
        self,
        embed_dim: int = 768,
        coord_dim: int = 2,
        num_frequencies: int = 4,
        include_input: bool = True,
        hidden_dim: int | None = None,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        if coord_dim != 2:
            raise ValueError(f"coord_dim must be 2, got {coord_dim}")
        if num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be > 0, got {num_frequencies}")
        if coord_scale <= 0:
            raise ValueError(f"coord_scale must be > 0, got {coord_scale}")

        self.coord_dim = int(coord_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.coord_scale = float(coord_scale)

        self.register_buffer(
            "freq_bands",
            (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)),
            persistent=False,
        )

        fourier_dim = self.coord_dim * self.num_frequencies * 2
        raw_dim = self.coord_dim if self.include_input else 0
        in_dim = raw_dim + fourier_dim

        if hidden_dim is None:
            hidden_dim = max(embed_dim // 2, 64)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"coords last dim must be {self.coord_dim}, got shape {tuple(coords.shape)}"
            )

        coords = coords / self.coord_scale
        freq = self.freq_bands.to(device=coords.device, dtype=coords.dtype)

        phase = coords.unsqueeze(-1) * freq
        phase = phase * (2.0 * math.pi)

        pieces = []
        if self.include_input:
            pieces.append(coords)
        pieces.append(torch.sin(phase).reshape(*coords.shape[:-1], -1))
        pieces.append(torch.cos(phase).reshape(*coords.shape[:-1], -1))

        enc = torch.cat(pieces, dim=-1)
        return self.proj(enc)


class RelativeCoordBias(nn.Module):
    """Map relative 2D offsets to per-head attention bias."""

    def __init__(self, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, delta_coords: torch.Tensor) -> torch.Tensor:
        """
        delta_coords: (B, Nq, Nc, 2)
        returns:      (B, H, Nq, Nc)
        """
        bias = self.mlp(delta_coords)  # (B, Nq, Nc, H)
        return bias.permute(0, 3, 1, 2).contiguous()


class CrossAttentionWithBias(nn.Module):
    """Custom cross-attention so we can inject relative coordinate bias."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
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
        query: torch.Tensor,          # (B, Nq, D)
        key: torch.Tensor,            # (B, Nc, D)
        value: torch.Tensor,          # (B, Nc, D)
        attn_bias: torch.Tensor | None = None,          # (B, H, Nq, Nc)
        key_padding_mask: torch.Tensor | None = None,   # (B, Nc), True means PAD/invalid
    ) -> torch.Tensor:
        b, nq, d = query.shape
        _, nc, _ = key.shape

        q = self.q_proj(query).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,Nq,Dh
        k = self.k_proj(key).view(b, nc, self.num_heads, self.head_dim).transpose(1, 2)    # B,H,Nc,Dh
        v = self.v_proj(value).view(b, nc, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,Nc,Dh

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B,H,Nq,Nc

        if attn_bias is not None:
            logits = logits + attn_bias

        if key_padding_mask is not None:
            if key_padding_mask.shape != (b, nc):
                raise ValueError(
                    f"key_padding_mask shape must be {(b, nc)}, got {tuple(key_padding_mask.shape)}"
                )
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # B,H,Nq,Dh
        out = out.transpose(1, 2).contiguous().view(b, nq, d)
        out = self.out_proj(out)
        return out


class USAQueryDecoderBlock(nn.Module):
    """Pre-norm decoder block: self-attn -> cross-attn(with rel bias) -> FFN."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_ratio: float = 4.0,
        rel_bias_hidden_dim: int = 64,
    ):
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
        self.rel_bias = RelativeCoordBias(
            num_heads=num_heads,
            hidden_dim=rel_bias_hidden_dim,
        )

        hidden_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_state: torch.Tensor,                # (B, Nq, D)
        context_k: torch.Tensor,                  # (B, Nc, D)
        context_v: torch.Tensor,                  # (B, Nc, D)
        query_coords_norm: torch.Tensor,          # (B, Nq, 2)
        context_coords_norm: torch.Tensor,        # (B, Nc, 2)
        context_key_padding_mask: torch.Tensor | None = None,   # (B, Nc)
        query_key_padding_mask: torch.Tensor | None = None,     # (B, Nq)
        query_valid_mask: torch.Tensor | None = None,           # (B, Nq), True=valid
    ) -> torch.Tensor:
        # Self-attention on query tokens
        q_norm = self.self_norm(query_state)
        self_out, _ = self.self_attn(
            query=q_norm,
            key=q_norm,
            value=q_norm,
            key_padding_mask=query_key_padding_mask,
            need_weights=False,
        )
        query_state = query_state + self.dropout(self_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        # Relative coordinate bias for cross-attention
        delta = query_coords_norm[:, :, None, :] - context_coords_norm[:, None, :, :]  # B,Nq,Nc,2
        rel_bias = self.rel_bias(delta)  # B,H,Nq,Nc

        # Cross-attention
        q_norm = self.cross_norm(query_state)
        cross_out = self.cross_attn(
            query=q_norm,
            key=context_k,
            value=context_v,
            attn_bias=rel_bias,
            key_padding_mask=context_key_padding_mask,
        )
        query_state = query_state + self.dropout(cross_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        # FFN
        ffn_out = self.ffn(self.ffn_norm(query_state))
        query_state = query_state + self.dropout(ffn_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        return query_state


class UniversalScaleAdapter(nn.Module):
    """
    USA v2:
    - coordinate-conditioned latent resampler
    - relative coordinate attention bias
    - kNN interpolation residual
    - scale token
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        coord_num_frequencies: int = 4,
        coord_include_input: bool = True,
        coord_scale_mm: float = 10.0,
        use_final_norm: bool = True,
        interp_k: int = 4,
        rel_bias_hidden_dim: int = 64,
        use_scale_token: bool = True,
        learnable_output_gate: bool = True,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if interp_k <= 0:
            raise ValueError(f"interp_k must be > 0, got {interp_k}")

        self.embed_dim = int(embed_dim)
        self.interp_k = int(interp_k)
        self.coord_scale_mm = float(coord_scale_mm)
        self.use_scale_token = bool(use_scale_token)

        self.coord_encoder = FourierCoordinateEncoder(
            embed_dim=self.embed_dim,
            coord_dim=2,
            num_frequencies=coord_num_frequencies,
            include_input=coord_include_input,
            coord_scale=1.0,  # coords are already normalized before entering encoder
        )

        self.query_bias = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.use_scale_token:
            self.scale_mlp = nn.Sequential(
                nn.Linear(2, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.scale_mlp = None

        self.layers = nn.ModuleList(
            [
                USAQueryDecoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_ratio=4.0,
                    rel_bias_hidden_dim=rel_bias_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.context_k_norm = nn.LayerNorm(self.embed_dim)
        self.use_final_norm = bool(use_final_norm)
        self.final_norm = nn.LayerNorm(self.embed_dim) if self.use_final_norm else None

        if learnable_output_gate:
            self.interp_gate = nn.Parameter(torch.tensor(1.0))
            self.adapter_gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("interp_gate", torch.tensor(1.0), persistent=False)
            self.register_buffer("adapter_gate", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _flatten_feat(name: str, feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if feat.dim() == 4:
            b, h, w, d = feat.shape
            return feat.reshape(b, h * w, d), (h, w)
        if feat.dim() == 3:
            return feat, None
        raise ValueError(f"{name} must be 3D (B,N,D) or 4D (B,H,W,D), got shape {tuple(feat.shape)}")

    @staticmethod
    def _flatten_coords(name: str, coords: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if coords.shape[-1] != 2:
            raise ValueError(f"{name} last dim must be 2, got shape {tuple(coords.shape)}")
        if coords.dim() == 4:
            b, h, w, _ = coords.shape
            return coords.reshape(b, h * w, 2), (h, w)
        if coords.dim() == 3:
            return coords, None
        raise ValueError(f"{name} must be 3D (B,N,2) or 4D (B,H,W,2), got shape {tuple(coords.shape)}")

    @staticmethod
    def _flatten_mask(name: str, mask: torch.Tensor | None, batch: int, tokens: int) -> torch.Tensor | None:
        if mask is None:
            return None

        if mask.dim() == 3:
            b, h, w = mask.shape
            mask = mask.reshape(b, h * w)
        elif mask.dim() != 2:
            raise ValueError(f"{name} must be 2D (B,N) or 3D (B,H,W), got shape {tuple(mask.shape)}")

        if mask.shape[0] != batch:
            raise ValueError(f"{name} batch mismatch: expected {batch}, got {mask.shape[0]}")
        if mask.shape[1] != tokens:
            raise ValueError(f"{name} token mismatch: expected {tokens}, got {mask.shape[1]}")

        return mask.to(dtype=torch.bool)

    @staticmethod
    def _prepare_scale_tensor(
        scale_mm: float | int | torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor | None:
        if scale_mm is None:
            return None

        if isinstance(scale_mm, (float, int)):
            out = torch.full((batch, 1), float(scale_mm), device=device, dtype=dtype)
            return out

        if not isinstance(scale_mm, torch.Tensor):
            raise TypeError(f"{name} must be float/int/tensor/None, got {type(scale_mm)}")

        t = scale_mm.to(device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(batch).reshape(batch, 1)
        elif t.dim() == 1:
            if t.shape[0] != batch:
                raise ValueError(f"{name} length must be {batch}, got {t.shape[0]}")
            t = t.reshape(batch, 1)
        elif t.dim() == 2 and t.shape == (batch, 1):
            pass
        else:
            raise ValueError(f"{name} must have shape (), (B,), or (B,1), got {tuple(t.shape)}")
        return t

    @staticmethod
    def _normalize_coords_with_scale(
        coords_mm: torch.Tensor,          # (B,N,2)
        scale_mm: torch.Tensor | None,    # (B,1) or None
        fallback_scale_mm: float,
    ) -> torch.Tensor:
        if scale_mm is None:
            return coords_mm / float(fallback_scale_mm)

        denom = (scale_mm / 2.0).clamp_min(1e-6)  # half-width/half-height
        return coords_mm / denom.unsqueeze(-1)

    @staticmethod
    def _knn_interpolate(
        context_feat: torch.Tensor,              # (B,Nc,D)
        context_coords_mm: torch.Tensor,         # (B,Nc,2)
        query_coords_mm: torch.Tensor,           # (B,Nq,2)
        context_valid_mask: torch.Tensor | None, # (B,Nc), True=valid
        k: int,
    ) -> torch.Tensor:
        b, nc, d = context_feat.shape
        _, nq, _ = query_coords_mm.shape

        delta = query_coords_mm[:, :, None, :] - context_coords_mm[:, None, :, :]  # (B,Nq,Nc,2)
        dist2 = (delta ** 2).sum(dim=-1)  # (B,Nq,Nc)

        if context_valid_mask is not None:
            invalid = ~context_valid_mask[:, None, :]  # (B,1,Nc)
            dist2 = dist2.masked_fill(invalid, float("inf"))

        k_eff = min(k, nc)
        knn_dist2, knn_idx = torch.topk(dist2, k=k_eff, dim=-1, largest=False)  # (B,Nq,K)

        knn_feat = torch.gather(
            context_feat[:, None, :, :].expand(b, nq, nc, d),
            dim=2,
            index=knn_idx[..., None].expand(b, nq, k_eff, d),
        )  # (B,Nq,K,D)

        valid_knn = torch.isfinite(knn_dist2)
        inv_dist = torch.where(valid_knn, 1.0 / (knn_dist2 + 1e-8), torch.zeros_like(knn_dist2))
        weight_sum = inv_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = inv_dist / weight_sum

        out = (weights[..., None] * knn_feat).sum(dim=2)  # (B,Nq,D)

        # If some query has zero valid neighbors, make it zero.
        has_any = valid_knn.any(dim=-1, keepdim=True)
        out = torch.where(has_any, out, torch.zeros_like(out))
        return out

    def forward(
        self,
        context_feat: torch.Tensor | None = None,
        context_coord_map_mm: torch.Tensor | None = None,
        query_coord_map_mm: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        context_scale_mm: float | int | torch.Tensor | None = None,
        query_scale_mm: float | int | torch.Tensor | None = None,
        **legacy_kwargs,
    ) -> torch.Tensor:
        """
        Generic direction:
            context_feat/context_coord_map_mm  ->  query_coord_map_mm

        If you want target-size -> source-size adaptation:
            context = target
            query   = source

        Args:
            context_feat:         (B,Hc,Wc,D) or (B,Nc,D)
            context_coord_map_mm: (B,Hc,Wc,2) or (B,Nc,2)
            query_coord_map_mm:   (B,Hq,Wq,2) or (B,Nq,2)
            context_valid_mask:   optional bool mask, True=valid
            context_scale_mm:     sensor size for context branch
            query_scale_mm:       sensor size for query branch
        Returns:
            (B,Hq,Wq,D) if query_coord_map_mm is 4D, else (B,Nq,D)
        """
        used_legacy = False
        if legacy_kwargs:
            if context_feat is None and "source_feat" in legacy_kwargs:
                context_feat = legacy_kwargs.pop("source_feat")
                used_legacy = True
            if context_coord_map_mm is None and "source_coords" in legacy_kwargs:
                context_coord_map_mm = legacy_kwargs.pop("source_coords")
                used_legacy = True
            if query_coord_map_mm is None and "target_coords" in legacy_kwargs:
                query_coord_map_mm = legacy_kwargs.pop("target_coords")
                used_legacy = True
            if context_scale_mm is None and "source_scale_mm" in legacy_kwargs:
                context_scale_mm = legacy_kwargs.pop("source_scale_mm")
                used_legacy = True
            if query_scale_mm is None and "target_scale_mm" in legacy_kwargs:
                query_scale_mm = legacy_kwargs.pop("target_scale_mm")
                used_legacy = True

            if legacy_kwargs:
                unknown = ", ".join(sorted(legacy_kwargs.keys()))
                raise TypeError(f"Unknown keyword arguments: {unknown}")

        if used_legacy:
            warnings.warn(
                "Using legacy args (source_feat/source_coords/target_coords). "
                "Please switch to context_feat/context_coord_map_mm/query_coord_map_mm.",
                stacklevel=2,
            )

        if context_feat is None or context_coord_map_mm is None or query_coord_map_mm is None:
            raise ValueError("context_feat, context_coord_map_mm, and query_coord_map_mm are required")

        context_feat_flat, _ = self._flatten_feat("context_feat", context_feat)
        context_coords_flat, _ = self._flatten_coords("context_coord_map_mm", context_coord_map_mm)
        query_coords_flat, query_hw = self._flatten_coords("query_coord_map_mm", query_coord_map_mm)

        b, nc, d = context_feat_flat.shape
        if d != self.embed_dim:
            raise ValueError(f"context_feat embed dim mismatch: expected {self.embed_dim}, got {d}")

        if context_coords_flat.shape[0] != b:
            raise ValueError(
                f"Batch mismatch: context_feat batch={b}, context_coord_map_mm batch={context_coords_flat.shape[0]}"
            )
        if query_coords_flat.shape[0] != b:
            raise ValueError(
                f"Batch mismatch: context_feat batch={b}, query_coord_map_mm batch={query_coords_flat.shape[0]}"
            )
        if context_coords_flat.shape[1] != nc:
            raise ValueError(
                f"Token mismatch: context_feat tokens={nc}, context_coord_map_mm tokens={context_coords_flat.shape[1]}"
            )

        nq = query_coords_flat.shape[1]
        context_valid_flat = self._flatten_mask("context_valid_mask", context_valid_mask, b, nc)

        device = context_feat_flat.device
        dtype = context_feat_flat.dtype
        context_coords_flat = context_coords_flat.to(device=device, dtype=dtype)
        query_coords_flat = query_coords_flat.to(device=device, dtype=dtype)

        if context_valid_flat is not None:
            context_valid_flat = context_valid_flat.to(device=device)

        context_scale_mm_t = self._prepare_scale_tensor(
            context_scale_mm, b, device, dtype, "context_scale_mm"
        )
        query_scale_mm_t = self._prepare_scale_tensor(
            query_scale_mm, b, device, dtype, "query_scale_mm"
        )

        # Valid region check in raw mm space
        eps = 1e-4
        if context_valid_flat is None:
            min_bounds = context_coords_flat.min(dim=1, keepdim=True).values
            max_bounds = context_coords_flat.max(dim=1, keepdim=True).values
        else:
            if not torch.all(context_valid_flat.any(dim=1)):
                raise ValueError("context_valid_mask leaves at least one batch element with zero valid tokens")

            valid_context = context_valid_flat.unsqueeze(-1)
            coord_max = torch.finfo(dtype).max
            coord_min = torch.finfo(dtype).min

            min_bounds = torch.where(
                valid_context,
                context_coords_flat,
                torch.full_like(context_coords_flat, coord_max),
            ).min(dim=1, keepdim=True).values

            max_bounds = torch.where(
                valid_context,
                context_coords_flat,
                torch.full_like(context_coords_flat, coord_min),
            ).max(dim=1, keepdim=True).values

        is_valid = (
            (query_coords_flat >= (min_bounds - eps))
            & (query_coords_flat <= (max_bounds + eps))
        )
        query_valid_flat = is_valid[..., 0] & is_valid[..., 1]

        # Normalize coordinates for encoding / relative bias
        context_coords_norm = self._normalize_coords_with_scale(
            context_coords_flat, context_scale_mm_t, self.coord_scale_mm
        )
        query_coords_norm = self._normalize_coords_with_scale(
            query_coords_flat, query_scale_mm_t, self.coord_scale_mm
        )

        # Interpolation residual baseline
        interp_feat = self._knn_interpolate(
            context_feat=context_feat_flat,
            context_coords_mm=context_coords_flat,
            query_coords_mm=query_coords_flat,
            context_valid_mask=context_valid_flat,
            k=self.interp_k,
        )
        interp_feat = interp_feat.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        # Coordinate encodings
        context_pos_enc = self.coord_encoder(context_coords_norm)
        query_pos_enc = self.coord_encoder(query_coords_norm)

        # Scale token
        scale_token = 0.0
        if self.use_scale_token and (context_scale_mm_t is not None) and (query_scale_mm_t is not None):
            scale_pair = torch.cat([context_scale_mm_t, query_scale_mm_t], dim=-1)  # (B,2)
            scale_token = self.scale_mlp(scale_pair).unsqueeze(1)  # (B,1,D)

        # Query initialization = coord encoding + interpolation prior + scale token
        query_state = query_pos_enc + self.query_bias + interp_feat + scale_token
        query_state = query_state.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        # Context for cross-attention
        context_k = self.context_k_norm(context_feat_flat + context_pos_enc)
        context_v = context_feat_flat

        context_key_padding_mask = None if context_valid_flat is None else (~context_valid_flat)
        query_key_padding_mask = ~query_valid_flat

        for layer in self.layers:
            query_state = layer(
                query_state=query_state,
                context_k=context_k,
                context_v=context_v,
                query_coords_norm=query_coords_norm,
                context_coords_norm=context_coords_norm,
                context_key_padding_mask=context_key_padding_mask,
                query_key_padding_mask=query_key_padding_mask,
                query_valid_mask=query_valid_flat,
            )

        if self.final_norm is not None:
            query_state = self.final_norm(query_state)

        query_state = query_state.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        # Final output = interpolation prior + learned residual
        out = self.interp_gate * interp_feat + self.adapter_gate * query_state
        out = out.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        if query_hw is not None:
            hq, wq = query_hw
            return out.reshape(b, hq, wq, self.embed_dim)
        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    b, d = 2, 128
    model = UniversalScaleAdapter(
        embed_dim=d,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        coord_num_frequencies=4,
        coord_scale_mm=10.0,
        interp_k=4,
        use_scale_token=True,
        use_final_norm=True,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", params)

    # Example: context=25mm target sensor, query=15mm source sensor
    context_feat = torch.randn(b, 14, 14, d)
    context_coords = torch.randn(b, 14, 14, 2)
    query_coords = torch.randn(b, 14, 14, 2)

    out = model(
        context_feat=context_feat,
        context_coord_map_mm=context_coords,
        query_coord_map_mm=query_coords,
        context_scale_mm=torch.tensor([25.0, 25.0]),
        query_scale_mm=torch.tensor([15.0, 15.0]),
    )
    print("context 14x14 -> query 14x14:", tuple(out.shape))

    # Different token grid
    query_coords_16 = torch.randn(b, 16, 16, 2)
    out_16 = model(
        context_feat=context_feat,
        context_coord_map_mm=context_coords,
        query_coord_map_mm=query_coords_16,
        context_scale_mm=25.0,
        query_scale_mm=20.0,
    )
    print("context 14x14 -> query 16x16:", tuple(out_16.shape))

    # Flattened
    context_feat_flat = context_feat.reshape(b, 14 * 14, d)
    context_coords_flat = context_coords.reshape(b, 14 * 14, 2)
    query_coords_flat = query_coords_16.reshape(b, 16 * 16, 2)
    out_flat = model(
        context_feat=context_feat_flat,
        context_coord_map_mm=context_coords_flat,
        query_coord_map_mm=query_coords_flat,
        context_scale_mm=25.0,
        query_scale_mm=20.0,
    )
    print("flattened:", tuple(out_flat.shape))