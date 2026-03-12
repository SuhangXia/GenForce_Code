import math
import warnings

import torch
import torch.nn as nn


class FourierCoordinateEncoder(nn.Module):
    """Encode (X_mm, Y_mm) coordinates with compact Fourier features + MLP."""

    def __init__(
        self,
        embed_dim: int = 768,
        coord_dim: int = 2,
        num_frequencies: int = 4,
        include_input: bool = True,
        hidden_dim: int | None = None,
        coord_scale_mm: float = 10.0,
    ):
        super().__init__()
        if coord_dim != 2:
            raise ValueError(f"coord_dim must be 2, got {coord_dim}")
        if num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be > 0, got {num_frequencies}")
        if coord_scale_mm <= 0:
            raise ValueError(f"coord_scale_mm must be > 0, got {coord_scale_mm}")

        self.coord_dim = int(coord_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.coord_scale_mm = float(coord_scale_mm)

        # Modest frequency bands keep the encoder expressive without becoming heavy.
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

    def forward(self, coords_mm: torch.Tensor) -> torch.Tensor:
        if coords_mm.shape[-1] != self.coord_dim:
            raise ValueError(
                f"coords_mm last dim must be {self.coord_dim} for (X_mm, Y_mm), got shape {tuple(coords_mm.shape)}"
            )

        coords = coords_mm / self.coord_scale_mm
        freq = self.freq_bands.to(device=coords.device, dtype=coords.dtype)

        # (..., 2, F)
        phase = coords.unsqueeze(-1) * freq
        phase = phase * (2.0 * math.pi)

        pieces = []
        if self.include_input:
            pieces.append(coords)
        pieces.append(torch.sin(phase).reshape(*coords.shape[:-1], -1))
        pieces.append(torch.cos(phase).reshape(*coords.shape[:-1], -1))

        enc = torch.cat(pieces, dim=-1)
        return self.proj(enc)


class USAQueryDecoderBlock(nn.Module):
    """Pre-norm decoder block: self-attn -> cross-attn -> FFN."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_ratio: float = 4.0,
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
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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
        query_state: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
        query_key_padding_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q = q + SelfAttn(LN(q))
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

        # q = q + CrossAttn(LN(q), K, V)
        q_norm = self.cross_norm(query_state)
        cross_out, _ = self.cross_attn(
            query=q_norm,
            key=context_k,
            value=context_v,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )
        query_state = query_state + self.dropout(cross_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        # q = q + FFN(LN(q))
        ffn_out = self.ffn(self.ffn_norm(query_state))
        query_state = query_state + self.dropout(ffn_out)
        if query_valid_mask is not None:
            query_state = query_state.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

        return query_state


class UniversalScaleAdapter(nn.Module):
    """Lightweight coordinate-conditioned latent resampler for cross-sensor transfer."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        coord_num_frequencies: int = 4,
        coord_include_input: bool = True,
        coord_scale_mm: float = 10.0,
        use_final_norm: bool = False,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        self.embed_dim = int(embed_dim)

        self.coord_encoder = FourierCoordinateEncoder(
            embed_dim=self.embed_dim,
            coord_dim=2,
            num_frequencies=coord_num_frequencies,
            include_input=coord_include_input,
            coord_scale_mm=coord_scale_mm,
        )

        self.query_bias = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.layers = nn.ModuleList(
            [
                USAQueryDecoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_ratio=4.0,
                )
                for _ in range(num_layers)
            ]
        )

        self.context_k_norm = nn.LayerNorm(self.embed_dim)
        self.use_final_norm = bool(use_final_norm)
        self.final_norm = nn.LayerNorm(self.embed_dim) if self.use_final_norm else None

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
            raise ValueError(f"{name} last dim must be 2 for (X_mm, Y_mm), got shape {tuple(coords.shape)}")
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

    def forward(
        self,
        context_feat: torch.Tensor | None = None,
        context_coord_map_mm: torch.Tensor | None = None,
        query_coord_map_mm: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        **legacy_kwargs,
    ) -> torch.Tensor:
        """
        Args:
            context_feat: (B,Hc,Wc,D) or (B,Nc,D)
            context_coord_map_mm: (B,Hc,Wc,2) or (B,Nc,2)
            query_coord_map_mm: (B,Hq,Wq,2) or (B,Nq,2)
            context_valid_mask: optional bool mask, True=valid, shape (B,Hc,Wc) or (B,Nc)
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
                f"Batch mismatch: context_feat has batch {b}, context_coord_map_mm has batch {context_coords_flat.shape[0]}"
            )
        if query_coords_flat.shape[0] != b:
            raise ValueError(
                f"Batch mismatch: context_feat has batch {b}, query_coord_map_mm has batch {query_coords_flat.shape[0]}"
            )
        if context_coords_flat.shape[1] != nc:
            raise ValueError(
                "Token mismatch: context_feat token count and context_coord_map_mm token count differ "
                f"({nc} vs {context_coords_flat.shape[1]})"
            )

        nq = query_coords_flat.shape[1]

        context_valid_flat = self._flatten_mask("context_valid_mask", context_valid_mask, b, nc)

        device = context_feat_flat.device
        dtype = context_feat_flat.dtype
        context_coords_flat = context_coords_flat.to(device=device, dtype=dtype)
        query_coords_flat = query_coords_flat.to(device=device, dtype=dtype)

        if context_valid_flat is not None:
            context_valid_flat = context_valid_flat.to(device=device)

        eps = 1e-4
        if context_valid_flat is None:
            min_bounds = context_coords_flat.min(dim=1, keepdim=True).values
            max_bounds = context_coords_flat.max(dim=1, keepdim=True).values
        else:
            if not torch.all(context_valid_flat.any(dim=1)):
                raise ValueError("context_valid_mask leaves at least one batch element with zero valid context tokens")

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

        context_pos_enc = self.coord_encoder(context_coords_flat)
        query_pos_enc = self.coord_encoder(query_coords_flat)

        # Decoder-style query latent initialization.
        query_state = query_pos_enc + self.query_bias
        context_k = self.context_k_norm(context_feat_flat + context_pos_enc)
        context_v = context_feat_flat

        context_key_padding_mask = None if context_valid_flat is None else (~context_valid_flat)
        query_key_padding_mask = ~query_valid_flat
        query_state = query_state.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        for layer in self.layers:
            query_state = layer(
                query_state=query_state,
                context_k=context_k,
                context_v=context_v,
                context_key_padding_mask=context_key_padding_mask,
                query_key_padding_mask=query_key_padding_mask,
                query_valid_mask=query_valid_flat,
            )

        if self.final_norm is not None:
            query_state = self.final_norm(query_state)
        query_state = query_state.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        if query_hw is not None:
            hq, wq = query_hw
            return query_state.reshape(b, hq, wq, self.embed_dim)
        return query_state


if __name__ == "__main__":
    torch.manual_seed(42)

    b, d = 2, 128
    model = UniversalScaleAdapter(
        embed_dim=d,
        num_heads=8,
        num_layers=2,
        dropout=0.0,
        coord_num_frequencies=4,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", params)

    # 1) context 14x14 -> query 14x14
    context_feat = torch.randn(b, 14, 14, d)
    context_coords = torch.randn(b, 14, 14, 2)
    query_coords_14 = torch.randn(b, 14, 14, 2)
    out_14 = model(
        context_feat=context_feat,
        context_coord_map_mm=context_coords,
        query_coord_map_mm=query_coords_14,
    )
    print("case1 context 14x14 -> query 14x14:", tuple(out_14.shape))

    # 2) context 14x14 -> query 16x16
    query_coords_16 = torch.randn(b, 16, 16, 2)
    out_16 = model(
        context_feat=context_feat,
        context_coord_map_mm=context_coords,
        query_coord_map_mm=query_coords_16,
    )
    print("case2 context 14x14 -> query 16x16:", tuple(out_16.shape))

    # 3) flattened input tensors
    context_feat_flat = context_feat.reshape(b, 14 * 14, d)
    context_coords_flat = context_coords.reshape(b, 14 * 14, 2)
    query_coords_flat = query_coords_16.reshape(b, 16 * 16, 2)
    out_flat = model(
        context_feat=context_feat_flat,
        context_coord_map_mm=context_coords_flat,
        query_coord_map_mm=query_coords_flat,
    )
    print("case3 flattened:", tuple(out_flat.shape))
