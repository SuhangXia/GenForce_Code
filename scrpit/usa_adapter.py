import warnings

import torch
import torch.nn as nn


class CoordinateEncoder(nn.Module):
    """Encode physical coordinates (X_mm, Y_mm) into model embedding space."""

    def __init__(self, coord_dim: int = 2, embed_dim: int = 768):
        super().__init__()
        hidden_dim = max(embed_dim // 2, 32)
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class USACrossAttentionRefineBlock(nn.Module):
    """Decoder-style cross-attention block for query latent refinement."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_state: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_in = self.norm_q(query_state)
        attn_out, _ = self.cross_attn(
            query=attn_in,
            key=context_k,
            value=context_v,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )
        query_state = query_state + self.dropout(attn_out)
        query_state = query_state + self.dropout(self.ffn(self.norm_ffn(query_state)))
        return query_state


class UniversalScaleAdapter(nn.Module):
    """
    Coordinate-conditioned latent resampler:
        (context_feat, context_coord_map_mm, query_coord_map_mm) -> query_feat
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.coord_encoder = CoordinateEncoder(coord_dim=2, embed_dim=embed_dim)
        self.layers = nn.ModuleList(
            [USACrossAttentionRefineBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.context_k_norm = nn.LayerNorm(embed_dim)

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
        query_valid_mask: torch.Tensor | None = None,
        **legacy_kwargs,
    ) -> torch.Tensor:
        """
        Args:
            context_feat:
                (B,Hc,Wc,D) or (B,Nc,D)
            context_coord_map_mm:
                (B,Hc,Wc,2) or (B,Nc,2)
            query_coord_map_mm:
                (B,Hq,Wq,2) or (B,Nq,2)
            context_valid_mask (optional):
                bool mask, True=valid; shape (B,Hc,Wc) or (B,Nc)
            query_valid_mask (optional):
                bool mask, True=valid; shape (B,Hq,Wq) or (B,Nq)
        Returns:
            Query-aligned features:
                (B,Hq,Wq,D) if query_coord_map_mm is 4D, else (B,Nq,D)
        """
        if legacy_kwargs:
            if context_feat is None and "source_feat" in legacy_kwargs:
                context_feat = legacy_kwargs.pop("source_feat")
            if context_coord_map_mm is None and "source_coords" in legacy_kwargs:
                context_coord_map_mm = legacy_kwargs.pop("source_coords")
            if query_coord_map_mm is None and "target_coords" in legacy_kwargs:
                query_coord_map_mm = legacy_kwargs.pop("target_coords")
            if legacy_kwargs:
                unknown = ", ".join(sorted(legacy_kwargs.keys()))
                raise TypeError(f"Unknown keyword arguments: {unknown}")
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
        query_valid_flat = self._flatten_mask("query_valid_mask", query_valid_mask, b, nq)

        context_pos_enc = self.coord_encoder(context_coords_flat)
        query_state = self.coord_encoder(query_coords_flat)

        context_k = self.context_k_norm(context_feat_flat + context_pos_enc)
        context_v = context_feat_flat
        context_key_padding_mask = None if context_valid_flat is None else (~context_valid_flat)

        if query_valid_flat is not None:
            query_state = query_state.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        for layer in self.layers:
            query_state = layer(
                query_state=query_state,
                context_k=context_k,
                context_v=context_v,
                context_key_padding_mask=context_key_padding_mask,
            )
            if query_valid_flat is not None:
                query_state = query_state.masked_fill(~query_valid_flat.unsqueeze(-1), 0.0)

        if query_hw is not None:
            hq, wq = query_hw
            return query_state.reshape(b, hq, wq, self.embed_dim)
        return query_state


if __name__ == "__main__":
    torch.manual_seed(42)

    b, d = 2, 128
    model = UniversalScaleAdapter(embed_dim=d, num_heads=8, num_layers=2)

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

    # 3) flattened context/query tensors
    context_feat_flat = context_feat.reshape(b, 14 * 14, d)
    context_coords_flat = context_coords.reshape(b, 14 * 14, 2)
    query_coords_flat = query_coords_16.reshape(b, 16 * 16, 2)
    out_flat = model(
        context_feat=context_feat_flat,
        context_coord_map_mm=context_coords_flat,
        query_coord_map_mm=query_coords_flat,
    )
    print("case3 flattened:", tuple(out_flat.shape))
