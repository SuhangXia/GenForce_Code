from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from standalone_qformer.qformer_adapter import (
        DEFAULT_EMBED_DIM,
        DEFAULT_NUM_QUERIES,
        ScaleConditionedQFormer,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from qformer_adapter import DEFAULT_EMBED_DIM, DEFAULT_NUM_QUERIES, ScaleConditionedQFormer


def strip_cls_token_if_present(features: torch.Tensor) -> torch.Tensor:
    """Normalize ViT token output to exactly (B, 196, 768)."""
    if features.dim() != 3:
        raise ValueError(
            'Expected ViT features to have shape (B, N, D), '
            f'got {tuple(features.shape)}'
        )

    if features.shape[1] == 197:
        features = features[:, 1:, :]
    elif features.shape[1] != DEFAULT_NUM_QUERIES:
        raise ValueError(
            'Expected either 197 tokens (CLS + 196 patches) or 196 patch tokens, '
            f'got {tuple(features.shape)}'
        )

    if features.shape[-1] != DEFAULT_EMBED_DIM:
        raise ValueError(
            f'Expected token embedding dim {DEFAULT_EMBED_DIM}, got {features.shape[-1]}'
        )
    return features.contiguous()


def _extract_qformer_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if 'qformer_state_dict' in payload:
            state_dict = payload['qformer_state_dict']
        elif 'model_state_dict' in payload:
            state_dict = payload['model_state_dict']
        else:
            state_dict = payload
        if isinstance(state_dict, dict):
            return state_dict
    raise ValueError('Checkpoint does not contain a valid Q-Former state dict')


def _infer_qformer_kwargs(
    state_dict: dict[str, torch.Tensor],
    checkpoint_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_config = checkpoint_config or {}
    learnable_queries = state_dict.get('learnable_queries')
    if learnable_queries is None or learnable_queries.dim() != 3:
        raise KeyError('Q-Former state dict is missing learnable_queries')

    num_queries = int(learnable_queries.shape[1])
    embed_dim = int(learnable_queries.shape[2])
    layer_ids = sorted(
        {
            int(key.split('.')[2])
            for key in state_dict
            if key.startswith('decoder.layers.')
        }
    )
    if not layer_ids:
        raise ValueError('Could not infer decoder depth from Q-Former state dict')

    num_layers = int(checkpoint_config.get('num_layers', max(layer_ids) + 1))
    num_heads = int(checkpoint_config.get('num_heads', 8))
    dropout = float(checkpoint_config.get('dropout', 0.0))
    scale_hidden_dim = int(
        checkpoint_config.get(
            'scale_hidden_dim',
            state_dict['scale_embedding.mlp.0.weight'].shape[0],
        )
    )
    ffn_hidden = int(state_dict['decoder.layers.0.linear1.weight'].shape[0])
    mlp_ratio = float(checkpoint_config.get('mlp_ratio', ffn_hidden / max(embed_dim, 1)))

    return {
        'embed_dim': embed_dim,
        'num_queries': num_queries,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'dropout': dropout,
        'scale_hidden_dim': scale_hidden_dim,
    }


def load_qformer_from_checkpoint(
    checkpoint_reference: str | Path | dict[str, Any],
    device: torch.device | str = 'cpu',
    *,
    freeze: bool = True,
) -> tuple[ScaleConditionedQFormer, dict[str, Any]]:
    if isinstance(checkpoint_reference, (str, Path)):
        payload = torch.load(
            Path(checkpoint_reference).expanduser(),
            map_location='cpu',
            weights_only=False,
        )
    elif isinstance(checkpoint_reference, dict):
        payload = checkpoint_reference
    else:
        raise TypeError(
            f'checkpoint_reference must be a path or dict, got {type(checkpoint_reference)}'
        )

    state_dict = _extract_qformer_state_dict(payload)
    checkpoint_config = payload.get('config', {}) if isinstance(payload, dict) else {}
    constructor_kwargs = _infer_qformer_kwargs(state_dict, checkpoint_config)
    qformer = ScaleConditionedQFormer(**constructor_kwargs).to(device)
    qformer.load_state_dict(state_dict)
    qformer.eval()
    if freeze:
        for param in qformer.parameters():
            param.requires_grad = False

    metadata = {
        'constructor_kwargs': constructor_kwargs,
        'checkpoint_config': checkpoint_config,
        'epoch': payload.get('epoch') if isinstance(payload, dict) else None,
    }
    return qformer, metadata


class QFormerPluginAdapter(nn.Module):
    """Optional adapter to preserve compatibility with legacy plugin-style code."""

    def __init__(self, qformer: ScaleConditionedQFormer) -> None:
        super().__init__()
        self.qformer = qformer

    @staticmethod
    def _flatten_context(context_feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if context_feat.dim() == 4:
            batch_size, height, width, channels = context_feat.shape
            if channels != DEFAULT_EMBED_DIM:
                raise ValueError(
                    f'Expected context_feat channels={DEFAULT_EMBED_DIM}, got {channels}'
                )
            return context_feat.reshape(batch_size, height * width, channels), (height, width)
        if context_feat.dim() == 3:
            return context_feat, None
        raise ValueError(
            'context_feat must be (B, N, D) or (B, H, W, D), '
            f'got {tuple(context_feat.shape)}'
        )

    def forward(
        self,
        context_feat: torch.Tensor,
        context_coord_map_mm: torch.Tensor | None = None,
        query_coord_map_mm: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        context_scale_mm: float | int | torch.Tensor | None = None,
        query_scale_mm: float | int | torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        del context_coord_map_mm, query_coord_map_mm, context_valid_mask
        flat_context, hw = self._flatten_context(context_feat)
        source_scale = 20.0 if context_scale_mm is None else context_scale_mm
        target_scale = 20.0 if query_scale_mm is None else query_scale_mm
        adapted = self.qformer(
            flat_context,
            source_scale_mm=source_scale,
            target_scale_mm=target_scale,
        )
        if hw is None:
            return adapted
        batch_size = adapted.shape[0]
        height, width = hw
        return adapted.reshape(batch_size, height, width, DEFAULT_EMBED_DIM).contiguous()


def build_frozen_qformer_plugin(
    checkpoint_reference: str | Path | dict[str, Any],
    device: torch.device | str,
) -> tuple[nn.Module, dict[str, Any]]:
    qformer, metadata = load_qformer_from_checkpoint(
        checkpoint_reference,
        device=device,
        freeze=True,
    )
    plugin = QFormerPluginAdapter(qformer).to(device)
    plugin.eval()
    for param in plugin.parameters():
        param.requires_grad = False
    metadata = {**metadata, 'adapter_kind': 'scale_conditioned_qformer'}
    return plugin, metadata


class ModelWithQFormer(nn.Module):
    """Simple plug-and-play wrapper: image -> frozen ViT -> Q-Former -> flatten -> regressor.

    The downstream regressor is assumed to consume flattened token features with shape
    (B, 196 * 768). This keeps the wrapper fully external and avoids modifying any
    existing ViT or downstream model definitions.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        qformer: ScaleConditionedQFormer,
        downstream_regressor: nn.Module,
        *,
        freeze_modules: bool = True,
    ) -> None:
        super().__init__()
        self.vit_model = vit_model
        self.qformer = qformer
        self.downstream_regressor = downstream_regressor
        self.freeze_modules = bool(freeze_modules)

        if self.freeze_modules:
            self._freeze_module(self.vit_model)
            self._freeze_module(self.qformer)
            self._freeze_module(self.downstream_regressor)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _extract_vit_tokens(self, img: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vit_model, 'forward_features'):
            features = self.vit_model.forward_features(img)
        else:
            features = self.vit_model(img)
        return strip_cls_token_if_present(features)

    def _forward_impl(
        self,
        img: torch.Tensor,
        source_scale_mm: float | int | torch.Tensor,
        target_scale_mm: float | int | torch.Tensor,
    ) -> torch.Tensor:
        vit_tokens = self._extract_vit_tokens(img)
        adapted_tokens = self.qformer(
            vit_tokens,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
        )
        flat_features = adapted_tokens.reshape(adapted_tokens.shape[0], -1).contiguous()
        return self.downstream_regressor(flat_features)

    def forward(
        self,
        img: torch.Tensor,
        source_scale_mm: float | int | torch.Tensor,
        target_scale_mm: float | int | torch.Tensor = 20.0,
    ) -> torch.Tensor:
        if img.dim() != 4:
            raise ValueError(f'img must have shape (B, C, H, W), got {tuple(img.shape)}')

        if self.freeze_modules:
            with torch.no_grad():
                return self._forward_impl(img, source_scale_mm, target_scale_mm)
        return self._forward_impl(img, source_scale_mm, target_scale_mm)
