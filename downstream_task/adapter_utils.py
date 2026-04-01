from __future__ import annotations

import torch
import torch.nn as nn

from dd_usa_adapter import DirectDriveUniversalScaleAdapter
from usa_adapter import UniversalScaleAdapter


class DDUSAPluginAdapter(nn.Module):
    """Wrap DD-USA with the legacy downstream plugin interface."""

    def __init__(self, adapter: DirectDriveUniversalScaleAdapter) -> None:
        super().__init__()
        self.adapter = adapter

    def forward(
        self,
        context_feat: torch.Tensor,
        context_coord_map_mm: torch.Tensor,
        query_coord_map_mm: torch.Tensor,
        context_valid_mask: torch.Tensor | None = None,
        context_scale_mm: float | int | torch.Tensor | None = None,
        query_scale_mm: float | int | torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        return self.adapter(
            source_feat=context_feat,
            source_coord_map_mm=context_coord_map_mm,
            source_scale_mm=context_scale_mm,
            target_scale_mm=query_scale_mm,
            target_coord_map_mm=query_coord_map_mm,
            source_valid_mask=context_valid_mask,
        )


def _sorted_layer_ids(state_dict: dict[str, torch.Tensor]) -> list[int]:
    return sorted({int(key.split('.')[1]) for key in state_dict if key.startswith('layers.')})


def detect_adapter_kind(state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith('source_key_norm.') for key in state_dict):
        return 'dd_usa'
    if any(key.startswith('context_k_norm.') for key in state_dict) or any(key.startswith('scale_mlp.') for key in state_dict):
        return 'usa'
    raise ValueError('Could not infer adapter kind from checkpoint state_dict')


def infer_usa_kwargs(state_dict: dict[str, torch.Tensor], embed_dim: int = 768) -> dict[str, object]:
    layer_ids = _sorted_layer_ids(state_dict)
    if not layer_ids:
        raise ValueError('USA checkpoint does not contain any adapter layers.')

    return {
        'embed_dim': embed_dim,
        'num_heads': 8,
        'num_layers': max(layer_ids) + 1,
        'dropout': 0.0,
        'coord_num_frequencies': 4,
        'coord_scale_mm': 10.0,
        'interp_k': 4,
        'use_scale_token': any(key.startswith('scale_mlp.') for key in state_dict),
        'use_final_norm': any(key.startswith('final_norm.') for key in state_dict),
    }


def infer_dd_usa_kwargs(state_dict: dict[str, torch.Tensor], embed_dim: int = 768) -> dict[str, object]:
    layer_ids = _sorted_layer_ids(state_dict)
    if not layer_ids:
        raise ValueError('DD-USA checkpoint does not contain any adapter layers.')

    first_layer = layer_ids[0]
    num_heads = int(state_dict[f'layers.{first_layer}.rel_bias.mlp.2.weight'].shape[0])
    rel_bias_hidden_dim = int(state_dict[f'layers.{first_layer}.rel_bias.mlp.0.weight'].shape[0])
    convffn_hidden_dim = int(state_dict[f'layers.{first_layer}.conv_ffn.fc1.weight'].shape[0])
    convffn_ratio = float(convffn_hidden_dim) / float(embed_dim)

    coord_in_dim = int(state_dict['coord_encoder.proj.0.weight'].shape[1])
    coord_include_input = coord_in_dim % 4 == 2
    raw_dim = 2 if coord_include_input else 0
    fourier_dim = coord_in_dim - raw_dim
    if fourier_dim < 0 or fourier_dim % 4 != 0:
        raise ValueError(f'Could not infer DD-USA Fourier settings from coord encoder input dim {coord_in_dim}')
    coord_num_frequencies = fourier_dim // 4

    return {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': max(layer_ids) + 1,
        'dropout': 0.0,
        'coord_num_frequencies': coord_num_frequencies,
        'coord_include_input': coord_include_input,
        'convffn_ratio': convffn_ratio,
        'rel_bias_hidden_dim': rel_bias_hidden_dim,
        'use_final_norm': any(key.startswith('final_norm.') for key in state_dict),
        'learnable_mix': 'mix_logits' in state_dict,
    }


def build_frozen_adapter_plugin(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    *,
    embed_dim: int = 768,
) -> tuple[nn.Module, str]:
    adapter_kind = detect_adapter_kind(state_dict)
    if adapter_kind == 'dd_usa':
        core_adapter = DirectDriveUniversalScaleAdapter(**infer_dd_usa_kwargs(state_dict, embed_dim=embed_dim)).to(device)
        core_adapter.load_state_dict(state_dict)
        plugin: nn.Module = DDUSAPluginAdapter(core_adapter).to(device)
    else:
        plugin = UniversalScaleAdapter(**infer_usa_kwargs(state_dict, embed_dim=embed_dim)).to(device)
        plugin.load_state_dict(state_dict)

    plugin.eval()
    for param in plugin.parameters():
        param.requires_grad = False
    return plugin, adapter_kind
