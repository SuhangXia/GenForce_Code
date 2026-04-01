#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from downstream_task.datasets import MultiscaleTactileDataset
from downstream_task.models import StaticPoseRegressor
try:
    from standalone_qformer.eval_wrapper import build_frozen_qformer_plugin
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from eval_wrapper import build_frozen_qformer_plugin


DEFAULT_DOCKER_DATASET = Path('/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
DEFAULT_HOST_DATASET = Path('/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
REFERENCE_SCALE_MM = 20.0
SEEN_INDENTERS = [
    'cone',
    'cylinder',
    'cylinder_sh',
    'cylinder_si',
    'dotin',
    'dots',
    'hemisphere',
    'line',
    'moon',
    'prism',
    'random',
    'sphere',
    'sphere_s',
    'triangle',
]
UNSEEN_INDENTERS = ['pacman', 'wave', 'torus', 'hexagon']


def default_dataset_root() -> Path:
    if DEFAULT_DOCKER_DATASET.exists():
        return DEFAULT_DOCKER_DATASET
    return DEFAULT_HOST_DATASET


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


class FrozenViTBackbone(nn.Module):
    """Frozen ViT that returns patch tokens without the CLS token."""

    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True) -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)
        return x[:, 1:, :]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate the standalone Q-Former downstream regressor across scales and indenter splits.'
    )
    parser.add_argument('--dataset-root', type=str, default=str(default_dataset_root()))
    parser.add_argument('--test-scale', type=int, required=True)
    parser.add_argument('--indenter-split', type=str, choices=['seen', 'unseen'], required=True)
    parser.add_argument('--use-qformer', type=str2bool, default=False)
    parser.add_argument('--qformer-ckpt', type=str, default='')
    parser.add_argument('--regressor-ckpt', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--output-json', type=str, default='')
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {'model_state_dict': checkpoint}


def strip_prefix_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefix)
    }


def load_regressor_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    filtered = strip_prefix_from_state_dict(state_dict, 'usa_plugin.')
    load_result = model.load_state_dict(filtered, strict=False)
    unexpected_non_plugin = [key for key in load_result.unexpected_keys if not key.startswith('usa_plugin.')]
    missing_non_plugin = [key for key in load_result.missing_keys if not key.startswith('usa_plugin.')]
    if unexpected_non_plugin or missing_non_plugin:
        raise RuntimeError(
            'Failed to load regressor checkpoint cleanly. '
            f'missing={missing_non_plugin} unexpected={unexpected_non_plugin}'
        )


def build_eval_dataset(dataset_root: Path, test_scale: int, indenter_split: str) -> MultiscaleTactileDataset:
    indenters = SEEN_INDENTERS if indenter_split == 'seen' else UNSEEN_INDENTERS
    return MultiscaleTactileDataset(
        root_dir=str(dataset_root),
        scale_mm=float(test_scale),
        indenters=indenters,
        reference_scale_mm=REFERENCE_SCALE_MM,
    )


def unpack_eval_batch(
    batch: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 4:
        imgs, targets, target_coords, source_coords = batch[:4]
    elif isinstance(batch, dict):
        imgs = batch['image']
        targets = batch['target']
        target_coords = batch['target_coords']
        source_coords = batch['source_coords']
    else:
        raise TypeError(
            'Expected each batch to yield (imgs, targets, target_coords, source_coords) or a dict variant.'
        )

    imgs = imgs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    target_coords = target_coords.to(device=device, dtype=torch.float32)
    source_coords = source_coords.to(device=device, dtype=torch.float32)

    if imgs.ndim != 4:
        raise ValueError(f'Expected imgs to have shape (B, 3, 224, 224), got {tuple(imgs.shape)}')
    if targets.ndim != 2 or targets.shape[-1] != 3:
        raise ValueError(f'Expected targets to have shape (B, 3), got {tuple(targets.shape)}')
    return imgs, targets, target_coords, source_coords


def evaluate(
    model: StaticPoseRegressor,
    loader: DataLoader[Any],
    device: torch.device,
    test_scale: int,
    use_qformer: bool,
    *,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    sum_sq_depth = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            imgs, targets, target_coords, source_coords = unpack_eval_batch(batch, device)

            if use_qformer:
                baseline_target_scale_ts = torch.full(
                    (imgs.shape[0],),
                    float(REFERENCE_SCALE_MM),
                    device=device,
                    dtype=torch.float32,
                )
                current_test_source_scale_ts = torch.full(
                    (imgs.shape[0],),
                    float(test_scale),
                    device=device,
                    dtype=torch.float32,
                )
                preds = model(
                    src_imgs=imgs,
                    target_coords=target_coords,
                    source_coords=source_coords,
                    target_scale=baseline_target_scale_ts,
                    source_scale=current_test_source_scale_ts,
                )
            else:
                preds = model(src_imgs=imgs)

            err = (preds - targets).pow(2)
            sum_sq_x += err[:, 0].sum().item()
            sum_sq_y += err[:, 1].sum().item()
            sum_sq_depth += err[:, 2].sum().item()
            total_samples += imgs.shape[0]

    avg_mse_x = sum_sq_x / max(total_samples, 1)
    avg_mse_y = sum_sq_y / max(total_samples, 1)
    avg_mse_depth = sum_sq_depth / max(total_samples, 1)
    avg_total_mse = (avg_mse_x + avg_mse_y + avg_mse_depth) / 3.0
    return {
        'mse_x': float(avg_mse_x),
        'mse_y': float(avg_mse_y),
        'mse_depth': float(avg_mse_depth),
        'mse_total': float(avg_total_mse),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    vit_backbone = FrozenViTBackbone(model_name=args.model_name, pretrained=True)

    qformer_plugin = None
    adapter_kind = 'none'
    if args.use_qformer:
        if not args.qformer_ckpt:
            raise ValueError('--qformer-ckpt is required when --use-qformer is True')
        qformer_plugin, qformer_meta = build_frozen_qformer_plugin(resolve_path(args.qformer_ckpt), device)
        adapter_kind = str(qformer_meta.get('adapter_kind', 'scale_conditioned_qformer'))

    model = StaticPoseRegressor(vit_backbone=vit_backbone, usa_plugin=qformer_plugin, out_dim=3).to(device)

    regressor_ckpt = resolve_path(args.regressor_ckpt)
    reg_payload = load_checkpoint(regressor_ckpt, device)
    if 'model_state_dict' not in reg_payload:
        raise KeyError(f'Regressor checkpoint is missing model_state_dict: {regressor_ckpt}')
    load_regressor_state(model, reg_payload['model_state_dict'])

    dataset = build_eval_dataset(Path(args.dataset_root), args.test_scale, args.indenter_split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate(
        model,
        loader,
        device,
        args.test_scale,
        args.use_qformer,
        max_batches=args.max_batches,
    )
    metrics.update(
        {
            'test_scale': int(args.test_scale),
            'indenter_split': args.indenter_split,
            'use_qformer': bool(args.use_qformer),
            'adapter_kind': adapter_kind,
            'regressor_ckpt': str(regressor_ckpt),
            'qformer_ckpt': str(resolve_path(args.qformer_ckpt)) if args.qformer_ckpt else '',
        }
    )

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    print(
        f"Eval scale={args.test_scale} split={args.indenter_split} use_qformer={args.use_qformer} adapter_kind={adapter_kind} | "
        f"mse_x={metrics['mse_x']:.6f} mse_y={metrics['mse_y']:.6f} "
        f"mse_depth={metrics['mse_depth']:.6f} mse_total={metrics['mse_total']:.6f}"
    )


if __name__ == '__main__':
    main()
