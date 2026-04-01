#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from standalone_stea.data import build_downstream_split_dataset, create_loader, unpack_downstream_batch
from standalone_stea.models.stea_adapter import STEAAdapter
from standalone_stea.utils import (
    CANONICAL_SCALE_MM,
    FrozenViTPatchExtractor,
    default_downstream_dataset_root,
    load_trained_head_checkpoint,
    resolve_path,
    wandb,
)


def load_stea_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[STEAAdapter, dict[str, Any]]:
    payload = torch.load(resolve_path(checkpoint_path, base_dir=PROJECT_ROOT), map_location=device, weights_only=False)
    state_dict = payload.get('model_state_dict') or payload.get('stea_state_dict') or payload
    if not isinstance(state_dict, dict):
        raise ValueError(f'Checkpoint does not contain a valid STEA state dict: {checkpoint_path}')
    bg_map = state_dict.get('background_latent_map')
    if bg_map is None:
        raise KeyError(f'STEA checkpoint is missing background_latent_map: {checkpoint_path}')
    use_boundary_smoothing = 'boundary_smoother.weight' in state_dict
    hidden_dim = int(state_dict['modulation_mlp.0.weight'].shape[0])
    model = STEAAdapter(
        background_latent_map=bg_map,
        condition_hidden_dim=hidden_dim,
        use_boundary_smoothing=use_boundary_smoothing,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, payload


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if text in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a downstream regressor head with or without frozen STEA.')
    parser.add_argument('--dataset-root', type=str, default=str(default_downstream_dataset_root()))
    parser.add_argument('--regressor-ckpt', type=str, required=True)
    parser.add_argument('--stea-ckpt', type=str, default='')
    parser.add_argument('--use-stea', type=str2bool, default=False)
    parser.add_argument('--scale-mm', type=float, required=True)
    parser.add_argument('--split', type=str, choices=['seen', 'unseen'], required=True)
    parser.add_argument('--canonical-scale-mm', type=float, default=CANONICAL_SCALE_MM)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb-project', type=str, default='')
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    return parser.parse_args()


def evaluate(
    *,
    vit: FrozenViTPatchExtractor,
    head: torch.nn.Module,
    stea: STEAAdapter | None,
    loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
    scale_mm: float,
    canonical_scale_mm: float,
) -> dict[str, float]:
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    sum_sq_depth = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images, targets, _target_coords, _source_coords, source_scale_mm = unpack_downstream_batch(batch, device)
            features = vit(images)
            if stea is not None:
                adapted_features, _ = stea(
                    features,
                    source_scale_mm=source_scale_mm,
                    target_scale_mm=float(canonical_scale_mm),
                )
            else:
                adapted_features = features
            preds = head(adapted_features)
            sq = (preds - targets).pow(2)
            sum_sq_x += sq[:, 0].sum().item()
            sum_sq_y += sq[:, 1].sum().item()
            sum_sq_depth += sq[:, 2].sum().item()
            total_samples += images.shape[0]

    mse_x = sum_sq_x / max(total_samples, 1)
    mse_y = sum_sq_y / max(total_samples, 1)
    mse_depth = sum_sq_depth / max(total_samples, 1)
    mse_total = (mse_x + mse_y + mse_depth) / 3.0
    return {
        'mse_x': float(mse_x),
        'mse_y': float(mse_y),
        'mse_depth': float(mse_depth),
        'mse_total': float(mse_total),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vit = FrozenViTPatchExtractor(model_name=args.model_name, pretrained=True).to(device)
    vit.eval()
    head, _ = load_trained_head_checkpoint(args.regressor_ckpt, device, freeze=True)

    stea = None
    adapter_kind = 'none'
    if args.use_stea:
        if not args.stea_ckpt:
            raise ValueError('--stea-ckpt is required when --use-stea is True')
        stea, _ = load_stea_from_checkpoint(args.stea_ckpt, device)
        adapter_kind = 'stea'

    dataset = build_downstream_split_dataset(
        root_dir=args.dataset_root,
        scales_mm=[args.scale_mm],
        split=args.split,
        reference_scale_mm=args.canonical_scale_mm,
    )
    loader = create_loader(dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False)
    metrics = evaluate(
        vit=vit,
        head=head,
        stea=stea,
        loader=loader,
        device=device,
        scale_mm=args.scale_mm,
        canonical_scale_mm=args.canonical_scale_mm,
    )

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=f'eval_stea_{args.scale_mm:g}_{args.split}', config=vars(args))
        wandb.log({
            'mse_x': metrics['mse_x'],
            'mse_y': metrics['mse_y'],
            'mse_depth': metrics['mse_depth'],
            'mse_total': metrics['mse_total'],
            'use_stea': bool(args.use_stea),
            'adapter_kind': adapter_kind,
            'scale_mm': float(args.scale_mm),
            'split': args.split,
        })
        wandb.finish()

    scale_text = str(int(args.scale_mm)) if float(args.scale_mm).is_integer() else f'{args.scale_mm:g}'
    print(
        f'Eval scale={scale_text} split={args.split} use_stea={bool(args.use_stea)} adapter_kind={adapter_kind} | '
        f'mse_x={metrics["mse_x"]:.6f} mse_y={metrics["mse_y"]:.6f} '
        f'mse_depth={metrics["mse_depth"]:.6f} mse_total={metrics["mse_total"]:.6f}'
    )


if __name__ == '__main__':
    main()
