from __future__ import annotations

import argparse
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency in lightweight runtime environments.
    class _WandbStub:
        @staticmethod
        def init(*args, **kwargs):
            print('wandb is not installed; continuing with local logging only.')
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def finish(*args, **kwargs):
            return None

    wandb = _WandbStub()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRPIT_DIR = PROJECT_ROOT / 'scrpit'
if str(SCRPIT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRPIT_DIR))

from downstream_task.datasets import MultiscaleTactileDataset
from downstream_task.models import StaticPoseRegressor
from usa_adapter import UniversalScaleAdapter


def _default_dataset_root() -> Path:
    container_path = Path('/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
    host_path = Path('/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
    if container_path.exists():
        return container_path
    return host_path


DEFAULT_DATASET_ROOT = _default_dataset_root()
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
TARGET_NAMES = ['mse_x', 'mse_y', 'mse_depth']


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


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the static downstream regressor across scales and indenter splits.')
    parser.add_argument('--dataset-root', type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument('--test_scale', type=int, required=True)
    parser.add_argument('--indenter_split', type=str, choices=['seen', 'unseen'], required=True)
    parser.add_argument('--use_usa', type=str2bool, default=False)
    parser.add_argument('--usa_ckpt', type=str, default='')
    parser.add_argument('--regressor_ckpt', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def infer_usa_kwargs(state_dict: dict[str, torch.Tensor], embed_dim: int = 768) -> dict[str, object]:
    layer_ids = sorted({int(key.split('.')[1]) for key in state_dict if key.startswith('layers.')})
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


def build_usa_plugin(state_dict: dict[str, torch.Tensor], embed_dim: int = 768) -> UniversalScaleAdapter:
    return UniversalScaleAdapter(**infer_usa_kwargs(state_dict, embed_dim=embed_dim))


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {'model_state_dict': checkpoint}


def strip_prefix_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefix)
    }


def build_eval_dataset(dataset_root: Path, test_scale: int, indenter_split: str):
    indenters = SEEN_INDENTERS if indenter_split == 'seen' else UNSEEN_INDENTERS
    return MultiscaleTactileDataset(
        root_dir=str(dataset_root),
        scale_mm=float(test_scale),
        indenters=indenters,
        reference_scale_mm=REFERENCE_SCALE_MM,
    )


def unpack_eval_batch(batch: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    loader: DataLoader,
    device: torch.device,
    test_scale: int,
    use_usa: bool,
) -> dict[str, float]:
    model.eval()
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    sum_sq_depth = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            imgs, targets, target_coords, source_coords = unpack_eval_batch(batch, device)

            if use_usa:
                # Physical meaning:
                # - the incoming image is observed at the current test scale (for example 16mm or 23mm)
                # - the downstream regressor was trained strictly on 20mm features
                # - USA must therefore map: current test scale -> 20mm baseline scale
                baseline_target_coords = target_coords
                current_test_source_coords = source_coords
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
                    target_coords=baseline_target_coords,
                    source_coords=current_test_source_coords,
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

    vit_backbone = FrozenViTBackbone()

    usa_plugin = None
    if args.use_usa:
        if not args.usa_ckpt:
            raise ValueError('--usa_ckpt is required when --use_usa is True')
        usa_payload = load_checkpoint(Path(args.usa_ckpt), device)
        if 'model_state_dict' not in usa_payload:
            raise KeyError('USA checkpoint must contain model_state_dict')
        usa_plugin = build_usa_plugin(usa_payload['model_state_dict'], embed_dim=768).to(device)
        usa_plugin.load_state_dict(usa_payload['model_state_dict'])
        usa_plugin.eval()
        for param in usa_plugin.parameters():
            param.requires_grad = False

    model = StaticPoseRegressor(vit_backbone=vit_backbone, usa_plugin=usa_plugin, out_dim=3).to(device)

    regressor_ckpt = Path(args.regressor_ckpt)
    reg_payload = load_checkpoint(regressor_ckpt, device)
    reg_state_dict = reg_payload['model_state_dict']
    # The downstream regressor checkpoint may have been trained with a frozen USA plugin.
    # We always load the non-USA weights from the regressor checkpoint, while USA itself is
    # controlled explicitly by --use_usa/--usa_ckpt for evaluation.
    reg_state_dict = strip_prefix_from_state_dict(reg_state_dict, 'usa_plugin.')
    load_result = model.load_state_dict(reg_state_dict, strict=False)
    unexpected_non_usa = [key for key in load_result.unexpected_keys if not key.startswith('usa_plugin.')]
    missing_non_usa = [key for key in load_result.missing_keys if not key.startswith('usa_plugin.')]
    if unexpected_non_usa or missing_non_usa:
        raise RuntimeError(
            'Failed to load regressor checkpoint cleanly. '
            f'missing={missing_non_usa} unexpected={unexpected_non_usa}'
        )

    dataset = build_eval_dataset(Path(args.dataset_root), args.test_scale, args.indenter_split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    run_name = f'Eval_{args.test_scale}mm_{args.indenter_split}_USA_{args.use_usa}'
    wandb.init(project='Tactile_Downstream_Eval', name=run_name, config=vars(args))
    try:
        metrics = evaluate(model, loader, device, args.test_scale, args.use_usa)
        wandb.log(metrics)
        print(
            f"Eval scale={args.test_scale} split={args.indenter_split} use_usa={args.use_usa} | "
            f"mse_x={metrics['mse_x']:.6f} mse_y={metrics['mse_y']:.6f} "
            f"mse_depth={metrics['mse_depth']:.6f} mse_total={metrics['mse_total']:.6f}"
        )
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
