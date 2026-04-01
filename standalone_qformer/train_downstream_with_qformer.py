#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

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
DEFAULT_DOCKER_CHECKPOINT_ROOT = Path('/datasets/checkpoints')
DEFAULT_HOST_CHECKPOINT_ROOT = Path('/home/suhang/datasets/checkpoints')
REFERENCE_SCALE_MM = 20.0
TRAIN_INDENTERS = [
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
]
VAL_INDENTERS = ['sphere_s', 'triangle']
TARGET_NAMES = ['command_x_norm', 'command_y_norm', 'frame_actual_max_down_mm']


def default_dataset_root() -> Path:
    if DEFAULT_DOCKER_DATASET.exists():
        return DEFAULT_DOCKER_DATASET
    return DEFAULT_HOST_DATASET


def default_save_path() -> Path:
    if DEFAULT_DOCKER_CHECKPOINT_ROOT.exists():
        return DEFAULT_DOCKER_CHECKPOINT_ROOT / 'downstream_regressor_20mm_with_qformer' / 'best.pt'
    return DEFAULT_HOST_CHECKPOINT_ROOT / 'downstream_regressor_20mm_with_qformer' / 'best.pt'


def resolve_path(path_str: str | Path | None) -> Path | None:
    if path_str is None or path_str == '':
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f'{hours:d}h{minutes:02d}m{secs:02d}s'
    return f'{minutes:02d}m{secs:02d}s'


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Standalone downstream regressor training with an optional frozen Q-Former plugin.'
    )
    parser.add_argument('--dataset-root', type=str, default=str(default_dataset_root()))
    parser.add_argument('--qformer-ckpt', type=str, default='')
    parser.add_argument('--train-scale-mm', type=float, default=REFERENCE_SCALE_MM)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default=str(default_save_path()))
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def build_datasets(dataset_root: Path, train_scale_mm: float) -> tuple[MultiscaleTactileDataset, MultiscaleTactileDataset]:
    train_dataset = MultiscaleTactileDataset(
        root_dir=str(dataset_root),
        scale_mm=float(train_scale_mm),
        indenters=TRAIN_INDENTERS,
    )
    val_dataset = MultiscaleTactileDataset(
        root_dir=str(dataset_root),
        scale_mm=float(train_scale_mm),
        indenters=VAL_INDENTERS,
    )
    return train_dataset, val_dataset


def create_loaders(
    train_dataset: MultiscaleTactileDataset,
    val_dataset: MultiscaleTactileDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def unpack_supervised_batch(
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


def strip_prefix_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefix)
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        return payload
    return {'model_state_dict': payload}


def load_model_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    ignore_prefix: str = 'usa_plugin.',
) -> None:
    filtered = strip_prefix_from_state_dict(state_dict, ignore_prefix)
    load_result = model.load_state_dict(filtered, strict=False)
    unexpected = [key for key in load_result.unexpected_keys if not key.startswith(ignore_prefix)]
    missing = [key for key in load_result.missing_keys if not key.startswith(ignore_prefix)]
    if unexpected or missing:
        raise RuntimeError(
            'Failed to load regressor checkpoint cleanly. '
            f'missing={missing} unexpected={unexpected}'
        )


def run_epoch(
    model: StaticPoseRegressor,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    desc: str,
    progress: bool,
    reference_scale_mm: float,
    max_batches: int | None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs, targets, target_coords, _source_coords = unpack_supervised_batch(batch, device)

        if model.usa_plugin is not None:
            # Head training is still anchored to the 20mm manifold.
            # We therefore feed 20mm coordinates/scales on both source and target sides,
            # so the regressor learns the statistics of Q-Former-processed 20mm features.
            baseline_coords = target_coords
            baseline_scale_ts = torch.full(
                (imgs.shape[0],),
                float(reference_scale_mm),
                device=device,
                dtype=torch.float32,
            )
            preds = model(
                src_imgs=imgs,
                target_coords=baseline_coords,
                source_coords=baseline_coords,
                target_scale=baseline_scale_ts,
                source_scale=baseline_scale_ts,
            )
        else:
            preds = model(src_imgs=imgs)

        loss = criterion(preds, targets)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        if progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(loss=f'{loss.item():.4f}', avg=f'{avg_loss:.4f}')

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    qformer_meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'best_val_loss': float(best_val_loss),
            'config': vars(args),
            'target_names': TARGET_NAMES,
            'train_indenters': TRAIN_INDENTERS,
            'val_indenters': VAL_INDENTERS,
            'scale_mm': float(args.train_scale_mm),
            'using_qformer': bool(args.qformer_ckpt),
            'qformer_ckpt': args.qformer_ckpt,
            'adapter_kind': qformer_meta.get('adapter_kind', 'none'),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    progress = not args.no_progress

    dataset_root = Path(args.dataset_root)
    train_dataset, val_dataset = build_datasets(dataset_root, args.train_scale_mm)
    train_loader, val_loader = create_loaders(train_dataset, val_dataset, args)

    vit_backbone = FrozenViTBackbone(model_name=args.model_name, pretrained=True)
    qformer_plugin = None
    qformer_meta: dict[str, Any] = {'adapter_kind': 'none'}
    qformer_ckpt = resolve_path(args.qformer_ckpt)
    if qformer_ckpt is not None:
        qformer_plugin, qformer_meta = build_frozen_qformer_plugin(qformer_ckpt, device)

    model = StaticPoseRegressor(vit_backbone=vit_backbone, usa_plugin=qformer_plugin, out_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_path = resolve_path(args.save_path)
    if save_path is None:
        raise ValueError('--save-path must not be empty')
    ckpt_dir = save_path.parent
    best_val_loss = float('inf')
    train_start_time = time.time()
    start_epoch = 1

    resume_path = resolve_path(args.resume_from)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        payload = load_checkpoint(resume_path, device)
        if 'model_state_dict' not in payload:
            raise KeyError(f'Checkpoint is missing model_state_dict: {resume_path}')
        load_model_state(model, payload['model_state_dict'])
        if 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        best_val_loss = float(payload.get('best_val_loss', payload.get('val_loss', best_val_loss)))
        start_epoch = int(payload.get('epoch', 0)) + 1
        print(
            f'Resumed from {resume_path} | start_epoch={start_epoch} | '
            f'best_val_loss={best_val_loss:.6f}'
        )

    print(
        f'Training setup | device={device} | train_samples={len(train_dataset)} | '
        f'val_samples={len(val_dataset)} | batch_size={args.batch_size} | '
        f'qformer_ckpt={qformer_ckpt if qformer_ckpt is not None else "disabled"} | '
        f'adapter_kind={qformer_meta.get("adapter_kind", "none")}'
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f'Train {epoch}/{args.epochs}',
            progress=progress,
            reference_scale_mm=args.train_scale_mm,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                desc=f'Val {epoch}/{args.epochs}',
                progress=progress,
                reference_scale_mm=args.train_scale_mm,
                max_batches=args.max_val_batches,
            )

        epoch_seconds = time.time() - epoch_start_time
        elapsed_seconds = time.time() - train_start_time
        print(
            f'Epoch {epoch:03d}/{args.epochs} | '
            f'train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | '
            f'epoch_t={_format_duration(epoch_seconds)} | '
            f'elapsed={_format_duration(elapsed_seconds)}'
        )

        latest_path = ckpt_dir / 'last.pt'
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            args,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=min(best_val_loss, val_loss),
            qformer_meta=qformer_meta,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_path,
                model,
                optimizer,
                args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                qformer_meta=qformer_meta,
            )
            print(f'Saved new best checkpoint to {save_path} (val_loss={val_loss:.6f})')

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = ckpt_dir / f'epoch_{epoch:04d}.pt'
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                qformer_meta=qformer_meta,
            )
            print(f'Saved periodic checkpoint to {epoch_path}')


if __name__ == '__main__':
    main()
