#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCRPIT_DIR = PROJECT_ROOT / 'scrpit'
for candidate in (SCRIPT_DIR, PROJECT_ROOT, SCRPIT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import timm  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency is expected in the target runtime
    raise ImportError('train_qformer_distill.py requires timm to be installed.') from exc

try:
    from standalone_qformer.qformer_adapter import (
        DEFAULT_EMBED_DIM,
        DEFAULT_NUM_QUERIES,
        ScaleConditionedQFormer,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from qformer_adapter import DEFAULT_EMBED_DIM, DEFAULT_NUM_QUERIES, ScaleConditionedQFormer


DEFAULT_HOST_DATASET = Path('/home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix')
DEFAULT_DOCKER_DATASET = Path('/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix')
DEFAULT_HOST_CHECKPOINT_ROOT = Path('/home/suhang/datasets/checkpoints')
DEFAULT_DOCKER_CHECKPOINT_ROOT = Path('/datasets/checkpoints')
DEFAULT_SOURCE_SCALES = (15.0, 18.0, 20.0, 22.0, 25.0)
TARGET_SCALE_MM = 20.0
TRAIN_INDENTERS = (
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
)
VAL_INDENTERS = ('sphere_s', 'triangle')


def resolve_default_dataset_root() -> Path:
    if DEFAULT_DOCKER_DATASET.exists():
        return DEFAULT_DOCKER_DATASET
    return DEFAULT_HOST_DATASET


def resolve_default_checkpoint_dir() -> Path:
    if DEFAULT_DOCKER_CHECKPOINT_ROOT.exists():
        return DEFAULT_DOCKER_CHECKPOINT_ROOT / 'standalone_qformer_real_distill'
    return DEFAULT_HOST_CHECKPOINT_ROOT / 'standalone_qformer_real_distill'


def resolve_path(path_str: str | Path) -> Path:
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


def format_seconds(total_seconds: float) -> str:
    total = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}h{minutes:02d}m{seconds:02d}s'
    return f'{minutes:02d}m{seconds:02d}s'


def scale_distort_without_padding(
    image: torch.Tensor,
    source_scale_mm: float,
    target_scale_mm: float,
) -> torch.Tensor:
    """Create a simple no-padding scale-illusion surrogate for the dummy dataset."""
    if image.dim() != 3:
        raise ValueError(f'image must be CHW, got {tuple(image.shape)}')

    _, height, width = image.shape
    apparent_factor = float(target_scale_mm) / max(float(source_scale_mm), 1e-6)
    scaled_h = max(1, int(round(height * apparent_factor)))
    scaled_w = max(1, int(round(width * apparent_factor)))

    scaled = F.interpolate(
        image.unsqueeze(0),
        size=(scaled_h, scaled_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

    if scaled_h >= height and scaled_w >= width:
        top = (scaled_h - height) // 2
        left = (scaled_w - width) // 2
        return scaled[:, top : top + height, left : left + width].contiguous()

    return F.interpolate(
        scaled.unsqueeze(0),
        size=(height, width),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0).contiguous()


class DummyPairedTactileDataset(Dataset[dict[str, torch.Tensor]]):
    """Dummy paired tactile dataset preserving the requested public batch keys."""

    def __init__(
        self,
        num_samples: int,
        image_size: int = 224,
        source_scales_mm: tuple[float, ...] = DEFAULT_SOURCE_SCALES,
        target_scale_mm: float = TARGET_SCALE_MM,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f'num_samples must be positive, got {num_samples}')
        if not source_scales_mm:
            raise ValueError('source_scales_mm must not be empty')

        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.source_scales_mm = tuple(float(scale) for scale in source_scales_mm)
        self.target_scale_mm = float(target_scale_mm)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        img_target_20mm = torch.rand(
            3,
            self.image_size,
            self.image_size,
            generator=generator,
            dtype=torch.float32,
        )
        source_scale_mm = float(self.source_scales_mm[index % len(self.source_scales_mm)])
        img_source = scale_distort_without_padding(
            img_target_20mm,
            source_scale_mm=source_scale_mm,
            target_scale_mm=self.target_scale_mm,
        )
        noise = 0.01 * torch.randn(img_source.shape, generator=generator, dtype=img_source.dtype)
        img_source = torch.clamp(img_source + noise, 0.0, 1.0)
        return {
            'img_source': img_source,
            'img_target_20mm': img_target_20mm,
            'source_scale_mm': torch.tensor(source_scale_mm, dtype=torch.float32),
        }


class ReferenceTargetPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Filter the real DD-USA pair dataset to source -> fixed 20mm target pairs."""

    def __init__(
        self,
        base_dataset: Any,
        target_scale_mm: float,
        pairs_per_epoch: int | None,
        *,
        exclude_identity_pairs: bool = False,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.target_scale_mm = float(target_scale_mm)
        self.exclude_identity_pairs = bool(exclude_identity_pairs)
        self._pairs: list[dict[str, Any]] = []

        for pair in getattr(base_dataset, '_all_pairs', []):
            source_scale = float(pair['source']['scale_mm'])
            target_scale = float(pair['target']['scale_mm'])
            if not np.isclose(target_scale, self.target_scale_mm):
                continue
            if self.exclude_identity_pairs and np.isclose(source_scale, target_scale):
                continue
            self._pairs.append(pair)

        if not self._pairs:
            raise RuntimeError(
                'No valid source->target pairs found for '
                f'target_scale_mm={self.target_scale_mm} '
                f'(exclude_identity_pairs={self.exclude_identity_pairs})'
            )

        if pairs_per_epoch is None:
            self._len = len(self._pairs)
        else:
            self._len = int(pairs_per_epoch)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self._pairs[index % len(self._pairs)]
        source = pair['source']
        target = pair['target']
        img_source = self.base_dataset.transform(Image.open(source['image_path']).convert('RGB'))
        img_target_20mm = self.base_dataset.transform(Image.open(target['image_path']).convert('RGB'))
        return {
            'img_source': img_source,
            'img_target_20mm': img_target_20mm,
            'source_scale_mm': torch.tensor(float(source['scale_mm']), dtype=torch.float32),
        }


class FrozenViTFeatureExtractor(nn.Module):
    """Frozen timm ViT wrapper exposing forward_features for CLS stripping."""

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.vit_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit_model.eval()
        for param in self.vit_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.vit_model.forward_features(images)


def strip_cls_token_if_present(features: torch.Tensor) -> torch.Tensor:
    """Ensure ViT tokens are exactly (B, 196, 768) before distillation."""
    if features.dim() != 3:
        raise ValueError(
            'Expected ViT features to have shape (B, N, D), '
            f'got {tuple(features.shape)}'
        )

    if features.shape[1] == 197:
        features = features[:, 1:, :]
    elif features.shape[1] != 196:
        raise ValueError(
            'Expected ViT forward_features to return either 197 tokens (with CLS) '
            f'or 196 patch tokens, got shape {tuple(features.shape)}'
        )

    if features.shape[1:] != (DEFAULT_NUM_QUERIES, DEFAULT_EMBED_DIM):
        raise ValueError(
            f'Patch tokens must end up as (196, 768), got {tuple(features.shape[1:])}'
        )
    return features.contiguous()


def build_real_pair_datasets(args: argparse.Namespace) -> tuple[Dataset[Any], Dataset[Any]]:
    try:
        from train_dd_usa_pretrain import DDUSAPretrainPairDataset
    except ImportError as exc:
        raise ImportError(
            'Real-data Q-Former training requires scrpit/train_dd_usa_pretrain.py '
            'to provide DDUSAPretrainPairDataset.'
        ) from exc

    dataset_root = resolve_path(args.dataset_root)
    print(
        'Building real tactile pair datasets | '
        f'dataset_root={dataset_root} | train_scales={list(args.train_scales)} | '
        f'eval_scales={list(args.eval_scales)} | target_scale={args.target_scale_mm:.1f}mm'
    )
    train_base = DDUSAPretrainPairDataset(
        dataset_root=dataset_root,
        indenters=TRAIN_INDENTERS,
        scales_mm=args.train_scales,
        mode='train',
        pairs_per_epoch=None,
        seed=args.seed,
    )
    val_base = DDUSAPretrainPairDataset(
        dataset_root=dataset_root,
        indenters=VAL_INDENTERS,
        scales_mm=args.eval_scales or args.train_scales,
        mode='val',
        pairs_per_epoch=None,
        seed=args.seed + 1,
    )
    train_dataset = ReferenceTargetPairDataset(
        train_base,
        target_scale_mm=args.target_scale_mm,
        pairs_per_epoch=args.pairs_per_epoch,
        exclude_identity_pairs=args.exclude_identity_pairs,
    )
    val_dataset = ReferenceTargetPairDataset(
        val_base,
        target_scale_mm=args.target_scale_mm,
        pairs_per_epoch=args.val_pairs_per_epoch,
        exclude_identity_pairs=args.exclude_identity_pairs,
    )
    print(
        'Finished building real tactile pair datasets | '
        f'train_pairs={len(train_dataset)} | val_pairs={len(val_dataset)}'
    )
    return train_dataset, val_dataset


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader[Any], DataLoader[Any], str]:
    if args.use_dummy_data:
        train_dataset = DummyPairedTactileDataset(
            num_samples=args.train_samples,
            image_size=args.image_size,
            source_scales_mm=tuple(args.source_scales_mm),
            target_scale_mm=args.target_scale_mm,
            seed=args.seed,
        )
        val_dataset = DummyPairedTactileDataset(
            num_samples=args.val_samples,
            image_size=args.image_size,
            source_scales_mm=tuple(args.source_scales_mm),
            target_scale_mm=args.target_scale_mm,
            seed=args.seed + 10000,
        )
        dataset_mode = 'dummy'
    else:
        train_dataset, val_dataset = build_real_pair_datasets(args)
        dataset_mode = 'real'

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
    return train_loader, val_loader, dataset_mode


def save_checkpoint(
    checkpoint_path: Path,
    *,
    qformer: ScaleConditionedQFormer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': int(epoch),
            'qformer_state_dict': qformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'best_val_loss': float(best_val_loss),
            'config': vars(args),
            'adapter_kind': 'scale_conditioned_qformer',
            'dataset_mode': 'dummy' if args.use_dummy_data else 'real',
        },
        checkpoint_path,
    )


def run_epoch(
    *,
    qformer: ScaleConditionedQFormer,
    vit_model: FrozenViTFeatureExtractor,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    target_scale_mm: float,
    desc: str,
    progress: bool,
) -> float:
    is_training = optimizer is not None
    qformer.train(is_training)
    total_loss = 0.0
    total_batches = 0
    total_steps = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress, total=total_steps)

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        img_source = batch['img_source'].to(device=device, dtype=torch.float32)
        img_target_20mm = batch['img_target_20mm'].to(device=device, dtype=torch.float32)
        source_scale_mm = batch['source_scale_mm'].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            target_features = strip_cls_token_if_present(
                vit_model.forward_features(img_target_20mm).detach()
            )
            source_features = strip_cls_token_if_present(
                vit_model.forward_features(img_source).detach()
            )

        adapted_features = qformer(
            source_features,
            source_scale_mm=source_scale_mm,
            target_scale_mm=float(target_scale_mm),
        )
        loss = criterion(adapted_features, target_features)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        if progress and hasattr(iterator, 'set_postfix'):
            postfix: dict[str, str] = {
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
            }
            if optimizer is not None:
                postfix['lr'] = f"{optimizer.param_groups[0]['lr']:.2e}"
            iterator.set_postfix(postfix)

    return total_loss / max(total_batches, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a standalone Scale-Conditioned Q-Former with real tactile data or a dummy fallback.'
    )
    parser.add_argument('--dataset-root', '--dataset', dest='dataset_root', type=str, default=str(resolve_default_dataset_root()))
    parser.add_argument('--use-dummy-data', action='store_true')
    parser.add_argument('--train-samples', type=int, default=256)
    parser.add_argument('--val-samples', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--source-scales-mm', type=float, nargs='*', default=list(DEFAULT_SOURCE_SCALES))
    parser.add_argument('--target-scale-mm', type=float, default=TARGET_SCALE_MM)
    parser.add_argument('--train-scales', type=float, nargs='*', default=list(DEFAULT_SOURCE_SCALES))
    parser.add_argument('--eval-scales', type=float, nargs='*', default=list(DEFAULT_SOURCE_SCALES))
    parser.add_argument('--exclude-identity-pairs', action='store_true')
    parser.add_argument('--pairs-per-epoch', type=int, default=24000)
    parser.add_argument('--val-pairs-per-epoch', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--checkpoint-dir', type=str, default=str(resolve_default_checkpoint_dir()))
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    progress = not args.no_progress

    train_loader, val_loader, dataset_mode = build_dataloaders(args)
    vit_model = FrozenViTFeatureExtractor(model_name=args.model_name, pretrained=True).to(device)
    qformer = ScaleConditionedQFormer(
        embed_dim=DEFAULT_EMBED_DIM,
        num_queries=DEFAULT_NUM_QUERIES,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(qformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from:
        payload = torch.load(resolve_path(args.resume_from), map_location=device, weights_only=False)
        qformer.load_state_dict(payload['qformer_state_dict'])
        if 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        if 'scheduler_state_dict' in payload:
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        start_epoch = int(payload.get('epoch', 0)) + 1
        best_val_loss = float(payload.get('best_val_loss', payload.get('val_loss', best_val_loss)))
        print(
            f'Resumed from {args.resume_from} | start_epoch={start_epoch} | '
            f'best_val_loss={best_val_loss:.6f}'
        )

    train_start_time = time.time()
    print(
        'Training setup | '
        f'device={device} | dataset_mode={dataset_mode} | '
        f'train_batches={len(train_loader)} | val_batches={len(val_loader)} | '
        f'batch_size={args.batch_size} | target_scale={args.target_scale_mm:.1f}mm | '
        f'validate_every=1 | save_every={args.save_every} | checkpoint_dir={checkpoint_dir}'
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = run_epoch(
            qformer=qformer,
            vit_model=vit_model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            target_scale_mm=args.target_scale_mm,
            desc=f'Train {epoch}/{args.epochs}',
            progress=progress,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                qformer=qformer,
                vit_model=vit_model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                max_batches=args.max_val_batches,
                target_scale_mm=args.target_scale_mm,
                desc=f'Val   {epoch}/{args.epochs}',
                progress=progress,
            )
        scheduler.step()

        epoch_seconds = time.time() - epoch_start
        elapsed_seconds = time.time() - train_start_time
        epochs_done = epoch - start_epoch + 1
        remaining_epochs = args.epochs - epoch
        avg_epoch_seconds = elapsed_seconds / max(epochs_done, 1)
        eta_seconds = avg_epoch_seconds * max(remaining_epochs, 0)
        current_lr = optimizer.param_groups[0]['lr']
        candidate_best = min(best_val_loss, val_loss)
        print(
            f'Epoch {epoch:03d}/{args.epochs} | '
            f'train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_val={candidate_best:.6f} | '
            f'lr={current_lr:.2e} | epoch_t={format_seconds(epoch_seconds)} | '
            f'elapsed={format_seconds(elapsed_seconds)} | eta={format_seconds(eta_seconds)}'
        )

        save_checkpoint(
            checkpoint_dir / 'last.pt',
            qformer=qformer,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=candidate_best,
        )
        save_checkpoint(
            checkpoint_dir / 'latest.pt',
            qformer=qformer,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=candidate_best,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / 'best.pt',
                qformer=qformer,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
            )
            save_checkpoint(
                checkpoint_dir / 'best_loss.pt',
                qformer=qformer,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
            )
            print(f'Saved new best checkpoint to {checkpoint_dir / "best.pt"}')
            print(f'Saved best-loss alias to {checkpoint_dir / "best_loss.pt"}')

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = checkpoint_dir / f'epoch_{epoch:04d}.pt'
            save_checkpoint(
                epoch_path,
                qformer=qformer,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
            )
            print(f'Saved periodic checkpoint to {epoch_path}')

    print(f'Training finished. Latest checkpoint: {checkpoint_dir / "latest.pt"}')


if __name__ == '__main__':
    main()
