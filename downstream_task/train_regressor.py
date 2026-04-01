from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency in lightweight runtime environments.
    def tqdm(iterable, *args, **kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRPIT_DIR = PROJECT_ROOT / 'scrpit'
if str(SCRPIT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRPIT_DIR))

from downstream_task.datasets import MultiscaleTactileDataset
from downstream_task.models import StaticPoseRegressor
from downstream_task.adapter_utils import build_frozen_adapter_plugin


def _default_dataset_root() -> Path:
    container_path = Path('/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
    host_path = Path('/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
    if container_path.exists():
        return container_path
    return host_path


DEFAULT_DATASET_ROOT = _default_dataset_root()
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
SEEN_INDENTERS = TRAIN_INDENTERS + VAL_INDENTERS
TARGET_NAMES = ['command_x_norm', 'command_y_norm', 'frame_actual_max_down_mm']
DEFAULT_SAVE_PATH = PROJECT_ROOT / 'logs' / 'regressor_20mm_best.pt'


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
    parser = argparse.ArgumentParser(description='Train the static downstream regressor on 20mm seen-indenter data.')
    parser.add_argument('--dataset-root', type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default=str(DEFAULT_SAVE_PATH))
    parser.add_argument('--save-every', type=int, default=5, help='Save an epoch checkpoint every N epochs.')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume training from a saved checkpoint.')
    parser.add_argument('--adapter-ckpt', '--usa-ckpt', dest='usa_ckpt', type=str, default='', help='Optional frozen adapter checkpoint (USA or DD-USA) for frozen-plugin downstream training.')
    parser.add_argument('--wandb-project', type=str, default='Tactile_Downstream')
    parser.add_argument('--wandb-name', type=str, default='Train_Baseline_20mm')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars.')
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(dataset_root: Path):
    train_dataset = MultiscaleTactileDataset(
        root_dir=str(dataset_root),
        scale_mm=20.0,
        indenters=TRAIN_INDENTERS,
    )
    val_dataset = MultiscaleTactileDataset(
        root_dir=str(dataset_root),
        scale_mm=20.0,
        indenters=VAL_INDENTERS,
    )
    return train_dataset, val_dataset


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


def create_loaders(train_dataset, val_dataset, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
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


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {'model_state_dict': checkpoint}


def run_epoch(
    model: StaticPoseRegressor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    desc: str,
    progress: bool,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_batches = 0

    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)
    for batch in iterator:
        imgs, targets, target_coords, _source_coords = unpack_supervised_batch(batch, device)

        if model.usa_plugin is not None:
            # Identity training on the adapter manifold: 20mm -> 20mm.
            # We intentionally feed the same 20mm reference grid on both sides so the
            # downstream head learns USA-processed feature statistics without scale shift.
            baseline_coords = target_coords
            baseline_scale_ts = torch.full(
                (imgs.shape[0],),
                float(REFERENCE_SCALE_MM),
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
            optimizer.zero_grad()
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
    val_loss: float,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': float(val_loss),
            'best_val_loss': float(best_val_loss),
            'config': vars(args),
            'target_names': TARGET_NAMES,
            'train_indenters': TRAIN_INDENTERS,
            'val_indenters': VAL_INDENTERS,
            'scale_mm': 20.0,
        },
        path,
    )


def _resolve_checkpoint_path(raw_path: str | None) -> Path | None:
    if raw_path is None or raw_path == '':
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f'{hours:d}h{minutes:02d}m{secs:02d}s'
    return f'{minutes:02d}m{secs:02d}s'


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    progress = not args.no_progress

    dataset_root = Path(args.dataset_root)
    train_dataset, val_dataset = build_datasets(dataset_root)
    train_loader, val_loader = create_loaders(train_dataset, val_dataset, args)

    vit_backbone = FrozenViTBackbone()
    usa_plugin = None
    adapter_kind = 'none'
    resolved_usa_ckpt = _resolve_checkpoint_path(args.usa_ckpt)
    if resolved_usa_ckpt is not None:
        usa_payload = load_checkpoint(resolved_usa_ckpt, device)
        if 'model_state_dict' not in usa_payload:
            raise KeyError(f'USA checkpoint is missing model_state_dict: {resolved_usa_ckpt}')
        usa_plugin, adapter_kind = build_frozen_adapter_plugin(usa_payload['model_state_dict'], device, embed_dim=768)

    model = StaticPoseRegressor(vit_backbone=vit_backbone, usa_plugin=usa_plugin, out_dim=3).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_path = _resolve_checkpoint_path(args.save_path)
    if save_path is None:
        raise ValueError('--save-path must not be empty')
    ckpt_dir = save_path.parent
    best_val_loss = float('inf')
    train_start_time = time.time()
    start_epoch = 1

    if args.resume_from is not None:
        resume_path = _resolve_checkpoint_path(args.resume_from)
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume_from}')
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if 'model_state_dict' not in payload:
            raise KeyError(f'Checkpoint is missing model_state_dict: {resume_path}')
        model.load_state_dict(payload['model_state_dict'])
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
        f'adapter_ckpt={resolved_usa_ckpt if resolved_usa_ckpt is not None else "disabled"} | '
        f'adapter_kind={adapter_kind}'
    )

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    try:
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
                )

            epoch_seconds = time.time() - epoch_start_time
            elapsed_seconds = time.time() - train_start_time
            wandb.log(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch_seconds': epoch_seconds,
                    'elapsed_seconds': elapsed_seconds,
                    'using_adapter': resolved_usa_ckpt is not None,
                    'adapter_kind': adapter_kind,
                }
            )
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
                val_loss=val_loss,
                best_val_loss=min(best_val_loss, val_loss),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    save_path,
                    model,
                    optimizer,
                    args,
                    epoch=epoch,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
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
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                )
                print(f'Saved periodic checkpoint to {epoch_path}')
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
