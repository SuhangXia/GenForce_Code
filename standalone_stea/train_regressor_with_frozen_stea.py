#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
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

from standalone_stea.data import build_downstream_split_dataset, create_loader, unpack_downstream_batch
from standalone_stea.utils import (
    CANONICAL_SCALE_MM,
    FrozenViTPatchExtractor,
    STEARegressorHead,
    default_checkpoint_dir,
    default_downstream_dataset_root,
    format_seconds,
    resolve_path,
    seed_everything,
    wandb,
)


def load_stea_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    freeze: bool = True,
):
    from standalone_stea.models.stea_adapter import STEAAdapter

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
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a new downstream regressor head on top of frozen ViT + frozen STEA.')
    parser.add_argument('--dataset-root', type=str, default=str(default_downstream_dataset_root()))
    parser.add_argument('--stea-ckpt', type=str, required=True)
    parser.add_argument('--canonical-scale-mm', type=float, default=CANONICAL_SCALE_MM)
    parser.add_argument('--train-scales', type=float, nargs='*', default=[16.0, 20.0, 23.0])
    parser.add_argument('--checkpoint-dir', type=str, default=str(default_checkpoint_dir('standalone_stea_regressor')))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb-project', type=str, default='STEA_Downstream')
    parser.add_argument('--wandb-name', type=str, default='stea_regressor_train')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--eval-test', action='store_true')
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    *,
    model: STEARegressorHead,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'head_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': float(best_val_loss),
            'config': vars(args),
            'out_dim': 3,
            'adapter_kind': 'stea',
        },
        path,
    )


def run_epoch(
    *,
    head: STEARegressorHead,
    vit: FrozenViTPatchExtractor,
    stea: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    canonical_scale_mm: float,
    max_batches: int | None,
    desc: str,
    progress: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    head.train(is_training)
    metric_rows: list[dict[str, float]] = []
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, targets, _target_coords, _source_coords, source_scale_mm = unpack_downstream_batch(batch, device)

        with torch.no_grad():
            source_features = vit(images)
            adapted_features, _ = stea(
                source_features,
                source_scale_mm=source_scale_mm,
                target_scale_mm=float(canonical_scale_mm),
            )

        preds = head(adapted_features)
        sq = (preds - targets).pow(2)
        mse_x = sq[:, 0].mean()
        mse_y = sq[:, 1].mean()
        mse_depth = sq[:, 2].mean()
        mse_total = (mse_x + mse_y + mse_depth) / 3.0

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            mse_total.backward()
            optimizer.step()

        row = {
            'mse_x': float(mse_x.item()),
            'mse_y': float(mse_y.item()),
            'mse_depth': float(mse_depth.item()),
            'mse_total': float(mse_total.item()),
        }
        metric_rows.append(row)
        running = {
            key: sum(item[key] for item in metric_rows) / max(len(metric_rows), 1)
            for key in row
        }
        if progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(total=f"{running['mse_total']:.4f}", x=f"{running['mse_x']:.4f}", y=f"{running['mse_y']:.4f}")

    if not metric_rows:
        return {'mse_x': 0.0, 'mse_y': 0.0, 'mse_depth': 0.0, 'mse_total': 0.0}
    return {
        key: float(sum(item[key] for item in metric_rows) / max(len(metric_rows), 1))
        for key in metric_rows[0]
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    progress = not args.no_progress
    checkpoint_dir = resolve_path(args.checkpoint_dir, base_dir=PROJECT_ROOT)

    train_dataset = build_downstream_split_dataset(
        root_dir=args.dataset_root,
        scales_mm=args.train_scales,
        split='train',
        reference_scale_mm=args.canonical_scale_mm,
    )
    val_dataset = build_downstream_split_dataset(
        root_dir=args.dataset_root,
        scales_mm=args.train_scales,
        split='val',
        reference_scale_mm=args.canonical_scale_mm,
    )
    train_loader = create_loader(train_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=True)
    val_loader = create_loader(val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False)

    vit = FrozenViTPatchExtractor(model_name=args.model_name, pretrained=True).to(device)
    vit.eval()
    stea, _ = load_stea_from_checkpoint(args.stea_ckpt, device, freeze=True)
    head = STEARegressorHead(out_dim=3).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from:
        payload = torch.load(resolve_path(args.resume_from, base_dir=PROJECT_ROOT), map_location=device, weights_only=False)
        state_dict = payload.get('model_state_dict') or payload.get('head_state_dict')
        if state_dict is None:
            raise KeyError(f'Resume checkpoint is missing model_state_dict/head_state_dict: {args.resume_from}')
        head.load_state_dict(state_dict)
        if 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        start_epoch = int(payload.get('epoch', 0)) + 1
        best_val_loss = float(payload.get('best_val_loss', best_val_loss))
        print(f'Resumed from {args.resume_from} | start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}')

    print(
        'Training downstream regressor with frozen STEA | '
        f'device={device} | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | '
        f'train_scales={list(args.train_scales)} | canonical_scale={args.canonical_scale_mm:.1f}'
    )

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    train_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(
                head=head,
                vit=vit,
                stea=stea,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                canonical_scale_mm=args.canonical_scale_mm,
                max_batches=args.max_train_batches,
                desc=f'Train {epoch}/{args.epochs}',
                progress=progress,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    head=head,
                    vit=vit,
                    stea=stea,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    canonical_scale_mm=args.canonical_scale_mm,
                    max_batches=args.max_val_batches,
                    desc=f'Val   {epoch}/{args.epochs}',
                    progress=progress,
                )

            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - train_start
            epochs_done = epoch - start_epoch + 1
            eta_seconds = (elapsed_seconds / max(epochs_done, 1)) * max(args.epochs - epoch, 0)
            candidate_best = min(best_val_loss, val_metrics['mse_total'])

            wandb.log(
                {
                    'epoch': epoch,
                    'train_mse_x': train_metrics['mse_x'],
                    'train_mse_y': train_metrics['mse_y'],
                    'train_mse_depth': train_metrics['mse_depth'],
                    'train_mse_total': train_metrics['mse_total'],
                    'mse_x': val_metrics['mse_x'],
                    'mse_y': val_metrics['mse_y'],
                    'mse_depth': val_metrics['mse_depth'],
                    'mse_total': val_metrics['mse_total'],
                    'epoch_seconds': epoch_seconds,
                    'elapsed_seconds': elapsed_seconds,
                }
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train_total={train_metrics['mse_total']:.6f} | "
                f"val_total={val_metrics['mse_total']:.6f} | mse_x={val_metrics['mse_x']:.6f} | "
                f"mse_y={val_metrics['mse_y']:.6f} | mse_depth={val_metrics['mse_depth']:.6f} | "
                f"best_val={candidate_best:.6f} | epoch_t={format_seconds(epoch_seconds)} | "
                f"elapsed={format_seconds(elapsed_seconds)} | eta={format_seconds(eta_seconds)}"
            )

            save_checkpoint(
                checkpoint_dir / 'last.pt',
                model=head,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=candidate_best,
            )
            if val_metrics['mse_total'] < best_val_loss:
                best_val_loss = val_metrics['mse_total']
                save_checkpoint(
                    checkpoint_dir / 'best.pt',
                    model=head,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                )
                print(f'Saved new best checkpoint to {checkpoint_dir / "best.pt"}')
            if args.save_every > 0 and epoch % args.save_every == 0:
                epoch_path = checkpoint_dir / f'epoch_{epoch:04d}.pt'
                save_checkpoint(
                    epoch_path,
                    model=head,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                )
                print(f'Saved periodic checkpoint to {epoch_path}')
    finally:
        wandb.finish()

    if args.eval_test:
        print('Final validation metrics:', val_metrics)


if __name__ == '__main__':
    main()
