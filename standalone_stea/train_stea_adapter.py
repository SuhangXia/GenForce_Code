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

from standalone_stea.data import build_stea_pair_datasets, create_loader
from standalone_stea.losses import aggregate_epoch_metrics, compute_stea_losses
from standalone_stea.models.stea_adapter import STEAAdapter
from standalone_stea.utils import (
    CANONICAL_SCALE_MM,
    FrozenViTPatchExtractor,
    default_checkpoint_dir,
    default_stea_train_dataset_root,
    format_seconds,
    load_canonical_head_from_regressor_ckpt,
    resolve_path,
    seed_everything,
    wandb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the standalone STEA adapter with paired cross-scale teacher-student distillation.')
    parser.add_argument('--dataset-root', type=str, default=str(default_stea_train_dataset_root()))
    parser.add_argument('--canonical-scale-mm', type=float, default=CANONICAL_SCALE_MM)
    parser.add_argument('--source-scales', type=float, nargs='*', default=[15.0, 18.0, 20.0, 22.0, 25.0])
    parser.add_argument('--target-scale-mm', type=float, default=CANONICAL_SCALE_MM)
    parser.add_argument('--checkpoint-dir', type=str, default=str(default_checkpoint_dir('standalone_stea_adapter')))
    parser.add_argument('--regressor-ckpt', type=str, required=True)
    parser.add_argument('--background-latent-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb-project', type=str, default='STEA_Adapter')
    parser.add_argument('--wandb-name', type=str, default='stea_adapter_train')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--eval-test', action='store_true')
    parser.add_argument('--use-boundary-smoothing', action='store_true')
    parser.add_argument('--lambda-latent', type=float, default=1.0)
    parser.add_argument('--lambda-task', type=float, default=1.0)
    parser.add_argument('--lambda-id', type=float, default=0.5)
    parser.add_argument('--lambda-mod', type=float, default=0.01)
    parser.add_argument('--pairs-per-epoch', type=int, default=24000)
    parser.add_argument('--val-pairs-per-epoch', type=int, default=2000)
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
    model: STEAAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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
            'stea_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': float(best_val_loss),
            'config': vars(args),
            'adapter_kind': 'stea',
        },
        path,
    )


def run_epoch(
    *,
    model: STEAAdapter,
    vit: FrozenViTPatchExtractor,
    canonical_head: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    desc: str,
    max_batches: int | None,
    progress: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    metric_rows: list[dict[str, float]] = []
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        source_image = batch['source_image'].to(device=device, dtype=torch.float32)
        target_image = batch['target_image'].to(device=device, dtype=torch.float32)
        source_scale_mm = batch['source_scale_mm'].to(device=device, dtype=torch.float32)
        target_scale_mm = batch['target_scale_mm'].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            source_features = vit(source_image)
            target_features = vit(target_image)
            teacher_pred = canonical_head(target_features)

        adapted_features, aux = model(
            source_features,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
        )
        student_pred = canonical_head(adapted_features)
        total_loss, metrics = compute_stea_losses(
            adapted_features=adapted_features,
            target_features=target_features,
            student_pred=student_pred,
            teacher_pred=teacher_pred,
            source_features=source_features,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
            aux=aux,
            lambda_latent=args.lambda_latent,
            lambda_task=args.lambda_task,
            lambda_id=args.lambda_id,
            lambda_mod=args.lambda_mod,
        )

        grad_norm_value: float | None = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            optimizer.step()

        row = {key: float(value.item()) for key, value in metrics.items()}
        if grad_norm_value is not None:
            row['grad_norm'] = grad_norm_value
        metric_rows.append(row)
        running = aggregate_epoch_metrics(metric_rows)
        if progress and hasattr(iterator, 'set_postfix'):
            postfix = {
                'loss': f"{running.get('loss', 0.0):.4f}",
                'latent': f"{running.get('latent_loss', 0.0):.4f}",
                'task': f"{running.get('task_loss', 0.0):.4f}",
                'valid': f"{running.get('valid_ratio', 0.0):.3f}",
            }
            if grad_norm_value is not None:
                postfix['grad'] = f"{running.get('grad_norm', 0.0):.2f}"
            iterator.set_postfix(postfix)

    return aggregate_epoch_metrics(metric_rows)


def main() -> None:
    args = parse_args()
    if abs(args.canonical_scale_mm - args.target_scale_mm) > 1e-6:
        raise ValueError(
            '--canonical-scale-mm and --target-scale-mm must match for STEA. '
            f'Got {args.canonical_scale_mm} vs {args.target_scale_mm}'
        )

    seed_everything(args.seed)
    device = torch.device(args.device)
    progress = not args.no_progress
    checkpoint_dir = resolve_path(args.checkpoint_dir, base_dir=PROJECT_ROOT)

    train_dataset, val_dataset = build_stea_pair_datasets(
        dataset_root=args.dataset_root,
        target_scale_mm=args.target_scale_mm,
        source_scales=args.source_scales,
        pairs_per_epoch=args.pairs_per_epoch,
        val_pairs_per_epoch=args.val_pairs_per_epoch,
        seed=args.seed,
    )
    train_loader = create_loader(train_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=True)
    val_loader = create_loader(val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False)

    vit = FrozenViTPatchExtractor(model_name=args.model_name, pretrained=True).to(device)
    vit.eval()
    canonical_head = load_canonical_head_from_regressor_ckpt(args.regressor_ckpt, device, freeze=True)
    model = STEAAdapter(
        background_latent_map=resolve_path(args.background_latent_path, base_dir=PROJECT_ROOT),
        use_boundary_smoothing=args.use_boundary_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from:
        payload = torch.load(resolve_path(args.resume_from, base_dir=PROJECT_ROOT), map_location=device, weights_only=False)
        state_dict = payload.get('model_state_dict') or payload.get('stea_state_dict')
        if state_dict is None:
            raise KeyError(f'Resume checkpoint is missing model_state_dict/stea_state_dict: {args.resume_from}')
        model.load_state_dict(state_dict)
        if 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        if 'scheduler_state_dict' in payload:
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        start_epoch = int(payload.get('epoch', 0)) + 1
        best_val_loss = float(payload.get('best_val_loss', best_val_loss))
        print(f'Resumed from {args.resume_from} | start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}')

    print(
        'Training STEA adapter | '
        f'device={device} | train_pairs={len(train_dataset)} | val_pairs={len(val_dataset)} | '
        f'canonical_scale={args.canonical_scale_mm:.1f} | source_scales={list(args.source_scales)} | '
        f'use_boundary_smoothing={args.use_boundary_smoothing}'
    )

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    train_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model=model,
                vit=vit,
                canonical_head=canonical_head,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                args=args,
                desc=f'Train {epoch}/{args.epochs}',
                max_batches=args.max_train_batches,
                progress=progress,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    vit=vit,
                    canonical_head=canonical_head,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    args=args,
                    desc=f'Val   {epoch}/{args.epochs}',
                    max_batches=args.max_val_batches,
                    progress=progress,
                )
            scheduler.step()

            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - train_start
            epochs_done = epoch - start_epoch + 1
            eta_seconds = (elapsed_seconds / max(epochs_done, 1)) * max(args.epochs - epoch, 0)
            current_lr = optimizer.param_groups[0]['lr']
            candidate_best = min(best_val_loss, val_metrics.get('loss', float('inf')))

            wandb.log(
                {
                    'epoch': epoch,
                    'lr': current_lr,
                    'epoch_seconds': epoch_seconds,
                    'elapsed_seconds': elapsed_seconds,
                    'train_loss': train_metrics.get('loss', 0.0),
                    'val_loss': val_metrics.get('loss', 0.0),
                    'latent_loss': val_metrics.get('latent_loss', 0.0),
                    'task_loss': val_metrics.get('task_loss', 0.0),
                    'id_loss': val_metrics.get('id_loss', 0.0),
                    'mod_loss': val_metrics.get('mod_loss', 0.0),
                    'task_x': val_metrics.get('task_x', 0.0),
                    'task_y': val_metrics.get('task_y', 0.0),
                    'task_depth': val_metrics.get('task_depth', 0.0),
                    'valid_ratio': val_metrics.get('valid_ratio', 0.0),
                    'residual_norm': val_metrics.get('residual_norm', 0.0),
                    'gamma_abs': val_metrics.get('gamma_abs', 0.0),
                    'beta_abs': val_metrics.get('beta_abs', 0.0),
                    'train_grad_norm': train_metrics.get('grad_norm', 0.0),
                }
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_metrics.get('loss', 0.0):.6f} | "
                f"val_loss={val_metrics.get('loss', 0.0):.6f} | latent_loss={val_metrics.get('latent_loss', 0.0):.6f} | "
                f"task_loss={val_metrics.get('task_loss', 0.0):.6f} | id_loss={val_metrics.get('id_loss', 0.0):.6f} | "
                f"mod_loss={val_metrics.get('mod_loss', 0.0):.6f} | best_val={candidate_best:.6f} | lr={current_lr:.2e} | "
                f"epoch_t={format_seconds(epoch_seconds)} | elapsed={format_seconds(elapsed_seconds)} | eta={format_seconds(eta_seconds)}"
            )

            save_checkpoint(
                checkpoint_dir / 'last.pt',
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=candidate_best,
            )

            if val_metrics.get('loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    checkpoint_dir / 'best.pt',
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
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
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
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
