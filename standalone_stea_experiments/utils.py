from __future__ import annotations

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

from standalone_stea.data import build_downstream_split_dataset, create_loader, unpack_downstream_batch
from standalone_stea.models.stea_adapter import STEAAdapter
from standalone_stea.utils import (
    CANONICAL_SCALE_MM,
    DOCKER_CHECKPOINT_ROOT,
    EMBED_DIM,
    FrozenViTPatchExtractor,
    HOST_CHECKPOINT_ROOT,
    STEARegressorHead,
    default_downstream_dataset_root,
    format_seconds,
    load_trained_head_checkpoint,
    resolve_path,
    seed_everything,
    wandb,
)

EXPERIMENT_CHECKPOINT_DIRNAME = 'standalone_stea_experiments'
DEFAULT_STEA_CKPT = '/datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt'


def default_experiment_checkpoint_root() -> Path:
    host = HOST_CHECKPOINT_ROOT / EXPERIMENT_CHECKPOINT_DIRNAME
    docker = DOCKER_CHECKPOINT_ROOT / EXPERIMENT_CHECKPOINT_DIRNAME
    if host.parent.exists():
        return host
    return docker


def default_experiment_checkpoint_dir(name: str) -> Path:
    return default_experiment_checkpoint_root() / name


def default_results_dir() -> Path:
    return default_experiment_checkpoint_root() / 'results'


def load_frozen_stea(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[STEAAdapter, dict[str, Any]]:
    payload = torch.load(resolve_path(checkpoint_path), map_location=device, weights_only=False)
    state_dict = payload.get('model_state_dict') or payload.get('stea_state_dict') or payload
    if not isinstance(state_dict, dict):
        raise ValueError(f'Checkpoint does not contain a valid STEA state dict: {checkpoint_path}')
    bg_map = state_dict.get('background_latent_map')
    if bg_map is None:
        raise KeyError(f'STEA checkpoint is missing background_latent_map: {checkpoint_path}')
    hidden_dim = int(state_dict['modulation_mlp.0.weight'].shape[0])
    use_boundary_smoothing = 'boundary_smoother.weight' in state_dict
    stea = STEAAdapter(
        embed_dim=EMBED_DIM,
        background_latent_map=bg_map,
        condition_hidden_dim=hidden_dim,
        use_boundary_smoothing=use_boundary_smoothing,
    ).to(device)
    stea.load_state_dict(state_dict, strict=True)
    stea.eval()
    for param in stea.parameters():
        param.requires_grad = False
    return stea, payload


def create_train_val_loaders(
    *,
    dataset_root: str | Path,
    train_scales: list[float],
    canonical_scale_mm: float,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader[Any], DataLoader[Any], Any, Any]:
    train_dataset = build_downstream_split_dataset(
        root_dir=dataset_root,
        scales_mm=train_scales,
        split='train',
        reference_scale_mm=canonical_scale_mm,
    )
    val_dataset = build_downstream_split_dataset(
        root_dir=dataset_root,
        scales_mm=train_scales,
        split='val',
        reference_scale_mm=canonical_scale_mm,
    )
    train_loader = create_loader(train_dataset, batch_size=batch_size, workers=workers, shuffle=True)
    val_loader = create_loader(val_dataset, batch_size=batch_size, workers=workers, shuffle=False)
    return train_loader, val_loader, train_dataset, val_dataset


def save_head_checkpoint(
    path: Path,
    *,
    model: STEARegressorHead,
    optimizer: torch.optim.Optimizer,
    args: Any,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    pipeline_kind: str,
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
            'adapter_kind': pipeline_kind,
        },
        path,
    )


def run_head_epoch(
    *,
    head: STEARegressorHead,
    vit: FrozenViTPatchExtractor,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    canonical_scale_mm: float,
    stea: STEAAdapter | None,
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
            key: float(sum(item[key] for item in metric_rows) / max(len(metric_rows), 1))
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


def train_regressor_pipeline(
    *,
    args: Any,
    use_stea: bool,
    description: str,
    pipeline_kind: str,
) -> None:
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    progress = not args.no_progress
    checkpoint_dir = resolve_path(args.checkpoint_dir)

    train_loader, val_loader, train_dataset, val_dataset = create_train_val_loaders(
        dataset_root=args.dataset_root,
        train_scales=[float(scale) for scale in args.train_scales],
        canonical_scale_mm=float(args.canonical_scale_mm),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )

    vit = FrozenViTPatchExtractor(model_name=args.model_name, pretrained=True).to(device)
    vit.eval()
    stea = None
    if use_stea:
        stea, _ = load_frozen_stea(args.stea_ckpt, device)
    head = STEARegressorHead(out_dim=3).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from:
        payload = torch.load(resolve_path(args.resume_from), map_location=device, weights_only=False)
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
        f'{description} | '
        f'device={device} | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | '
        f'train_scales={[float(scale) for scale in args.train_scales]} | canonical_scale={float(args.canonical_scale_mm):.1f}'
    )

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    train_start = time.time()
    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            epoch_start = time.time()
            train_metrics = run_head_epoch(
                head=head,
                vit=vit,
                stea=stea,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                canonical_scale_mm=float(args.canonical_scale_mm),
                max_batches=args.max_train_batches,
                desc=f'Train {epoch}/{args.epochs}',
                progress=progress,
            )
            with torch.no_grad():
                val_metrics = run_head_epoch(
                    head=head,
                    vit=vit,
                    stea=stea,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    canonical_scale_mm=float(args.canonical_scale_mm),
                    max_batches=args.max_val_batches,
                    desc=f'Val   {epoch}/{args.epochs}',
                    progress=progress,
                )

            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - train_start
            epochs_done = epoch - start_epoch + 1
            eta_seconds = (elapsed_seconds / max(epochs_done, 1)) * max(int(args.epochs) - epoch, 0)
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

            save_head_checkpoint(
                checkpoint_dir / 'last.pt',
                model=head,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=candidate_best,
                pipeline_kind=pipeline_kind,
            )
            if val_metrics['mse_total'] < best_val_loss:
                best_val_loss = val_metrics['mse_total']
                save_head_checkpoint(
                    checkpoint_dir / 'best.pt',
                    model=head,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                    pipeline_kind=pipeline_kind,
                )
                print(f'Saved new best checkpoint to {checkpoint_dir / "best.pt"}')
            if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
                epoch_path = checkpoint_dir / f'epoch_{epoch:04d}.pt'
                save_head_checkpoint(
                    epoch_path,
                    model=head,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                    pipeline_kind=pipeline_kind,
                )
                print(f'Saved periodic checkpoint to {epoch_path}')
    finally:
        wandb.finish()

    if args.eval_test:
        print('Final validation metrics:', val_metrics)


def build_common_train_parser(
    *,
    description: str,
    default_train_scales: list[float],
    default_checkpoint_name: str,
    default_wandb_project: str,
    default_wandb_name: str,
    require_stea: bool,
) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset-root', type=str, default=str(default_downstream_dataset_root()))
    if require_stea:
        parser.add_argument('--stea-ckpt', type=str, default=DEFAULT_STEA_CKPT)
    parser.add_argument('--train-scales', type=float, nargs='*', default=default_train_scales)
    parser.add_argument('--canonical-scale-mm', type=float, default=CANONICAL_SCALE_MM)
    parser.add_argument('--checkpoint-dir', type=str, default=str(default_experiment_checkpoint_dir(default_checkpoint_name)))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--wandb-project', type=str, default=default_wandb_project)
    parser.add_argument('--wandb-name', type=str, default=default_wandb_name)
    parser.add_argument('--eval-test', action='store_true')
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--no-progress', action='store_true')
    return parser


def evaluate_pipeline(
    *,
    dataset_root: str | Path,
    regressor_ckpt: str | Path,
    pipeline_kind: str,
    scale_mm: float,
    split: str,
    canonical_scale_mm: float,
    device: torch.device,
    batch_size: int,
    workers: int,
    model_name: str,
    stea_ckpt: str | Path | None = None,
) -> dict[str, float]:
    vit = FrozenViTPatchExtractor(model_name=model_name, pretrained=True).to(device)
    vit.eval()
    head, _ = load_trained_head_checkpoint(regressor_ckpt, device, freeze=True)

    stea = None
    adapter_kind = 'none'
    if pipeline_kind == 'with_stea':
        if not stea_ckpt:
            raise ValueError('stea_ckpt is required when pipeline_kind=with_stea')
        stea, _ = load_frozen_stea(stea_ckpt, device)
        adapter_kind = 'stea'
    elif pipeline_kind != 'no_adapter':
        raise ValueError(f'Unsupported pipeline_kind: {pipeline_kind}')

    dataset = build_downstream_split_dataset(
        root_dir=dataset_root,
        scales_mm=[float(scale_mm)],
        split=split,
        reference_scale_mm=float(canonical_scale_mm),
    )
    loader = create_loader(dataset, batch_size=int(batch_size), workers=int(workers), shuffle=False)

    sum_sq_x = 0.0
    sum_sq_y = 0.0
    sum_sq_depth = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            images, targets, _target_coords, _source_coords, source_scale_mm = unpack_downstream_batch(batch, device)
            features = vit(images)
            if stea is not None:
                features, _ = stea(
                    features,
                    source_scale_mm=source_scale_mm,
                    target_scale_mm=float(canonical_scale_mm),
                )
            preds = head(features)
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
        'use_stea': bool(stea is not None),
        'adapter_kind': adapter_kind,
    }

