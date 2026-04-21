#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.metrics.regression_metrics import compute_regression_metrics
from sccwm.train.common import (
    _safe_loader_workers,
    _tqdm_metric_postfix,
    build_stage_argparser,
    default_device,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    wandb,
)
from sccwm.utils.config import dump_json, format_seconds, load_config_with_overrides, resolve_path, seed_everything
from sccwm_force_grounded_v21.datasets import OverlayPairedSequenceDataset
from sccwm_force_grounded_v21.losses import compute_force_grounded_v21_losses
from sccwm_force_grounded_v21.models import SCCWMForceGroundedV21

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def build_force_grounded_v21_model(cfg: dict[str, Any], device: torch.device) -> SCCWMForceGroundedV21:
    model_cfg = cfg.get("model", {})
    model = SCCWMForceGroundedV21(
        input_channels=int(model_cfg.get("input_channels", 3)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        sensor_dim=int(model_cfg.get("sensor_dim", 64)),
        world_hidden_dim=int(model_cfg.get("world_hidden_dim", 128)),
        geometry_dim=int(model_cfg.get("geometry_dim", 64)),
        visibility_dim=int(model_cfg.get("visibility_dim", 32)),
        lattice_size=int(model_cfg.get("lattice_size", 32)),
        reconstruct_observation=bool(model_cfg.get("reconstruct_observation", False)),
        enable_contact_mask=bool(model_cfg.get("enable_contact_mask", False)),
        z_force_dim=int(model_cfg.get("z_force_dim", 6)),
        adv_grl_lambda=float(model_cfg.get("adv_grl_lambda", 0.25)),
        scale_bucket_edges_mm=[float(v) for v in model_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])],
    )
    return model.to(device)


def _select_overlay_root(ds_cfg: dict[str, Any], *, split: str, kind: str) -> Path:
    split = str(split)
    assert kind in {"image", "asset"}
    prefix = "dataset_a_root" if kind == "image" else "dataset_a_assets_root"
    per_split_key = {
        "train": f"train_{prefix}",
        "val": f"val_{prefix}",
        "test": f"test_{prefix}",
    }.get(split, prefix)
    root_value = ds_cfg.get(per_split_key)
    if root_value is None:
        root_value = ds_cfg.get("dataset_a_root" if kind == "image" else "dataset_a_assets_root")
    if root_value is None:
        raise KeyError(f"Missing dataset root for split={split!r}, kind={kind!r}")
    return resolve_path(root_value)


def _select_index_file(ds_cfg: dict[str, Any], *, split: str) -> str | Path | None:
    return {
        "train": ds_cfg.get("train_index_file"),
        "val": ds_cfg.get("val_index_file"),
        "test": ds_cfg.get("test_index_file"),
    }.get(str(split))


def _build_overlay_loader(
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    workers: int,
    sequence_length: int,
    shuffle: bool,
    device: torch.device,
    episode_id_subset: set[int] | None = None,
) -> DataLoader[Any]:
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    dataset = OverlayPairedSequenceDataset(
        _select_overlay_root(ds_cfg, split=split, kind="image"),
        _select_overlay_root(ds_cfg, split=split, kind="asset"),
        split=split,
        sequence_length=sequence_length,
        index_file=_select_index_file(ds_cfg, split=split),
        episode_id_subset=episode_id_subset,
        gray_cache_max_items=int(ds_cfg.get("gray_cache_max_items", 256)),
    )
    dataset_workers = _safe_loader_workers(
        workers,
        dataset,
        label=split,
        allow_large_index_workers=bool(train_cfg.get("allow_large_index_workers", False)) and shuffle and episode_id_subset is not None,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": dataset_workers,
        "pin_memory": device.type == "cuda",
    }
    if dataset_workers > 0:
        loader_kwargs["persistent_workers"] = episode_id_subset is None
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def _discover_train_episode_buckets(asset_root: Path) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for meta_path in sorted(asset_root.glob("episode_*/metadata.json")):
        episode_id = int(meta_path.parent.name.split("_")[-1])
        payload = json.loads(meta_path.read_text())
        indenter = str(payload.get("indenter", "unknown"))
        buckets.setdefault(indenter, []).append(episode_id)
    for values in buckets.values():
        values.sort()
    return buckets


def _sample_train_episode_subset(cfg: dict[str, Any], *, epoch: int) -> set[int] | None:
    train_cfg = cfg.get("train", {})
    ds_cfg = cfg.get("dataset", {})
    per_indenter = int(train_cfg.get("sample_episodes_per_indenter_per_epoch", 0))
    total = int(train_cfg.get("sample_episodes_per_epoch", 0))
    if per_indenter <= 0 and total <= 0:
        return None
    train_root = _select_overlay_root(ds_cfg, split=str(ds_cfg.get("train_split", "train")), kind="asset")
    buckets = _discover_train_episode_buckets(train_root)
    rng = __import__("random").Random(int(cfg.get("experiment", {}).get("seed", 42)) + int(epoch))
    if per_indenter > 0:
        selected: set[int] = set()
        for episode_ids in buckets.values():
            if not episode_ids:
                continue
            if per_indenter >= len(episode_ids):
                selected.update(episode_ids)
            else:
                selected.update(rng.sample(episode_ids, per_indenter))
        return selected
    all_episode_ids = sorted({episode_id for values in buckets.values() for episode_id in values})
    if total >= len(all_episode_ids):
        return set(all_episode_ids)
    return set(rng.sample(all_episode_ids, total))


def _aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1)) for key in keys}


def _extract_checkpoint_state(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_state_dict", "state_dict"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload


def _load_compatible_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and torch.is_tensor(value) and current[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return len(compatible), list(missing), list(unexpected)


def _load_init_weights(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = load_checkpoint(path, device)
    state = _extract_checkpoint_state(payload)
    compatible_count, missing, unexpected = _load_compatible_state_dict(model, state)
    print(
        f"[stage2_force_grounded_v21] init_from={path} loaded {compatible_count} compatible tensors "
        f"(missing={len(missing)} unexpected={len(unexpected)})"
    )
    return payload


def run_force_grounded_v21_epoch(
    model: SCCWMForceGroundedV21,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    loss_cfg: dict[str, Any],
    desc: str,
    progress: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[dict[str, float]] = []
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress, total=total)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        abs_contact = batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None
        outputs = model.forward_pair(
            source_obs=batch["source_obs"],
            target_obs=batch["target_obs"],
            source_coord_map=batch["source_coord_map"],
            target_coord_map=batch["target_coord_map"],
            source_scale_mm=batch["source_scale_mm"],
            target_scale_mm=batch["target_scale_mm"],
            source_valid_mask=batch["seq_valid_mask"],
            target_valid_mask=batch["seq_valid_mask"],
            absolute_contact_xy_mm=abs_contact,
            world_origin_xy_mm=batch["world_origin_xy_mm"],
        )
        loss_out = compute_force_grounded_v21_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(loss_cfg.get("grad_clip_norm", 1.0)))
            optimizer.step()
        pred = torch.stack(
            [outputs["source"]["pred_x_norm"], outputs["source"]["pred_y_norm"], outputs["source"]["pred_depth_mm"]],
            dim=1,
        )
        target = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
        metrics = {key: float(value.item()) for key, value in loss_out.metrics.items()}
        state_metrics = compute_regression_metrics(pred.detach(), target.detach())
        metrics.update({f"pred_{k}": v for k, v in state_metrics.items()})
        metrics.update({f"state_{k}": v for k, v in state_metrics.items()})
        rows.append(metrics)
        if progress:
            postfix = _tqdm_metric_postfix(rows)
            if postfix:
                iterator.set_postfix(postfix)
    return _aggregate_rows(rows)


def train_force_grounded_stage2_v21(cfg: dict[str, Any]) -> None:
    device = default_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    train_cfg = cfg.get("train", {})
    ds_cfg = cfg.get("dataset", {})
    loss_cfg = cfg.get("loss", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    val_batch_size = int(train_cfg.get("val_batch_size", batch_size))
    workers = int(train_cfg.get("workers", 4))
    seq_len = int(ds_cfg.get("sequence_length", 3))
    train_split = str(ds_cfg.get("train_split", "train"))
    val_split = str(ds_cfg.get("val_split", "val"))
    model = build_force_grounded_v21_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    total_epochs = int(train_cfg.get("epochs", 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    output_dir = resolve_path(train_cfg["output_dir"])
    metrics_history_path = output_dir / "metrics_history.jsonl"
    max_train_samples = int(train_cfg.get("max_train_samples_per_epoch", 0))
    max_val_samples = int(train_cfg.get("max_val_samples", 0))
    max_train_batches = math.ceil(max_train_samples / max(batch_size, 1)) if max_train_samples > 0 else None
    max_val_batches = math.ceil(max_val_samples / max(val_batch_size, 1)) if max_val_samples > 0 else None
    best_metric_name = str(train_cfg.get("best_metric", "state_mae_depth"))
    best_mode = str(train_cfg.get("best_mode", "min"))
    best_value = float("inf") if best_mode == "min" else float("-inf")
    start_epoch = 1
    init_from = str(train_cfg.get("init_from", "")).strip()
    resume_from = str(train_cfg.get("resume_from", "")).strip()
    if resume_from:
        payload = load_checkpoint(resume_from, device)
        model.load_state_dict(payload["model_state_dict"], strict=False)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        best_value = float(payload.get("best_value", best_value))
        start_epoch = int(payload.get("epoch", 0)) + 1
    elif init_from:
        _load_init_weights(model, init_from, device)

    val_loader = _build_overlay_loader(
        cfg,
        split=val_split,
        batch_size=val_batch_size,
        workers=workers,
        sequence_length=seq_len,
        shuffle=False,
        device=device,
    )
    wandb.init(
        project=str(cfg.get("system", {}).get("wandb_project", "SCCWM")),
        name=str(cfg.get("system", {}).get("wandb_name", cfg.get("experiment", {}).get("name", "sccwm_force_grounded_v21"))),
        config=copy.deepcopy(cfg),
    )
    start_time = time.time()
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.time()
            epoch_episode_subset = _sample_train_episode_subset(cfg, epoch=epoch)
            if epoch_episode_subset is not None:
                print(
                    f"[stage2_force_grounded_v21] epoch={epoch} sampled {len(epoch_episode_subset)} train episodes for this epoch "
                    f"(dynamic episode subset enabled)"
                )
            train_loader = _build_overlay_loader(
                cfg,
                split=train_split,
                batch_size=batch_size,
                workers=workers,
                sequence_length=seq_len,
                shuffle=True,
                device=device,
                episode_id_subset=epoch_episode_subset,
            )
            train_metrics = run_force_grounded_v21_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_cfg=loss_cfg,
                desc=f"Train stage2_force_grounded_v21 {epoch}",
                progress=not bool(train_cfg.get("no_progress", False)),
                max_batches=max_train_batches,
            )
            with torch.no_grad():
                val_metrics = run_force_grounded_v21_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    loss_cfg=loss_cfg,
                    desc=f"Val stage2_force_grounded_v21 {epoch}",
                    progress=not bool(train_cfg.get("no_progress", False)),
                    max_batches=max_val_batches,
                )
            scheduler.step()
            log_row = {f"train/{k}": v for k, v in train_metrics.items()}
            log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_row["epoch"] = epoch
            log_row["epoch_seconds"] = time.time() - epoch_start
            wandb.log(log_row)
            output_dir.mkdir(parents=True, exist_ok=True)
            with metrics_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            current = float(val_metrics.get(best_metric_name, val_metrics.get("loss", 0.0)))
            improved = current < best_value if best_mode == "min" else current > best_value
            current_best = current if improved else best_value
            payload = {
                "epoch": epoch,
                "stage": "stage2_force_grounded_v21",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": cfg,
                "best_value": current_best,
            }
            save_checkpoint(output_dir / "last.pt", payload)
            print(f"[stage2_force_grounded_v21] saved last checkpoint -> {output_dir / 'last.pt'}")
            if improved:
                best_value = current
                save_checkpoint(output_dir / "best.pt", payload)
                print(
                    f"[stage2_force_grounded_v21] updated best checkpoint -> {output_dir / 'best.pt'} "
                    f"({best_metric_name}={best_value:.4f})"
                )
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
                print(f"[stage2_force_grounded_v21] saved periodic checkpoint -> {output_dir / f'epoch_{epoch:04d}.pt'}")
            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            avg_epoch_seconds = elapsed / max(epochs_done, 1)
            remaining_epochs = max(total_epochs - epoch, 0)
            eta_seconds = avg_epoch_seconds * remaining_epochs
            print(
                f"[stage2_force_grounded_v21] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
                f"val_loss={val_metrics.get('loss', 0.0):.4f} "
                f"best_{best_metric_name}={best_value:.4f} "
                f"epoch_time={format_seconds(time.time() - epoch_start)} "
                f"elapsed={format_seconds(elapsed)} "
                f"remaining={format_seconds(eta_seconds)} "
                f"eta={format_seconds(elapsed + eta_seconds)}"
            )
    finally:
        wandb.finish()
        dump_json(
            {
                "stage": "stage2_force_grounded_v21",
                "best_value": best_value,
                "elapsed_seconds": time.time() - start_time,
                "config": cfg,
            },
            output_dir / "train_summary.json",
        )


def main() -> None:
    parser = build_stage_argparser(
        "Train the force-grounded SCCWM v2.1 stage-2 model.",
        "sccwm_force_grounded_v21/configs/sccwm_stage2_force_grounded_v21.yaml",
    )
    args = parser.parse_args()
    cfg = load_config_with_overrides(args.config, args.override)
    train_force_grounded_stage2_v21(cfg)


if __name__ == "__main__":
    main()
