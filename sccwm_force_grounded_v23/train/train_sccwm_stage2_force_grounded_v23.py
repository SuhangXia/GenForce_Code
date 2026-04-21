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
    _tqdm_metric_postfix,
    build_stage_argparser,
    default_device,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    wandb,
)
from sccwm.utils.config import dump_json, format_seconds, load_config_with_overrides, resolve_path, seed_everything
from sccwm_force_grounded_v21a.train.train_sccwm_stage2_force_grounded_v21a import (
    _build_overlay_loader,
    _first_batch,
    _load_init_weights,
    _sample_train_episode_subset,
)
from sccwm_force_grounded_v23.losses import compute_force_grounded_v23_losses, summarize_force_grounded_v23_batch
from sccwm_force_grounded_v23.models import build_force_grounded_v23_model

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _stage_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("experiment", {}).get("name", "stage2_force_grounded_v23"))


def build_force_grounded_v23_model_for_train(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    return build_force_grounded_v23_model(cfg.get("model", {})).to(device)


def _preflight_forward_metrics_v23(
    model: torch.nn.Module,
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    batch = move_batch_to_device(batch, device)
    abs_contact = batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None
    with torch.no_grad():
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
        loss_out = compute_force_grounded_v23_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
    return {key: float(value.item()) for key, value in loss_out.metrics.items()}


def _print_preflight_summary_v23(stage_name: str, label: str, summary: dict[str, Any], forward_metrics: dict[str, float]) -> None:
    print(f"[{stage_name}] preflight {label}: anchor_mode={summary['anchor_selection_mode']} anchor_stats={summary['anchor_index_stats']}")
    print(
        f"[{stage_name}] preflight {label}: penetration_stats={summary['penetration_proxy_target_stats']} "
        f"contact_stats={summary['contact_intensity_target_stats']} load_stats={summary['load_progress_target_stats']}"
    )
    if "marker_metadata" in summary:
        print(f"[{stage_name}] preflight {label}: marker_metadata={summary['marker_metadata']}")
    for example in summary.get("examples", []):
        print(
            f"[{stage_name}] preflight {label}: batch_idx={example['batch_index']} "
            f"episode={example['episode_id']} anchor={example['anchor_index_in_window']} "
            f"phase={example['phase_name']} progress={example['phase_progress']:.4f} "
            f"load_target={example['load_progress_target']:.4f}"
        )
    print(
        f"[{stage_name}] preflight {label}: "
        f"force_scale_adv_acc={forward_metrics.get('force_scale_adv_acc', float('nan')):.4f} "
        f"force_branch_adv_acc={forward_metrics.get('force_branch_adv_acc', float('nan')):.4f} "
        f"geometry_scale_adv_acc={forward_metrics.get('geometry_scale_adv_acc', float('nan')):.4f} "
        f"geometry_branch_adv_acc={forward_metrics.get('geometry_branch_adv_acc', float('nan')):.4f} "
        f"geometry_marker_adv_acc={forward_metrics.get('geometry_marker_adv_acc', float('nan')):.4f} "
        f"canonical_scale_adv_acc={forward_metrics.get('canonical_scale_adv_acc', float('nan')):.4f} "
        f"canonical_branch_adv_acc={forward_metrics.get('canonical_branch_adv_acc', float('nan')):.4f} "
        f"canonical_marker_adv_acc={forward_metrics.get('canonical_marker_adv_acc', float('nan')):.4f} "
        f"operator_scale_acc={forward_metrics.get('operator_scale_acc', float('nan')):.4f} "
        f"operator_branch_acc={forward_metrics.get('operator_branch_acc', float('nan')):.4f} "
        f"operator_marker_acc={forward_metrics.get('operator_marker_acc', float('nan')):.4f}"
    )


def run_force_grounded_v23_epoch(
    stage_name: str,
    model: torch.nn.Module,
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
        loss_out = compute_force_grounded_v23_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
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
    return {key: float(sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1)) for key in rows[0].keys()} if rows else {}


def train_force_grounded_stage2_v23(cfg: dict[str, Any], *, sanity_only: bool = False) -> None:
    stage_name = _stage_name(cfg)
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
    model = build_force_grounded_v23_model_for_train(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    total_epochs = int(train_cfg.get("epochs", 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    output_dir = resolve_path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
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

    preflight_epoch = start_epoch
    preflight_episode_subset = _sample_train_episode_subset(cfg, epoch=preflight_epoch)
    if preflight_episode_subset is not None:
        print(
            f"[{stage_name}] preflight epoch={preflight_epoch} sampled {len(preflight_episode_subset)} train episodes "
            f"(dynamic episode subset enabled)"
        )
    preflight_train_loader = _build_overlay_loader(
        cfg,
        split=train_split,
        batch_size=batch_size,
        workers=workers,
        sequence_length=seq_len,
        shuffle=True,
        device=device,
        episode_id_subset=preflight_episode_subset,
    )
    val_loader = _build_overlay_loader(
        cfg,
        split=val_split,
        batch_size=val_batch_size,
        workers=workers,
        sequence_length=seq_len,
        shuffle=False,
        device=device,
    )
    train_preflight_batch = _first_batch(preflight_train_loader)
    val_preflight_batch = _first_batch(val_loader)
    train_preflight_summary = summarize_force_grounded_v23_batch(train_preflight_batch, loss_cfg)
    val_preflight_summary = summarize_force_grounded_v23_batch(val_preflight_batch, loss_cfg)
    train_preflight_forward = _preflight_forward_metrics_v23(model, train_preflight_batch, loss_cfg, device)
    val_preflight_forward = _preflight_forward_metrics_v23(model, val_preflight_batch, loss_cfg, device)
    preflight_report = {
        "stage": stage_name,
        "train": {"summary": train_preflight_summary, "forward_metrics": train_preflight_forward},
        "val": {"summary": val_preflight_summary, "forward_metrics": val_preflight_forward},
    }
    dump_json(preflight_report, output_dir / "preflight_sanity.json")
    _print_preflight_summary_v23(stage_name, "train", train_preflight_summary, train_preflight_forward)
    _print_preflight_summary_v23(stage_name, "val", val_preflight_summary, val_preflight_forward)
    if sanity_only:
        dump_json({"stage": stage_name, "status": "sanity_only_completed", "output_dir": str(output_dir)}, output_dir / "sanity_only_summary.json")
        print(f"[{stage_name}] sanity-only completed -> {output_dir / 'preflight_sanity.json'}")
        return

    wandb.init(
        project=str(cfg.get("system", {}).get("wandb_project", "SCCWM")),
        name=str(cfg.get("system", {}).get("wandb_name", cfg.get("experiment", {}).get("name", stage_name))),
        config=copy.deepcopy(cfg),
    )
    start_time = time.time()
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.time()
            if epoch == preflight_epoch:
                epoch_episode_subset = preflight_episode_subset
                train_loader = preflight_train_loader
            else:
                epoch_episode_subset = _sample_train_episode_subset(cfg, epoch=epoch)
                if epoch_episode_subset is not None:
                    print(
                        f"[{stage_name}] epoch={epoch} sampled {len(epoch_episode_subset)} train episodes "
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
            train_metrics = run_force_grounded_v23_epoch(
                stage_name,
                model,
                train_loader,
                optimizer,
                device,
                loss_cfg=loss_cfg,
                desc=f"Train {stage_name} {epoch}",
                progress=not bool(train_cfg.get("no_progress", False)),
                max_batches=max_train_batches,
            )
            with torch.no_grad():
                val_metrics = run_force_grounded_v23_epoch(
                    stage_name,
                    model,
                    val_loader,
                    None,
                    device,
                    loss_cfg=loss_cfg,
                    desc=f"Val {stage_name} {epoch}",
                    progress=not bool(train_cfg.get("no_progress", False)),
                    max_batches=max_val_batches,
                )
            scheduler.step()
            log_row = {f"train/{k}": v for k, v in train_metrics.items()}
            log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_row["epoch"] = epoch
            log_row["epoch_seconds"] = time.time() - epoch_start
            wandb.log(log_row)
            with metrics_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            current = float(val_metrics.get(best_metric_name, val_metrics.get("loss", 0.0)))
            improved = current < best_value if best_mode == "min" else current > best_value
            current_best = current if improved else best_value
            payload = {
                "epoch": epoch,
                "stage": stage_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": cfg,
                "best_value": current_best,
            }
            save_checkpoint(output_dir / "last.pt", payload)
            print(f"[{stage_name}] saved last checkpoint -> {output_dir / 'last.pt'}")
            if improved:
                best_value = current
                save_checkpoint(output_dir / "best.pt", payload)
                print(f"[{stage_name}] updated best checkpoint -> {output_dir / 'best.pt'} ({best_metric_name}={best_value:.4f})")
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
                print(f"[{stage_name}] saved periodic checkpoint -> {output_dir / f'epoch_{epoch:04d}.pt'}")
            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            avg_epoch_seconds = elapsed / max(epochs_done, 1)
            remaining_epochs = max(total_epochs - epoch, 0)
            eta_seconds = avg_epoch_seconds * remaining_epochs
            print(
                f"[{stage_name}] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
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
                "stage": stage_name,
                "output_dir": str(output_dir),
                "metrics_history": str(metrics_history_path),
                "best_metric": best_metric_name,
                "best_mode": best_mode,
                "best_value": best_value,
            },
            output_dir / "train_summary.json",
        )


def main() -> None:
    parser = build_stage_argparser(
        "Train geometry-focused force-grounded SCCWM v23 variants.",
        "sccwm_force_grounded_v23/configs/sccwm_stage2_force_grounded_v23gof.yaml",
    )
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args.config, args.override)
    train_force_grounded_stage2_v23(cfg, sanity_only=bool(args.sanity_only))


if __name__ == "__main__":
    main()
