#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.metrics.regression_metrics import compute_regression_metrics
from sccwm.models import DWLTR
from sccwm.train.common import (
    _build_paired_loader,
    _sample_train_episode_subset,
    _tqdm_metric_postfix,
    build_stage_argparser,
    default_device,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    wandb,
)
from sccwm.utils.config import dump_json, format_seconds, load_config_with_overrides, resolve_path, seed_everything

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def build_dwl_tr_model(cfg: dict[str, Any], device: torch.device) -> DWLTR:
    model_cfg = cfg.get("model", {})
    model = DWLTR(
        input_channels=int(model_cfg.get("input_channels", 3)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        world_hidden_dim=int(model_cfg.get("world_hidden_dim", 128)),
        lattice_size=int(model_cfg.get("lattice_size", 32)),
    )
    return model.to(device)


def _state_supervision_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_x = F.smooth_l1_loss(pred[:, 0], target[:, 0])
    loss_y = F.smooth_l1_loss(pred[:, 1], target[:, 1])
    loss_depth = F.smooth_l1_loss(pred[:, 2], target[:, 2])
    total = loss_x + loss_y + loss_depth
    return total, {
        "state_x": loss_x.detach(),
        "state_y": loss_y.detach(),
        "state_depth": loss_depth.detach(),
    }


def _temporal_depth_regularizer(pred_depth_seq: torch.Tensor, phase_names: list[list[str]] | list[tuple[str, ...]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if pred_depth_seq.shape[1] <= 1:
        zero = pred_depth_seq.new_zeros(())
        return zero, {"temporal_depth_smooth": zero.detach(), "temporal_monotonic": zero.detach()}
    depth_smooth = (pred_depth_seq[:, 1:] - pred_depth_seq[:, :-1]).abs().mean()
    monotonic_terms: list[torch.Tensor] = []
    for batch_idx, phases in enumerate(phase_names):
        for step in range(1, len(phases)):
            prev_depth = pred_depth_seq[batch_idx, step - 1]
            cur_depth = pred_depth_seq[batch_idx, step]
            phase = str(phases[step])
            if phase == "press":
                monotonic_terms.append(F.relu(prev_depth - cur_depth))
            elif phase == "release":
                monotonic_terms.append(F.relu(cur_depth - prev_depth))
    monotonic = torch.stack(monotonic_terms).mean() if monotonic_terms else pred_depth_seq.new_zeros(())
    return depth_smooth + monotonic, {
        "temporal_depth_smooth": depth_smooth.detach(),
        "temporal_monotonic": monotonic.detach(),
    }


def _compute_dwl_tr_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = torch.stack([outputs["pred_x_norm"], outputs["pred_y_norm"], outputs["pred_depth_mm"]], dim=1)
    target = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
    state_loss, state_parts = _state_supervision_loss(pred, target)
    temporal_loss, temporal_parts = _temporal_depth_regularizer(outputs["pred_depth_mm_seq"], batch["phase_names"])
    total = loss_cfg.get("state_supervision", 1.0) * state_loss
    total = total + loss_cfg.get("temporal_consistency", 0.0) * temporal_loss
    metrics: dict[str, torch.Tensor] = {
        "loss": total.detach(),
        "state_supervision": state_loss.detach(),
        **state_parts,
        **temporal_parts,
    }
    reg = compute_regression_metrics(pred.detach(), target.detach())
    metrics.update({f"pred_{k}": torch.tensor(v, device=pred.device) for k, v in reg.items()})
    metrics.update({f"state_{k}": torch.tensor(v, device=pred.device) for k, v in reg.items()})
    return total, metrics


def _aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1)) for key in keys}


def run_dwl_tr_epoch(
    model: DWLTR,
    loader: torch.utils.data.DataLoader[Any],
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
        outputs = model(
            batch["source_obs"],
            batch["source_coord_map"],
            batch["source_scale_mm"],
            valid_mask=batch["seq_valid_mask"],
            absolute_contact_xy_mm=abs_contact,
            world_origin_xy_mm=batch["world_origin_xy_mm"],
        )
        total_loss, metrics_t = _compute_dwl_tr_losses(outputs, batch, loss_cfg)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(loss_cfg.get("grad_clip_norm", 1.0)))
            optimizer.step()
        metrics = {key: float(value.item()) for key, value in metrics_t.items()}
        rows.append(metrics)
        if progress:
            postfix = _tqdm_metric_postfix(rows)
            if postfix:
                iterator.set_postfix(postfix)
    return _aggregate_rows(rows)


def train_dwl_tr(cfg: dict[str, Any]) -> None:
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

    val_loader = _build_paired_loader(
        cfg,
        split=val_split,
        batch_size=val_batch_size,
        workers=workers,
        sequence_length=seq_len,
        shuffle=False,
    )
    model = build_dwl_tr_model(cfg, device)
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
    best_metric_name = str(train_cfg.get("best_metric", "state_mae_mean"))
    best_mode = str(train_cfg.get("best_mode", "min"))
    best_value = float("inf") if best_mode == "min" else float("-inf")
    start_epoch = 1

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

    wandb.init(
        project=str(cfg.get("system", {}).get("wandb_project", "SCCWM")),
        name=str(cfg.get("system", {}).get("wandb_name", cfg.get("experiment", {}).get("name", "dwl_tr"))),
        config=cfg,
    )
    start_time = time.time()
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.time()
            epoch_episode_subset = _sample_train_episode_subset(cfg, epoch=epoch)
            if epoch_episode_subset is not None:
                print(
                    f"[dwl_tr] epoch={epoch} sampled {len(epoch_episode_subset)} train episodes for this epoch "
                    f"(dynamic episode subset enabled)"
                )
            train_loader = _build_paired_loader(
                cfg,
                split=train_split,
                batch_size=batch_size,
                workers=workers,
                sequence_length=seq_len,
                shuffle=True,
                episode_id_subset=epoch_episode_subset,
            )
            train_metrics = run_dwl_tr_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_cfg=loss_cfg,
                desc=f"Train dwl_tr {epoch}",
                progress=not bool(train_cfg.get("no_progress", False)),
                max_batches=max_train_batches,
            )
            with torch.no_grad():
                val_metrics = run_dwl_tr_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    loss_cfg=loss_cfg,
                    desc=f"Val dwl_tr {epoch}",
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
            with metrics_history_path.open("a", encoding="utf-8") as f:
                import json

                f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

            current = float(val_metrics.get(best_metric_name, val_metrics.get("loss", 0.0)))
            improved = current < best_value if best_mode == "min" else current > best_value
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": cfg,
                "best_value": best_value,
            }
            save_checkpoint(output_dir / "last.pt", payload)
            print(f"[dwl_tr] saved last checkpoint -> {output_dir / 'last.pt'}")
            if improved:
                best_value = current
                payload["best_value"] = best_value
                save_checkpoint(output_dir / "best.pt", payload)
                print(f"[dwl_tr] updated best checkpoint -> {output_dir / 'best.pt'} ({best_metric_name}={best_value:.4f})")
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
                print(f"[dwl_tr] saved periodic checkpoint -> {output_dir / f'epoch_{epoch:04d}.pt'}")

            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            avg_epoch_seconds = elapsed / max(epochs_done, 1)
            remaining_epochs = max(total_epochs - epoch, 0)
            eta_seconds = avg_epoch_seconds * remaining_epochs
            print(
                f"[dwl_tr] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
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
                "stage": "dwl_tr",
                "best_value": best_value,
                "elapsed_seconds": time.time() - start_time,
                "config": cfg,
            },
            output_dir / "train_summary.json",
        )


def main() -> None:
    parser = build_stage_argparser("Train the DWL-TR baseline.", "sccwm/configs/dwl_tr.yaml")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args.config, args.override)
    train_dwl_tr(cfg)


if __name__ == "__main__":
    main()
