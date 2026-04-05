from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable

try:
    import wandb as _wandb  # type: ignore
except ImportError:  # pragma: no cover
    class _WandbStub:
        @staticmethod
        def init(*args: Any, **kwargs: Any) -> None:
            print("wandb is not installed; continuing with local logging only.")
            return None

        @staticmethod
        def log(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def finish(*args: Any, **kwargs: Any) -> None:
            return None
    wandb = _WandbStub()
else:
    wandb = _wandb

from sccwm.datasets import LegacyClipDataset, PairedSequenceDataset
from sccwm.losses import build_negative_labels, build_state_embedding, compute_sccwm_losses
from sccwm.metrics.regression_metrics import compute_regression_metrics
from sccwm.metrics.ccauc_metric import compute_ccauc
from sccwm.models import LegacyStaticRegressor, LegacyTemporalRegressor, SCCWM
from sccwm.utils.config import dump_json, format_seconds, resolve_path, seed_everything


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1)) for key in keys}


def build_sccwm_model(cfg: dict[str, Any], device: torch.device) -> SCCWM:
    model_cfg = cfg.get("model", {})
    model = SCCWM(
        input_channels=int(model_cfg.get("input_channels", 3)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        sensor_dim=int(model_cfg.get("sensor_dim", 64)),
        world_hidden_dim=int(model_cfg.get("world_hidden_dim", 128)),
        geometry_dim=int(model_cfg.get("geometry_dim", 64)),
        visibility_dim=int(model_cfg.get("visibility_dim", 32)),
        lattice_size=int(model_cfg.get("lattice_size", 32)),
        reconstruct_observation=bool(model_cfg.get("reconstruct_observation", True)),
        enable_contact_mask=bool(model_cfg.get("enable_contact_mask", False)),
    )
    return model.to(device)


def build_legacy_regressor(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    reg_cfg = cfg.get("legacy_regressor", {})
    reg_type = str(reg_cfg.get("type", "temporal"))
    if reg_type == "static":
        model = LegacyStaticRegressor(
            observation_channels=int(reg_cfg.get("observation_channels", 3)),
            feature_dim=int(reg_cfg.get("feature_dim", 128)),
            output_dim=int(reg_cfg.get("output_dim", 3)),
        )
    elif reg_type in {"temporal", "pooled"}:
        model = LegacyTemporalRegressor(
            observation_channels=int(reg_cfg.get("observation_channels", 3)),
            feature_dim=int(reg_cfg.get("feature_dim", 128)),
            hidden_dim=int(reg_cfg.get("hidden_dim", 128)),
            output_dim=int(reg_cfg.get("output_dim", 3)),
        )
    else:
        raise ValueError(f"Unsupported legacy_regressor.type={reg_type!r}")
    return model.to(device)


def default_device(cfg: dict[str, Any]) -> torch.device:
    sys_cfg = cfg.get("system", {})
    raw = str(sys_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    requested = torch.device(raw)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested in config but not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


def normalize_phase_batch(batch_phase_names: Any) -> list[list[str]]:
    if isinstance(batch_phase_names, list) and batch_phase_names:
        first = batch_phase_names[0]
        if isinstance(first, tuple):
            t = len(batch_phase_names)
            b = len(first)
            out = [[str(batch_phase_names[step][idx]) for step in range(t)] for idx in range(b)]
            return out
        if isinstance(first, list):
            return [[str(v) for v in row] for row in batch_phase_names]
    return []


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
        else:
            out[key] = value
    out["phase_names"] = normalize_phase_batch(batch.get("phase_names", []))
    return out


def build_paired_loaders(cfg: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    ds_cfg = cfg.get("dataset", {})
    batch_size = int(cfg.get("train", {}).get("batch_size", 8))
    val_batch_size = int(cfg.get("train", {}).get("val_batch_size", batch_size))
    workers = int(cfg.get("train", {}).get("workers", 4))
    root = resolve_path(ds_cfg["dataset_a_root"])
    seq_len = int(ds_cfg.get("sequence_length", 3))
    train_ds = PairedSequenceDataset(root, split=str(ds_cfg.get("train_split", "train")), sequence_length=seq_len)
    val_ds = PairedSequenceDataset(root, split=str(ds_cfg.get("val_split", "val")), sequence_length=seq_len)
    pin_memory = default_device(cfg).type == "cuda"
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
        DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
    )


def build_clip_loaders(cfg: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    ds_cfg = cfg.get("dataset", {})
    batch_size = int(cfg.get("train", {}).get("batch_size", 8))
    val_batch_size = int(cfg.get("train", {}).get("val_batch_size", batch_size))
    workers = int(cfg.get("train", {}).get("workers", 4))
    root = resolve_path(ds_cfg["dataset_b_root"])
    train_ds = LegacyClipDataset(root, split=str(ds_cfg.get("clip_train_split", "train")))
    val_ds = LegacyClipDataset(root, split=str(ds_cfg.get("clip_val_split", "val")))
    pin_memory = default_device(cfg).type == "cuda"
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
        DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
    )


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {path} is not a dict payload")
    return payload


def load_model_state(model: torch.nn.Module, path: str | Path, device: torch.device, key_candidates: list[str] | None = None) -> dict[str, Any]:
    payload = load_checkpoint(path, device)
    keys = key_candidates or ["model_state_dict", "state_dict", "legacy_regressor_state_dict"]
    state = None
    for key in keys:
        if key in payload:
            state = payload[key]
            break
    if state is None:
        state = payload
    model.load_state_dict(state, strict=False)
    return payload


def run_legacy_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    desc: str,
    progress: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[dict[str, float]] = []
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        outputs = model(clip_obs=batch["clip_obs"])
        pred = outputs["pred"]
        target = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = compute_regression_metrics(pred.detach(), target.detach())
        metrics["loss"] = float(loss.item())
        rows.append(metrics)
    return aggregate_metric_rows(rows)


def _plugin_predictions(
    plugin_model: torch.nn.Module | None,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if plugin_model is None:
        return None, None
    with torch.no_grad():
        target_pred = plugin_model(feature_grids=outputs["target"]["patch_features"], valid_mask=batch["seq_valid_mask"])["pred"]
    plugin_pred = plugin_model(feature_grids=outputs["source_to_target"]["decoded_target_features"], valid_mask=batch["seq_valid_mask"])["pred"]
    return plugin_pred, target_pred


def run_sccwm_epoch(
    model: SCCWM,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    stage: str,
    loss_cfg: dict[str, Any],
    plugin_model: torch.nn.Module | None,
    desc: str,
    progress: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if plugin_model is not None:
        plugin_model.eval()
    rows: list[dict[str, float]] = []
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)
    for batch in iterator:
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
        plugin_pred, plugin_target_pred = _plugin_predictions(plugin_model, outputs, batch)
        loss_out = compute_sccwm_losses(
            outputs=outputs,
            batch=batch,
            stage=stage,
            loss_cfg=loss_cfg,
            plugin_pred=plugin_pred,
            plugin_target_pred=plugin_target_pred,
        )
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
        if plugin_pred is not None:
            plugin_metrics = compute_regression_metrics(plugin_pred.detach(), target.detach())
            metrics.update({f"plugin_{k}": v for k, v in plugin_metrics.items()})
        if stage in {"stage3", "stage4"}:
            source_state = build_state_embedding(outputs["source"])
            target_state = build_state_embedding(outputs["target"])
            pos = torch.nn.functional.cosine_similarity(source_state, target_state, dim=1)
            occ = (batch["source_obs"][:, -1, 0] - batch["source_obs"][:, 0, 0]).abs().mean(dim=(1, 2))
            neg_idx = build_negative_labels(occ.detach(), batch["depth_mm"].detach())
            neg = torch.nn.functional.cosine_similarity(source_state, target_state[neg_idx], dim=1)
            metrics.update(compute_ccauc(pos.detach().cpu().numpy(), neg.detach().cpu().numpy()))
        rows.append(metrics)
    return aggregate_metric_rows(rows)


def train_sccwm_stage(cfg: dict[str, Any], *, stage: str) -> None:
    device = default_device(cfg)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    train_loader, val_loader = build_paired_loaders(cfg)
    model = build_sccwm_model(cfg, device)
    plugin_model = None
    if stage == "stage4":
        plugin_model = build_legacy_regressor(cfg, device)
        plugin_cfg = cfg.get("plugin", {})
        ckpt = plugin_cfg.get("legacy_ckpt", "")
        if not ckpt:
            raise ValueError("Stage 4 requires plugin.legacy_ckpt")
        load_model_state(plugin_model, ckpt, device)
        for param in plugin_model.parameters():
            param.requires_grad = False

    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    output_dir = resolve_path(train_cfg["output_dir"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg.get("epochs", 1)), 1))
    best_metric_name = str(train_cfg.get("best_metric", "loss"))
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

    run_name = str(cfg.get("experiment", {}).get("name", f"sccwm_{stage}"))
    wandb_project = str(cfg.get("system", {}).get("wandb_project", "SCCWM"))
    wandb_name = str(cfg.get("system", {}).get("wandb_name", run_name))
    wandb.init(project=wandb_project, name=wandb_name, config=copy.deepcopy(cfg))
    start_time = time.time()
    try:
        for epoch in range(start_epoch, int(train_cfg.get("epochs", 1)) + 1):
            epoch_start = time.time()
            train_metrics = run_sccwm_epoch(
                model,
                train_loader,
                optimizer,
                device,
                stage=stage,
                loss_cfg=loss_cfg,
                plugin_model=plugin_model,
                desc=f"Train {stage} {epoch}",
                progress=not bool(train_cfg.get("no_progress", False)),
            )
            with torch.no_grad():
                val_metrics = run_sccwm_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    stage=stage,
                    loss_cfg=loss_cfg,
                    plugin_model=plugin_model,
                    desc=f"Val {stage} {epoch}",
                    progress=not bool(train_cfg.get("no_progress", False)),
                )
            scheduler.step()
            log_row = {f"train/{k}": v for k, v in train_metrics.items()}
            log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_row["epoch"] = epoch
            log_row["epoch_seconds"] = time.time() - epoch_start
            wandb.log(log_row)
            current = float(val_metrics.get(best_metric_name, val_metrics.get("loss", 0.0)))
            improved = current < best_value if best_mode == "min" else current > best_value
            payload = {
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": cfg,
                "best_value": best_value,
            }
            save_checkpoint(output_dir / "last.pt", payload)
            if improved:
                best_value = current
                payload["best_value"] = best_value
                save_checkpoint(output_dir / "best.pt", payload)
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
            print(
                f"[{stage}] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
                f"val_loss={val_metrics.get('loss', 0.0):.4f} "
                f"best_{best_metric_name}={best_value:.4f} "
                f"epoch_time={format_seconds(time.time() - epoch_start)} "
                f"elapsed={format_seconds(time.time() - start_time)}"
            )
    finally:
        wandb.finish()
        dump_json({"stage": stage, "best_value": best_value, "elapsed_seconds": time.time() - start_time, "config": cfg}, output_dir / "train_summary.json")


def train_legacy_regressor(cfg: dict[str, Any]) -> None:
    device = default_device(cfg)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    train_loader, val_loader = build_clip_loaders(cfg)
    model = build_legacy_regressor(cfg, device)
    train_cfg = cfg.get("train", {})
    output_dir = resolve_path(train_cfg["output_dir"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-2)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg.get("epochs", 1)), 1))
    best_value = float("inf")
    wandb.init(
        project=str(cfg.get("system", {}).get("wandb_project", "SCCWM_Legacy")),
        name=str(cfg.get("system", {}).get("wandb_name", cfg.get("experiment", {}).get("name", "legacy_regressor"))),
        config=copy.deepcopy(cfg),
    )
    try:
        for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
            train_metrics = run_legacy_epoch(model, train_loader, optimizer, device, desc=f"Legacy train {epoch}", progress=not bool(train_cfg.get("no_progress", False)))
            with torch.no_grad():
                val_metrics = run_legacy_epoch(model, val_loader, None, device, desc=f"Legacy val {epoch}", progress=not bool(train_cfg.get("no_progress", False)))
            scheduler.step()
            wandb.log({f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "legacy_regressor_state_dict": model.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": cfg,
                "best_value": best_value,
            }
            save_checkpoint(output_dir / "last.pt", payload)
            if val_metrics.get("mae_mean", float("inf")) < best_value:
                best_value = float(val_metrics["mae_mean"])
                payload["best_value"] = best_value
                save_checkpoint(output_dir / "best.pt", payload)
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
            print(f"[legacy] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} val_mae={val_metrics.get('mae_mean', 0.0):.4f}")
    finally:
        wandb.finish()
        dump_json({"best_value": best_value, "config": cfg}, output_dir / "train_summary.json")


def build_stage_argparser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--override", action="append", default=[], help="YAML-style key=value override; can be passed multiple times.")
    return parser
