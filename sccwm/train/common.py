from __future__ import annotations

import argparse
import copy
import json
import math
import random
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

class _WandbStub:
    @staticmethod
    def init(*args: Any, **kwargs: Any) -> None:
        print("wandb SDK is unavailable; continuing with local logging only.")
        return None

    @staticmethod
    def log(*args: Any, **kwargs: Any) -> None:
        return None

    @staticmethod
    def finish(*args: Any, **kwargs: Any) -> None:
        return None


try:
    import wandb as _wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = _WandbStub()
else:
    wandb = _wandb if hasattr(_wandb, "init") and callable(getattr(_wandb, "init", None)) else _WandbStub()

from sccwm.datasets import LegacyClipDataset, PairedSequenceDataset
from sccwm.losses import build_negative_labels, build_state_embedding, compute_sccwm_losses
from sccwm.metrics.regression_metrics import compute_regression_metrics
from sccwm.metrics.ccauc_metric import compute_ccauc
from sccwm.models import LegacyStaticRegressor, LegacyTemporalRegressor, SCCWM
from sccwm.utils.config import dump_json, format_seconds, resolve_path, seed_everything


def _tqdm_metric_postfix(rows: list[dict[str, float]]) -> dict[str, str]:
    if not rows:
        return {}
    latest = aggregate_metric_rows(rows[-20:])
    keys = [
        "loss",
        "feature_recon",
        "state_supervision_source",
        "state_consistency",
        "counterfactual_ranking",
        "plugin_gt",
        "pred_mae_mean",
    ]
    postfix: dict[str, str] = {}
    for key in keys:
        if key in latest:
            postfix[key] = f"{latest[key]:.4f}"
    return postfix


def _safe_loader_workers(
    requested_workers: int,
    dataset: Any,
    *,
    label: str,
    allow_large_index_workers: bool = False,
) -> int:
    workers = int(requested_workers)
    index_path = getattr(dataset, "index_path", None)
    if workers <= 0 or index_path is None:
        return max(workers, 0)
    try:
        index_size_mb = Path(index_path).stat().st_size / (1024.0 * 1024.0)
    except OSError:
        return max(workers, 0)
    if index_size_mb >= 512.0:
        subset_filtered = getattr(dataset, "episode_id_subset", None) is not None
        streaming_safe = bool(getattr(dataset, "uses_streaming_index", False))
        if allow_large_index_workers or (streaming_safe and subset_filtered):
            print(
                f"{label} index is large ({index_size_mb:.1f} MB); honoring DataLoader workers={workers} "
                "because this dataset is using the streaming/subset path."
            )
            return max(workers, 0)
        print(
            f"{label} index is large ({index_size_mb:.1f} MB); forcing DataLoader workers=0 "
            "to avoid duplicating the dataset index across worker processes."
        )
        return 0
    return max(workers, 0)


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
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(
                device=device,
                dtype=torch.float32 if value.is_floating_point() else value.dtype,
                non_blocking=non_blocking,
            )
        else:
            out[key] = value
    out["phase_names"] = normalize_phase_batch(batch.get("phase_names", []))
    return out


def _select_split_root(ds_cfg: dict[str, Any], *, split: str, dataset_key: str) -> Path:
    split = str(split)
    if dataset_key == "a":
        per_split_key = {
            "train": "train_dataset_a_root",
            "val": "val_dataset_a_root",
            "test": "test_dataset_a_root",
        }.get(split, "")
        fallback_key = "dataset_a_root"
    elif dataset_key == "b":
        per_split_key = {
            "train": "train_dataset_b_root",
            "val": "val_dataset_b_root",
            "test": "test_dataset_b_root",
        }.get(split, "")
        fallback_key = "dataset_b_root"
    else:
        raise ValueError(f"Unsupported dataset_key={dataset_key!r}")

    root_value = ds_cfg.get(per_split_key) if per_split_key else None
    if root_value is None:
        root_value = ds_cfg[fallback_key]
    return resolve_path(root_value)


def _select_split_index_file(ds_cfg: dict[str, Any], *, split: str, dataset_key: str) -> str | Path | None:
    split = str(split)
    if dataset_key == "a":
        return {
            "train": ds_cfg.get("train_index_file"),
            "val": ds_cfg.get("val_index_file"),
            "test": ds_cfg.get("test_index_file"),
        }.get(split)
    if dataset_key == "b":
        return {
            "train": ds_cfg.get("clip_train_index_file"),
            "val": ds_cfg.get("clip_val_index_file"),
            "test": ds_cfg.get("clip_test_index_file"),
        }.get(split)
    raise ValueError(f"Unsupported dataset_key={dataset_key!r}")


def _build_paired_loader(
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    workers: int,
    sequence_length: int,
    shuffle: bool,
    episode_id_subset: set[int] | None = None,
) -> DataLoader[Any]:
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    root = _select_split_root(ds_cfg, split=split, dataset_key="a")
    index_file = _select_split_index_file(ds_cfg, split=split, dataset_key="a")
    gray_cache_max_items = int(ds_cfg.get("gray_cache_max_items", 256))
    allow_large_index_workers = bool(train_cfg.get("allow_large_index_workers", False))
    is_dynamic_train_subset = shuffle and episode_id_subset is not None
    dataset = PairedSequenceDataset(
        root,
        split=split,
        sequence_length=sequence_length,
        index_file=index_file,
        episode_id_subset=episode_id_subset,
        gray_cache_max_items=gray_cache_max_items,
    )
    dataset_workers = _safe_loader_workers(
        workers,
        dataset,
        label=f"{split}",
        allow_large_index_workers=allow_large_index_workers and is_dynamic_train_subset,
    )
    pin_memory = default_device(cfg).type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": dataset_workers,
        "pin_memory": pin_memory,
    }
    if dataset_workers > 0:
        loader_kwargs["persistent_workers"] = not is_dynamic_train_subset
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def build_paired_loaders(cfg: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    ds_cfg = cfg.get("dataset", {})
    batch_size = int(cfg.get("train", {}).get("batch_size", 8))
    val_batch_size = int(cfg.get("train", {}).get("val_batch_size", batch_size))
    workers = int(cfg.get("train", {}).get("workers", 4))
    seq_len = int(ds_cfg.get("sequence_length", 3))
    train_split = str(ds_cfg.get("train_split", "train"))
    val_split = str(ds_cfg.get("val_split", "val"))
    return (
        _build_paired_loader(cfg, split=train_split, batch_size=batch_size, workers=workers, sequence_length=seq_len, shuffle=True),
        _build_paired_loader(cfg, split=val_split, batch_size=val_batch_size, workers=workers, sequence_length=seq_len, shuffle=False),
    )


def _discover_train_episode_buckets(train_root: Path) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for meta_path in sorted(train_root.glob("episode_*/metadata.json")):
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
    train_split = str(ds_cfg.get("train_split", "train"))
    train_root = _select_split_root(ds_cfg, split=train_split, dataset_key="a")
    buckets = _discover_train_episode_buckets(train_root)
    rng = random.Random(int(cfg.get("experiment", {}).get("seed", 42)) + int(epoch))
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


def build_clip_loaders(cfg: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    ds_cfg = cfg.get("dataset", {})
    batch_size = int(cfg.get("train", {}).get("batch_size", 8))
    val_batch_size = int(cfg.get("train", {}).get("val_batch_size", batch_size))
    workers = int(cfg.get("train", {}).get("workers", 4))
    train_split = str(ds_cfg.get("clip_train_split", "train"))
    val_split = str(ds_cfg.get("clip_val_split", "val"))
    train_root = _select_split_root(ds_cfg, split=train_split, dataset_key="b")
    val_root = _select_split_root(ds_cfg, split=val_split, dataset_key="b")
    train_index_file = _select_split_index_file(ds_cfg, split=train_split, dataset_key="b")
    val_index_file = _select_split_index_file(ds_cfg, split=val_split, dataset_key="b")
    gray_cache_max_items = int(ds_cfg.get("gray_cache_max_items", 256))
    train_ds = LegacyClipDataset(train_root, split=train_split, clip_index_file=train_index_file, gray_cache_max_items=gray_cache_max_items)
    val_ds = LegacyClipDataset(val_root, split=val_split, clip_index_file=val_index_file, gray_cache_max_items=gray_cache_max_items)
    train_workers = _safe_loader_workers(workers, train_ds, label="clip-train")
    val_workers = _safe_loader_workers(workers, val_ds, label="clip-val")
    pin_memory = default_device(cfg).type == "cuda"
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=pin_memory),
        DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=val_workers, pin_memory=pin_memory),
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
        if progress:
            postfix = _tqdm_metric_postfix(rows)
            if postfix:
                iterator.set_postfix(postfix)
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
    max_batches: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if plugin_model is not None:
        plugin_model.eval()
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
        if progress:
            postfix = _tqdm_metric_postfix(rows)
            if postfix:
                iterator.set_postfix(postfix)
    return aggregate_metric_rows(rows)


def train_sccwm_stage(cfg: dict[str, Any], *, stage: str) -> None:
    device = default_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
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
    loss_cfg = cfg.get("loss", {})
    output_dir = resolve_path(train_cfg["output_dir"])
    metrics_history_path = output_dir / "metrics_history.jsonl"
    max_train_samples = int(train_cfg.get("max_train_samples_per_epoch", 0))
    max_val_samples = int(train_cfg.get("max_val_samples", 0))
    max_train_batches = math.ceil(max_train_samples / max(batch_size, 1)) if max_train_samples > 0 else None
    max_val_batches = math.ceil(max_val_samples / max(val_batch_size, 1)) if max_val_samples > 0 else None
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
        total_epochs = int(train_cfg.get("epochs", 1))
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.time()
            epoch_episode_subset = _sample_train_episode_subset(cfg, epoch=epoch)
            if epoch_episode_subset is not None:
                print(
                    f"[{stage}] epoch={epoch} sampled {len(epoch_episode_subset)} train episodes for this epoch "
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
                max_batches=max_train_batches,
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
                f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
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
            print(f"[{stage}] saved last checkpoint -> {output_dir / 'last.pt'}")
            if improved:
                best_value = current
                payload["best_value"] = best_value
                save_checkpoint(output_dir / "best.pt", payload)
                print(f"[{stage}] updated best checkpoint -> {output_dir / 'best.pt'} ({best_metric_name}={best_value:.4f})")
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
                print(f"[{stage}] saved periodic checkpoint -> {output_dir / f'epoch_{epoch:04d}.pt'}")
            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            avg_epoch_seconds = elapsed / max(epochs_done, 1)
            remaining_epochs = max(total_epochs - epoch, 0)
            eta_seconds = avg_epoch_seconds * remaining_epochs
            print(
                f"[{stage}] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
                f"val_loss={val_metrics.get('loss', 0.0):.4f} "
                f"best_{best_metric_name}={best_value:.4f} "
                f"epoch_time={format_seconds(time.time() - epoch_start)} "
                f"elapsed={format_seconds(elapsed)} "
                f"remaining={format_seconds(eta_seconds)} "
                f"eta={format_seconds(elapsed + eta_seconds)}"
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
    metrics_history_path = output_dir / "metrics_history.jsonl"
    batch_size = int(train_cfg.get("batch_size", 8))
    val_batch_size = int(train_cfg.get("val_batch_size", batch_size))
    max_train_samples = int(train_cfg.get("max_train_samples_per_epoch", 0))
    max_val_samples = int(train_cfg.get("max_val_samples", 0))
    max_train_batches = math.ceil(max_train_samples / max(batch_size, 1)) if max_train_samples > 0 else None
    max_val_batches = math.ceil(max_val_samples / max(val_batch_size, 1)) if max_val_samples > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-2)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg.get("epochs", 1)), 1))
    best_value = float("inf")
    wandb.init(
        project=str(cfg.get("system", {}).get("wandb_project", "SCCWM_Legacy")),
        name=str(cfg.get("system", {}).get("wandb_name", cfg.get("experiment", {}).get("name", "legacy_regressor"))),
        config=copy.deepcopy(cfg),
    )
    start_time = time.time()
    try:
        total_epochs = int(train_cfg.get("epochs", 1))
        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()
            train_metrics = run_legacy_epoch(
                model,
                train_loader,
                optimizer,
                device,
                desc=f"Legacy train {epoch}",
                progress=not bool(train_cfg.get("no_progress", False)),
                max_batches=max_train_batches,
            )
            with torch.no_grad():
                val_metrics = run_legacy_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    desc=f"Legacy val {epoch}",
                    progress=not bool(train_cfg.get("no_progress", False)),
                    max_batches=max_val_batches,
                )
            scheduler.step()
            log_row = {f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch}
            wandb.log(log_row)
            output_dir.mkdir(parents=True, exist_ok=True)
            with metrics_history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
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
            print(f"[legacy] saved last checkpoint -> {output_dir / 'last.pt'}")
            if val_metrics.get("mae_mean", float("inf")) < best_value:
                best_value = float(val_metrics["mae_mean"])
                payload["best_value"] = best_value
                save_checkpoint(output_dir / "best.pt", payload)
                print(f"[legacy] updated best checkpoint -> {output_dir / 'best.pt'} (mae_mean={best_value:.4f})")
            if int(train_cfg.get("save_every", 0)) > 0 and epoch % int(train_cfg.get("save_every", 1)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", payload)
                print(f"[legacy] saved periodic checkpoint -> {output_dir / f'epoch_{epoch:04d}.pt'}")
            elapsed = time.time() - start_time
            avg_epoch_seconds = elapsed / max(epoch, 1)
            remaining_epochs = max(total_epochs - epoch, 0)
            eta_seconds = avg_epoch_seconds * remaining_epochs
            print(
                f"[legacy] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
                f"val_mae={val_metrics.get('mae_mean', 0.0):.4f} "
                f"epoch_time={format_seconds(time.time() - epoch_start)} "
                f"elapsed={format_seconds(elapsed)} "
                f"remaining={format_seconds(eta_seconds)} "
                f"eta={format_seconds(elapsed + eta_seconds)}"
            )
    finally:
        wandb.finish()
        dump_json({"best_value": best_value, "config": cfg}, output_dir / "train_summary.json")


def build_stage_argparser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--override", action="append", default=[], help="YAML-style key=value override; can be passed multiple times.")
    return parser
