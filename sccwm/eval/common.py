from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable

from sccwm.datasets import PairedSequenceDataset
from sccwm.losses import build_negative_labels, build_state_embedding
from sccwm.metrics import compute_ccauc, compute_plugin_metrics, compute_regression_metrics, compute_sass
from sccwm.models import DeterministicTransportTemporalPredictor, FeatureSpaceSCTABaseline, LegacyStaticRegressor, LegacyTemporalRegressor, SCCWM
from sccwm.train.common import default_device, load_checkpoint
from sccwm.utils.config import dump_json, load_config_with_overrides, resolve_path


def _timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_elapsed(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_write(progress: Any, message: str) -> None:
    line = f"[eval-ts {_timestamp_str()}] {message}"
    writer = getattr(progress, "write", None)
    if callable(writer):
        writer(line)
    else:
        print(line)


def build_eval_argparser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def _select_eval_root(ds_cfg: dict[str, Any], *, split: str) -> Path:
    split = str(split)
    per_split_key = {
        "train": "train_dataset_a_root",
        "val": "val_dataset_a_root",
        "test": "test_dataset_a_root",
    }.get(split, "")
    root_value = ds_cfg.get(per_split_key) if per_split_key else None
    if root_value is None:
        root_value = ds_cfg["dataset_a_root"]
    return resolve_path(root_value)


def build_eval_loader(cfg: dict[str, Any], *, split: str, sequence_length: int | None) -> DataLoader[Any]:
    ds_cfg = cfg.get("dataset", {})
    root = _select_eval_root(ds_cfg, split=split)
    seq_len = int(sequence_length or ds_cfg.get("sequence_length", 3))
    dataset = PairedSequenceDataset(root, split=split, sequence_length=seq_len)
    batch_size = int(cfg.get("eval", {}).get("batch_size", cfg.get("train", {}).get("val_batch_size", 8)))
    workers = int(cfg.get("eval", {}).get("workers", cfg.get("train", {}).get("workers", 4)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


def load_sccwm_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> SCCWM:
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
    ).to(device)
    payload = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model


def load_legacy_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    reg_cfg = cfg.get("legacy_regressor", {})
    reg_type = str(reg_cfg.get("type", "temporal"))
    if reg_type == "static":
        model = LegacyStaticRegressor(
            observation_channels=int(reg_cfg.get("observation_channels", 3)),
            feature_dim=int(reg_cfg.get("feature_dim", 128)),
            output_dim=int(reg_cfg.get("output_dim", 3)),
        )
    else:
        model = LegacyTemporalRegressor(
            observation_channels=int(reg_cfg.get("observation_channels", 3)),
            feature_dim=int(reg_cfg.get("feature_dim", 128)),
            hidden_dim=int(reg_cfg.get("hidden_dim", 128)),
            output_dim=int(reg_cfg.get("output_dim", 3)),
        )
    payload = load_checkpoint(checkpoint_path, device)
    state = payload.get("legacy_regressor_state_dict") or payload.get("model_state_dict") or payload
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def _move_eval_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
        else:
            out[key] = value
    phase_names = batch.get("phase_names", [])
    if isinstance(phase_names, list) and phase_names and isinstance(phase_names[0], tuple):
        out["phase_names"] = [[str(phase_names[t][b]) for t in range(len(phase_names))] for b in range(len(phase_names[0]))]
    else:
        out["phase_names"] = phase_names
    return out


def _batch_item(value: Any, index: int) -> Any:
    if isinstance(value, list):
        return value[index]
    if isinstance(value, tuple):
        return value[index]
    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.ndim == 0 else item
    return value


def _is_same_scale(record: dict[str, Any], eps: float = 1e-5) -> bool:
    return abs(float(record["source_scale_mm"]) - float(record["target_scale_mm"])) <= float(eps)


def _scale_is_exact(scale_mm: float, exact_scales: list[float], eps: float = 1e-5) -> bool:
    return any(abs(float(scale_mm) - float(scale)) <= float(eps) for scale in exact_scales)


def _scale_in_band(scale_mm: float, bands: list[tuple[float, float]]) -> bool:
    return any(float(lo) <= float(scale_mm) <= float(hi) for lo, hi in bands)


def _protocol_filter(record: dict[str, Any], protocol: str, cfg: dict[str, Any]) -> bool:
    exact = [float(v) for v in cfg.get("eval", {}).get("exact_scales_mm", [])]
    bands = [(float(lo), float(hi)) for lo, hi in cfg.get("eval", {}).get("band_ranges_mm", [])]
    same_scale = _is_same_scale(record)
    target_scale = float(record["target_scale_mm"])
    target_exact = _scale_is_exact(target_scale, exact) if exact else False
    target_in_band = _scale_in_band(target_scale, bands) if bands else False
    target_heldout = target_exact or target_in_band or bool(record["is_unseen_scale_target"]) or str(record.get("target_scale_split", "")) in {"val", "test"}
    if protocol == "same_scale_sanity":
        return same_scale
    if protocol == "heldout_exact_scales":
        if exact:
            return (not same_scale) and target_exact
        return (not same_scale) and str(record.get("target_scale_split", "")) in {"val", "test"}
    if protocol == "heldout_scale_bands":
        if bands:
            return (not same_scale) and target_in_band
        return (not same_scale) and str(record.get("target_scale_split", "")) in {"val", "test"}
    if protocol == "unseen_indenters_seen_scales":
        return bool(record["is_unseen_indenter"]) and not target_heldout
    if protocol == "unseen_indenters_heldout_scales":
        return bool(record["is_unseen_indenter"]) and target_heldout
    if protocol == "boundary_clean":
        return str(record["boundary_subset"]) == "clean"
    if protocol == "boundary_near_boundary":
        return str(record["boundary_subset"]) == "near_boundary"
    if protocol == "boundary_partial_crop":
        return str(record["boundary_subset"]) == "partial_crop"
    return True


def _scale_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _evenly_spaced_subset(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        return records
    indices = np.linspace(0, len(records) - 1, num=limit, dtype=np.int64)
    unique_indices: list[int] = []
    last = -1
    for idx in indices.tolist():
        idx = int(idx)
        if idx != last:
            unique_indices.append(idx)
            last = idx
    return [records[idx] for idx in unique_indices]


def _summarize_direct_records(filtered: list[dict[str, Any]], *, protocol: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if not filtered:
        raise RuntimeError(f"No evaluation samples remained for protocol: {protocol}")
    pred = torch.tensor([record["pred_source"] for record in filtered], dtype=torch.float32)
    target = torch.tensor([record["target"] for record in filtered], dtype=torch.float32)
    metrics = compute_regression_metrics(pred, target)
    sass_rows: list[dict[str, object]] = []
    ccauc_max_samples = int(cfg.get("eval", {}).get("ccauc_max_samples", 4096))
    ccauc_records = _evenly_spaced_subset(filtered, ccauc_max_samples)
    pos_scores: list[float] = []
    occupancy = np.asarray([record["occupancy"] for record in ccauc_records], dtype=np.float32)
    depth = torch.tensor([record["target"][2] for record in ccauc_records], dtype=torch.float32)
    neg_idx = build_negative_labels(torch.tensor(occupancy, dtype=torch.float32), depth)
    state_source = np.asarray([record["state_source"] for record in ccauc_records], dtype=np.float32)
    state_target = np.asarray([record["state_target"] for record in ccauc_records], dtype=np.float32)
    neg_scores = [
        float(
            np.dot(state_source[idx], state_target[int(neg_idx[idx].item())])
            / (np.linalg.norm(state_source[idx]) * np.linalg.norm(state_target[int(neg_idx[idx].item())]) + 1e-6)
        )
        for idx in range(len(ccauc_records))
    ]
    for idx, record in enumerate(filtered):
        event_key = str(record["event_key"])
        sass_rows.append({"event_key": event_key, "pred": record["pred_source"]})
        sass_rows.append({"event_key": event_key, "pred": record["pred_target"]})
    for record in ccauc_records:
        pos_scores.append(record["positive_score"])
    metrics.update(compute_sass(sass_rows))
    metrics.update(compute_ccauc(np.asarray(pos_scores, dtype=np.float32), np.asarray(neg_scores, dtype=np.float32)))
    metrics["sample_count"] = len(filtered)
    metrics["ccauc_sample_count"] = len(ccauc_records)
    return {
        "protocol_name": protocol,
        "filtered_sample_count": len(filtered),
        "source_scale_stats": _scale_stats([float(record["source_scale_mm"]) for record in filtered]),
        "target_scale_stats": _scale_stats([float(record["target_scale_mm"]) for record in filtered]),
        "metrics": metrics,
        "records": filtered,
    }


def _finalize_direct_records(records: list[dict[str, Any]], *, protocol: str, cfg: dict[str, Any]) -> dict[str, Any]:
    filtered = [record for record in records if _protocol_filter(record, protocol, cfg)]
    return _summarize_direct_records(filtered, protocol=protocol, cfg=cfg)


def run_direct_eval(cfg: dict[str, Any], *, checkpoint_path: str | Path, split: str, sequence_length: int | None, protocol: str, limit: int = 0) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_sccwm_for_eval(cfg, checkpoint_path, device)
    loader = build_eval_loader(cfg, split=split, sequence_length=sequence_length)
    records: list[dict[str, Any]] = []
    eval_cfg = cfg.get("eval", {})
    log_interval_seconds = max(float(eval_cfg.get("timestamp_log_interval_seconds", 600.0)), 1.0)
    started_at = time.time()
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Direct eval {protocol} {split}", total=len(loader), leave=False)
        _progress_write(
            iterator,
            f"direct_eval started protocol={protocol} split={split} checkpoint={checkpoint_path} "
            f"total_batches={len(loader)} limit={limit if limit > 0 else 'none'}",
        )
        next_log_time = started_at + log_interval_seconds
        for batch_idx, batch in enumerate(iterator, start=1):
            if limit > 0 and len(records) >= limit:
                break
            batch = _move_eval_batch(batch, device)
            source = model.forward_single(
                batch["source_obs"],
                batch["source_coord_map"],
                batch["source_scale_mm"],
                absolute_contact_xy_mm=batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
                valid_mask=batch["seq_valid_mask"],
            )
            target = model.forward_single(
                batch["target_obs"],
                batch["target_coord_map"],
                batch["target_scale_mm"],
                absolute_contact_xy_mm=batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
                valid_mask=batch["seq_valid_mask"],
            )
            occ = (batch["source_obs"][:, -1, 0] - batch["source_obs"][:, 0, 0]).abs().mean(dim=(1, 2))
            pred_source = torch.stack([source["pred_x_norm"], source["pred_y_norm"], source["pred_depth_mm"]], dim=1)
            pred_target = torch.stack([target["pred_x_norm"], target["pred_y_norm"], target["pred_depth_mm"]], dim=1)
            target_gt = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
            source_state = build_state_embedding(source)
            target_state = build_state_embedding(target)
            pos = torch.nn.functional.cosine_similarity(source_state, target_state, dim=1)
            for sample_idx in range(pred_source.shape[0]):
                records.append(
                    {
                        "event_key": f"{int(batch['episode_id'][sample_idx].item())}::{int(batch['global_seq_index'][sample_idx].item())}",
                        "pred_source": pred_source[sample_idx].cpu().tolist(),
                        "pred_target": pred_target[sample_idx].cpu().tolist(),
                        "target": target_gt[sample_idx].cpu().tolist(),
                        "occupancy": float(occ[sample_idx].item()),
                        "positive_score": float(pos[sample_idx].item()),
                        "state_source": source_state[sample_idx].cpu().numpy().tolist(),
                        "state_target": target_state[sample_idx].cpu().numpy().tolist(),
                        "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
                        "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
                        "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
                        "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
                        "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
                        "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
                        "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
                    }
                )
            iterator.set_postfix({"records": len(records)})
            now = time.time()
            if now >= next_log_time:
                _progress_write(
                    iterator,
                    f"direct_eval heartbeat protocol={protocol} split={split} batch={batch_idx}/{len(loader)} "
                    f"records={len(records)} elapsed={_format_elapsed(now - started_at)}",
                )
                while next_log_time <= now:
                    next_log_time += log_interval_seconds
    result = _finalize_direct_records(records, protocol=protocol, cfg=cfg)
    _progress_write(
        iterator,
        f"direct_eval finished protocol={protocol} split={split} records={len(records)} "
        f"filtered={result['filtered_sample_count']} elapsed={_format_elapsed(time.time() - started_at)}",
    )
    return result


def run_plugin_eval(
    cfg: dict[str, Any],
    *,
    method: str,
    legacy_checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    sccwm_checkpoint_path: str | Path = "",
    scta_checkpoint_path: str | Path = "",
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    legacy = load_legacy_for_eval(cfg, legacy_checkpoint_path, device)
    loader = build_eval_loader(cfg, split=split, sequence_length=sequence_length)
    eval_cfg = cfg.get("eval", {})
    log_interval_seconds = max(float(eval_cfg.get("timestamp_log_interval_seconds", 600.0)), 1.0)
    started_at = time.time()
    sccwm_model = None
    if method in {"deterministic_transport", "sccwm_plugin"}:
        if not sccwm_checkpoint_path:
            raise ValueError(f"Method {method} requires --checkpoint")
        sccwm_model = load_sccwm_for_eval(cfg, sccwm_checkpoint_path, device)
    scta_model = None
    if method == "scta":
        scta_model = FeatureSpaceSCTABaseline().to(device)
        if scta_checkpoint_path:
            payload = load_checkpoint(scta_checkpoint_path, device)
            state = payload.get("model_state_dict", payload)
            scta_model.load_state_dict(state, strict=False)
        scta_model.eval()
    transport_model = DeterministicTransportTemporalPredictor().to(device).eval() if method == "deterministic_transport" else None

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Plugin eval {method} {split}", total=len(loader), leave=False)
        _progress_write(
            iterator,
            f"plugin_eval started method={method} split={split} checkpoint={legacy_checkpoint_path} "
            f"total_batches={len(loader)} limit={limit if limit > 0 else 'none'}",
        )
        next_log_time = started_at + log_interval_seconds
        for batch_idx, batch in enumerate(iterator, start=1):
            if limit > 0 and len(rows) >= limit:
                break
            batch = _move_eval_batch(batch, device)
            target = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
            baseline_pred = legacy(clip_obs=batch["source_obs"], valid_mask=batch["seq_valid_mask"])["pred"]
            if method == "legacy_no_adaptation":
                pred = baseline_pred
            elif method == "deterministic_transport":
                assert sccwm_model is not None and transport_model is not None
                features = sccwm_model.patch_encoder(batch["source_obs"])
                world, _ = sccwm_model.projector.splat_to_world_lattice(features, batch["source_coord_map"])
                transported, _ = sccwm_model.projector.gather_from_world_lattice(world, batch["target_coord_map"])
                pred = legacy(feature_grids=transported, valid_mask=batch["seq_valid_mask"])["pred"]
            elif method == "scta":
                assert scta_model is not None
                scta_out = scta_model(
                    batch["source_obs"],
                    batch["source_coord_map"],
                    batch["source_scale_mm"],
                    batch["target_coord_map"],
                    batch["target_scale_mm"],
                )
                pred = legacy(feature_grids=scta_out["decoded_features"], valid_mask=batch["seq_valid_mask"])["pred"]
            elif method == "sccwm_plugin":
                assert sccwm_model is not None
                outputs = sccwm_model.forward_pair(
                    source_obs=batch["source_obs"],
                    target_obs=batch["target_obs"],
                    source_coord_map=batch["source_coord_map"],
                    target_coord_map=batch["target_coord_map"],
                    source_scale_mm=batch["source_scale_mm"],
                    target_scale_mm=batch["target_scale_mm"],
                    source_valid_mask=batch["seq_valid_mask"],
                    target_valid_mask=batch["seq_valid_mask"],
                    absolute_contact_xy_mm=batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None,
                    world_origin_xy_mm=batch["world_origin_xy_mm"],
                )
                pred = legacy(feature_grids=outputs["source_to_target"]["decoded_target_features"], valid_mask=batch["seq_valid_mask"])["pred"]
            else:
                raise ValueError(f"Unsupported plugin method: {method}")
            batch_metrics = compute_plugin_metrics(pred.detach(), target.detach(), baseline_pred.detach())
            rows.append(batch_metrics)
            iterator.set_postfix({"batches": len(rows)})
            now = time.time()
            if now >= next_log_time:
                _progress_write(
                    iterator,
                    f"plugin_eval heartbeat method={method} split={split} batch={batch_idx}/{len(loader)} "
                    f"rows={len(rows)} elapsed={_format_elapsed(now - started_at)}",
                )
                while next_log_time <= now:
                    next_log_time += log_interval_seconds
    metrics = aggregate_rows(rows)
    result = {"metrics": metrics, "method": method}
    _progress_write(
        iterator,
        f"plugin_eval finished method={method} split={split} rows={len(rows)} "
        f"elapsed={_format_elapsed(time.time() - started_at)}",
    )
    return result


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / max(len(rows), 1)) for key in keys}


def save_eval_result(payload: dict[str, Any], output_path: str | Path) -> None:
    target = resolve_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    return load_config_with_overrides(args.config, args.override)
