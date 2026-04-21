#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import (
    _batch_item,
    _finalize_direct_records,
    _format_elapsed,
    _move_eval_batch,
    _progress_write,
    _summarize_direct_records,
    build_eval_argparser,
    load_eval_config,
    save_eval_result,
)
from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs
from sccwm.train.common import _safe_loader_workers, default_device, load_checkpoint
from sccwm.utils.config import resolve_path
from sccwm_force_grounded_v21.datasets import OverlayPairedSequenceDataset
from sccwm_force_grounded_v21.models import SCCWMForceGroundedV21

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


STANDARD_PROTOCOLS = [
    "same_scale_sanity",
    "heldout_exact_scales",
    "heldout_scale_bands",
    "unseen_indenters_seen_scales",
    "unseen_indenters_heldout_scales",
    "boundary_clean",
    "boundary_near_boundary",
    "boundary_partial_crop",
]
STRICT_CROSS_BAND_PROTOCOLS = [
    "cross_band_16_to_23",
    "cross_band_23_to_16",
    "cross_band_16_23_bidirectional",
]


def load_force_grounded_v21_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> SCCWMForceGroundedV21:
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
    ).to(device)
    payload = load_checkpoint(checkpoint_path, device)
    current = model.state_dict()
    source_state = payload["model_state_dict"]
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in current and torch.is_tensor(value) and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model


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
        raise KeyError(f"Missing {kind} root for split={split}")
    return resolve_path(root_value)


def _select_index_file(ds_cfg: dict[str, Any], *, split: str) -> str | Path | None:
    return {
        "train": ds_cfg.get("train_index_file"),
        "val": ds_cfg.get("val_index_file"),
        "test": ds_cfg.get("test_index_file"),
    }.get(str(split))


def build_force_grounded_v21_eval_loader(cfg: dict[str, Any], *, split: str, sequence_length: int | None, device: torch.device) -> DataLoader[Any]:
    ds_cfg = cfg.get("dataset", {})
    seq_len = int(sequence_length or ds_cfg.get("sequence_length", 3))
    dataset = OverlayPairedSequenceDataset(
        _select_overlay_root(ds_cfg, split=split, kind="image"),
        _select_overlay_root(ds_cfg, split=split, kind="asset"),
        split=split,
        sequence_length=seq_len,
        index_file=_select_index_file(ds_cfg, split=split),
        gray_cache_max_items=int(ds_cfg.get("gray_cache_max_items", 256)),
    )
    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("val_batch_size", 8)))
    workers = _safe_loader_workers(
        int(eval_cfg.get("workers", train_cfg.get("workers", 4))),
        dataset,
        label=f"eval-{split}",
        allow_large_index_workers=False,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")


def run_force_grounded_direct_eval_v21(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_force_grounded_v21_for_eval(cfg, checkpoint_path, device)
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        return _run_force_grounded_cross_band_eval_v21(cfg, model=model, protocol=protocol, sequence_length=sequence_length, limit=limit)
    loader = build_force_grounded_v21_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
    records: list[dict[str, Any]] = []
    eval_cfg = cfg.get("eval", {})
    log_interval_seconds = max(float(eval_cfg.get("timestamp_log_interval_seconds", 600.0)), 1.0)
    started_at = __import__("time").time()
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"ForceGroundedV21 direct eval {protocol} {split}", total=len(loader), leave=False)
        _progress_write(
            iterator,
            f"force_grounded_v21_direct_eval started protocol={protocol} split={split} checkpoint={checkpoint_path} "
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
            pos = torch.nn.functional.cosine_similarity(source["state_embedding"], target["state_embedding"], dim=1)
            for sample_idx in range(pred_source.shape[0]):
                records.append(
                    {
                        "event_key": f"{int(batch['episode_id'][sample_idx].item())}::{int(batch['global_seq_index'][sample_idx].item())}",
                        "pred_source": pred_source[sample_idx].cpu().tolist(),
                        "pred_target": pred_target[sample_idx].cpu().tolist(),
                        "target": target_gt[sample_idx].cpu().tolist(),
                        "occupancy": float(occ[sample_idx].item()),
                        "positive_score": float(pos[sample_idx].item()),
                        "state_source": source["state_embedding"][sample_idx].cpu().tolist(),
                        "state_target": target["state_embedding"][sample_idx].cpu().tolist(),
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
            now = __import__("time").time()
            if now >= next_log_time:
                _progress_write(
                    iterator,
                    f"force_grounded_v21_direct_eval heartbeat protocol={protocol} split={split} "
                    f"batch={batch_idx}/{len(loader)} records={len(records)} elapsed={_format_elapsed(now - started_at)}",
                )
                while next_log_time <= now:
                    next_log_time += log_interval_seconds
    result = _finalize_direct_records(records, protocol=protocol, cfg=cfg)
    _progress_write(
        iterator,
        f"force_grounded_v21_direct_eval finished protocol={protocol} split={split} records={len(records)} "
        f"filtered={result['filtered_sample_count']} elapsed={_format_elapsed(__import__('time').time() - started_at)}",
    )
    return result


def _run_force_grounded_cross_band_eval_v21(
    cfg: dict[str, Any],
    *,
    model: SCCWMForceGroundedV21,
    protocol: str,
    sequence_length: int | None,
    limit: int,
) -> dict[str, Any]:
    pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
    loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    records: list[dict[str, Any]] = []
    eval_cfg = cfg.get("eval", {})
    log_interval_seconds = max(float(eval_cfg.get("timestamp_log_interval_seconds", 600.0)), 1.0)
    started_at = __import__("time").time()
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"ForceGroundedV21 cross-band eval {protocol}", total=len(loader), leave=False)
        _progress_write(
            iterator,
            f"force_grounded_v21_cross_band_eval started protocol={protocol} matched_pairs={len(pair_specs)} "
            f"total_batches={len(loader)} limit={limit if limit > 0 else 'none'}",
        )
        next_log_time = started_at + log_interval_seconds
        for batch_idx, batch in enumerate(iterator, start=1):
            batch = _move_eval_batch(batch, next(model.parameters()).device)
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
            pos = torch.nn.functional.cosine_similarity(source["state_embedding"], target["state_embedding"], dim=1)
            pair_ids = batch["pair_id"]
            for sample_idx in range(pred_source.shape[0]):
                records.append(
                    {
                        "event_key": str(pair_ids[sample_idx]),
                        "pred_source": pred_source[sample_idx].cpu().tolist(),
                        "pred_target": pred_target[sample_idx].cpu().tolist(),
                        "target": target_gt[sample_idx].cpu().tolist(),
                        "occupancy": float(occ[sample_idx].item()),
                        "positive_score": float(pos[sample_idx].item()),
                        "state_source": source["state_embedding"][sample_idx].cpu().tolist(),
                        "state_target": target["state_embedding"][sample_idx].cpu().tolist(),
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
            now = __import__("time").time()
            if now >= next_log_time:
                _progress_write(
                    iterator,
                    f"force_grounded_v21_cross_band_eval heartbeat protocol={protocol} "
                    f"batch={batch_idx}/{len(loader)} records={len(records)} elapsed={_format_elapsed(now - started_at)}",
                )
                while next_log_time <= now:
                    next_log_time += log_interval_seconds
    result = _summarize_direct_records(records, protocol=protocol, cfg=cfg)
    result["matching_stats"] = matching_stats
    result["model_type"] = "sccwm_force_grounded_v21"
    _progress_write(
        iterator,
        f"force_grounded_v21_cross_band_eval finished protocol={protocol} records={len(records)} "
        f"elapsed={_format_elapsed(__import__('time').time() - started_at)}",
    )
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate force-grounded SCCWM v2.1 direct prediction protocols.",
        "sccwm_force_grounded_v21/configs/sccwm_stage2_force_grounded_v21.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_direct_eval_v21(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v21_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
