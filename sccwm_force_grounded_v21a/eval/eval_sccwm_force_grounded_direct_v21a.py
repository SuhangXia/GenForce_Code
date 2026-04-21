#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
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
from sccwm_force_grounded_v21a.datasets import OverlayPairedSequenceDataset
from sccwm_force_grounded_v21a.models import EMBEDDING_VIEW_TO_KEY, SCCWMForceGroundedV21A, select_embedding_view_v21a

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
EMBEDDING_VIEW_CHOICES = list(EMBEDDING_VIEW_TO_KEY.keys())


def load_force_grounded_v21a_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> SCCWMForceGroundedV21A:
    model_cfg = cfg.get("model", {})
    model = SCCWMForceGroundedV21A(
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


def build_force_grounded_v21a_eval_loader(cfg: dict[str, Any], *, split: str, sequence_length: int | None, device: torch.device) -> DataLoader[Any]:
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


def _metadata_field(batch: dict[str, Any], key: str, index: int) -> Any | None:
    metadata = batch.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return _batch_item(metadata[key], index)
    return None


def _extract_pair_metadata(batch: dict[str, Any], index: int, *, strict_cross_band: bool) -> dict[str, Any]:
    if strict_cross_band:
        source_episode = int(batch["source_episode_id"][index].item()) if "source_episode_id" in batch else None
        target_episode = int(batch["target_episode_id"][index].item()) if "target_episode_id" in batch else None
        source_seq = int(batch["source_global_seq_index"][index].item()) if "source_global_seq_index" in batch else None
        target_seq = int(batch["target_global_seq_index"][index].item()) if "target_global_seq_index" in batch else None
        pair_id = str(_batch_item(batch["pair_id"], index)) if "pair_id" in batch else None
        return {
            "event_key": pair_id or f"{source_episode}::{target_episode}::{source_seq}::{target_seq}",
            "pair_id": pair_id,
            "episode_id": None,
            "source_episode_id": source_episode,
            "target_episode_id": target_episode,
            "source_global_seq_index": source_seq,
            "target_global_seq_index": target_seq,
            "source_marker_name": _batch_item(batch["source_marker_name"], index) if "source_marker_name" in batch else None,
            "target_marker_name": _batch_item(batch["target_marker_name"], index) if "target_marker_name" in batch else None,
        }
    episode_id = int(batch["episode_id"][index].item()) if "episode_id" in batch else None
    global_seq_index = int(batch["global_seq_index"][index].item()) if "global_seq_index" in batch else None
    pair_id = _metadata_field(batch, "frame_key", index)
    return {
        "event_key": f"{episode_id}::{global_seq_index}",
        "pair_id": str(pair_id) if pair_id is not None else None,
        "episode_id": episode_id,
        "source_episode_id": episode_id,
        "target_episode_id": episode_id,
        "source_global_seq_index": global_seq_index,
        "target_global_seq_index": global_seq_index,
        "source_marker_name": _metadata_field(batch, "source_marker_name", index),
        "target_marker_name": _metadata_field(batch, "target_marker_name", index),
    }


def _build_direct_record(
    *,
    batch: dict[str, Any],
    sample_idx: int,
    pred_source: torch.Tensor,
    pred_target: torch.Tensor,
    target_gt: torch.Tensor,
    occupancy: torch.Tensor,
    positive_score: torch.Tensor,
    source_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    embedding_view: str,
    strict_cross_band: bool,
) -> dict[str, Any]:
    meta = _extract_pair_metadata(batch, sample_idx, strict_cross_band=strict_cross_band)
    record = {
        **meta,
        "embedding_view": embedding_view,
        "pred_source": pred_source[sample_idx].detach().cpu().tolist(),
        "pred_target": pred_target[sample_idx].detach().cpu().tolist(),
        "target": target_gt[sample_idx].detach().cpu().tolist(),
        "occupancy": float(occupancy[sample_idx].item()),
        "positive_score": float(positive_score[sample_idx].item()),
        "state_source": source_embedding[sample_idx].detach().cpu().tolist(),
        "state_target": target_embedding[sample_idx].detach().cpu().tolist(),
        "embedding_source": source_embedding[sample_idx].detach().cpu().tolist(),
        "embedding_target": target_embedding[sample_idx].detach().cpu().tolist(),
        "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
        "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
        "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
        "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
        "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
        "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
        "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
    }
    return record


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_force_grounded_eval_loop(
    *,
    cfg: dict[str, Any],
    model: SCCWMForceGroundedV21A,
    loader: DataLoader[Any],
    protocol: str,
    split_label: str,
    checkpoint_path: str | Path,
    limit: int,
    embedding_view: str,
    strict_cross_band: bool,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    eval_cfg = cfg.get("eval", {})
    log_interval_seconds = max(float(eval_cfg.get("timestamp_log_interval_seconds", 600.0)), 1.0)
    started_at = time.time()
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"ForceGroundedV21A direct eval {protocol} {split_label}", total=len(loader), leave=False)
        prefix = "force_grounded_v21a_cross_band_eval" if strict_cross_band else "force_grounded_v21a_direct_eval"
        _progress_write(
            iterator,
            f"{prefix} started protocol={protocol} split={split_label} checkpoint={checkpoint_path} "
            f"embedding_view={embedding_view} total_batches={len(loader)} limit={limit if limit > 0 else 'none'}",
        )
        next_log_time = started_at + log_interval_seconds
        for batch_idx, batch in enumerate(iterator, start=1):
            if limit > 0 and len(records) >= limit and not strict_cross_band:
                break
            batch = _move_eval_batch(batch, next(model.parameters()).device)
            absolute_contact = batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None
            source = model.forward_single(
                batch["source_obs"],
                batch["source_coord_map"],
                batch["source_scale_mm"],
                absolute_contact_xy_mm=absolute_contact,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
                valid_mask=batch["seq_valid_mask"],
            )
            target = model.forward_single(
                batch["target_obs"],
                batch["target_coord_map"],
                batch["target_scale_mm"],
                absolute_contact_xy_mm=absolute_contact,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
                valid_mask=batch["seq_valid_mask"],
            )
            occ = (batch["source_obs"][:, -1, 0] - batch["source_obs"][:, 0, 0]).abs().mean(dim=(1, 2))
            pred_source = torch.stack([source["pred_x_norm"], source["pred_y_norm"], source["pred_depth_mm"]], dim=1)
            pred_target = torch.stack([target["pred_x_norm"], target["pred_y_norm"], target["pred_depth_mm"]], dim=1)
            target_gt = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
            source_embedding = select_embedding_view_v21a(source, embedding_view)
            target_embedding = select_embedding_view_v21a(target, embedding_view)
            positive_score = torch.nn.functional.cosine_similarity(source_embedding, target_embedding, dim=1)
            for sample_idx in range(pred_source.shape[0]):
                records.append(
                    _build_direct_record(
                        batch=batch,
                        sample_idx=sample_idx,
                        pred_source=pred_source,
                        pred_target=pred_target,
                        target_gt=target_gt,
                        occupancy=occ,
                        positive_score=positive_score,
                        source_embedding=source_embedding,
                        target_embedding=target_embedding,
                        embedding_view=embedding_view,
                        strict_cross_band=strict_cross_band,
                    )
                )
            iterator.set_postfix({"records": len(records)})
            now = time.time()
            if now >= next_log_time:
                _progress_write(
                    iterator,
                    f"{prefix} heartbeat protocol={protocol} split={split_label} batch={batch_idx}/{len(loader)} "
                    f"records={len(records)} elapsed={_format_elapsed(now - started_at)}",
                )
                while next_log_time <= now:
                    next_log_time += log_interval_seconds
    result = _summarize_direct_records(records, protocol=protocol, cfg=cfg) if strict_cross_band else _finalize_direct_records(records, protocol=protocol, cfg=cfg)
    result["embedding_view"] = embedding_view
    result["model_type"] = "sccwm_force_grounded_v21a"
    _progress_write(
        iterator,
        f"{prefix} finished protocol={protocol} split={split_label} records={len(records)} "
        f"filtered={result['filtered_sample_count']} elapsed={_format_elapsed(time.time() - started_at)}",
    )
    return result


def run_force_grounded_direct_eval_v21a(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
    embedding_view: str = "full_state",
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_force_grounded_v21a_for_eval(cfg, checkpoint_path, device)
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        return _run_force_grounded_cross_band_eval_v21a(
            cfg,
            model=model,
            protocol=protocol,
            sequence_length=sequence_length,
            limit=limit,
            embedding_view=embedding_view,
            checkpoint_path=checkpoint_path,
        )
    loader = build_force_grounded_v21a_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
    return _run_force_grounded_eval_loop(
        cfg=cfg,
        model=model,
        loader=loader,
        protocol=protocol,
        split_label=split,
        checkpoint_path=checkpoint_path,
        limit=limit,
        embedding_view=embedding_view,
        strict_cross_band=False,
    )


def _run_force_grounded_cross_band_eval_v21a(
    cfg: dict[str, Any],
    *,
    model: SCCWMForceGroundedV21A,
    protocol: str,
    sequence_length: int | None,
    limit: int,
    embedding_view: str,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
    loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    result = _run_force_grounded_eval_loop(
        cfg=cfg,
        model=model,
        loader=loader,
        protocol=protocol,
        split_label="strict_cross_band",
        checkpoint_path=checkpoint_path,
        limit=limit,
        embedding_view=embedding_view,
        strict_cross_band=True,
    )
    result["matching_stats"] = matching_stats
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate force-grounded SCCWM v2.1a direct prediction protocols.",
        "sccwm_force_grounded_v21a/configs/sccwm_stage2_force_grounded_v21a.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--embedding-view", type=str, default="full_state", choices=EMBEDDING_VIEW_CHOICES)
    parser.add_argument("--save-records-jsonl", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_direct_eval_v21a(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
        embedding_view=args.embedding_view,
    )
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v21a_{args.protocol}_{args.split}_{args.embedding_view}.json"
    save_eval_result(result, output)
    if args.save_records_jsonl:
        _write_jsonl(result.get("records", []), args.save_records_jsonl)
    print(result["metrics"])


if __name__ == "__main__":
    main()
