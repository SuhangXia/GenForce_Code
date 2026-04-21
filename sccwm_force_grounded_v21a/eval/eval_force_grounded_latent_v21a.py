#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import _batch_item, _move_eval_batch, _protocol_filter, build_eval_argparser, load_eval_config, save_eval_result
from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs
from sccwm.train.common import default_device
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import (
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    build_force_grounded_v21a_eval_loader,
    load_force_grounded_v21a_for_eval,
)
from sccwm_force_grounded_v21a.losses import (
    contact_intensity_target_v21a,
    penetration_target_v21a,
    phase_load_progress_target_v21a,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _scale_bucket_labels(scale_mm: torch.Tensor, edges_mm: list[float]) -> torch.Tensor:
    edges = scale_mm.new_tensor(edges_mm, dtype=torch.float32)
    return torch.bucketize(scale_mm.to(torch.float32), edges)


def _metadata_field(batch: dict[str, Any], key: str, index: int) -> Any | None:
    metadata = batch.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return _batch_item(metadata[key], index)
    if key in batch:
        return _batch_item(batch[key], index)
    return None


def _latent_row_common(batch: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    source_episode = int(batch["source_episode_id"][sample_idx].item()) if "source_episode_id" in batch else None
    target_episode = int(batch["target_episode_id"][sample_idx].item()) if "target_episode_id" in batch else None
    pair_id = str(_batch_item(batch["pair_id"], sample_idx)) if "pair_id" in batch else None
    episode_id = int(batch["episode_id"][sample_idx].item()) if "episode_id" in batch else None
    if episode_id is None and source_episode is not None and target_episode is not None and source_episode == target_episode:
        episode_id = source_episode
    return {
        "event_key": pair_id
        or (
            f"{episode_id}::{int(batch['global_seq_index'][sample_idx].item())}"
            if "global_seq_index" in batch
            else f"{source_episode}::{target_episode}::{sample_idx}"
        ),
        "pair_id": pair_id,
        "episode_id": episode_id,
        "source_episode_id": source_episode if source_episode is not None else episode_id,
        "target_episode_id": target_episode if target_episode is not None else episode_id,
        "source_marker_name": _metadata_field(batch, "source_marker_name", sample_idx),
        "target_marker_name": _metadata_field(batch, "target_marker_name", sample_idx),
        "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
        "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
        "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
        "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
        "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
        "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
        "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
    }


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_filtered_rows(rows: list[dict[str, Any]], *, protocol: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No latent-eval samples remained for protocol: {protocol}")
    z_source = np.asarray([row["z_source"] for row in rows], dtype=np.float32)
    z_target = np.asarray([row["z_target"] for row in rows], dtype=np.float32)
    z_map_source = np.asarray([row["z_map_source"] for row in rows], dtype=np.float32)
    z_map_target = np.asarray([row["z_map_target"] for row in rows], dtype=np.float32)
    z_all = np.concatenate([z_source, z_target], axis=0)
    penetration_proxy = np.asarray([row["pred_penetration_proxy"] for row in rows], dtype=np.float32)
    contact_proxy = np.asarray([row["pred_contact_intensity_proxy"] for row in rows], dtype=np.float32)
    load_proxy = np.asarray([row["pred_load_progress_proxy"] for row in rows], dtype=np.float32)
    penetration_target = np.asarray([row["penetration_target"] for row in rows], dtype=np.float32)
    contact_target = np.asarray([row["contact_target"] for row in rows], dtype=np.float32)
    load_target = np.asarray([row["load_progress_target"] for row in rows], dtype=np.float32)
    scale_adv_correct = np.asarray([row["scale_adv_correct"] for row in rows], dtype=np.float32)
    branch_adv_correct = np.asarray([row["branch_adv_correct"] for row in rows], dtype=np.float32)
    cosine = np.sum(z_source * z_target, axis=1) / (np.linalg.norm(z_source, axis=1) * np.linalg.norm(z_target, axis=1) + 1e-6)
    pair_mse = np.mean((z_source - z_target) ** 2, axis=1)
    map_pair_mse = np.mean((z_map_source - z_map_target) ** 2, axis=(1, 2, 3))
    per_dim_std = z_all.std(axis=0)
    metrics = {
        "z_force_global_pair_mse": float(pair_mse.mean()),
        "z_force_global_pair_cosine": float(cosine.mean()),
        "z_force_map_pair_mse": float(map_pair_mse.mean()),
        "z_force_dim_std_mean": float(per_dim_std.mean()),
        "z_force_dim_std_min": float(per_dim_std.min()),
        "penetration_proxy_mae": float(np.abs(penetration_proxy - penetration_target).mean()),
        "penetration_proxy_pearson": _pearson(penetration_proxy, penetration_target),
        "contact_intensity_proxy_mae": float(np.abs(contact_proxy - contact_target).mean()),
        "contact_intensity_proxy_pearson": _pearson(contact_proxy, contact_target),
        "load_progress_proxy_mae": float(np.abs(load_proxy - load_target).mean()),
        "load_progress_proxy_pearson": _pearson(load_proxy, load_target),
        "scale_adv_accuracy": float(scale_adv_correct.mean()),
        "branch_adv_accuracy": float(branch_adv_correct.mean()),
        "sample_count": int(len(rows)),
    }
    return {
        "protocol_name": protocol,
        "filtered_sample_count": len(rows),
        "metrics": metrics,
    }


def run_force_grounded_latent_eval_v21a(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_force_grounded_v21a_for_eval(cfg, checkpoint_path, device)
    loss_cfg = cfg.get("loss", {})
    scale_bucket_edges_mm = [float(v) for v in cfg.get("loss", {}).get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])]
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
        loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    else:
        loader = build_force_grounded_v21a_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
        matching_stats = None
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"ForceGroundedV21A latent eval {protocol}", total=len(loader), leave=False)
        for batch in iterator:
            batch = _move_eval_batch(batch, device)
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
            penetration_target = penetration_target_v21a(batch, proxy_scale=float(loss_cfg.get("penetration_proxy_scale", 20.0)))
            contact_target = contact_intensity_target_v21a(batch, threshold=float(loss_cfg.get("contact_intensity_threshold", 0.03)))
            load_target = phase_load_progress_target_v21a(batch) if ("phase_names" in batch and "phase_progress" in batch) else penetration_target
            source_scale_bucket = _scale_bucket_labels(batch["source_scale_mm"], scale_bucket_edges_mm)
            target_scale_bucket = _scale_bucket_labels(batch["target_scale_mm"], scale_bucket_edges_mm)
            scale_labels = torch.cat([source_scale_bucket, target_scale_bucket], dim=0)
            branch_labels = torch.cat(
                [
                    torch.zeros(source["branch_adv_logits"].shape[0], device=device, dtype=torch.long),
                    torch.ones(target["branch_adv_logits"].shape[0], device=device, dtype=torch.long),
                ],
                dim=0,
            )
            scale_logits = torch.cat([source["scale_adv_logits"], target["scale_adv_logits"]], dim=0)
            branch_logits = torch.cat([source["branch_adv_logits"], target["branch_adv_logits"]], dim=0)
            scale_correct = (scale_logits.argmax(dim=1) == scale_labels).to(torch.float32)
            branch_correct = (branch_logits.argmax(dim=1) == branch_labels).to(torch.float32)
            batch_n = source["z_force_global"].shape[0]
            for sample_idx in range(batch_n):
                row = {
                    **_latent_row_common(batch, sample_idx),
                    "z_source": source["z_force_global"][sample_idx].detach().cpu().tolist(),
                    "z_target": target["z_force_global"][sample_idx].detach().cpu().tolist(),
                    "z_map_source": source["z_force_map"][sample_idx].detach().cpu().tolist(),
                    "z_map_target": target["z_force_map"][sample_idx].detach().cpu().tolist(),
                    "pred_penetration_proxy_source": float(source["pred_penetration_proxy"][sample_idx].item()),
                    "pred_penetration_proxy_target": float(target["pred_penetration_proxy"][sample_idx].item()),
                    "pred_penetration_proxy": float(0.5 * (source["pred_penetration_proxy"][sample_idx] + target["pred_penetration_proxy"][sample_idx]).item()),
                    "pred_contact_intensity_proxy_source": float(source["pred_contact_intensity_proxy"][sample_idx].item()),
                    "pred_contact_intensity_proxy_target": float(target["pred_contact_intensity_proxy"][sample_idx].item()),
                    "pred_contact_intensity_proxy": float(
                        0.5 * (source["pred_contact_intensity_proxy"][sample_idx] + target["pred_contact_intensity_proxy"][sample_idx]).item()
                    ),
                    "pred_load_progress_proxy_source": float(source["pred_load_progress_proxy"][sample_idx].item()),
                    "pred_load_progress_proxy_target": float(target["pred_load_progress_proxy"][sample_idx].item()),
                    "pred_load_progress_proxy": float(0.5 * (source["pred_load_progress_proxy"][sample_idx] + target["pred_load_progress_proxy"][sample_idx]).item()),
                    "penetration_target": float(penetration_target[sample_idx].item()),
                    "contact_target": float(contact_target[sample_idx].item()),
                    "load_progress_target": float(load_target[sample_idx].item()),
                    "source_scale_bucket": int(source_scale_bucket[sample_idx].item()),
                    "target_scale_bucket": int(target_scale_bucket[sample_idx].item()),
                    "source_branch_label": 0,
                    "target_branch_label": 1,
                    "scale_adv_correct": float(0.5 * (scale_correct[sample_idx].item() + scale_correct[sample_idx + batch_n].item())),
                    "branch_adv_correct": float(0.5 * (branch_correct[sample_idx].item() + branch_correct[sample_idx + batch_n].item())),
                }
                rows.append(row)
            iterator.set_postfix({"rows": len(rows)})
            if limit > 0 and len(rows) >= limit and protocol not in STRICT_CROSS_BAND_PROTOCOLS:
                break
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        filtered = rows
    else:
        filtered = [row for row in rows if _protocol_filter(row, protocol, cfg)]
    result = _summarize_filtered_rows(filtered, protocol=protocol)
    result["rows"] = filtered
    result["model_type"] = "sccwm_force_grounded_v21a"
    if matching_stats is not None:
        result["matching_stats"] = matching_stats
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate z_force / force-proxy behavior for force-grounded SCCWM v2.1a.",
        "sccwm_force_grounded_v21a/configs/sccwm_stage2_force_grounded_v21a.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--save-rows-jsonl", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_latent_eval_v21a(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    rows = result.pop("rows", [])
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v21a_latent_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    if args.save_rows_jsonl:
        _write_jsonl(rows, args.save_rows_jsonl)
    print(result["metrics"])


if __name__ == "__main__":
    main()
