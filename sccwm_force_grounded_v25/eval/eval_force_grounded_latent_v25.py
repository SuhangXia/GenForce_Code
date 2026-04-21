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

from sccwm.eval.common import _move_eval_batch, _protocol_filter, build_eval_argparser, load_eval_config, save_eval_result
from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs
from sccwm.train.common import default_device
from sccwm_force_grounded_v21a.eval.eval_force_grounded_latent_v21a import (
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    _latent_row_common,
    _pearson,
    _scale_bucket_labels,
    _write_jsonl,
)
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import build_force_grounded_v21a_eval_loader
from sccwm_force_grounded_v21a.losses import contact_intensity_target_v21a, penetration_target_v21a, phase_load_progress_target_v21a
from sccwm_force_grounded_v25.eval.eval_sccwm_force_grounded_direct_v25 import load_force_grounded_v25_for_eval

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _cosine_mean(a_rows: list[list[float]], b_rows: list[list[float]]) -> float:
    if not a_rows:
        return float("nan")
    a = torch.tensor(a_rows, dtype=torch.float32)
    b = torch.tensor(b_rows, dtype=torch.float32)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item())


def _mse_mean(a_rows: list[list[float]], b_rows: list[list[float]]) -> float:
    if not a_rows:
        return float("nan")
    a = torch.tensor(a_rows, dtype=torch.float32)
    b = torch.tensor(b_rows, dtype=torch.float32)
    return float(torch.mean((a - b).pow(2)).item())


def _summarize_filtered_rows_v25(rows: list[dict[str, Any]], *, protocol: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No rows available after filtering for protocol={protocol}")
    z_source = [row["z_source"] for row in rows]
    z_target = [row["z_target"] for row in rows]
    z_map_source = [row["z_map_source_flat"] for row in rows]
    z_map_target = [row["z_map_target_flat"] for row in rows]
    penetration_proxy = np.asarray([row["pred_penetration_proxy"] for row in rows], dtype=np.float32)
    penetration_target = np.asarray([row["penetration_target"] for row in rows], dtype=np.float32)
    contact_proxy = np.asarray([row["pred_contact_intensity_proxy"] for row in rows], dtype=np.float32)
    contact_target = np.asarray([row["contact_target"] for row in rows], dtype=np.float32)
    load_proxy = np.asarray([row["pred_load_progress_proxy"] for row in rows], dtype=np.float32)
    load_target = np.asarray([row["load_progress_target"] for row in rows], dtype=np.float32)

    metrics = {
        "z_force_global_pair_mse": _mse_mean(z_source, z_target),
        "z_force_global_pair_cosine": _cosine_mean(z_source, z_target),
        "z_force_map_pair_mse": _mse_mean(z_map_source, z_map_target),
        "z_force_dim_std_mean": float(torch.tensor(z_source, dtype=torch.float32).std(dim=0, unbiased=False).mean().item()),
        "z_force_dim_std_min": float(torch.tensor(z_source, dtype=torch.float32).std(dim=0, unbiased=False).min().item()),
        "penetration_proxy_mae": float(np.mean(np.abs(penetration_proxy - penetration_target))),
        "penetration_proxy_pearson": _pearson(penetration_proxy, penetration_target),
        "contact_intensity_proxy_mae": float(np.mean(np.abs(contact_proxy - contact_target))),
        "contact_intensity_proxy_pearson": _pearson(contact_proxy, contact_target),
        "load_progress_proxy_mae": float(np.mean(np.abs(load_proxy - load_target))),
        "load_progress_proxy_pearson": _pearson(load_proxy, load_target),
        "sample_count": len(rows),
    }
    metrics["canonical_load_pair_cosine"] = _cosine_mean([row["canonical_load_source"] for row in rows], [row["canonical_load_target"] for row in rows])
    metrics["canonical_pose_pair_cosine"] = _cosine_mean([row["canonical_pose_source"] for row in rows], [row["canonical_pose_target"] for row in rows])
    metrics["canonical_load_pair_mse"] = _mse_mean([row["canonical_load_source"] for row in rows], [row["canonical_load_target"] for row in rows])
    metrics["canonical_pose_pair_mse"] = _mse_mean([row["canonical_pose_source"] for row in rows], [row["canonical_pose_target"] for row in rows])
    metrics["canonical_force_projected_pair_cosine"] = _cosine_mean(
        [row["canonical_force_projected_source"] for row in rows],
        [row["canonical_force_projected_target"] for row in rows],
    )
    metrics["canonical_load_force_alignment_cosine"] = 0.5 * (
        _cosine_mean([row["canonical_force_projected_source"] for row in rows], [row["force_teacher_source"] for row in rows])
        + _cosine_mean([row["canonical_force_projected_target"] for row in rows], [row["force_teacher_target"] for row in rows])
    )
    metrics["pose_load_within_source_cosine"] = _cosine_mean([row["canonical_pose_source"] for row in rows], [row["canonical_load_source"] for row in rows])
    metrics["pose_load_within_target_cosine"] = _cosine_mean([row["canonical_pose_target"] for row in rows], [row["canonical_load_target"] for row in rows])
    metrics["load_scale_adv_accuracy"] = float(sum(row.get("load_scale_adv_correct", 0.0) for row in rows) / len(rows))
    metrics["pose_scale_adv_accuracy"] = float(sum(row.get("pose_scale_adv_correct", 0.0) for row in rows) / len(rows))
    metrics["operator_scale_accuracy"] = float(sum(row.get("operator_scale_correct", 0.0) for row in rows) / len(rows))
    return {
        "protocol_name": protocol,
        "filtered_sample_count": len(rows),
        "metrics": metrics,
    }


def run_force_grounded_latent_eval_v25(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_force_grounded_v25_for_eval(cfg, checkpoint_path, device)
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
        iterator = tqdm(loader, desc=f"ForceGroundedV25 latent eval {protocol}", total=len(loader), leave=False)
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
            batch_n = source["z_force_global"].shape[0]

            branch_labels = torch.cat(
                [
                    torch.zeros(batch_n, device=device, dtype=torch.long),
                    torch.ones(batch_n, device=device, dtype=torch.long),
                ],
                dim=0,
            )
            scale_labels = torch.cat([source_scale_bucket, target_scale_bucket], dim=0)
            load_scale_logits = torch.cat([source["canonical_load_scale_adv_logits"], target["canonical_load_scale_adv_logits"]], dim=0)
            pose_scale_logits = torch.cat([source["canonical_pose_scale_adv_logits"], target["canonical_pose_scale_adv_logits"]], dim=0)
            operator_scale_logits = torch.cat([source["operator_scale_logits"], target["operator_scale_logits"]], dim=0)
            load_branch_logits = torch.cat([source["canonical_load_branch_adv_logits"], target["canonical_load_branch_adv_logits"]], dim=0)
            pose_branch_logits = torch.cat([source["canonical_pose_branch_adv_logits"], target["canonical_pose_branch_adv_logits"]], dim=0)
            operator_branch_logits = torch.cat([source["operator_branch_logits"], target["operator_branch_logits"]], dim=0)
            load_scale_correct = (load_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32)
            pose_scale_correct = (pose_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32)
            operator_scale_correct = (operator_scale_logits.argmax(dim=1) == scale_labels).to(torch.float32)
            load_branch_correct = (load_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32)
            pose_branch_correct = (pose_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32)
            operator_branch_correct = (operator_branch_logits.argmax(dim=1) == branch_labels).to(torch.float32)

            for sample_idx in range(batch_n):
                row = {
                    **_latent_row_common(batch, sample_idx),
                    "z_source": source["z_force_global"][sample_idx].detach().cpu().tolist(),
                    "z_target": target["z_force_global"][sample_idx].detach().cpu().tolist(),
                    "z_map_source": source["z_force_map"][sample_idx].detach().cpu().tolist(),
                    "z_map_target": target["z_force_map"][sample_idx].detach().cpu().tolist(),
                    "z_map_source_flat": source["z_force_map"][sample_idx].detach().flatten().cpu().tolist(),
                    "z_map_target_flat": target["z_force_map"][sample_idx].detach().flatten().cpu().tolist(),
                    "pred_penetration_proxy": float(0.5 * (source["pred_penetration_proxy"][sample_idx] + target["pred_penetration_proxy"][sample_idx]).item()),
                    "pred_contact_intensity_proxy": float(
                        0.5 * (source["pred_contact_intensity_proxy"][sample_idx] + target["pred_contact_intensity_proxy"][sample_idx]).item()
                    ),
                    "pred_load_progress_proxy": float(0.5 * (source["pred_load_progress_proxy"][sample_idx] + target["pred_load_progress_proxy"][sample_idx]).item()),
                    "penetration_target": float(penetration_target[sample_idx].item()),
                    "contact_target": float(contact_target[sample_idx].item()),
                    "load_progress_target": float(load_target[sample_idx].item()),
                    "source_scale_bucket": int(source_scale_bucket[sample_idx].item()),
                    "target_scale_bucket": int(target_scale_bucket[sample_idx].item()),
                    "canonical_load_source": source["z_canon_load"][sample_idx].detach().cpu().tolist(),
                    "canonical_load_target": target["z_canon_load"][sample_idx].detach().cpu().tolist(),
                    "canonical_pose_source": source["z_canon_pose"][sample_idx].detach().cpu().tolist(),
                    "canonical_pose_target": target["z_canon_pose"][sample_idx].detach().cpu().tolist(),
                    "operator_source": source["operator_code"][sample_idx].detach().cpu().tolist(),
                    "operator_target": target["operator_code"][sample_idx].detach().cpu().tolist(),
                    "canonical_force_projected_source": source["canonical_force_projected"][sample_idx].detach().cpu().tolist(),
                    "canonical_force_projected_target": target["canonical_force_projected"][sample_idx].detach().cpu().tolist(),
                    "force_teacher_source": source["force_teacher_latent"][sample_idx].detach().cpu().tolist(),
                    "force_teacher_target": target["force_teacher_latent"][sample_idx].detach().cpu().tolist(),
                    "load_scale_adv_correct": float(0.5 * (load_scale_correct[sample_idx].item() + load_scale_correct[sample_idx + batch_n].item())),
                    "pose_scale_adv_correct": float(0.5 * (pose_scale_correct[sample_idx].item() + pose_scale_correct[sample_idx + batch_n].item())),
                    "operator_scale_correct": float(0.5 * (operator_scale_correct[sample_idx].item() + operator_scale_correct[sample_idx + batch_n].item())),
                    "load_branch_adv_correct": float(0.5 * (load_branch_correct[sample_idx].item() + load_branch_correct[sample_idx + batch_n].item())),
                    "pose_branch_adv_correct": float(0.5 * (pose_branch_correct[sample_idx].item() + pose_branch_correct[sample_idx + batch_n].item())),
                    "operator_branch_correct": float(0.5 * (operator_branch_correct[sample_idx].item() + operator_branch_correct[sample_idx + batch_n].item())),
                }
                rows.append(row)
            iterator.set_postfix({"rows": len(rows)})
            if limit > 0 and len(rows) >= limit and protocol not in STRICT_CROSS_BAND_PROTOCOLS:
                break
    filtered = rows if protocol in STRICT_CROSS_BAND_PROTOCOLS else [row for row in rows if _protocol_filter(row, protocol, cfg)]
    result = _summarize_filtered_rows_v25(filtered, protocol=protocol)
    result["rows"] = filtered
    result["model_type"] = str(cfg.get("model", {}).get("variant", "v25"))
    if matching_stats is not None:
        result["matching_stats"] = matching_stats
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate split-canonical/force/operator latent behavior for SCCWM v25.",
        "sccwm_force_grounded_v25/configs/sccwm_stage2_force_grounded_v25scfgof.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--save-rows-jsonl", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_latent_eval_v25(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    rows = result.pop("rows", [])
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v25_latent_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    if args.save_rows_jsonl:
        _write_jsonl(rows, args.save_rows_jsonl)
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
