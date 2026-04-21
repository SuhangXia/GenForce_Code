#!/usr/bin/env python3
from __future__ import annotations

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
from sccwm_force_grounded_v21.eval.eval_sccwm_force_grounded_direct_v21 import (
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    build_force_grounded_v21_eval_loader,
    load_force_grounded_v21_for_eval,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _anchor_indices(batch: dict[str, Any]) -> torch.Tensor:
    return batch["seq_valid_mask"].to(torch.float32).argmax(dim=1)


def _phase_load_progress_target(batch: dict[str, Any]) -> torch.Tensor:
    anchor_idx = _anchor_indices(batch)
    targets: list[torch.Tensor] = []
    for batch_idx, anchor in enumerate(anchor_idx.tolist()):
        phase_name = str(batch["phase_names"][batch_idx][anchor])
        progress = batch["phase_progress"][batch_idx, anchor]
        if phase_name == "precontact":
            target = progress.new_zeros(())
        elif phase_name == "press":
            target = progress.clamp(0.0, 1.0)
        elif phase_name == "dwell":
            target = progress.new_ones(())
        elif phase_name == "release":
            target = (1.0 - progress).clamp(0.0, 1.0)
        else:
            target = progress.clamp(0.0, 1.0)
        targets.append(target)
    return torch.stack(targets, dim=0)


def _contact_target_from_batch(batch: dict[str, Any], *, threshold: float = 0.03) -> torch.Tensor:
    source_delta = batch["source_obs"][:, -1, 1].abs()
    target_delta = batch["target_obs"][:, -1, 1].abs()
    source_ratio = (source_delta > float(threshold)).to(torch.float32).mean(dim=(1, 2))
    target_ratio = (target_delta > float(threshold)).to(torch.float32).mean(dim=(1, 2))
    return 0.5 * (source_ratio + target_ratio)


def _penetration_target(batch: dict[str, Any], *, proxy_scale: float = 20.0) -> torch.Tensor:
    mean_scale = 0.5 * (batch["source_scale_mm"] + batch["target_scale_mm"])
    normalized = batch["depth_mm"] / mean_scale.clamp_min(1e-6)
    return (normalized * float(proxy_scale)).clamp(0.0, 1.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _scale_bucket_labels(scale_mm: torch.Tensor, edges_mm: list[float]) -> torch.Tensor:
    edges = scale_mm.new_tensor(edges_mm, dtype=torch.float32)
    return torch.bucketize(scale_mm.to(torch.float32), edges)


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


def run_force_grounded_latent_eval_v21(
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
    scale_bucket_edges_mm = [float(v) for v in cfg.get("loss", {}).get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])]
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
        loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    else:
        loader = build_force_grounded_v21_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
        matching_stats = None
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"ForceGroundedV21 latent eval {protocol}", total=len(loader), leave=False)
        for batch in iterator:
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
            penetration_target = _penetration_target(batch)
            contact_target = _contact_target_from_batch(batch)
            if "phase_names" in batch and "phase_progress" in batch:
                load_target = _phase_load_progress_target(batch)
            else:
                load_target = penetration_target
            scale_labels = torch.cat(
                [
                    _scale_bucket_labels(batch["source_scale_mm"], scale_bucket_edges_mm),
                    _scale_bucket_labels(batch["target_scale_mm"], scale_bucket_edges_mm),
                ],
                dim=0,
            )
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
            for sample_idx in range(source["z_force_global"].shape[0]):
                rows.append(
                    {
                        "z_source": source["z_force_global"][sample_idx].cpu().tolist(),
                        "z_target": target["z_force_global"][sample_idx].cpu().tolist(),
                        "z_map_source": source["z_force_map"][sample_idx].cpu().tolist(),
                        "z_map_target": target["z_force_map"][sample_idx].cpu().tolist(),
                        "pred_penetration_proxy": float(0.5 * (source["pred_penetration_proxy"][sample_idx] + target["pred_penetration_proxy"][sample_idx])),
                        "pred_contact_intensity_proxy": float(
                            0.5 * (source["pred_contact_intensity_proxy"][sample_idx] + target["pred_contact_intensity_proxy"][sample_idx])
                        ),
                        "pred_load_progress_proxy": float(0.5 * (source["pred_load_progress_proxy"][sample_idx] + target["pred_load_progress_proxy"][sample_idx])),
                        "penetration_target": float(penetration_target[sample_idx].item()),
                        "contact_target": float(contact_target[sample_idx].item()),
                        "load_progress_target": float(load_target[sample_idx].item()),
                        "scale_adv_correct": float(0.5 * (scale_correct[sample_idx].item() + scale_correct[sample_idx + source["z_force_global"].shape[0]].item())),
                        "branch_adv_correct": float(0.5 * (branch_correct[sample_idx].item() + branch_correct[sample_idx + source["z_force_global"].shape[0]].item())),
                        "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
                        "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
                        "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
                        "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
                        "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
                        "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
                        "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
                    }
                )
            iterator.set_postfix({"rows": len(rows)})
            if limit > 0 and len(rows) >= limit and protocol not in STRICT_CROSS_BAND_PROTOCOLS:
                break
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        filtered = rows
    else:
        filtered = [row for row in rows if _protocol_filter(row, protocol, cfg)]
    result = _summarize_filtered_rows(filtered, protocol=protocol)
    if matching_stats is not None:
        result["matching_stats"] = matching_stats
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate z_force / force-proxy behavior for force-grounded SCCWM v2.1.",
        "sccwm_force_grounded_v21/configs/sccwm_stage2_force_grounded_v21.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_latent_eval_v21(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v21_latent_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
