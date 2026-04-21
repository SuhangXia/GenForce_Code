#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    _summarize_filtered_rows,
    _write_jsonl,
)
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import build_force_grounded_v21a_eval_loader
from sccwm_force_grounded_v21a.losses import (
    contact_intensity_target_v21a,
    penetration_target_v21a,
    phase_load_progress_target_v21a,
)
from sccwm_force_grounded_v22.eval.eval_sccwm_force_grounded_direct_v22 import load_force_grounded_v22_for_eval

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def run_force_grounded_latent_eval_v22(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_force_grounded_v22_for_eval(cfg, checkpoint_path, device)
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
        iterator = tqdm(loader, desc=f"ForceGroundedV22 latent eval {protocol}", total=len(loader), leave=False)
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
    filtered = rows if protocol in STRICT_CROSS_BAND_PROTOCOLS else [row for row in rows if _protocol_filter(row, protocol, cfg)]
    result = _summarize_filtered_rows(filtered, protocol=protocol)
    result["rows"] = filtered
    result["model_type"] = str(cfg.get("model", {}).get("variant", "v22"))
    if matching_stats is not None:
        result["matching_stats"] = matching_stats
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate z_force / force-proxy behavior for conservative force-grounded SCCWM v22 variants.",
        "sccwm_force_grounded_v22/configs/sccwm_stage2_force_grounded_v22ts.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--save-rows-jsonl", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_latent_eval_v22(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    rows = result.pop("rows", [])
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v22_latent_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    if args.save_rows_jsonl:
        _write_jsonl(rows, args.save_rows_jsonl)
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
