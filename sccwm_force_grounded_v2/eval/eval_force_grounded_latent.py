#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import _batch_item, _move_eval_batch, build_eval_argparser, load_eval_config, save_eval_result
from sccwm.eval.common import _protocol_filter
from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs
from sccwm.train.common import default_device
from sccwm_force_grounded_v2.eval.eval_sccwm_force_grounded_direct import (
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    build_force_grounded_eval_loader,
    load_force_grounded_for_eval,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _contact_target_from_batch(batch: dict[str, Any], *, threshold: float = 0.03) -> torch.Tensor:
    source_delta = batch["source_obs"][:, -1, 1].abs()
    target_delta = batch["target_obs"][:, -1, 1].abs()
    source_ratio = (source_delta > float(threshold)).to(torch.float32).mean(dim=(1, 2))
    target_ratio = (target_delta > float(threshold)).to(torch.float32).mean(dim=(1, 2))
    return 0.5 * (source_ratio + target_ratio)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _summarize_filtered_rows(rows: list[dict[str, Any]], *, protocol: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No latent-eval samples remained for protocol: {protocol}")
    z_source = np.asarray([row["z_source"] for row in rows], dtype=np.float32)
    z_target = np.asarray([row["z_target"] for row in rows], dtype=np.float32)
    z_all = np.concatenate([z_source, z_target], axis=0)
    proxy_force = np.asarray([row["pred_normal_force_proxy"] for row in rows], dtype=np.float32)
    proxy_contact = np.asarray([row["pred_contact_intensity_proxy"] for row in rows], dtype=np.float32)
    depth_target = np.asarray([row["depth_target"] for row in rows], dtype=np.float32)
    contact_target = np.asarray([row["contact_target"] for row in rows], dtype=np.float32)
    cosine = np.sum(z_source * z_target, axis=1) / (
        np.linalg.norm(z_source, axis=1) * np.linalg.norm(z_target, axis=1) + 1e-6
    )
    pair_mse = np.mean((z_source - z_target) ** 2, axis=1)
    per_dim_std = z_all.std(axis=0)
    metrics = {
        "z_force_pair_mse": float(pair_mse.mean()),
        "z_force_pair_cosine": float(cosine.mean()),
        "z_force_dim_std_mean": float(per_dim_std.mean()),
        "z_force_dim_std_min": float(per_dim_std.min()),
        "normal_force_proxy_mae": float(np.abs(proxy_force - depth_target).mean()),
        "normal_force_proxy_pearson": _pearson(proxy_force, depth_target),
        "contact_intensity_proxy_mae": float(np.abs(proxy_contact - contact_target).mean()),
        "contact_intensity_proxy_pearson": _pearson(proxy_contact, contact_target),
        "sample_count": int(len(rows)),
    }
    return {
        "protocol_name": protocol,
        "filtered_sample_count": len(rows),
        "metrics": metrics,
    }


def run_force_grounded_latent_eval(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_force_grounded_for_eval(cfg, checkpoint_path, device)
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
        loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    else:
        loader = build_force_grounded_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
        matching_stats = None
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"ForceGrounded latent eval {protocol}", total=len(loader), leave=False)
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
            force_proxy = 0.5 * (source["pred_normal_force_proxy"] + target["pred_normal_force_proxy"])
            contact_proxy = 0.5 * (source["pred_contact_intensity_proxy"] + target["pred_contact_intensity_proxy"])
            contact_target = _contact_target_from_batch(batch)
            for sample_idx in range(source["z_force"].shape[0]):
                row = {
                    "z_source": source["z_force"][sample_idx].cpu().tolist(),
                    "z_target": target["z_force"][sample_idx].cpu().tolist(),
                    "pred_normal_force_proxy": float(force_proxy[sample_idx].item()),
                    "pred_contact_intensity_proxy": float(contact_proxy[sample_idx].item()),
                    "depth_target": float(batch["depth_mm"][sample_idx].item()),
                    "contact_target": float(contact_target[sample_idx].item()),
                    "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
                    "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
                    "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
                    "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
                    "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
                    "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
                    "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
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
    if matching_stats is not None:
        result["matching_stats"] = matching_stats
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate z_force / force-proxy behavior for force-grounded SCCWM v2.",
        "sccwm_force_grounded_v2/configs/sccwm_stage2_force_grounded.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_latent_eval(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v2_latent_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
