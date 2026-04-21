#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import (
    _finalize_direct_records,
    _format_elapsed,
    _move_eval_batch,
    _progress_write,
    _summarize_direct_records,
    build_eval_argparser,
    load_eval_config,
    save_eval_result,
)
from sccwm.train.common import default_device, load_checkpoint
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import (
    EMBEDDING_VIEW_CHOICES,
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    _build_direct_record,
    _write_jsonl,
    build_force_grounded_v21a_eval_loader,
)
from sccwm_force_grounded_v24.models import EMBEDDING_VIEW_TO_KEY, build_force_grounded_v24_model, select_embedding_view_v24

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def load_force_grounded_v24_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_force_grounded_v24_model(cfg.get("model", {})).to(device)
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


def _run_force_grounded_eval_loop_v24(
    *,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    loader: Any,
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
        iterator = tqdm(loader, desc=f"ForceGroundedV24 direct eval {protocol} {split_label}", total=len(loader), leave=False)
        prefix = "force_grounded_v24_cross_band_eval" if strict_cross_band else "force_grounded_v24_direct_eval"
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
            source_embedding = select_embedding_view_v24(source, embedding_view)
            target_embedding = select_embedding_view_v24(target, embedding_view)
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
    result["model_type"] = str(cfg.get("model", {}).get("variant", "v24"))
    _progress_write(
        iterator,
        f"{prefix} finished protocol={protocol} split={split_label} records={len(records)} "
        f"filtered={result['filtered_sample_count']} elapsed={_format_elapsed(time.time() - started_at)}",
    )
    return result


def run_force_grounded_direct_eval_v24(
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
    model = load_force_grounded_v24_for_eval(cfg, checkpoint_path, device)
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs

        pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
        loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
        result = _run_force_grounded_eval_loop_v24(
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
    else:
        loader = build_force_grounded_v21a_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
        result = _run_force_grounded_eval_loop_v24(
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
    result["model_type"] = str(cfg.get("model", {}).get("variant", "v24"))
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate force-guided geometry/operator factorized SCCWM v24 direct prediction protocols.",
        "sccwm_force_grounded_v24/configs/sccwm_stage2_force_grounded_v24fgof.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--embedding-view", type=str, default="full_state", choices=list(EMBEDDING_VIEW_TO_KEY.keys()) or EMBEDDING_VIEW_CHOICES)
    parser.add_argument("--save-records-jsonl", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_direct_eval_v24(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
        embedding_view=args.embedding_view,
    )
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v24_{args.protocol}_{args.split}_{args.embedding_view}.json"
    save_eval_result(result, output)
    if args.save_records_jsonl:
        _write_jsonl(result.get("records", []), args.save_records_jsonl)
    print(result["metrics"])


if __name__ == "__main__":
    main()
