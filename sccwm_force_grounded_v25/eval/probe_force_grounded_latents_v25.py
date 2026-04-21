#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import save_eval_result
from sccwm.utils.config import load_config_with_overrides
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import (
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    build_force_grounded_v21a_eval_loader,
)
from sccwm_force_grounded_v21a.eval.probe_force_grounded_latents_v21a import (
    PROBE_TASK_CHOICES,
    _balanced_sample,
    _expand_branch_rows,
    _fit_probe,
    _pair_row_common,
    _scale_bucket_labels,
    _strict_partition,
)
from sccwm_force_grounded_v25.eval.eval_sccwm_force_grounded_direct_v25 import load_force_grounded_v25_for_eval
from sccwm_force_grounded_v25.models import EMBEDDING_VIEW_TO_KEY, select_embedding_view_v25

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _collect_pair_rows(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    protocol: str | None,
    sequence_length: int | None,
    embedding_view: str,
) -> list[dict[str, Any]]:
    from sccwm.eval.common import _move_eval_batch, _protocol_filter
    from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs
    from sccwm.train.common import default_device

    device = default_device(cfg)
    model = load_force_grounded_v25_for_eval(cfg, checkpoint_path, device)
    model.eval()
    loss_cfg = cfg.get("loss", {})
    scale_bucket_edges_mm = [float(v) for v in loss_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])]
    strict_protocol = protocol in STRICT_CROSS_BAND_PROTOCOLS if protocol else False
    if strict_protocol:
        pair_specs, _ = _build_cross_band_pair_specs(cfg, protocol=str(protocol), limit=0)
        loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    else:
        loader = build_force_grounded_v21a_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Collect v25 probe rows {split} {protocol or 'all'} {embedding_view}", total=len(loader), leave=False)
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
            source_embedding = select_embedding_view_v25(source, embedding_view)
            target_embedding = select_embedding_view_v25(target, embedding_view)
            source_scale_bucket = _scale_bucket_labels(batch["source_scale_mm"], scale_bucket_edges_mm)
            target_scale_bucket = _scale_bucket_labels(batch["target_scale_mm"], scale_bucket_edges_mm)
            for sample_idx in range(source_embedding.shape[0]):
                rows.append(
                    {
                        **_pair_row_common(batch, sample_idx),
                        "source_embedding": source_embedding[sample_idx].detach().cpu().tolist(),
                        "target_embedding": target_embedding[sample_idx].detach().cpu().tolist(),
                        "source_scale_bucket": int(source_scale_bucket[sample_idx].item()),
                        "target_scale_bucket": int(target_scale_bucket[sample_idx].item()),
                    }
                )
    if protocol and not strict_protocol:
        rows = [row for row in rows if _protocol_filter(row, protocol, cfg)]
    return rows


def _run_probe_task(
    branch_rows_train: list[dict[str, Any]],
    branch_rows_eval: list[dict[str, Any]],
    *,
    task: str,
    seed: int,
    max_train_samples: int,
    max_eval_samples: int,
) -> dict[str, Any]:
    if task == "scale":
        label_key = "scale_bucket"
    elif task == "branch":
        label_key = "branch_label"
    elif task == "marker":
        label_key = "marker_label"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported probe task: {task}")
    train_rows = [row for row in branch_rows_train if row.get(label_key) is not None]
    eval_rows = [row for row in branch_rows_eval if row.get(label_key) is not None]
    if task == "marker" and (not train_rows or not eval_rows):
        return {"status": "skipped", "reason": "marker_labels_unavailable"}
    if len({row[label_key] for row in train_rows}) <= 1 or len({row[label_key] for row in eval_rows}) <= 1:
        return {"status": "skipped", "reason": f"{task}_labels_insufficient_variation"}
    train_rows = _balanced_sample(train_rows, label_key=label_key, seed=seed, max_samples=max_train_samples)
    eval_rows = _balanced_sample(eval_rows, label_key=label_key, seed=seed + 1, max_samples=max_eval_samples)
    result = _fit_probe(train_rows, eval_rows, label_key=label_key, seed=seed)
    result["status"] = "ok"
    result["label_key"] = label_key
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen post-hoc leakage probes on split-canonical SCCWM v25 embeddings.")
    parser.add_argument("--config", type=str, default="sccwm_force_grounded_v25/configs/sccwm_stage2_force_grounded_v25scfgof.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--protocol", type=str, default="", choices=[""] + STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--embedding-view", type=str, default="canonical_pose_only", choices=list(EMBEDDING_VIEW_TO_KEY.keys()))
    parser.add_argument("--probe-tasks", type=str, default="scale,branch,marker")
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--max-eval-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)
    protocol = args.protocol or None
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        strict_rows = _collect_pair_rows(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.eval_split,
            protocol=protocol,
            sequence_length=args.sequence_length,
            embedding_view=args.embedding_view,
        )
        train_pair_rows = _strict_partition(strict_rows, partition="train")
        eval_pair_rows = _strict_partition(strict_rows, partition="eval")
        split_note = "strict protocol rows were partitioned 70/30 by deterministic hash of pair_id/event_key"
    else:
        train_pair_rows = _collect_pair_rows(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.train_split,
            protocol=protocol,
            sequence_length=args.sequence_length,
            embedding_view=args.embedding_view,
        )
        eval_pair_rows = _collect_pair_rows(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.eval_split,
            protocol=protocol,
            sequence_length=args.sequence_length,
            embedding_view=args.embedding_view,
        )
        split_note = "standard split loaders used directly"

    branch_rows_train = _expand_branch_rows(train_pair_rows)
    branch_rows_eval = _expand_branch_rows(eval_pair_rows)
    tasks = [task.strip() for task in args.probe_tasks.split(",") if task.strip()]
    invalid = sorted(set(tasks) - set(PROBE_TASK_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported probe tasks: {invalid}")

    results: dict[str, Any] = {}
    for task in tasks:
        results[task] = _run_probe_task(
            branch_rows_train,
            branch_rows_eval,
            task=task,
            seed=int(args.seed),
            max_train_samples=int(args.max_train_samples),
            max_eval_samples=int(args.max_eval_samples),
        )

    payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "embedding_view": args.embedding_view,
        "protocol": protocol,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "split_note": split_note,
        "train_pair_count": len(train_pair_rows),
        "eval_pair_count": len(eval_pair_rows),
        "train_branch_count": len(branch_rows_train),
        "eval_branch_count": len(branch_rows_eval),
        "probe_tasks": tasks,
        "results": results,
    }
    save_eval_result(payload, args.output)
    print(json.dumps(payload["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
