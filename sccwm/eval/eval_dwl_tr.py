#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import (
    _batch_item,
    _finalize_direct_records,
    _move_eval_batch,
    build_eval_argparser,
    build_eval_loader,
    default_device,
    load_eval_config,
    save_eval_result,
)
from sccwm.models import DWLTR
from sccwm.train.common import load_checkpoint

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def load_dwl_tr_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> DWLTR:
    model_cfg = cfg.get("model", {})
    model = DWLTR(
        input_channels=int(model_cfg.get("input_channels", 3)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        world_hidden_dim=int(model_cfg.get("world_hidden_dim", 128)),
        lattice_size=int(model_cfg.get("lattice_size", 32)),
    ).to(device)
    payload = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model


def run_dwl_tr_eval(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    sequence_length: int | None,
    protocol: str,
    limit: int = 0,
) -> dict[str, Any]:
    device = default_device(cfg)
    model = load_dwl_tr_for_eval(cfg, checkpoint_path, device)
    loader = build_eval_loader(cfg, split=split, sequence_length=sequence_length)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"DWL-TR eval {protocol} {split}", total=len(loader), leave=False)
        for batch in iterator:
            if limit > 0 and len(records) >= limit:
                break
            batch = _move_eval_batch(batch, device)
            abs_contact = batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None
            source = model.forward_single(
                batch["source_obs"],
                batch["source_coord_map"],
                batch["source_scale_mm"],
                valid_mask=batch["seq_valid_mask"],
                absolute_contact_xy_mm=abs_contact,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
            )
            target = model.forward_single(
                batch["target_obs"],
                batch["target_coord_map"],
                batch["target_scale_mm"],
                valid_mask=batch["seq_valid_mask"],
                absolute_contact_xy_mm=abs_contact,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
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
    return _finalize_direct_records(records, protocol=protocol, cfg=cfg)


def main() -> None:
    parser = build_eval_argparser("Evaluate the DWL-TR baseline.", "sccwm/configs/dwl_tr.yaml")
    parser.add_argument(
        "--protocol",
        type=str,
        default="unseen_indenters_heldout_scales",
        choices=[
            "same_scale_sanity",
            "heldout_exact_scales",
            "heldout_scale_bands",
            "unseen_indenters_seen_scales",
            "unseen_indenters_heldout_scales",
            "boundary_clean",
            "boundary_near_boundary",
            "boundary_partial_crop",
        ],
    )
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_dwl_tr_eval(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/dwl_tr_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
