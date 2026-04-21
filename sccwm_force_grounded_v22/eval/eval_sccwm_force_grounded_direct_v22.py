#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import build_eval_argparser, load_eval_config, save_eval_result
from sccwm.train.common import default_device, load_checkpoint
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import (
    EMBEDDING_VIEW_CHOICES,
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    _run_force_grounded_eval_loop,
    _write_jsonl,
    build_force_grounded_v21a_eval_loader,
)
from sccwm_force_grounded_v22.models import EMBEDDING_VIEW_TO_KEY, build_force_grounded_v22_model


def load_force_grounded_v22_for_eval(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_force_grounded_v22_model(cfg.get("model", {})).to(device)
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


def run_force_grounded_direct_eval_v22(
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
    model = load_force_grounded_v22_for_eval(cfg, checkpoint_path, device)
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs

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
    else:
        loader = build_force_grounded_v21a_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
        result = _run_force_grounded_eval_loop(
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
    result["model_type"] = str(cfg.get("model", {}).get("variant", "v22"))
    return result


def main() -> None:
    parser = build_eval_argparser(
        "Evaluate conservative force-grounded SCCWM v22 direct prediction protocols.",
        "sccwm_force_grounded_v22/configs/sccwm_stage2_force_grounded_v22ts.yaml",
    )
    parser.add_argument("--protocol", type=str, default="cross_band_16_23_bidirectional", choices=STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--embedding-view", type=str, default="full_state", choices=list(EMBEDDING_VIEW_TO_KEY.keys()) or EMBEDDING_VIEW_CHOICES)
    parser.add_argument("--save-records-jsonl", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_force_grounded_direct_eval_v22(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
        embedding_view=args.embedding_view,
    )
    output = args.output or f"sccwm/eval_outputs/sccwm_force_grounded_v22_{args.protocol}_{args.split}_{args.embedding_view}.json"
    save_eval_result(result, output)
    if args.save_records_jsonl:
        _write_jsonl(result.get("records", []), args.save_records_jsonl)
    print(result["metrics"])


if __name__ == "__main__":
    main()
