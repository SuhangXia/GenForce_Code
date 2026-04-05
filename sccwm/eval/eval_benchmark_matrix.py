#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import load_eval_config, run_direct_eval, run_plugin_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCCWM benchmark matrix.")
    parser.add_argument("--config", type=str, default="sccwm/configs/sccwm_stage4.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--sccwm-ckpt", type=str, default="")
    parser.add_argument("--legacy-ckpt", type=str, required=True)
    parser.add_argument("--scta-ckpt", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="sccwm/eval_outputs/benchmark_matrix.csv")
    parser.add_argument("--output-json", type=str, default="sccwm/eval_outputs/benchmark_matrix.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args)
    rows: list[dict[str, Any]] = []
    direct_protocols = [
        "same_scale_sanity",
        "heldout_exact_scales",
        "heldout_scale_bands",
        "unseen_indenters_seen_scales",
        "unseen_indenters_heldout_scales",
        "boundary_clean",
        "boundary_near_boundary",
        "boundary_partial_crop",
    ]
    for protocol in direct_protocols:
        result = run_direct_eval(
            cfg,
            checkpoint_path=args.sccwm_ckpt,
            split=args.split,
            sequence_length=cfg.get("dataset", {}).get("sequence_length", 3),
            protocol=protocol,
            limit=args.limit,
        )
        rows.append({"family": "direct", "method": "sccwm", "protocol": protocol, **result["metrics"]})

    plugin_methods = ["legacy_no_adaptation", "deterministic_transport", "scta", "sccwm_plugin"]
    for method in plugin_methods:
        ckpt = args.sccwm_ckpt if method in {"deterministic_transport", "sccwm_plugin"} else ""
        result = run_plugin_eval(
            cfg,
            method=method,
            legacy_checkpoint_path=args.legacy_ckpt,
            split=args.split,
            sequence_length=cfg.get("dataset", {}).get("sequence_length", 3),
            sccwm_checkpoint_path=ckpt,
            scta_checkpoint_path=args.scta_ckpt,
            limit=args.limit,
        )
        rows.append({"family": "plugin", "method": method, "protocol": args.split, **result["metrics"]})

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.write_text(__import__("json").dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} benchmark rows to {output_csv}")


if __name__ == "__main__":
    main()
