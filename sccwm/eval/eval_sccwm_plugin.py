#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import build_eval_argparser, load_eval_config, run_plugin_eval, save_eval_result


def main() -> None:
    parser = build_eval_argparser("Evaluate SCCWM plugin and baseline methods.", "sccwm/configs/sccwm_stage4.yaml")
    parser.add_argument(
        "--method",
        type=str,
        default="sccwm_plugin",
        choices=["legacy_no_adaptation", "deterministic_transport", "scta", "sccwm_plugin"],
    )
    parser.add_argument("--legacy-ckpt", type=str, required=True)
    parser.add_argument("--scta-ckpt", type=str, default="")
    args = parser.parse_args()
    cfg = load_eval_config(args)
    result = run_plugin_eval(
        cfg,
        method=args.method,
        legacy_checkpoint_path=args.legacy_ckpt,
        split=args.split,
        sequence_length=args.sequence_length,
        sccwm_checkpoint_path=args.checkpoint,
        scta_checkpoint_path=args.scta_ckpt,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/plugin_{args.method}_{args.split}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
