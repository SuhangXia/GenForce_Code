#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import build_eval_argparser, load_eval_config, run_direct_eval, save_eval_result


def main() -> None:
    parser = build_eval_argparser("Evaluate SCCWM direct state prediction protocols.", "sccwm/configs/sccwm_stage3.yaml")
    parser.add_argument(
        "--protocol",
        type=str,
        default="same_scale_sanity",
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
    result = run_direct_eval(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sequence_length=args.sequence_length,
        protocol=args.protocol,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/direct_{args.protocol}_{args.split}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
