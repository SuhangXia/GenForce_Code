#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.train.common import build_stage_argparser, train_sccwm_stage
from sccwm.utils.config import load_config_with_overrides


def main() -> None:
    parser = build_stage_argparser("Train SCCWM stage 3 counterfactual model.", "sccwm/configs/sccwm_stage3.yaml")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args.config, args.override)
    train_sccwm_stage(cfg, stage="stage3")


if __name__ == "__main__":
    main()
