#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.train.common import build_stage_argparser, train_legacy_regressor
from sccwm.utils.config import load_config_with_overrides


def main() -> None:
    parser = build_stage_argparser("Train SCCWM legacy regressors on dataset B.", "sccwm/configs/legacy_regressor.yaml")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args.config, args.override)
    train_legacy_regressor(cfg)


if __name__ == "__main__":
    main()
