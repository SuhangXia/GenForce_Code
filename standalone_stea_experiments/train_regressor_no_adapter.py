#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from standalone_stea_experiments.utils import build_common_train_parser, train_regressor_pipeline


def parse_args():
    parser = build_common_train_parser(
        description='Train a fair no-adapter downstream baseline on scales 16/20/23.',
        default_train_scales=[16.0, 20.0, 23.0],
        default_checkpoint_name='no_adapter_baseline',
        default_wandb_project='STEA_Supp_NoAdapter',
        default_wandb_name='no_adapter_baseline_train',
        require_stea=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_regressor_pipeline(
        args=args,
        use_stea=False,
        description='Training fair no-adapter baseline regressor',
        pipeline_kind='none',
    )


if __name__ == '__main__':
    main()
