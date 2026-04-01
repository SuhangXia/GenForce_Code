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
        description='Train a strict canonical-20 downstream baseline with no adapter.',
        default_train_scales=[20.0],
        default_checkpoint_name='canonical20_no_adapter',
        default_wandb_project='STEA_Supp_Canonical20_NoAdapter',
        default_wandb_name='canonical20_no_adapter_train',
        require_stea=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_regressor_pipeline(
        args=args,
        use_stea=False,
        description='Training canonical-20 no-adapter baseline regressor',
        pipeline_kind='none',
    )


if __name__ == '__main__':
    main()
