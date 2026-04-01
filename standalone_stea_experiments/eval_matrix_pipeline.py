#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from standalone_stea.utils import CANONICAL_SCALE_MM, default_downstream_dataset_root, wandb
from standalone_stea_experiments.utils import DEFAULT_STEA_CKPT, evaluate_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a supplemental STEA experiment pipeline on one scale/split pair.')
    parser.add_argument('--dataset-root', type=str, default=str(default_downstream_dataset_root()))
    parser.add_argument('--regressor-ckpt', type=str, required=True)
    parser.add_argument('--pipeline-kind', type=str, choices=['no_adapter', 'with_stea'], required=True)
    parser.add_argument('--stea-ckpt', type=str, default=DEFAULT_STEA_CKPT)
    parser.add_argument('--canonical-scale-mm', type=float, default=CANONICAL_SCALE_MM)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', type=str, choices=['seen', 'unseen'], required=True)
    parser.add_argument('--scale-mm', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wandb-project', type=str, default='')
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    metrics = evaluate_pipeline(
        dataset_root=args.dataset_root,
        regressor_ckpt=args.regressor_ckpt,
        pipeline_kind=args.pipeline_kind,
        scale_mm=float(args.scale_mm),
        split=args.split,
        canonical_scale_mm=float(args.canonical_scale_mm),
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        model_name=args.model_name,
        stea_ckpt=args.stea_ckpt if args.pipeline_kind == 'with_stea' else None,
    )

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f'{args.pipeline_kind}_{float(args.scale_mm):g}_{args.split}',
            config=vars(args),
        )
        wandb.log(metrics)
        wandb.finish()

    scale_text = str(int(args.scale_mm)) if float(args.scale_mm).is_integer() else f'{args.scale_mm:g}'
    print(
        f'Eval scale={scale_text} split={args.split} use_stea={metrics["use_stea"]} '
        f'adapter_kind={metrics["adapter_kind"]} | mse_x={metrics["mse_x"]:.6f} '
        f'mse_y={metrics["mse_y"]:.6f} mse_depth={metrics["mse_depth"]:.6f} '
        f'mse_total={metrics["mse_total"]:.6f}'
    )


if __name__ == '__main__':
    main()
