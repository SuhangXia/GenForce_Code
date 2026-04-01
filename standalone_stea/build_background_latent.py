#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from standalone_stea.utils import (
    FrozenViTPatchExtractor,
    build_default_image_transform,
    default_checkpoint_dir,
    format_seconds,
    load_rgb_image,
    map_to_tokens,
    resolve_path,
    tokens_to_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a canonical 20mm no-contact background latent map for STEA.')
    parser.add_argument('--image-paths', type=str, nargs='+', required=True)
    parser.add_argument('--output-path', type=str, default=str(default_checkpoint_dir('stea_background') / 'canonical_20mm_background.pt'))
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--also-save-npy', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_path = resolve_path(args.output_path, base_dir=PROJECT_ROOT)
    if output_path.suffix.lower() != '.pt':
        raise ValueError('--output-path must end with .pt so the canonical latent map is saved as a PyTorch checkpoint.')

    transform = build_default_image_transform()
    vit = FrozenViTPatchExtractor(model_name=args.model_name, pretrained=True).to(device)
    vit.eval()

    token_maps = []
    for image_path in args.image_paths:
        image = load_rgb_image(image_path, transform=transform).unsqueeze(0).to(device)
        with torch.no_grad():
            tokens = vit(image)
        token_map = tokens_to_map(tokens).squeeze(0).cpu()
        token_maps.append(token_map)
        print(f'Encoded background candidate: {resolve_path(image_path)}')

    stacked = torch.stack(token_maps, dim=0)
    background_latent_map = stacked.mean(dim=0, keepdim=True).contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'background_latent_map': background_latent_map,
            'image_paths': [str(resolve_path(p)) for p in args.image_paths],
            'model_name': args.model_name,
        },
        output_path,
    )
    print(f'Saved canonical background latent map to {output_path}')

    if args.also_save_npy:
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, background_latent_map.squeeze(0).numpy())
        print(f'Saved NumPy copy to {npy_path}')

    print(f'Background latent map shape: {tuple(background_latent_map.shape)}')


if __name__ == '__main__':
    main()
