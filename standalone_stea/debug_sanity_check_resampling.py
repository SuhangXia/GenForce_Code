#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from standalone_stea.models.stea_adapter import STEAAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Quick sanity check for STEA token-space resampling direction.')
    parser.add_argument('--output-dir', type=str, default='/tmp/stea_sanity')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


def build_toy_token_map(embed_dim: int = 768, grid_size: int = 14) -> torch.Tensor:
    base = torch.zeros((1, 1, grid_size, grid_size), dtype=torch.float32)
    base[:, :, 5:9, 5:9] = 1.0
    return base.repeat(1, embed_dim, 1, 1)


def compute_center_of_mass(map_2d: torch.Tensor) -> tuple[float, float]:
    weights = map_2d.clamp_min(0.0)
    mass = float(weights.sum().item())
    if mass <= 1e-8:
        return float('nan'), float('nan')
    ys, xs = torch.meshgrid(
        torch.arange(map_2d.shape[0], device=map_2d.device, dtype=weights.dtype),
        torch.arange(map_2d.shape[1], device=map_2d.device, dtype=weights.dtype),
        indexing='ij',
    )
    center_y = float((weights * ys).sum().item() / mass)
    center_x = float((weights * xs).sum().item() / mass)
    return center_y, center_x


def occupied_area(map_2d: torch.Tensor, threshold: float) -> int:
    return int((map_2d >= threshold).sum().item())


def save_debug_image(map_2d: torch.Tensor, path: Path) -> None:
    array = map_2d.detach().cpu().numpy()
    if array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode='L').resize((224, 224), resample=Image.NEAREST)
    image.save(path)


def run_case(
    adapter: STEAAdapter,
    toy_map: torch.Tensor,
    *,
    source_scale_mm: float,
    target_scale_mm: float,
    output_dir: Path,
    threshold: float,
) -> None:
    device = toy_map.device
    dtype = toy_map.dtype
    ratio = torch.tensor([[float(source_scale_mm) / max(float(target_scale_mm), 1e-6)]], device=device, dtype=dtype)
    grid = adapter._build_sampling_grid(ratio, batch_size=1, device=device, dtype=dtype)
    sampled = F.grid_sample(
        toy_map,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    valid_mask = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )

    input_map = toy_map[0, 0]
    sampled_map = sampled[0, 0]
    input_com = compute_center_of_mass(input_map)
    sampled_com = compute_center_of_mass(sampled_map)
    input_area = occupied_area(input_map, threshold)
    sampled_area = occupied_area(sampled_map, threshold)
    valid_ratio = float(valid_mask.to(dtype=torch.float32).mean().item())

    prefix = f'sanity_{int(source_scale_mm)}_to_{int(target_scale_mm)}'
    save_debug_image(input_map, output_dir / f'{prefix}_input.png')
    save_debug_image(sampled_map, output_dir / f'{prefix}_sampled.png')
    save_debug_image(valid_mask[0].to(dtype=torch.float32), output_dir / f'{prefix}_valid_mask.png')

    print(
        f'{prefix} | '
        f'input_com=({input_com[0]:.2f}, {input_com[1]:.2f}) | '
        f'sampled_com=({sampled_com[0]:.2f}, {sampled_com[1]:.2f}) | '
        f'input_area={input_area} | sampled_area={sampled_area} | '
        f'valid_ratio={valid_ratio:.4f}'
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    adapter = STEAAdapter(
        background_latent_map=torch.zeros((1, 768, 14, 14), dtype=torch.float32),
        use_boundary_smoothing=False,
    ).to(device)
    adapter.eval()

    toy_map = build_toy_token_map().to(device)
    for source_scale_mm in (15.0, 20.0, 25.0):
        run_case(
            adapter,
            toy_map,
            source_scale_mm=source_scale_mm,
            target_scale_mm=20.0,
            output_dir=output_dir,
            threshold=float(args.threshold),
        )

    print(f'Saved sanity-check images to {output_dir}')
    print('Expected reading: 15->20 should shrink the inflated blob, 20->20 should stay stable, 25->20 should expand.')


if __name__ == '__main__':
    main()
