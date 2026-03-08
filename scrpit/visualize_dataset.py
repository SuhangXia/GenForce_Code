#!/usr/bin/env python3
"""
Visualize dataset samples to sanity-check scale differences.

Usage:
    python scrpit/visualize_dataset.py
    python scrpit/visualize_dataset.py --scale-a 25 --scale-b 20   # 排除15mm，对比25 vs 20
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from train_adapter import (
    DEFAULT_DATASET,
    TEST_INDENTERS,
    load_manifest_and_split,
    MultiscaleTactileDataset,
)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def denormalize(img: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """Revert normalization. img: (C, H, W)."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img * std + mean


def coords_to_dims(coords: torch.Tensor) -> tuple[float, float]:
    """Derive physical width/height (mm) from coord grid. coords: (N, 2)."""
    x_min, x_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    y_min, y_max = coords[:, 1].min().item(), coords[:, 1].max().item()
    width = round(x_max - x_min, 1)
    height = round(y_max - y_min, 1)
    return width, height


def load_fixed_scale_pair(
    dataset_root: Path,
    episode_meta: dict[int, dict],
    episode_ids: list[int],
    scale_a: int,
    scale_b: int,
    transform,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load one episode with fixed scale pair (A, B). Returns (img_a, img_b, scale_a, scale_b)."""
    ep_id = rng.choice(episode_ids)
    meta = episode_meta[ep_id]
    scales = meta.get("scales", {})
    key_a = f"scale_{scale_a}mm"
    key_b = f"scale_{scale_b}mm"
    if key_a not in scales or key_b not in scales:
        valid = [k for k in scales if k.startswith("scale_")]
        raise ValueError(f"Episode {ep_id} missing {key_a} or {key_b}. Available: {valid}")

    path_a = dataset_root / scales[key_a]["image"]
    path_b = dataset_root / scales[key_b]["image"]
    img_a = transform(Image.open(path_a).convert("RGB"))
    img_b = transform(Image.open(path_b).convert("RGB"))
    return img_a, img_b, scale_a, scale_b


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default="dataset_sanity_check.png")
    parser.add_argument("--num-samples", "-n", type=int, default=1)
    parser.add_argument("--scale-a", type=int, default=None,
                        help="Left image scale (mm). With --scale-b, 排除15mm对比")
    parser.add_argument("--scale-b", type=int, default=None,
                        help="Right image scale (mm). e.g. --scale-a 25 --scale-b 20")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    train_ids, val_ids, test_ids, episode_meta = load_manifest_and_split(
        manifest_path,
        dataset_root=dataset_root,
        test_indenters=TEST_INDENTERS,
        train_ratio=0.85,
        seed=args.seed,
    )

    ids = val_ids if val_ids else train_ids
    if not ids:
        raise RuntimeError("No episodes in train or val split")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    use_fixed_scales = args.scale_a is not None and args.scale_b is not None
    if use_fixed_scales:
        scale_a, scale_b = args.scale_a, args.scale_b
        if scale_a == scale_b:
            raise ValueError("--scale-a and --scale-b must differ")

    out_path = Path(args.output)
    out_stem = out_path.stem
    out_dir = out_path.parent

    for i in range(args.num_samples):
        if use_fixed_scales:
            img_a, img_b, sa, sb = load_fixed_scale_pair(
                dataset_root, episode_meta, ids, scale_a, scale_b, transform, rng
            )
            w_a, h_a = sa, sa
            w_b, h_b = sb, sb
        else:
            dataset = MultiscaleTactileDataset(
                dataset_root=dataset_root,
                episode_ids=ids,
                episode_meta=episode_meta,
                scales_mm=[15, 18, 20, 22, 25],
                mode="val",
                augment=False,
                pairs_per_epoch=max(args.num_samples, 100),
                seed=args.seed + i,
            )
            img_a, coords_a, img_b, coords_b = dataset[i]
            w_a, h_a = coords_to_dims(coords_a)
            w_b, h_b = coords_to_dims(coords_b)

        img_a = denormalize(img_a, NORM_MEAN, NORM_STD)
        img_b = denormalize(img_b, NORM_MEAN, NORM_STD)
        img_a = torch.clamp(img_a, 0, 1)
        img_b = torch.clamp(img_b, 0, 1)

        img_a_np = img_a.permute(1, 2, 0).numpy()
        img_b_np = img_b.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_a_np)
        axes[0].set_title(f"Source: {w_a:.0f}x{h_a:.0f} mm (larger sensor)")
        axes[0].axis("off")
        axes[1].imshow(img_b_np)
        axes[1].set_title(f"Target: {w_b:.0f}x{h_b:.0f} mm (smaller sensor)")
        axes[1].axis("off")
        plt.tight_layout()

        if args.num_samples == 1:
            save_path = out_path
        else:
            save_path = out_dir / f"{out_stem}_{i+1:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
