#!/usr/bin/env python3
"""Sample tactile images and test left/right black-border widths.

Outputs preview images under the GenForce repo root so border coverage can be
inspected quickly without touching the dataset itself.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path("/home/suhang/datasets/usa_static_v1_large_run/pilot50")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_border_outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview black side-border removal on sampled tactile images.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample-count", type=int, default=8, help="How many dataset images to sample.")
    p.add_argument(
        "--border-px",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="Left/right border widths to test.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_dataset_images(dataset_root: Path) -> list[Path]:
    index_csv = dataset_root / "image_index.csv"
    if index_csv.exists():
        images: list[Path] = []
        with open(index_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("image_relpath", "")
                if not rel:
                    continue
                path = dataset_root / rel
                if path.exists():
                    images.append(path)
        if images:
            return images

    return sorted(dataset_root.glob("episode_*/scale_*mm/frame_*/marker_*.jpg"))


def apply_black_side_border(img: Image.Image, border_px: int) -> Image.Image:
    out = img.convert("RGB").copy()
    if border_px <= 0:
        return out

    width, height = out.size
    px = min(int(border_px), max(1, width // 2))
    black = (0, 0, 0)
    for x in range(px):
        for y in range(height):
            out.putpixel((x, y), black)
            out.putpixel((width - 1 - x, y), black)
    return out


def save_variants(image_paths: Iterable[Path], dataset_root: Path, output_dir: Path, border_values: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, src_path in enumerate(image_paths):
        rel = src_path.relative_to(dataset_root)
        stem = "__".join(rel.with_suffix("").parts)

        with Image.open(src_path) as img:
            for border_px in border_values:
                out = apply_black_side_border(img, border_px)
                out_name = f"{sample_idx:02d}__border_{border_px:02d}px__{stem}.jpg"
                out.save(output_dir / out_name, format="JPEG", quality=100)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.sample_count <= 0:
        raise ValueError("--sample-count must be > 0")
    if any(v < 0 for v in args.border_px):
        raise ValueError("--border-px values must be >= 0")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    all_images = list_dataset_images(dataset_root)
    if not all_images:
        raise FileNotFoundError(f"No tactile images found under {dataset_root}")

    rng = random.Random(args.seed)
    sample_count = min(int(args.sample_count), len(all_images))
    sampled = rng.sample(all_images, sample_count)

    save_variants(sampled, dataset_root, output_dir, [int(v) for v in args.border_px])

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Sampled images: {sample_count}")
    print(f"Border widths tested: {[int(v) for v in args.border_px]}")
    for path in sampled:
        print(f"  - {path.relative_to(dataset_root)}")


if __name__ == "__main__":
    main()
