#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold a marker image into a pure black/white result."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--threshold",
        type=int,
        default=220,
        help="Pixels strictly above this grayscale value become white; others become black",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold < 0 or args.threshold > 255:
        raise ValueError("--threshold must be in [0, 255]")

    img = Image.open(args.input).convert("L")
    binary = img.point(lambda x: 255 if x > int(args.threshold) else 0, mode="L")
    rgb = Image.merge("RGB", (binary, binary, binary))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".png":
        rgb.save(args.output, format="PNG")
    else:
        rgb.save(args.output, format="JPEG", quality=100, subsampling=0)
    print(f"Saved thresholded image: {args.output}")


if __name__ == "__main__":
    main()
