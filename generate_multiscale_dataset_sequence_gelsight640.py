#!/usr/bin/env python3
"""Run multiscale sequence generation with a single 640x480 GelSight texture.

This wrapper is intentionally additive. It reuses the existing multiscale
sequence generator for physics, meshing, frame scheduling, and pair-index
creation, while overriding only:

- render resolution metadata and camera fitting to 640x480
- Blender render script output resolution to 640x480
- marker texture selection to a single GelSight reference texture

The GelSight texture is preprocessed into a square black-padded canvas before it
is passed into the existing square-gel UV pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image, ImageOps

import generate_multiscale_dataset as legacy
import generate_multiscale_dataset_sequence as seq


SCRIPT_DIR = Path(__file__).resolve().parent
TEXTURE_DIR = SCRIPT_DIR / "sim" / "marker" / "marker_pattern"
DEFAULT_GELSIGHT_TEXTURE_NAME = "Array_Gelsight"
DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 480
FORBIDDEN_SEQUENCE_ARGS = {
    "--marker-texture-names",
    "--marker-texture-count",
    "--marker-texture-seed",
}


def parse_wrapper_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper around generate_multiscale_dataset_sequence.py that keeps the "
            "existing multiscale/time-series logic, but renders a single "
            "GelSight texture at 640x480."
        ),
        epilog=(
            "All unknown arguments are forwarded to "
            "generate_multiscale_dataset_sequence.py unchanged."
        ),
    )
    parser.add_argument("--gelsight-texture-name", type=str, default=DEFAULT_GELSIGHT_TEXTURE_NAME)
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    return parser.parse_known_args(argv)


def validate_forwarded_args(forwarded: Sequence[str]) -> None:
    for item in forwarded:
        key = str(item).split("=", 1)[0]
        if key in FORBIDDEN_SEQUENCE_ARGS:
            raise ValueError(
                f"{key} is managed by generate_multiscale_dataset_sequence_gelsight640.py "
                "and must not be passed explicitly."
            )


def resolve_texture_path(texture_dir: Path, texture_name: str) -> Path:
    textures = legacy.discover_textures(texture_dir)
    by_stem = {path.stem: path for path in textures}
    by_name = {path.name: path for path in textures}

    if texture_name in by_stem:
        return by_stem[texture_name]
    if texture_name in by_name:
        return by_name[texture_name]

    raise FileNotFoundError(
        f"Could not find texture '{texture_name}' in {texture_dir}. "
        f"Available stems: {sorted(by_stem)}"
    )


def prepare_square_texture(source_path: Path, out_dir: Path) -> Path:
    with Image.open(source_path) as img:
        if img.mode not in {"L", "RGB"}:
            img = img.convert("RGB")
        else:
            img = img.copy()

    side = max(img.width, img.height)
    # Fill the square while preserving aspect ratio, then center-crop.
    canvas = ImageOps.fit(
        img,
        (side, side),
        method=Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
        centering=(0.5, 0.5),
    )

    out_path = out_dir / source_path.name
    save_kwargs = {"quality": 100}
    if out_path.suffix.lower() in {".jpg", ".jpeg"} and canvas.mode == "RGB":
        save_kwargs["subsampling"] = 0
    canvas.save(out_path, **save_kwargs)
    return out_path


def build_640_blender_script(
    original_builder,
    *,
    render_width: int,
    render_height: int,
):
    pattern = re.compile(r"DEFAULT_IMAGE_RES\s*=\s*\(\s*\d+\s*,\s*\d+\s*\)", re.MULTILINE)

    def _patched(*, render_device: str, render_gpu_backend: str) -> str:
        script = original_builder(
            render_device=render_device,
            render_gpu_backend=render_gpu_backend,
        )
        updated = pattern.sub(
            f"DEFAULT_IMAGE_RES = ({int(render_width)}, {int(render_height)})",
            script,
            count=1,
        )
        if updated == script:
            raise RuntimeError("Failed to patch Blender script render resolution.")
        return updated

    return _patched


@contextlib.contextmanager
def patched_sequence_runtime(
    *,
    texture_path: Path,
    render_width: int,
    render_height: int,
    forwarded_args: Sequence[str],
) -> Iterator[None]:
    original_resolution = legacy.DEFAULT_IMAGE_RES
    original_select_marker_textures = seq.select_marker_textures
    original_build_sequence_blender_script = seq.build_sequence_blender_script
    original_argv = sys.argv[:]

    def _select_marker_textures(*, texture_dir: Path, args: argparse.Namespace):
        return [texture_path], "explicit"

    try:
        legacy.DEFAULT_IMAGE_RES = (int(render_width), int(render_height))
        seq.select_marker_textures = _select_marker_textures
        seq.build_sequence_blender_script = build_640_blender_script(
            original_build_sequence_blender_script,
            render_width=render_width,
            render_height=render_height,
        )
        sys.argv = [str(Path(__file__).resolve()), *forwarded_args]
        yield
    finally:
        legacy.DEFAULT_IMAGE_RES = original_resolution
        seq.select_marker_textures = original_select_marker_textures
        seq.build_sequence_blender_script = original_build_sequence_blender_script
        sys.argv = original_argv


def main() -> None:
    wrapper_args, forwarded_args = parse_wrapper_args()
    validate_forwarded_args(forwarded_args)

    if wrapper_args.render_width <= 0 or wrapper_args.render_height <= 0:
        raise ValueError("--render-width and --render-height must be > 0")

    if not TEXTURE_DIR.exists():
        raise FileNotFoundError(f"Missing marker texture directory: {TEXTURE_DIR}")

    source_texture = resolve_texture_path(TEXTURE_DIR, wrapper_args.gelsight_texture_name)

    with tempfile.TemporaryDirectory(prefix="gelsight640_texture_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        square_texture = prepare_square_texture(source_texture, temp_dir)
        with patched_sequence_runtime(
            texture_path=square_texture,
            render_width=wrapper_args.render_width,
            render_height=wrapper_args.render_height,
            forwarded_args=forwarded_args,
        ):
            seq.main()


if __name__ == "__main__":
    main()
