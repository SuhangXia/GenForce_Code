#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INDENTERS = ["cone", "cylinder", "line", "prism", "random"]
DEFAULT_EPISODE_INDEX = 33
DEFAULT_FRAME_COUNT = 18
DEFAULT_COLS = 5
DEFAULT_TILE_WIDTH = 256
DEFAULT_TILE_HEIGHT = 192
LABEL_HEIGHT = 28
HEADER_HEIGHT = 48
PADDING = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export marker preview grids and GIFs for selected indenters from the "
            "same-scale rect-safe dataset."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--marker-source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--indenters",
        nargs="+",
        default=list(DEFAULT_INDENTERS),
        help="Indenter names, e.g. cone cylinder line prism random",
    )
    p.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    p.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    p.add_argument("--cols", type=int, default=DEFAULT_COLS)
    p.add_argument("--tile-width", type=int, default=DEFAULT_TILE_WIDTH)
    p.add_argument("--tile-height", type=int, default=DEFAULT_TILE_HEIGHT)
    p.add_argument(
        "--gif-duration-ms",
        type=int,
        default=180,
        help="Frame duration for the exported GIF in milliseconds.",
    )
    p.add_argument(
        "--episode-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of episode indices to concatenate into one longer GIF.",
    )
    p.add_argument(
        "--gif-only",
        action="store_true",
        help="Export only the final GIF files and skip writing per-frame PNG grids.",
    )
    return p.parse_args()


def list_marker_names(marker_source_dir: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [path.name for path in marker_source_dir.iterdir() if path.suffix.lower() in exts],
        key=lambda name: name.lower(),
    )


def ensure_single_scale_dir(episode_dir: Path) -> Path:
    scale_dirs = sorted([path for path in episode_dir.iterdir() if path.is_dir() and path.name.startswith("scale_")])
    if len(scale_dirs) != 1:
        raise RuntimeError(f"Expected exactly one scale dir under {episode_dir}, found {len(scale_dirs)}")
    return scale_dirs[0]


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def normalize_episode_indices(args: argparse.Namespace) -> List[int]:
    if args.episode_indices:
        return [int(x) for x in args.episode_indices]
    return [int(args.episode_index)]


def describe_episode_position(dataset_root: Path, indenter: str, episode_index: int) -> str:
    npz_root = (
        dataset_root
        / "_work"
        / f"run_{indenter}"
        / "_intermediates"
        / f"episode_{episode_index:06d}"
    )
    candidates = sorted(npz_root.rglob("*.npz"))
    if not candidates:
        return f"episode_{episode_index:06d}"
    return candidates[0].stem


def make_grid_frame(
    *,
    title: str,
    frame_label: str,
    image_paths: Iterable[Path],
    marker_labels: Iterable[str],
    cols: int,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    image_paths = list(image_paths)
    marker_labels = list(marker_labels)
    rows = (len(image_paths) + cols - 1) // cols
    canvas_width = cols * tile_width + (cols + 1) * PADDING
    canvas_height = HEADER_HEIGHT + rows * (tile_height + LABEL_HEIGHT) + (rows + 1) * PADDING
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(18)
    draw.text((PADDING, 10), f"{title} | {frame_label}", fill=(20, 20, 20), font=title_font)

    for idx, (image_path, marker_label) in enumerate(zip(image_paths, marker_labels)):
        row = idx // cols
        col = idx % cols
        x = PADDING + col * (tile_width + PADDING)
        y = HEADER_HEIGHT + PADDING + row * (tile_height + LABEL_HEIGHT)

        img = Image.open(image_path).convert("RGB")
        img = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(img, (x, y))
        draw.rectangle(
            [x - 1, y - 1, x + tile_width, y + tile_height],
            outline=(190, 190, 190),
            width=1,
        )
        draw.text((x + 6, y + tile_height + 4), marker_label, fill=(30, 30, 30), font=label_font)

    return canvas


def export_for_indenter(
    *,
    dataset_root: Path,
    output_dir: Path,
    indenter: str,
    episode_indices: List[int],
    frame_count: int,
    marker_names: List[str],
    cols: int,
    tile_width: int,
    tile_height: int,
    gif_duration_ms: int,
    gif_only: bool,
) -> List[str]:
    if not episode_indices:
        raise ValueError("episode_indices must not be empty")

    indenter_output_dir = output_dir / indenter
    indenter_output_dir.mkdir(parents=True, exist_ok=True)
    frames_output_dir = indenter_output_dir / "frames"
    if not gif_only:
        frames_output_dir.mkdir(parents=True, exist_ok=True)

    rendered_frames: List[Image.Image] = []
    episode_descriptions: List[str] = []
    global_frame_idx = 0
    for episode_index in episode_indices:
        episode_dir = dataset_root / "_work" / f"run_{indenter}" / f"episode_{episode_index:06d}"
        if not episode_dir.exists():
            raise FileNotFoundError(f"Missing episode dir: {episode_dir}")
        scale_dir = ensure_single_scale_dir(episode_dir)
        position_desc = describe_episode_position(dataset_root, indenter, episode_index)
        episode_descriptions.append(f"episode_{episode_index:06d}: {position_desc}")

        for frame_idx in range(frame_count):
            frame_dir = scale_dir / f"frame_{frame_idx:06d}"
            image_paths = [frame_dir / f"marker_{marker_name}" for marker_name in marker_names]
            missing = [str(path) for path in image_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing marker images for {indenter} frame {frame_idx:06d}: {missing[:3]}"
                )
            frame_img = make_grid_frame(
                title=f"{indenter} | episode_{episode_index:06d}",
                frame_label=f"{position_desc} | frame_{frame_idx:06d}",
                image_paths=image_paths,
                marker_labels=[Path(name).stem for name in marker_names],
                cols=cols,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            if not gif_only:
                frame_png = (
                    frames_output_dir
                    / f"grid_{global_frame_idx:06d}_ep_{episode_index:06d}_frame_{frame_idx:06d}.png"
                )
                frame_img.save(frame_png)
            rendered_frames.append(frame_img)
            global_frame_idx += 1

    gif_label = (
        f"{indenter}_episode_{episode_indices[0]:06d}_all_markers.gif"
        if len(episode_indices) == 1
        else f"{indenter}_episodes_{episode_indices[0]:06d}_to_{episode_indices[-1]:06d}_all_markers.gif"
    )
    gif_path = indenter_output_dir / gif_label
    rendered_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=gif_duration_ms,
        loop=0,
        optimize=False,
    )
    return episode_descriptions


def write_readme(
    output_dir: Path,
    dataset_root: Path,
    episode_indices: List[int],
    indenters: List[str],
    marker_names: List[str],
    episode_summary: Dict[str, List[str]],
    gif_only: bool,
) -> None:
    lines = [
        f"dataset_root: {dataset_root}",
        f"episode_indices: {', '.join(f'{idx:06d}' for idx in episode_indices)}",
        f"indenters: {', '.join(indenters)}",
        f"marker_count: {len(marker_names)}",
        f"markers: {', '.join(Path(name).stem for name in marker_names)}",
        f"gif_only: {int(gif_only)}",
        "",
        "Each indenter directory contains:",
        "- *.gif: animated GIF over consecutive frames",
        "",
    ]
    if not gif_only:
        lines.insert(-2, "- frames/: per-frame grid PNGs")
    for indenter in indenters:
        lines.append(f"{indenter}:")
        lines.extend(episode_summary.get(indenter, []))
        lines.append("")
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    episode_indices = normalize_episode_indices(args)
    marker_names = list_marker_names(args.marker_source_dir)
    if not marker_names:
        raise RuntimeError(f"No marker textures found in {args.marker_source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_summary: Dict[str, List[str]] = {}
    for indenter in args.indenters:
        episode_summary[indenter] = export_for_indenter(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            indenter=indenter,
            episode_indices=episode_indices,
            frame_count=int(args.frame_count),
            marker_names=marker_names,
            cols=int(args.cols),
            tile_width=int(args.tile_width),
            tile_height=int(args.tile_height),
            gif_duration_ms=int(args.gif_duration_ms),
            gif_only=bool(args.gif_only),
        )

    write_readme(
        args.output_dir,
        args.dataset_root,
        episode_indices,
        list(args.indenters),
        marker_names,
        episode_summary,
        bool(args.gif_only),
    )


if __name__ == "__main__":
    main()
