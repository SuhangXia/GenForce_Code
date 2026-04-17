#!/usr/bin/env python3
"""Export GenForce sequence data into a uniforce_corl-style dataset tree.

This script is intentionally additive: it does not modify the existing
GenForce generators or uniforce_corl code. It runs the current sequence
generator to a temporary stage directory, then reshapes the result into:

<output-root>/<marker_name>/<date_tag>/<sensor_l>_<sensor_r>/
  collection_log.txt
  marker/<sensor>/<indenter>/<000000>.jpg
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

import rectgel_common as rectgel

SCRIPT_DIR = Path(__file__).resolve().parent
SEQUENCE_GENERATOR = SCRIPT_DIR / "generate_multiscale_dataset_sequence_rectgel.py"
BORDER_PREPROCESS_SCRIPT = SCRIPT_DIR / "scripts" / "preprocess_marker_borders.py"
PIL_RESAMPLING = getattr(Image, "Resampling", Image)

DEFAULT_OBJECTS = [
    "cylinder",
    "cylinder_si",
    "cylinder_sh",
    "moon",
    "sphere",
    "triangle",
    "hexagon",
    "prism",
]

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
RAW_IMAGE_EXT = ".jpg"
JPEG_SAVE_KWARGS = {"quality": 100, "subsampling": 0}
DEFAULT_STAGE_DIRNAME = "_stage"
DEFAULT_FIXED_BORDER_WIDTH_PX = 4


@dataclass(frozen=True)
class LogicalPair:
    indenter: str
    episode_id: int
    global_seq_index: int
    phase_name: str
    left_frame_dir: Path
    right_frame_dir: Path
    left_depth_mm: float
    right_depth_mm: float
    shared_markers: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a uniforce_corl-style synthetic dataset from GenForce.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stage-root", type=Path, default=None)
    parser.add_argument("--sensor-l-name", type=str, default="digit")
    parser.add_argument("--sensor-r-name", type=str, default="gelsight")
    parser.add_argument("--gel-width-l-mm", type=float, default=15.0)
    parser.add_argument("--gel-width-r-mm", type=float, default=17.0)
    parser.add_argument("--date-tag", type=str, default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--objects", nargs="*", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--episodes-per-indenter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--genforce-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def scale_suffix(scale_mm: float) -> str:
    rounded = round(float(scale_mm), 4)
    if abs(rounded - int(round(rounded))) <= 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def format_scale_key(scale_mm: float) -> str:
    value = 0.0 if abs(float(scale_mm)) < 5e-8 else float(scale_mm)
    return f"scale_{value:.4f}".replace("-", "m").replace(".", "p") + "mm"


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    folder = (
        f"uniforce_synth_"
        f"{rectgel.make_sensor_size_slug(args.sensor_l_name, args.gel_width_l_mm)}_"
        f"{rectgel.make_sensor_size_slug(args.sensor_r_name, args.gel_width_r_mm)}"
    )
    return (Path("/home/suhang/datasets") / folder).resolve()


def resolve_stage_root(args: argparse.Namespace, output_root: Path) -> Path:
    if args.stage_root is not None:
        return args.stage_root.expanduser().resolve()
    return (output_root / DEFAULT_STAGE_DIRNAME).resolve()


def ensure_empty_dir(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"{label} exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"{label} must not already contain data: {path}")
    path.mkdir(parents=True, exist_ok=True)


def validate_output_and_stage_roots(output_root: Path, stage_root: Path) -> None:
    if output_root == stage_root:
        raise ValueError("--stage-root must not be the same as --output-root")
    if stage_root in output_root.parents:
        raise ValueError("--stage-root must not be a parent of --output-root")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_passthrough_args(genforce_args: Sequence[str]) -> None:
    forbidden = {
        "--dataset-root",
        "--dataset-profile",
        "--repo-root",
        "--gel-widths-mm",
        "--gel-width-min-mm",
        "--gel-width-max-mm",
        "--scale-mode",
        "--scales-per-episode",
        "--reference-gel-width-mm",
        "--objects",
        "--episodes-per-indenter",
        "--seed",
        "--pair-marker-policy",
        "--include-identity-pairs",
    }
    for item in genforce_args:
        key = str(item).split("=", 1)[0]
        if key in forbidden:
            raise ValueError(f"{key} is managed by generate_uniforce.py and cannot be passed via --genforce-args")


def build_generator_command(args: argparse.Namespace, stage_root: Path) -> list[str]:
    validate_passthrough_args(args.genforce_args)
    cmd = [
        sys.executable,
        str(SEQUENCE_GENERATOR),
        "--repo-root",
        str(SCRIPT_DIR),
        "--dataset-root",
        str(stage_root),
        "--dataset-profile",
        "A",
        "--gel-widths-mm",
        str(args.gel_width_l_mm),
        str(args.gel_width_r_mm),
        "--objects",
        *args.objects,
        "--episodes-per-indenter",
        str(args.episodes_per_indenter),
        "--seed",
        str(args.seed),
        "--pair-marker-policy",
        "same_texture_only",
        *args.genforce_args,
    ]
    return cmd


def run_stage_generation(args: argparse.Namespace, stage_root: Path) -> None:
    if not SEQUENCE_GENERATOR.exists():
        raise FileNotFoundError(f"Missing sequence generator: {SEQUENCE_GENERATOR}")
    cmd = build_generator_command(args, stage_root)
    print("Running stage generator:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def run_fixed_border_postprocess(marker_root: Path) -> None:
    if not BORDER_PREPROCESS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing border preprocessing script: {BORDER_PREPROCESS_SCRIPT}")
    cmd = [
        sys.executable,
        str(BORDER_PREPROCESS_SCRIPT),
        "--input-dir",
        str(marker_root),
        "--output-dir",
        str(marker_root),
        "--strategy",
        "fixed",
        "--fixed-border-width",
        str(DEFAULT_FIXED_BORDER_WIDTH_PX),
        "--mode",
        "full",
        "--overwrite",
    ]
    print("Running fixed-border postprocess:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def extract_marker_stems(manifest: Mapping[str, Any], logical_pairs: Sequence[LogicalPair]) -> list[str]:
    selected = [str(name) for name in manifest.get("marker_textures_selected", []) if str(name)]
    if selected:
        return sorted(selected)

    stems: set[str] = set()
    for pair in logical_pairs:
        for marker_file in pair.shared_markers:
            stem = marker_file
            if stem.startswith("marker_"):
                stem = stem[len("marker_") :]
            if stem.endswith(".jpg"):
                stem = stem[:-4]
            if stem:
                stems.add(stem)
    if not stems:
        raise RuntimeError("No marker textures found in stage dataset")
    return sorted(stems)


def marker_filename(marker_stem: str) -> str:
    return f"marker_{marker_stem}{RAW_IMAGE_EXT}"


def find_scale_metadata(
    episode_dir: Path,
    episode_meta: Mapping[str, Any],
    target_gel_width_mm: float,
) -> dict[str, Any]:
    scales = episode_meta.get("scales", {})
    if not isinstance(scales, Mapping):
        raise ValueError(f"Unexpected scales payload in {episode_dir / 'metadata.json'}")

    exact_key = format_scale_key(target_gel_width_mm)
    if exact_key in scales:
        seq_rel = scales[exact_key].get("sequence_metadata")
        if not seq_rel:
            raise FileNotFoundError(f"Missing sequence_metadata for scale {exact_key} in {episode_dir}")
        return load_json(episode_dir / str(seq_rel))

    for scale_summary in scales.values():
        simulated = float(scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm", -1)))
        if abs(simulated - float(target_scale_mm)) <= 5e-4:
            seq_rel = scale_summary.get("sequence_metadata")
            if not seq_rel:
                raise FileNotFoundError(f"Missing sequence_metadata for gel width {target_gel_width_mm} in {episode_dir}")
            return load_json(episode_dir / str(seq_rel))

    raise KeyError(f"Could not find gel width {target_gel_width_mm}mm in {episode_dir}")


def build_logical_pairs(stage_root: Path, gel_width_l_mm: float, gel_width_r_mm: float) -> tuple[list[LogicalPair], dict[str, Any]]:
    manifest = load_json(stage_root / "manifest.json")
    logical_pairs: list[LogicalPair] = []

    for episode_entry in manifest.get("episodes", []):
        episode_rel = str(episode_entry.get("path", ""))
        if not episode_rel:
            continue
        episode_dir = stage_root / episode_rel
        episode_meta = load_json(episode_dir / "metadata.json")
        left_scale_meta = find_scale_metadata(episode_dir, episode_meta, gel_width_l_mm)
        right_scale_meta = find_scale_metadata(episode_dir, episode_meta, gel_width_r_mm)

        left_frames = {
            int(frame["global_seq_index"]): frame
            for frame in left_scale_meta.get("frames", [])
        }
        right_frames = {
            int(frame["global_seq_index"]): frame
            for frame in right_scale_meta.get("frames", [])
        }
        shared_indices = sorted(set(left_frames) & set(right_frames))
        indenter = str(episode_meta["indenter"])
        episode_id = int(episode_meta["episode_id"])

        for seq_idx in shared_indices:
            left_frame = left_frames[seq_idx]
            right_frame = right_frames[seq_idx]
            left_phase = str(left_frame.get("phase_name", ""))
            right_phase = str(right_frame.get("phase_name", ""))
            if left_phase != right_phase:
                raise ValueError(
                    f"Phase mismatch at episode={episode_id} seq={seq_idx}: left={left_phase} right={right_phase}"
                )
            left_frame_dir = episode_dir / str(left_scale_meta["scale_key"]) / str(left_frame["frame_name"])
            right_frame_dir = episode_dir / str(right_scale_meta["scale_key"]) / str(right_frame["frame_name"])
            shared_markers = tuple(
                sorted(
                    set(str(name) for name in left_frame.get("rendered_markers", []))
                    & set(str(name) for name in right_frame.get("rendered_markers", []))
                )
            )
            if not shared_markers:
                raise RuntimeError(
                    f"No shared markers for episode={episode_id} seq={seq_idx} "
                    f"between {left_scale_meta['scale_key']} and {right_scale_meta['scale_key']}"
                )
            logical_pairs.append(
                LogicalPair(
                    indenter=indenter,
                    episode_id=episode_id,
                    global_seq_index=seq_idx,
                    phase_name=left_phase,
                    left_frame_dir=left_frame_dir,
                    right_frame_dir=right_frame_dir,
                    left_depth_mm=float(left_frame.get("frame_actual_max_down_mm", 0.0)),
                    right_depth_mm=float(right_frame.get("frame_actual_max_down_mm", 0.0)),
                    shared_markers=shared_markers,
                )
            )

    logical_pairs.sort(key=lambda item: (item.indenter, item.episode_id, item.global_seq_index))
    return logical_pairs, manifest


def prepare_marker_image(source_path: Path) -> Image.Image:
    with Image.open(source_path) as img:
        marker = img.convert("L")
        if marker.size == (TARGET_WIDTH, TARGET_HEIGHT):
            return marker.copy()
        return marker.resize((TARGET_WIDTH, TARGET_HEIGHT), PIL_RESAMPLING.NEAREST)


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, **JPEG_SAVE_KWARGS)


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def append_log_line(log_path: Path, indenter: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{timestamp_now()} - [indenter: {indenter}] - {message}\n")


def export_marker_dataset(
    *,
    output_root: Path,
    marker_stem: str,
    date_tag: str,
    sensor_l_name: str,
    sensor_r_name: str,
    gel_width_l_mm: float,
    gel_width_r_mm: float,
    logical_pairs: Sequence[LogicalPair],
) -> Path:
    dataset_root = output_root / marker_stem / date_tag / f"{sensor_l_name}_{sensor_r_name}"
    dataset_root.mkdir(parents=True, exist_ok=True)
    log_path = dataset_root / "collection_log.txt"

    if log_path.exists():
        log_path.unlink()

    append_log_line(
        log_path,
        "ALL",
        (
            f"Synthetic export started. Base dir: {dataset_root}. "
            f"Marker: {marker_stem}. Sensors: "
            f"{sensor_l_name}({gel_width_l_mm:g}x{rectgel.gel_dims_from_width_mm(gel_width_l_mm).height_mm:g}mm) -> "
            f"{sensor_r_name}({gel_width_r_mm:g}x{rectgel.gel_dims_from_width_mm(gel_width_r_mm).height_mm:g}mm)"
        ),
    )

    pairs_by_indenter: dict[str, list[LogicalPair]] = defaultdict(list)
    for pair in logical_pairs:
        expected_marker = marker_filename(marker_stem)
        if expected_marker not in pair.shared_markers:
            raise RuntimeError(
                f"Marker {expected_marker} is missing for indenter={pair.indenter} "
                f"episode={pair.episode_id} seq={pair.global_seq_index}"
            )
        pairs_by_indenter[pair.indenter].append(pair)

    total_exported = 0
    for indenter in sorted(pairs_by_indenter):
        pairs = pairs_by_indenter[indenter]
        append_log_line(
            log_path,
            indenter,
            f"Synthetic export started. Marker: {marker_stem}. Planned frames: {len(pairs)}",
        )

        marker_left_dir = dataset_root / "marker" / sensor_l_name / indenter
        marker_right_dir = dataset_root / "marker" / sensor_r_name / indenter
        for path in (marker_left_dir, marker_right_dir):
            path.mkdir(parents=True, exist_ok=True)

        for index, pair in enumerate(pairs):
            filename = f"{index:06d}{RAW_IMAGE_EXT}"
            marker_file = marker_filename(marker_stem)
            left_source = pair.left_frame_dir / marker_file
            right_source = pair.right_frame_dir / marker_file

            if not left_source.exists() or not right_source.exists():
                raise FileNotFoundError(
                    f"Missing rendered source for marker={marker_stem}: left={left_source.exists()} right={right_source.exists()}"
                )

            save_image(prepare_marker_image(left_source), marker_left_dir / filename)
            save_image(prepare_marker_image(right_source), marker_right_dir / filename)
            total_exported += 1

        append_log_line(
            log_path,
            indenter,
            f"Synthetic export completed. Marker: {marker_stem}. Final frames: {len(pairs)}",
        )

    append_log_line(
        log_path,
        "ALL",
        f"Synthetic export completed. Marker: {marker_stem}. Total paired frames: {total_exported}",
    )
    return dataset_root


def summarize_pairs(logical_pairs: Sequence[LogicalPair]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for pair in logical_pairs:
        counts[pair.indenter] += 1
    items = [f"{name}:{counts[name]}" for name in sorted(counts)]
    return ", ".join(items)


def verify_export_layout(
    output_root: Path,
    marker_stems: Sequence[str],
    date_tag: str,
    sensor_l_name: str,
    sensor_r_name: str,
) -> None:
    for marker_stem in marker_stems:
        dataset_root = output_root / marker_stem / date_tag / f"{sensor_l_name}_{sensor_r_name}"
        if not (dataset_root / "collection_log.txt").exists():
            raise FileNotFoundError(f"Missing collection_log.txt in {dataset_root}")


def cleanup_stage_root(stage_root: Path) -> None:
    if stage_root.exists():
        shutil.rmtree(stage_root)


def main() -> None:
    args = parse_args()
    if args.gel_width_l_mm <= 0 or args.gel_width_r_mm <= 0:
        raise ValueError("Gel width values must be > 0")
    if args.episodes_per_indenter <= 0:
        raise ValueError("--episodes-per-indenter must be > 0")
    if not args.objects:
        raise ValueError("--objects must not be empty")

    output_root = resolve_output_root(args)
    stage_root = resolve_stage_root(args, output_root)
    validate_output_and_stage_roots(output_root, stage_root)

    ensure_empty_dir(output_root, "Output root")
    ensure_empty_dir(stage_root, "Stage root")

    try:
        run_stage_generation(args, stage_root)
        logical_pairs, manifest = build_logical_pairs(stage_root, args.gel_width_l_mm, args.gel_width_r_mm)
        if not logical_pairs:
            raise RuntimeError("No paired frames were found in the generated stage dataset")

        marker_stems = extract_marker_stems(manifest, logical_pairs)
        print(f"Resolved {len(logical_pairs)} logical pairs across indenters: {summarize_pairs(logical_pairs)}")
        print(f"Exporting {len(marker_stems)} marker datasets: {', '.join(marker_stems)}")

        for marker_stem in marker_stems:
            dataset_root = export_marker_dataset(
                output_root=output_root,
                marker_stem=marker_stem,
                date_tag=args.date_tag,
                sensor_l_name=args.sensor_l_name,
                sensor_r_name=args.sensor_r_name,
                gel_width_l_mm=args.gel_width_l_mm,
                gel_width_r_mm=args.gel_width_r_mm,
                logical_pairs=logical_pairs,
            )
            run_fixed_border_postprocess(dataset_root / "marker")

        verify_export_layout(
            output_root=output_root,
            marker_stems=marker_stems,
            date_tag=args.date_tag,
            sensor_l_name=args.sensor_l_name,
            sensor_r_name=args.sensor_r_name,
        )
        print(f"Export complete: {output_root}")
    finally:
        if not args.keep_stage:
            cleanup_stage_root(stage_root)


if __name__ == "__main__":
    main()
