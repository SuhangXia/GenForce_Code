#!/usr/bin/env python3
"""Generate same-scale 4:3 UniForce data with an event-centric layout.

This script is additive-only. It does not modify existing generators or
uniforce_corl code. It builds a same-scale rectangular-gel dataset with:

- short edge fixed by `--short-edge-mm` and width inferred as 4:3
- per-indenter safe-depth caps with an extra safety margin
- 20 marker textures total
  - 12 top-level 1:1 marker textures, excluding Array_Gelsight
  - 8 pre-existing 4:3 textures under sim/marker/marker_pattern/4_3
- event-centric storage:
    episode_xxxxxx/scale_xxxxxmm/frame_xxxxxx/marker_*.jpg
- compact event_index.jsonl for runtime marker-pair sampling
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_multiscale_dataset as legacy
import generate_multiscale_dataset_sequence_rectgel as rectseq
import rectgel_common as rectgel


MARKER_PATTERN_DIR = SCRIPT_DIR / "sim" / "marker" / "marker_pattern"
MARKER_PATTERN_4_3_DIR = MARKER_PATTERN_DIR / "4_3"
PIL_RESAMPLING = getattr(Image, "Resampling", Image)

DEFAULT_BAD_DEPTH_MM = {
    "dots": 1.510032,
    "moon": 1.711395,
    "hemisphere": 1.790733,
    "cylinder_si": 1.810065,
    "cylinder_sh": 1.910038,
}
DEFAULT_OBJECTS = [
    "cone",
    "cylinder",
    "cylinder_sh",
    "cylinder_si",
    "dotin",
    "dots",
    "hemisphere",
    "hexagon",
    "line",
    "moon",
    "pacman",
    "prism",
    "random",
    "sphere",
    "sphere_s",
    "torus",
    "triangle",
    "wave",
]
DEFAULT_SAFETY_MARGIN_MM = 0.1
DEFAULT_DEPTH_MIN_MM = 0.4
DEFAULT_SHORT_EDGE_MM = 16.0
DEFAULT_KEEP_WORK = True
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TOP_LEVEL_MARKER_EXCLUDES = {"Array_Gelsight"}
FOUR_THREE_MARKER_EXCLUDES: set[str] = set()
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_WORK_DIRNAME = "_work"
DATASET_VARIANT = "uniforce_same_scale_rect_safe_depth_event_v1"


@dataclass(frozen=True)
class SafeDepthConfig:
    indenter: str
    requested_depth_mm: float
    safe_limit_mm: float
    safe_cap_mm: float
    effective_depth_mm: float
    depth_was_capped: bool
    threshold_configured: bool


@dataclass(frozen=True)
class GeneratorRun:
    indenter: str
    run_root: Path
    safe_depth: SafeDepthConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate same-scale 4:3 UniForce data with event-centric indexing."
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--short-edge-mm", type=float, default=DEFAULT_SHORT_EDGE_MM)
    parser.add_argument("--objects", nargs="*", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--episodes-per-indenter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN_MM)
    parser.add_argument("--requested-depth-max-mm", type=float, default=float(legacy.DEFAULT_DEPTH_RANGE[1]))
    parser.add_argument("--safety-margin-mm", type=float, default=DEFAULT_SAFETY_MARGIN_MM)
    parser.add_argument("--keep-work", action=argparse.BooleanOptionalAction, default=DEFAULT_KEEP_WORK)
    parser.add_argument("--genforce-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    short_suffix = rectgel.format_dim_suffix(float(args.short_edge_mm))
    return (Path("/home/suhang/datasets") / f"uniforce_same_scale_rect_safe_depth_short{short_suffix}").resolve()


def resolve_work_root(args: argparse.Namespace, output_root: Path) -> Path:
    if args.work_root is not None:
        return args.work_root.expanduser().resolve()
    return (output_root / DEFAULT_WORK_DIRNAME).resolve()


def ensure_empty_dir(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"{label} exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"{label} must not already contain data: {path}")
    path.mkdir(parents=True, exist_ok=True)


def validate_output_and_work_roots(output_root: Path, work_root: Path) -> None:
    if output_root == work_root:
        raise ValueError("--work-root must not be the same as --output-root")
    if work_root in output_root.parents:
        raise ValueError("--work-root must not be a parent of --output-root")


def discover_square_markers() -> list[Path]:
    if not MARKER_PATTERN_DIR.exists():
        raise FileNotFoundError(f"Missing marker directory: {MARKER_PATTERN_DIR}")
    paths = sorted(
        path
        for path in MARKER_PATTERN_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem not in TOP_LEVEL_MARKER_EXCLUDES
    )
    if len(paths) != 12:
        raise RuntimeError(f"Expected 12 top-level square marker textures, found {len(paths)} in {MARKER_PATTERN_DIR}")
    return paths


def discover_four_three_markers() -> list[Path]:
    if not MARKER_PATTERN_4_3_DIR.exists():
        raise FileNotFoundError(f"Missing 4:3 marker directory: {MARKER_PATTERN_4_3_DIR}")
    paths = sorted(
        path
        for path in MARKER_PATTERN_4_3_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem not in FOUR_THREE_MARKER_EXCLUDES
    )
    if len(paths) != 8:
        raise RuntimeError(f"Expected 8 4:3 marker textures, found {len(paths)} in {MARKER_PATTERN_4_3_DIR}")
    return paths


def cover_resize_to_four_three(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        fitted = ImageOps.fit(
            rgb,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=PIL_RESAMPLING.LANCZOS,
            centering=(0.5, 0.5),
        )
        fitted.save(dst, quality=100, subsampling=0)


def stage_marker_textures(staging_dir: Path) -> list[Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    square_markers = discover_square_markers()
    four_three_markers = discover_four_three_markers()
    out_paths: list[Path] = []
    seen_stems: set[str] = set()

    for src in square_markers:
        if src.stem in seen_stems:
            raise RuntimeError(f"Duplicate marker stem during staging: {src.stem}")
        dst = staging_dir / f"{src.stem}.jpg"
        cover_resize_to_four_three(src, dst)
        out_paths.append(dst)
        seen_stems.add(src.stem)

    for src in four_three_markers:
        if src.stem in seen_stems:
            raise RuntimeError(f"Duplicate marker stem during staging: {src.stem}")
        dst = staging_dir / src.name
        shutil.copy2(src, dst)
        out_paths.append(dst)
        seen_stems.add(src.stem)

    out_paths = sorted(out_paths, key=lambda path: path.stem)
    if len(out_paths) != 20:
        raise RuntimeError(f"Expected 20 staged marker textures, found {len(out_paths)}")
    return out_paths


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload)}")
    return payload


def compute_safe_depth(
    *,
    indenter: str,
    requested_depth_mm: float,
    safety_margin_mm: float,
    depth_min_mm: float,
) -> SafeDepthConfig:
    threshold_configured = indenter in DEFAULT_BAD_DEPTH_MM
    if threshold_configured:
        safe_limit_mm = float(DEFAULT_BAD_DEPTH_MM[indenter]) - float(safety_margin_mm)
        if safe_limit_mm <= 0:
            raise ValueError(f"Computed non-positive safe limit for {indenter}: {safe_limit_mm}")
        safe_cap_mm = math.floor(safe_limit_mm * 10.0 + 1e-9) / 10.0
        if safe_cap_mm < float(depth_min_mm):
            raise ValueError(
                f"Quantized safe depth {safe_cap_mm:.4f}mm for {indenter} is below depth-min {depth_min_mm:.4f}mm"
            )
        effective_depth_mm = min(float(requested_depth_mm), safe_cap_mm)
        depth_was_capped = float(requested_depth_mm) > safe_cap_mm + 1e-9
    else:
        safe_limit_mm = float(requested_depth_mm)
        safe_cap_mm = float(requested_depth_mm)
        effective_depth_mm = float(requested_depth_mm)
        depth_was_capped = False
    return SafeDepthConfig(
        indenter=indenter,
        requested_depth_mm=float(requested_depth_mm),
        safe_limit_mm=float(safe_limit_mm),
        safe_cap_mm=float(safe_cap_mm),
        effective_depth_mm=float(effective_depth_mm),
        depth_was_capped=bool(depth_was_capped),
        threshold_configured=bool(threshold_configured),
    )


def validate_passthrough_args(genforce_args: Sequence[str]) -> None:
    forbidden = {
        "--repo-root",
        "--dataset-root",
        "--dataset-profile",
        "--gel-widths-mm",
        "--gel-width-min-mm",
        "--gel-width-max-mm",
        "--scale-mode",
        "--objects",
        "--episodes-per-indenter",
        "--seed",
        "--depth-min",
        "--depth-max",
        "--marker-texture-names",
        "--marker-texture-count",
        "--pair-marker-policy",
        "--include-identity-pairs",
        "--render-width",
        "--render-height",
    }
    for item in genforce_args:
        key = str(item).split("=", 1)[0]
        if key in forbidden:
            raise ValueError(
                f"{key} is managed by generate_uniforce_same_scale_rect_safe_depth.py and cannot be passed via --genforce-args"
            )


def build_rectgel_argv(
    *,
    run_root: Path,
    indenter: str,
    marker_stems: Sequence[str],
    seed: int,
    episodes_per_indenter: int,
    gel_width_mm: float,
    depth_min_mm: float,
    depth_max_mm: float,
    passthrough: Sequence[str],
) -> list[str]:
    validate_passthrough_args(passthrough)
    return [
        str(rectseq.__file__),
        "--repo-root",
        str(SCRIPT_DIR),
        "--dataset-root",
        str(run_root),
        "--dataset-profile",
        "A",
        "--gel-widths-mm",
        f"{gel_width_mm:.4f}",
        "--reference-gel-width-mm",
        f"{gel_width_mm:.4f}",
        "--objects",
        indenter,
        "--episodes-per-indenter",
        str(int(episodes_per_indenter)),
        "--seed",
        str(seed),
        "--depth-min",
        f"{depth_min_mm:.4f}",
        "--depth-max",
        f"{depth_max_mm:.4f}",
        "--pair-marker-policy",
        "same_texture_only",
        "--marker-texture-names",
        *marker_stems,
        "--render-width",
        str(TARGET_WIDTH),
        "--render-height",
        str(TARGET_HEIGHT),
        *passthrough,
    ]


def run_rectgel_generation(argv: Sequence[str], *, staged_texture_dir: Path) -> None:
    staged_by_stem = {
        path.stem: path
        for path in sorted(staged_texture_dir.iterdir(), key=lambda path: path.stem)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not staged_by_stem:
        raise RuntimeError(f"No staged textures found in {staged_texture_dir}")

    original_select = rectseq.select_marker_textures
    original_argv = sys.argv[:]

    def patched_select_marker_textures(*, texture_dir: Path, args: argparse.Namespace) -> tuple[list[Path], str]:
        del texture_dir
        if args.marker_texture_names:
            missing = [name for name in args.marker_texture_names if name not in staged_by_stem]
            if missing:
                raise ValueError(f"Unknown staged marker texture names: {missing}")
            selected = [staged_by_stem[name] for name in args.marker_texture_names]
        else:
            selected = list(staged_by_stem.values())
        selected = sorted({path.stem: path for path in selected}.values(), key=lambda path: path.stem)
        return selected, "explicit"

    try:
        rectseq.select_marker_textures = patched_select_marker_textures
        sys.argv = list(argv)
        rectseq.main()
    finally:
        rectseq.select_marker_textures = original_select
        sys.argv = original_argv


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_json_atomic(path, dict(payload))


def update_merged_episode_metadata(
    episode_dir: Path,
    *,
    new_episode_id: int,
    safe_depth: SafeDepthConfig,
    short_edge_mm: float,
) -> dict[str, Any]:
    metadata_path = episode_dir / "metadata.json"
    metadata = load_json(metadata_path)
    metadata["episode_id"] = int(new_episode_id)
    metadata["requested_depth_mm"] = float(safe_depth.requested_depth_mm)
    metadata["safe_depth_cap_mm"] = float(safe_depth.safe_cap_mm)
    metadata["effective_depth_mm"] = float(safe_depth.effective_depth_mm)
    metadata["safe_limit_mm"] = float(safe_depth.safe_limit_mm)
    metadata["depth_was_capped"] = bool(safe_depth.depth_was_capped)
    metadata["short_edge_mm"] = float(short_edge_mm)

    scales = metadata.get("scales", {})
    if not isinstance(scales, dict):
        raise TypeError(f"metadata.scales must be a dict in {metadata_path}")

    for scale_key, scale_summary in scales.items():
        if not isinstance(scale_summary, dict):
            continue
        scale_summary["requested_depth_mm"] = float(safe_depth.requested_depth_mm)
        scale_summary["safe_depth_cap_mm"] = float(safe_depth.safe_cap_mm)
        scale_summary["effective_depth_mm"] = float(safe_depth.effective_depth_mm)
        scale_summary["safe_limit_mm"] = float(safe_depth.safe_limit_mm)
        scale_summary["depth_was_capped"] = bool(safe_depth.depth_was_capped)
        scale_summary["short_edge_mm"] = float(short_edge_mm)
        seq_rel = scale_summary.get("sequence_metadata")
        if not seq_rel:
            continue
        seq_meta_path = episode_dir / str(seq_rel)
        seq_meta = load_json(seq_meta_path)
        seq_meta["episode_id"] = int(new_episode_id)
        seq_meta["requested_depth_mm"] = float(safe_depth.requested_depth_mm)
        seq_meta["safe_depth_cap_mm"] = float(safe_depth.safe_cap_mm)
        seq_meta["effective_depth_mm"] = float(safe_depth.effective_depth_mm)
        seq_meta["safe_limit_mm"] = float(safe_depth.safe_limit_mm)
        seq_meta["depth_was_capped"] = bool(safe_depth.depth_was_capped)
        seq_meta["short_edge_mm"] = float(short_edge_mm)
        write_json(seq_meta_path, seq_meta)

    write_json(metadata_path, metadata)
    return metadata


def episode_has_complete_sequence(episode_dir: Path) -> bool:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = load_json(metadata_path)
    scales = metadata.get("scales", {})
    if not isinstance(scales, dict) or not scales:
        return False
    for scale_summary in scales.values():
        if not isinstance(scale_summary, dict):
            return False
        seq_rel = scale_summary.get("sequence_metadata")
        if not seq_rel:
            return False
        if not (episode_dir / str(seq_rel)).exists():
            return False
    return True


def determine_split(scale_split: str, indenter_split: str) -> str:
    for candidate in (str(scale_split), str(indenter_split)):
        if candidate in {"train", "val", "test"}:
            return candidate
    return "train"


def scan_event_records(dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    episodes_out: list[dict[str, Any]] = []
    marker_stems: set[str] = set()

    for episode_dir in sorted(dataset_root.glob("episode_*")):
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        scales = metadata.get("scales", {})
        if not isinstance(scales, dict):
            continue

        scale_entries: list[dict[str, Any]] = []
        for scale_key, scale_summary in sorted(scales.items()):
            if not isinstance(scale_summary, dict):
                continue
            seq_rel = scale_summary.get("sequence_metadata")
            if not seq_rel:
                continue
            seq_meta_path = episode_dir / str(seq_rel)
            if not seq_meta_path.exists():
                continue
            seq_meta = load_json(seq_meta_path)
            frame_rows = seq_meta.get("frames", [])
            adapter_rel = str(seq_meta.get("adapter_coord_map", scale_summary.get("adapter_coord_map", "adapter_coord_map.npy")))
            split_name = determine_split(str(scale_summary.get("scale_split", "")), str(metadata.get("indenter_split", "")))

            for frame in frame_rows:
                rendered_markers = sorted(str(v) for v in frame.get("rendered_markers", []))
                if len(rendered_markers) != 20:
                    raise RuntimeError(
                        f"Expected 20 rendered markers in {episode_dir / scale_key / str(frame.get('frame_name', ''))}, "
                        f"found {len(rendered_markers)}"
                    )
                marker_stems.update(
                    marker_name[len("marker_") : -4]
                    if marker_name.startswith("marker_") and marker_name.endswith(".jpg")
                    else Path(marker_name).stem
                    for marker_name in rendered_markers
                )
                row = {
                    "episode_id": int(metadata["episode_id"]),
                    "indenter": str(metadata["indenter"]),
                    "indenter_split": str(metadata.get("indenter_split", "")),
                    "is_unseen_indenter": bool(metadata.get("is_unseen_indenter", False)),
                    "scale_key": str(scale_key),
                    "scale_mm": float(scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm"))),
                    "scale_requested_mm": float(scale_summary.get("scale_requested_mm", scale_summary.get("scale_mm"))),
                    "scale_simulated_mm": float(scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm"))),
                    "gel_width_mm": float(scale_summary.get("gel_width_mm", scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm")))),
                    "gel_height_mm": float(
                        scale_summary.get(
                            "gel_height_mm",
                            rectgel.gel_dims_from_width_mm(scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm"))).height_mm,
                        )
                    ),
                    "global_seq_index": int(frame["global_seq_index"]),
                    "frame_name": str(frame["frame_name"]),
                    "phase_name": str(frame["phase_name"]),
                    "phase_index": int(frame.get("phase_index", 0)),
                    "phase_progress": float(frame["phase_progress"]),
                    "frame_depth_mm": float(frame["frame_actual_max_down_mm"]),
                    "contact_x_norm": float(scale_summary["contact_x_norm"]),
                    "contact_y_norm": float(scale_summary["contact_y_norm"]),
                    "contact_x_mm": float(scale_summary["contact_x_mm"]),
                    "contact_y_mm": float(scale_summary["contact_y_mm"]),
                    "frame_dir_relpath": legacy._safe_relpath(episode_dir / str(scale_key) / str(frame["frame_name"]), dataset_root),
                    "adapter_coord_map_relpath": legacy._safe_relpath(episode_dir / str(scale_key) / adapter_rel, dataset_root),
                    "marker_names": rendered_markers,
                    "split": split_name,
                    "scale_split": str(scale_summary.get("scale_split", "")),
                    "is_unseen_scale": bool(scale_summary.get("is_unseen_scale", False)),
                    "position_sample_type_requested": str(metadata.get("position_sample_type_requested", "")),
                    "position_sample_type_actual": str(metadata.get("position_sample_type_actual", "")),
                    "requested_depth_mm": float(metadata.get("requested_depth_mm", scale_summary.get("requested_depth_mm", frame["frame_actual_max_down_mm"]))),
                    "effective_depth_mm": float(metadata.get("effective_depth_mm", scale_summary.get("effective_depth_mm", frame["frame_actual_max_down_mm"]))),
                    "safe_depth_cap_mm": float(metadata.get("safe_depth_cap_mm", scale_summary.get("safe_depth_cap_mm", frame["frame_actual_max_down_mm"]))),
                    "depth_was_capped": bool(metadata.get("depth_was_capped", scale_summary.get("depth_was_capped", False))),
                    "short_edge_mm": float(metadata.get("short_edge_mm", scale_summary.get("short_edge_mm", 0.0))),
                }
                rows.append(row)

            scale_entries.append(
                {
                    "scale_key": str(scale_key),
                    "scale_requested_mm": float(scale_summary.get("scale_requested_mm", scale_summary.get("scale_mm"))),
                    "scale_simulated_mm": float(scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm"))),
                    "gel_width_mm": float(scale_summary.get("gel_width_mm", scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm")))),
                    "gel_height_mm": float(
                        scale_summary.get(
                            "gel_height_mm",
                            rectgel.gel_dims_from_width_mm(scale_summary.get("scale_simulated_mm", scale_summary.get("scale_mm"))).height_mm,
                        )
                    ),
                    "scale_split": str(scale_summary.get("scale_split", "")),
                    "is_unseen_scale": bool(scale_summary.get("is_unseen_scale", False)),
                    "sequence_metadata": str(scale_summary.get("sequence_metadata", "")),
                }
            )

        latent_contact = metadata.get("latent_contact", {})
        episodes_out.append(
            {
                "episode_id": int(metadata["episode_id"]),
                "path": episode_dir.name,
                "indenter": str(metadata["indenter"]),
                "indenter_split": str(metadata.get("indenter_split", "")),
                "is_unseen_indenter": bool(metadata.get("is_unseen_indenter", False)),
                "latent_contact": {
                    "contact_x_norm": float(latent_contact.get("contact_x_norm", 0.0)),
                    "contact_y_norm": float(latent_contact.get("contact_y_norm", 0.0)),
                    "final_depth_mm": float(latent_contact.get("final_depth_mm", metadata.get("effective_depth_mm", 0.0))),
                },
                "requested_depth_mm": float(metadata.get("requested_depth_mm", 0.0)),
                "effective_depth_mm": float(metadata.get("effective_depth_mm", 0.0)),
                "safe_depth_cap_mm": float(metadata.get("safe_depth_cap_mm", 0.0)),
                "depth_was_capped": bool(metadata.get("depth_was_capped", False)),
                "scales": scale_entries,
            }
        )

    rows.sort(key=lambda row: (int(row["episode_id"]), str(row["scale_key"]), int(row["global_seq_index"])))
    episodes_out.sort(key=lambda item: int(item["episode_id"]))
    return rows, episodes_out, sorted(marker_stems)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(legacy._json_safe(dict(row)), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_event_indexes(dataset_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "all": write_jsonl(dataset_root / "event_index.jsonl", rows),
        "train": 0,
        "val": 0,
        "test": 0,
    }
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if str(row.get("split", "")) == split]
        counts[split] = write_jsonl(dataset_root / f"event_index_{split}.jsonl", split_rows)
    return counts


def build_manifest(
    *,
    dataset_root: Path,
    args: argparse.Namespace,
    gel_width_mm: float,
    gel_height_mm: float,
    marker_stems: Sequence[str],
    episodes: Sequence[Mapping[str, Any]],
    event_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_variant": DATASET_VARIANT,
        "command": " ".join(sys.argv),
        "short_edge_mm": float(args.short_edge_mm),
        "gel_width_mm": float(gel_width_mm),
        "gel_height_mm": float(gel_height_mm),
        "scale_mm": float(gel_width_mm),
        "scale_key": rectgel.format_width_key(gel_width_mm),
        "image_resolution": [TARGET_WIDTH, TARGET_HEIGHT],
        "coordinate_convention": rectseq.COORDINATE_CONVENTION,
        "marker_textures_selected": [str(v) for v in marker_stems],
        "marker_texture_count": int(len(marker_stems)),
        "top_level_square_marker_count": 12,
        "four_three_marker_count": 8,
        "safe_depth_bad_thresholds_mm": {key: float(value) for key, value in DEFAULT_BAD_DEPTH_MM.items()},
        "safety_margin_mm": float(args.safety_margin_mm),
        "episodes_per_indenter": int(args.episodes_per_indenter),
        "event_index_files": {
            "all": "event_index.jsonl",
            "train": "event_index_train.jsonl",
            "val": "event_index_val.jsonl",
            "test": "event_index_test.jsonl",
        },
        "event_counts": {key: int(value) for key, value in event_counts.items()},
        "storage_layout": "event_centric_same_scale",
        "pair_materialization": "runtime_marker_sampling",
        "pair_policy": {
            "same_episode": True,
            "same_indenter": True,
            "same_scale_key": True,
            "same_global_seq_index": True,
            "same_phase": True,
            "exclude_identity": True,
        },
        "episodes": list(episodes),
    }


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args)
    work_root = resolve_work_root(args, output_root)
    validate_output_and_work_roots(output_root, work_root)
    ensure_empty_dir(output_root, "Output root")
    if work_root != output_root / DEFAULT_WORK_DIRNAME:
        ensure_empty_dir(work_root, "Work root")
    else:
        work_root.mkdir(parents=True, exist_ok=True)

    gel_height_mm = float(args.short_edge_mm)
    gel_width_mm = round(float(args.short_edge_mm) * 4.0 / 3.0, 4)
    if gel_height_mm <= 0:
        raise ValueError("--short-edge-mm must be > 0")
    if args.depth_min <= 0:
        raise ValueError("--depth-min must be > 0")
    if args.requested_depth_max_mm <= 0:
        raise ValueError("--requested-depth-max-mm must be > 0")
    if args.requested_depth_max_mm < args.depth_min:
        raise ValueError("--requested-depth-max-mm must be >= --depth-min")
    if args.episodes_per_indenter <= 0:
        raise ValueError("--episodes-per-indenter must be > 0")

    staged_texture_dir = work_root / "marker_textures_flat"
    staged_textures = stage_marker_textures(staged_texture_dir)
    marker_stems = [path.stem for path in staged_textures]

    runs: list[GeneratorRun] = []
    requested_depth_mm = float(args.requested_depth_max_mm)
    for indenter_index, indenter in enumerate(args.objects):
        safe_depth = compute_safe_depth(
            indenter=str(indenter),
            requested_depth_mm=requested_depth_mm,
            safety_margin_mm=float(args.safety_margin_mm),
            depth_min_mm=float(args.depth_min),
        )
        run_root = work_root / f"run_{indenter}"
        ensure_empty_dir(run_root, f"Run root for {indenter}")
        argv = build_rectgel_argv(
            run_root=run_root,
            indenter=str(indenter),
            marker_stems=marker_stems,
            seed=int(args.seed) + indenter_index,
            episodes_per_indenter=int(args.episodes_per_indenter),
            gel_width_mm=gel_width_mm,
            depth_min_mm=float(args.depth_min),
            depth_max_mm=float(safe_depth.effective_depth_mm),
            passthrough=args.genforce_args,
        )
        print(
            f"Generating {indenter}: requested_depth={safe_depth.requested_depth_mm:.4f}mm "
            f"safe_cap={safe_depth.safe_cap_mm:.4f}mm effective={safe_depth.effective_depth_mm:.4f}mm"
        )
        run_rectgel_generation(argv, staged_texture_dir=staged_texture_dir)
        runs.append(GeneratorRun(indenter=str(indenter), run_root=run_root, safe_depth=safe_depth))

    next_episode_id = 0
    for run in runs:
        source_episode_dirs = [path for path in sorted(run.run_root.glob("episode_*")) if episode_has_complete_sequence(path)]
        if not source_episode_dirs:
            raise RuntimeError(
                f"No completed episodes were produced for indenter {run.indenter} in {run.run_root}. "
                "Check the rectgel generator logs in the work directory."
            )
        for source_episode_dir in source_episode_dirs:
            destination_episode_dir = output_root / f"episode_{next_episode_id:06d}"
            shutil.move(str(source_episode_dir), str(destination_episode_dir))
            update_merged_episode_metadata(
                destination_episode_dir,
                new_episode_id=next_episode_id,
                safe_depth=run.safe_depth,
                short_edge_mm=gel_height_mm,
            )
            next_episode_id += 1

    rows, episodes, discovered_marker_stems = scan_event_records(output_root)
    if sorted(discovered_marker_stems) != sorted(marker_stems):
        raise RuntimeError(
            "Merged dataset marker set does not match staged marker set: "
            f"expected={marker_stems} actual={discovered_marker_stems}"
        )
    event_counts = write_event_indexes(output_root, rows)
    manifest = build_manifest(
        dataset_root=output_root,
        args=args,
        gel_width_mm=gel_width_mm,
        gel_height_mm=gel_height_mm,
        marker_stems=discovered_marker_stems,
        episodes=episodes,
        event_counts=event_counts,
    )
    write_json(output_root / "manifest.json", manifest)

    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)

    print(
        "Generation complete | episodes=%d events=%d train=%d val=%d test=%d markers=%d output=%s"
        % (
            len(episodes),
            event_counts["all"],
            event_counts["train"],
            event_counts["val"],
            event_counts["test"],
            len(discovered_marker_stems),
            output_root,
        )
    )


if __name__ == "__main__":
    main()
