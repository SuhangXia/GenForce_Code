#!/usr/bin/env python3
"""Generate a strict 16mm<->23mm shared-state cross-band benchmark.

This script is intentionally separate from the general-purpose multiscale
dataset generators. The benchmark semantics are different:

1. Sample a shared episode/state template first.
2. Render the same template at one or more 16-band scales and one or more
   23-band scales.
3. Rebuild benchmark-specific candidate pools and explicit cross-band indexes
   using a shared state key, so strict 16<->23 eval never degenerates into
   target-only held-out filtering.

The rendered sequences still come from the proven sequence generator. The new
logic here is the shared-state planning layer and the benchmark-specific index
construction.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_multiscale_dataset as legacy
import generate_multiscale_dataset_sequence as seqgen


BAND_16 = (15.8, 16.2)
BAND_23 = (22.8, 23.2)
DEFAULT_SMOKE_INDENTERS = ["cone", "cylinder", "line"]
@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def _json_safe(value: Any) -> Any:
    return legacy._json_safe(value)


def _in_band(scale_mm: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    value = float(scale_mm)
    return float(lo) <= value <= float(hi)


def _format_scale_list(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.4f}" for v in values)


def _marker_stems_to_files(marker_stems: Sequence[str]) -> list[str]:
    return [f"marker_{stem}.jpg" for stem in marker_stems]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a strict shared-state 16<->23 cross-band benchmark.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--mode", choices=["generate", "inspect"], default="generate")
    parser.add_argument("--particle", type=str, default="100000")
    parser.add_argument("--indenters", nargs="+", default=None)
    parser.add_argument("--episodes-per-indenter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--band16-scales-mm", nargs="+", type=float, default=[16.0])
    parser.add_argument("--band23-scales-mm", nargs="+", type=float, default=[23.0])
    parser.add_argument("--marker-texture-names", nargs="*", default=None)
    parser.add_argument("--marker-texture-count", type=int, default=2)
    parser.add_argument("--marker-texture-seed", type=int, default=None)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--physics-npz-cleanup", choices=["keep", "delete_after_scale_complete"], default="keep")

    parser.add_argument("--x-min", type=float, default=legacy.DEFAULT_X_RANGE[0])
    parser.add_argument("--x-max", type=float, default=legacy.DEFAULT_X_RANGE[1])
    parser.add_argument("--y-min", type=float, default=legacy.DEFAULT_Y_RANGE[0])
    parser.add_argument("--y-max", type=float, default=legacy.DEFAULT_Y_RANGE[1])
    parser.add_argument("--depth-min", type=float, default=legacy.DEFAULT_DEPTH_RANGE[0])
    parser.add_argument("--depth-max", type=float, default=legacy.DEFAULT_DEPTH_RANGE[1])
    parser.add_argument("--position-clean-ratio", type=float, default=legacy.DEFAULT_POSITION_SAMPLE_RATIOS["clean"])
    parser.add_argument("--position-near-boundary-ratio", type=float, default=legacy.DEFAULT_POSITION_SAMPLE_RATIOS["near_boundary"])
    parser.add_argument("--position-partial-crop-ratio", type=float, default=legacy.DEFAULT_POSITION_SAMPLE_RATIOS["partial_crop"])
    parser.add_argument("--near-boundary-max-margin-mm", type=float, default=legacy.DEFAULT_NEAR_BOUNDARY_MAX_MARGIN_MM)
    parser.add_argument("--partial-crop-min-overhang-mm", type=float, default=legacy.DEFAULT_PARTIAL_CROP_MIN_OVERHANG_MM)
    parser.add_argument("--partial-crop-max-overhang-mm", type=float, default=legacy.DEFAULT_PARTIAL_CROP_MAX_OVERHANG_MM)

    parser.add_argument("--patch-grid", type=str, default=legacy.DEFAULT_PATCH_GRID)
    parser.add_argument("--precontact-frames-min", type=int, default=seqgen.DEFAULT_PRECONTACT_FRAMES[0])
    parser.add_argument("--precontact-frames-max", type=int, default=seqgen.DEFAULT_PRECONTACT_FRAMES[1])
    parser.add_argument("--press-frames-min", type=int, default=seqgen.DEFAULT_PRESS_FRAMES[0])
    parser.add_argument("--press-frames-max", type=int, default=seqgen.DEFAULT_PRESS_FRAMES[1])
    parser.add_argument("--dwell-frames-min", type=int, default=seqgen.DEFAULT_DWELL_FRAMES[0])
    parser.add_argument("--dwell-frames-max", type=int, default=seqgen.DEFAULT_DWELL_FRAMES[1])
    parser.add_argument("--release-frames-min", type=int, default=seqgen.DEFAULT_RELEASE_FRAMES[0])
    parser.add_argument("--release-frames-max", type=int, default=seqgen.DEFAULT_RELEASE_FRAMES[1])
    parser.add_argument("--press-sampling-mode", choices=["uniform_index"], default="uniform_index")
    parser.add_argument("--release-mode", choices=["auto", "mirror_loading"], default="auto")
    parser.add_argument("--include-dwell", action="store_true", default=True)
    parser.add_argument("--no-include-dwell", dest="include_dwell", action="store_false")
    parser.add_argument("--include-release", action="store_true", default=True)
    parser.add_argument("--no-include-release", dest="include_release", action="store_false")

    parser.add_argument("--camera-mode", choices=["fixed_distance_variable_fov", "fixed_fov_variable_distance"], default="fixed_distance_variable_fov")
    parser.add_argument("--fov-deg", type=float, default=legacy.DEFAULT_FOV_DEG)
    parser.add_argument("--reference-scale-mm", type=float, default=legacy.DEFAULT_REFERENCE_SCALE_MM)
    parser.add_argument("--camera-distance-m", type=float, default=None)
    parser.add_argument("--distance-safety", type=float, default=legacy.DEFAULT_DISTANCE_SAFETY)
    parser.add_argument("--uv-inset-ratio", type=float, default=0.01)
    parser.add_argument("--uv-mode", choices=["unwrap_genforce", "physical_math"], default="unwrap_genforce")
    parser.add_argument("--force-black-side-border-px", type=int, default=2)

    parser.add_argument("--max-physics-workers", type=int, default=6)
    parser.add_argument("--max-meshing-workers", type=int, default=2)
    parser.add_argument("--max-render-workers", type=int, default=4)
    parser.add_argument("--physics-timeout-sec", type=int, default=900)
    parser.add_argument("--meshing-timeout-sec", type=int, default=600)
    parser.add_argument("--render-timeout-sec", type=int, default=900)
    parser.add_argument("--render-samples", type=int, default=32)
    parser.add_argument("--render-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--render-gpu-backend", choices=["auto", "optix", "cuda", "hip", "oneapi", "metal"], default="auto")
    parser.add_argument("--auto-balance-pipeline", action="store_true")
    parser.add_argument("--render-backlog-high-watermark", type=int, default=0)
    parser.add_argument("--render-backlog-low-watermark", type=int, default=0)

    parser.add_argument("--python-cmd", type=str, default=sys.executable)
    parser.add_argument("--blender-cmd", type=str, default="blender")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def _resolve_indenters(args: argparse.Namespace) -> list[str]:
    if args.indenters:
        return [str(v) for v in args.indenters]
    return list(DEFAULT_SMOKE_INDENTERS)


def _validate_band_scale_lists(band16_scales: Sequence[float], band23_scales: Sequence[float]) -> None:
    if not band16_scales or not band23_scales:
        raise ValueError("Both --band16-scales-mm and --band23-scales-mm must be non-empty")
    for value in band16_scales:
        if not _in_band(float(value), BAND_16):
            raise ValueError(f"16-band scale {value} is outside strict band {BAND_16}")
    for value in band23_scales:
        if not _in_band(float(value), BAND_23):
            raise ValueError(f"23-band scale {value} is outside strict band {BAND_23}")


def _build_sequence_args(args: argparse.Namespace) -> argparse.Namespace:
    indenters = _resolve_indenters(args)
    _validate_band_scale_lists(args.band16_scales_mm, args.band23_scales_mm)
    scales_mm = [float(v) for v in args.band16_scales_mm] + [float(v) for v in args.band23_scales_mm]
    cli = [
        "generate_multiscale_dataset_sequence.py",
        "--repo-root", str(args.repo_root),
        "--dataset-root", str(args.dataset_root),
        "--dataset-profile", "A",
        "--particle", str(args.particle),
        "--episodes-per-indenter", str(int(args.episodes_per_indenter)),
        "--seed", str(int(args.seed)),
        "--scale-mode", "discrete",
        "--allowed-scale-splits", "val", "test",
        "--heldout-val-scales-mm", "16.0",
        "--heldout-test-scales-mm", "23.0",
        "--heldout-val-scale-band-mm", "0.2",
        "--heldout-test-scale-band-mm", "0.2",
        "--pair-marker-policy", "same_texture_only",
        "--pair-split-policy", "all",
        "--include-identity-pairs",
        "--patch-grid", str(args.patch_grid),
        "--x-min", str(float(args.x_min)),
        "--x-max", str(float(args.x_max)),
        "--y-min", str(float(args.y_min)),
        "--y-max", str(float(args.y_max)),
        "--depth-min", str(float(args.depth_min)),
        "--depth-max", str(float(args.depth_max)),
        "--position-clean-ratio", str(float(args.position_clean_ratio)),
        "--position-near-boundary-ratio", str(float(args.position_near_boundary_ratio)),
        "--position-partial-crop-ratio", str(float(args.position_partial_crop_ratio)),
        "--near-boundary-max-margin-mm", str(float(args.near_boundary_max_margin_mm)),
        "--partial-crop-min-overhang-mm", str(float(args.partial_crop_min_overhang_mm)),
        "--partial-crop-max-overhang-mm", str(float(args.partial_crop_max_overhang_mm)),
        "--precontact-frames-min", str(int(args.precontact_frames_min)),
        "--precontact-frames-max", str(int(args.precontact_frames_max)),
        "--press-frames-min", str(int(args.press_frames_min)),
        "--press-frames-max", str(int(args.press_frames_max)),
        "--dwell-frames-min", str(int(args.dwell_frames_min)),
        "--dwell-frames-max", str(int(args.dwell_frames_max)),
        "--release-frames-min", str(int(args.release_frames_min)),
        "--release-frames-max", str(int(args.release_frames_max)),
        "--press-sampling-mode", str(args.press_sampling_mode),
        "--release-mode", str(args.release_mode),
        "--camera-mode", str(args.camera_mode),
        "--fov-deg", str(float(args.fov_deg)),
        "--reference-scale-mm", str(float(args.reference_scale_mm)),
        "--distance-safety", str(float(args.distance_safety)),
        "--uv-inset-ratio", str(float(args.uv_inset_ratio)),
        "--uv-mode", str(args.uv_mode),
        "--force-black-side-border-px", str(int(args.force_black_side_border_px)),
        "--max-physics-workers", str(int(args.max_physics_workers)),
        "--max-meshing-workers", str(int(args.max_meshing_workers)),
        "--max-render-workers", str(int(args.max_render_workers)),
        "--physics-timeout-sec", str(int(args.physics_timeout_sec)),
        "--meshing-timeout-sec", str(int(args.meshing_timeout_sec)),
        "--render-timeout-sec", str(int(args.render_timeout_sec)),
        "--render-samples", str(int(args.render_samples)),
        "--render-device", str(args.render_device),
        "--render-gpu-backend", str(args.render_gpu_backend),
        "--python-cmd", str(args.python_cmd),
        "--blender-cmd", str(args.blender_cmd),
        "--physics-npz-cleanup", str(args.physics_npz_cleanup),
        "--log-level", str(args.log_level),
        "--objects", *indenters,
        "--train-indenters", *indenters,
        "--scales-mm", *[f"{float(v):.4f}" for v in scales_mm],
    ]
    if args.marker_texture_names:
        cli.extend(["--marker-texture-names", *[str(v) for v in args.marker_texture_names]])
    else:
        cli.extend(["--marker-texture-count", str(int(args.marker_texture_count))])
    marker_seed = args.marker_texture_seed if args.marker_texture_seed is not None else args.seed
    cli.extend(["--marker-texture-seed", str(int(marker_seed))])
    if args.camera_distance_m is not None:
        cli.extend(["--camera-distance-m", str(float(args.camera_distance_m))])
    if args.keep_intermediates:
        cli.append("--keep-intermediates")
    if args.resume:
        cli.append("--resume")
    if args.clean_output:
        cli.append("--clean-output")
    if args.include_dwell:
        cli.append("--include-dwell")
    else:
        cli.append("--no-include-dwell")
    if args.include_release:
        cli.append("--include-release")
    else:
        cli.append("--no-include-release")
    if args.auto_balance_pipeline:
        cli.append("--auto-balance-pipeline")
    if int(args.render_backlog_high_watermark) > 0:
        cli.extend(["--render-backlog-high-watermark", str(int(args.render_backlog_high_watermark))])
    if int(args.render_backlog_low_watermark) > 0:
        cli.extend(["--render-backlog-low-watermark", str(int(args.render_backlog_low_watermark))])

    with _temporary_argv(cli):
        seq_args = seqgen.parse_args()
    seqgen.validate_args(seq_args)
    return seq_args


def _load_generation_context(
    seq_args: argparse.Namespace,
) -> tuple[list[str], Mapping[str, Dict[str, float]], list[str], int, int]:
    repo_root = seq_args.repo_root.resolve()
    parameters_path = repo_root / "sim" / "parameters.yml"
    texture_dir = repo_root / "sim" / "marker" / "marker_pattern"
    indenter_dir = repo_root / "sim" / "assets" / "indenters" / "input" / f"npy_{seq_args.particle}"
    if not parameters_path.exists():
        raise FileNotFoundError(f"Missing {parameters_path}")
    if not texture_dir.exists():
        raise FileNotFoundError(f"Missing marker texture directory: {texture_dir}")
    if not indenter_dir.exists():
        raise FileNotFoundError(f"Missing indenter directory: {indenter_dir}")

    patch_h, patch_w = legacy.parse_patch_grid(seq_args.patch_grid)
    indenter_names = legacy.discover_indenter_names(indenter_dir, seq_args.objects)

    with open(parameters_path, "r", encoding="utf-8") as f:
        parameters_cfg = yaml.safe_load(f)
    indenter_pose = parameters_cfg.get("indenter", {}).get("pose", {}) if isinstance(parameters_cfg, dict) else {}
    indenter_rotation = legacy.rotation_matrix_from_pose(indenter_pose, degrees=False)
    indenter_bboxes = {
        name: legacy.compute_indenter_contact_bbox_mm(indenter_dir / f"{name}.npy", indenter_rotation)
        for name in indenter_names
    }

    selected_textures, selection_mode = seqgen.select_marker_textures(texture_dir=texture_dir, args=seq_args)
    seq_args.marker_textures_selected = [str(path.stem) for path in selected_textures]
    seq_args.marker_texture_selection_mode = str(selection_mode)
    marker_files = [f"marker_{path.stem}.jpg" for path in selected_textures]
    return indenter_names, indenter_bboxes, marker_files, patch_h, patch_w


def _planned_press_depths(press_count: int, final_depth_mm: float) -> list[float]:
    if press_count <= 0:
        return []
    if press_count == 1:
        return [float(final_depth_mm)]
    return [float(final_depth_mm) * (float(idx) / float(press_count - 1)) for idx in range(press_count)]


def _planned_release_depths(press_depths: Sequence[float], release_count: int) -> list[float]:
    if release_count <= 0:
        return []
    mirrored = list(reversed(press_depths[:-1] if len(press_depths) > 1 else press_depths))
    if not mirrored:
        mirrored = [0.0]
    if len(mirrored) == 1:
        return [float(mirrored[0])] * release_count
    positions = [float(idx) * (float(len(mirrored) - 1) / float(max(release_count - 1, 1))) for idx in range(release_count)]
    depths: list[float] = []
    for pos in positions:
        depths.append(float(mirrored[int(round(pos))]))
    return depths


def _build_planned_shared_frames(
    temporal_plan: seqgen.TemporalPlan,
    *,
    final_depth_mm: float,
) -> list[dict[str, Any]]:
    phase_lengths = {
        "precontact": int(temporal_plan.precontact_frame_count),
        "press": int(temporal_plan.press_frame_count),
        "dwell": int(temporal_plan.dwell_frame_count),
        "release": int(temporal_plan.release_frame_count),
    }
    press_depths = _planned_press_depths(phase_lengths["press"], float(final_depth_mm))
    release_depths = _planned_release_depths(press_depths, phase_lengths["release"])
    phase_depth_lookup = {
        "precontact": [0.0] * phase_lengths["precontact"],
        "press": press_depths,
        "dwell": [float(final_depth_mm)] * phase_lengths["dwell"],
        "release": release_depths,
    }
    frames: list[dict[str, Any]] = []
    seq_idx = 0
    for phase_name in ("precontact", "press", "dwell", "release"):
        depths = phase_depth_lookup[phase_name]
        phase_length = len(depths)
        for phase_index, depth_mm in enumerate(depths):
            phase_progress = 1.0 if phase_length <= 1 else float(phase_index) / float(phase_length - 1)
            frames.append(
                {
                    "global_seq_index": int(seq_idx),
                    "phase_name": str(phase_name),
                    "phase_index": int(phase_index),
                    "phase_progress": float(round(phase_progress, 6)),
                    "shared_depth_mm": float(round(depth_mm, 6)),
                }
            )
            seq_idx += 1
    return frames


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_plan_manifest(
    dataset_root: Path,
    *,
    episode_states: Mapping[int, seqgen.SequenceEpisodeState],
    marker_files: Sequence[str],
) -> int:
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(episode_states):
        episode_state = episode_states[episode_id]
        band16 = [
            {
                "scale_key": str(meta["scale_key"]),
                "scale_mm": float(meta["scale_simulated_mm"]),
            }
            for meta in episode_state.scales.values()
            if _in_band(float(meta["scale_simulated_mm"]), BAND_16)
        ]
        band23 = [
            {
                "scale_key": str(meta["scale_key"]),
                "scale_mm": float(meta["scale_simulated_mm"]),
            }
            for meta in episode_state.scales.values()
            if _in_band(float(meta["scale_simulated_mm"]), BAND_23)
        ]
        frames = _build_planned_shared_frames(
            episode_state.temporal_plan,
            final_depth_mm=float(episode_state.final_depth_mm),
        )
        for frame in frames:
            for marker_name in marker_files:
                rows.append(
                    {
                        "state_id": f"ep{int(episode_id):06d}::{marker_name}::seq{int(frame['global_seq_index']):06d}",
                        "episode_id": int(episode_id),
                        "indenter": str(episode_state.indenter),
                        "indenter_split": str(episode_state.indenter_split),
                        "marker_name": str(marker_name),
                        "phase_name": str(frame["phase_name"]),
                        "phase_index": int(frame["phase_index"]),
                        "phase_progress": float(frame["phase_progress"]),
                        "global_seq_index": int(frame["global_seq_index"]),
                        "contact_x_norm": float(episode_state.contact_x_norm),
                        "contact_y_norm": float(episode_state.contact_y_norm),
                        "depth_mm": float(frame["shared_depth_mm"]),
                        "final_depth_mm": float(episode_state.final_depth_mm),
                        "band16_scales": band16,
                        "band23_scales": band23,
                    }
                )
    return _write_jsonl(dataset_root / "benchmark_manifest_plan.jsonl", rows)


def _load_top_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json under {dataset_root}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shared_depth_for_frame(
    phase_name: str,
    phase_index: int,
    phase_lengths: Mapping[str, Any],
    final_depth_mm: float,
) -> float:
    plan = seqgen.TemporalPlan(
        precontact_frame_count=int(phase_lengths.get("precontact", 0)),
        press_frame_count=int(phase_lengths.get("press", 0)),
        dwell_frame_count=int(phase_lengths.get("dwell", 0)),
        release_frame_count=int(phase_lengths.get("release", 0)),
        press_sampling_mode="uniform_index",
        release_mode_requested="mirror_loading",
        release_mode_actual="mirrored_loading",
        final_depth_mm=float(final_depth_mm),
        include_dwell=bool(int(phase_lengths.get("dwell", 0)) > 0),
        include_release=bool(int(phase_lengths.get("release", 0)) > 0),
    )
    frames = _build_planned_shared_frames(plan, final_depth_mm=float(final_depth_mm))
    for frame in frames:
        if str(frame["phase_name"]) == str(phase_name) and int(frame["phase_index"]) == int(phase_index):
            return float(frame["shared_depth_mm"])
    raise KeyError(f"Could not resolve shared depth for phase={phase_name} phase_index={phase_index}")


def _frame_lookup(scale_meta: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    frames = scale_meta.get("frames", [])
    if not isinstance(frames, list):
        raise TypeError("Expected sequence metadata frames list")
    mapped: dict[int, Mapping[str, Any]] = {}
    for frame in frames:
        mapped[int(frame["global_seq_index"])] = frame
    return mapped


def _make_pair_row(
    *,
    dataset_root: Path,
    episode_meta: Mapping[str, Any],
    source_scale_key: str,
    target_scale_key: str,
    source_scale_meta: Mapping[str, Any],
    target_scale_meta: Mapping[str, Any],
    source_frame: Mapping[str, Any],
    target_frame: Mapping[str, Any],
    marker_name: str,
    pair_split: str,
    shared_depth_mm: float,
) -> dict[str, Any]:
    episode_id = int(episode_meta["episode_id"])
    episode_dir = dataset_root / f"episode_{episode_id:06d}"
    source_frame_path = episode_dir / source_scale_key / str(source_frame["frame_name"]) / marker_name
    target_frame_path = episode_dir / target_scale_key / str(target_frame["frame_name"]) / marker_name
    source_coord_path = episode_dir / source_scale_key / str(source_scale_meta.get("adapter_coord_map", "adapter_coord_map.npy"))
    target_coord_path = episode_dir / target_scale_key / str(target_scale_meta.get("adapter_coord_map", "adapter_coord_map.npy"))
    return {
        "episode_id": episode_id,
        "indenter": str(episode_meta["indenter"]),
        "marker_name": str(marker_name),
        "source_marker_name": str(marker_name),
        "target_marker_name": str(marker_name),
        "source_scale_mm": float(source_scale_meta["scale_simulated_mm"]),
        "target_scale_mm": float(target_scale_meta["scale_simulated_mm"]),
        "source_scale_key": str(source_scale_key),
        "target_scale_key": str(target_scale_key),
        "global_seq_index": int(source_frame["global_seq_index"]),
        "phase_name": str(source_frame["phase_name"]),
        "phase_progress": float(source_frame["phase_progress"]),
        "source_frame_relpath": legacy._safe_relpath(source_frame_path, dataset_root),
        "target_frame_relpath": legacy._safe_relpath(target_frame_path, dataset_root),
        "source_adapter_coord_map_relpath": legacy._safe_relpath(source_coord_path, dataset_root),
        "target_adapter_coord_map_relpath": legacy._safe_relpath(target_coord_path, dataset_root),
        "contact_x_norm": float(episode_meta["latent_contact"]["contact_x_norm"]),
        "contact_y_norm": float(episode_meta["latent_contact"]["contact_y_norm"]),
        "source_contact_x_mm": float(source_scale_meta["contact_x_mm"]),
        "source_contact_y_mm": float(source_scale_meta["contact_y_mm"]),
        "target_contact_x_mm": float(target_scale_meta["contact_x_mm"]),
        "target_contact_y_mm": float(target_scale_meta["contact_y_mm"]),
        "frame_depth_mm": float(shared_depth_mm),
        "source_frame_depth_mm": float(shared_depth_mm),
        "target_frame_depth_mm": float(shared_depth_mm),
        "shared_state_depth_mm": float(shared_depth_mm),
        "indenter_split": str(episode_meta.get("indenter_split", "")),
        "source_scale_split": str(source_scale_meta.get("scale_split", "")),
        "target_scale_split": str(target_scale_meta.get("scale_split", "")),
        "scale_split_source": str(source_scale_meta.get("scale_split", "")),
        "scale_split_target": str(target_scale_meta.get("scale_split", "")),
        "pair_split": str(pair_split),
        "is_unseen_indenter": bool(episode_meta.get("is_unseen_indenter", False)),
        "is_unseen_scale_source": bool(source_scale_meta.get("is_unseen_scale", False)),
        "is_unseen_scale_target": bool(target_scale_meta.get("is_unseen_scale", False)),
    }


def _rebuild_benchmark_indexes(dataset_root: Path) -> dict[str, int]:
    manifest = _load_top_manifest(dataset_root)
    marker_files = _marker_stems_to_files([str(v) for v in manifest.get("marker_textures_selected", [])])
    if not marker_files:
        raise RuntimeError("Manifest did not record selected marker textures")

    records = seqgen.scan_sequence_dataset_records(dataset_root)
    benchmark_manifest_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    cross_16_to_23: list[dict[str, Any]] = []
    cross_23_to_16: list[dict[str, Any]] = []

    for item in records:
        episode_meta = item["episode_meta"]
        scale_payloads = item["scale_payloads"]
        band16 = {key: meta for key, meta in scale_payloads.items() if _in_band(float(meta["scale_simulated_mm"]), BAND_16)}
        band23 = {key: meta for key, meta in scale_payloads.items() if _in_band(float(meta["scale_simulated_mm"]), BAND_23)}
        if not band16 or not band23:
            continue
        ref_scale = next(iter(band16.values()))
        phase_lengths = ref_scale.get("phase_lengths", {})
        ref_frames = _frame_lookup(ref_scale)
        final_depth_mm = float(episode_meta["latent_contact"]["final_depth_mm"])

        for seq_idx, ref_frame in sorted(ref_frames.items()):
            shared_depth_mm = _shared_depth_for_frame(
                str(ref_frame["phase_name"]),
                int(ref_frame["phase_index"]),
                phase_lengths,
                final_depth_mm,
            )
            for marker_name in marker_files:
                benchmark_manifest_rows.append(
                    {
                        "state_id": f"ep{int(episode_meta['episode_id']):06d}::{marker_name}::seq{int(seq_idx):06d}",
                        "episode_id": int(episode_meta["episode_id"]),
                        "indenter": str(episode_meta["indenter"]),
                        "indenter_split": str(episode_meta.get("indenter_split", "")),
                        "marker_name": str(marker_name),
                        "phase_name": str(ref_frame["phase_name"]),
                        "phase_index": int(ref_frame["phase_index"]),
                        "phase_progress": float(ref_frame["phase_progress"]),
                        "global_seq_index": int(seq_idx),
                        "contact_x_norm": float(episode_meta["latent_contact"]["contact_x_norm"]),
                        "contact_y_norm": float(episode_meta["latent_contact"]["contact_y_norm"]),
                        "depth_mm": float(shared_depth_mm),
                        "final_depth_mm": float(final_depth_mm),
                        "band16": [
                            {
                                "scale_key": str(scale_key),
                                "scale_mm": float(scale_meta["scale_simulated_mm"]),
                                "frame_relpath": legacy._safe_relpath(
                                    dataset_root / f"episode_{int(episode_meta['episode_id']):06d}" / scale_key / str(ref_frame["frame_name"]) / marker_name,
                                    dataset_root,
                                ),
                                "coord_map_relpath": legacy._safe_relpath(
                                    dataset_root / f"episode_{int(episode_meta['episode_id']):06d}" / scale_key / str(scale_meta.get("adapter_coord_map", "adapter_coord_map.npy")),
                                    dataset_root,
                                ),
                            }
                            for scale_key, scale_meta in sorted(band16.items())
                        ],
                        "band23": [
                            {
                                "scale_key": str(scale_key),
                                "scale_mm": float(scale_meta["scale_simulated_mm"]),
                                "frame_relpath": legacy._safe_relpath(
                                    dataset_root / f"episode_{int(episode_meta['episode_id']):06d}" / scale_key / str(ref_frame["frame_name"]) / marker_name,
                                    dataset_root,
                                ),
                                "coord_map_relpath": legacy._safe_relpath(
                                    dataset_root / f"episode_{int(episode_meta['episode_id']):06d}" / scale_key / str(scale_meta.get("adapter_coord_map", "adapter_coord_map.npy")),
                                    dataset_root,
                                ),
                            }
                            for scale_key, scale_meta in sorted(band23.items())
                        ],
                    }
                )

            band16_frames = {scale_key: _frame_lookup(scale_meta)[seq_idx] for scale_key, scale_meta in band16.items()}
            band23_frames = {scale_key: _frame_lookup(scale_meta)[seq_idx] for scale_key, scale_meta in band23.items()}
            for marker_name in marker_files:
                for source_scale_key, source_scale_meta in sorted(band16.items()):
                    for target_scale_key, target_scale_meta in sorted(band16.items()):
                        val_rows.append(
                            _make_pair_row(
                                dataset_root=dataset_root,
                                episode_meta=episode_meta,
                                source_scale_key=source_scale_key,
                                target_scale_key=target_scale_key,
                                source_scale_meta=source_scale_meta,
                                target_scale_meta=target_scale_meta,
                                source_frame=band16_frames[source_scale_key],
                                target_frame=band16_frames[target_scale_key],
                                marker_name=marker_name,
                                pair_split="val",
                                shared_depth_mm=shared_depth_mm,
                            )
                        )
                for source_scale_key, source_scale_meta in sorted(band23.items()):
                    for target_scale_key, target_scale_meta in sorted(band23.items()):
                        test_rows.append(
                            _make_pair_row(
                                dataset_root=dataset_root,
                                episode_meta=episode_meta,
                                source_scale_key=source_scale_key,
                                target_scale_key=target_scale_key,
                                source_scale_meta=source_scale_meta,
                                target_scale_meta=target_scale_meta,
                                source_frame=band23_frames[source_scale_key],
                                target_frame=band23_frames[target_scale_key],
                                marker_name=marker_name,
                                pair_split="test",
                                shared_depth_mm=shared_depth_mm,
                            )
                        )
                for source_scale_key, source_scale_meta in sorted(band16.items()):
                    for target_scale_key, target_scale_meta in sorted(band23.items()):
                        cross_16_to_23.append(
                            _make_pair_row(
                                dataset_root=dataset_root,
                                episode_meta=episode_meta,
                                source_scale_key=source_scale_key,
                                target_scale_key=target_scale_key,
                                source_scale_meta=source_scale_meta,
                                target_scale_meta=target_scale_meta,
                                source_frame=band16_frames[source_scale_key],
                                target_frame=band23_frames[target_scale_key],
                                marker_name=marker_name,
                                pair_split="cross_split",
                                shared_depth_mm=shared_depth_mm,
                            )
                        )
                for source_scale_key, source_scale_meta in sorted(band23.items()):
                    for target_scale_key, target_scale_meta in sorted(band16.items()):
                        cross_23_to_16.append(
                            _make_pair_row(
                                dataset_root=dataset_root,
                                episode_meta=episode_meta,
                                source_scale_key=source_scale_key,
                                target_scale_key=target_scale_key,
                                source_scale_meta=source_scale_meta,
                                target_scale_meta=target_scale_meta,
                                source_frame=band23_frames[source_scale_key],
                                target_frame=band16_frames[target_scale_key],
                                marker_name=marker_name,
                                pair_split="cross_split",
                                shared_depth_mm=shared_depth_mm,
                            )
                        )

    bidirectional = list(cross_16_to_23) + list(cross_23_to_16)
    _write_jsonl(dataset_root / "benchmark_manifest.jsonl", benchmark_manifest_rows)
    _write_jsonl(dataset_root / "pair_index_val.jsonl", val_rows)
    _write_jsonl(dataset_root / "pair_index_test.jsonl", test_rows)
    _write_jsonl(dataset_root / "pair_index_train.jsonl", [])
    _write_jsonl(dataset_root / "cross_band_16_to_23.jsonl", cross_16_to_23)
    _write_jsonl(dataset_root / "cross_band_23_to_16.jsonl", cross_23_to_16)
    _write_jsonl(dataset_root / "cross_band_16_23_bidirectional.jsonl", bidirectional)
    _write_jsonl(dataset_root / "pair_index.jsonl", list(val_rows) + list(test_rows) + bidirectional)

    summary = {
        "dataset_root": str(dataset_root),
        "manifest_state_count": int(len(benchmark_manifest_rows)),
        "pair_index_val_count": int(len(val_rows)),
        "pair_index_test_count": int(len(test_rows)),
        "cross_band_16_to_23_count": int(len(cross_16_to_23)),
        "cross_band_23_to_16_count": int(len(cross_23_to_16)),
        "cross_band_16_23_bidirectional_count": int(len(bidirectional)),
        "band16_scales": sorted({round(float(row["source_scale_mm"]), 4) for row in val_rows}),
        "band23_scales": sorted({round(float(row["source_scale_mm"]), 4) for row in test_rows}),
    }
    legacy.write_json_atomic(dataset_root / "cross_band_benchmark_summary.json", summary)
    return summary


def _iter_strict_candidates(index_path: Path) -> Iterable[tuple[tuple[Any, ...], tuple[str, str, int]]]:
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            for branch in ("source", "target"):
                unique_id = (
                    str(row[f"{branch}_frame_relpath"]),
                    str(row[f"{branch}_adapter_coord_map_relpath"]),
                    int(row["global_seq_index"]),
                )
                state_key = (
                    str(row.get("indenter", "")),
                    str(row.get("marker_name", "")),
                    str(row.get("phase_name", "")),
                    round(float(row.get("phase_progress", 0.0)), 6),
                    round(float(row["contact_x_norm"]), 6),
                    round(float(row["contact_y_norm"]), 6),
                    round(float(row["frame_depth_mm"]), 6),
                )
                yield state_key, unique_id


def inspect_existing_benchmark(dataset_root: Path) -> dict[str, Any]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    files = {
        "manifest": dataset_root / "benchmark_manifest.jsonl",
        "val": dataset_root / "pair_index_val.jsonl",
        "test": dataset_root / "pair_index_test.jsonl",
        "x16to23": dataset_root / "cross_band_16_to_23.jsonl",
        "x23to16": dataset_root / "cross_band_23_to_16.jsonl",
        "xbidir": dataset_root / "cross_band_16_23_bidirectional.jsonl",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required benchmark file {name}: {path}")

    def _count_rows(path: Path) -> int:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    pool_16: dict[tuple[Any, ...], set[tuple[str, str, int]]] = defaultdict(set)
    pool_23: dict[tuple[Any, ...], set[tuple[str, str, int]]] = defaultdict(set)
    for state_key, unique_id in _iter_strict_candidates(files["val"]):
        pool_16[state_key].add(unique_id)
    for state_key, unique_id in _iter_strict_candidates(files["test"]):
        pool_23[state_key].add(unique_id)
    shared_keys = set(pool_16.keys()) & set(pool_23.keys())
    summary = {
        "dataset_root": str(dataset_root),
        "benchmark_manifest_count": _count_rows(files["manifest"]),
        "pair_index_val_count": _count_rows(files["val"]),
        "pair_index_test_count": _count_rows(files["test"]),
        "cross_band_16_to_23_count": _count_rows(files["x16to23"]),
        "cross_band_23_to_16_count": _count_rows(files["x23to16"]),
        "cross_band_16_23_bidirectional_count": _count_rows(files["xbidir"]),
        "candidate_16_state_key_count": len(pool_16),
        "candidate_23_state_key_count": len(pool_23),
        "strict_shared_state_key_count": len(shared_keys),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def generate_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    seq_args = _build_sequence_args(args)
    logging.info(
        "Cross-band benchmark planning | dataset_root=%s indenters=%s band16=%s band23=%s episodes_per_indenter=%d",
        seq_args.dataset_root,
        _resolve_indenters(args),
        _format_scale_list(args.band16_scales_mm),
        _format_scale_list(args.band23_scales_mm),
        int(args.episodes_per_indenter),
    )
    _, indenter_bboxes, marker_files, patch_h, patch_w = _load_generation_context(seq_args)
    rng = random.Random(int(seq_args.seed))
    episode_states, _ = seqgen.build_episode_states_for_generation(
        args=seq_args,
        dataset_root=seq_args.dataset_root.resolve(),
        indenter_names=_resolve_indenters(args),
        indenter_bboxes=indenter_bboxes,
        expected_marker_files=marker_files,
        patch_h=patch_h,
        patch_w=patch_w,
        rng=rng,
        materialize_dirs=False,
        persist_initial_metadata=False,
    )

    dataset_root = seq_args.dataset_root.resolve()
    if args.clean_output and dataset_root.exists() and not args.resume:
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    plan_count = _write_plan_manifest(dataset_root, episode_states=episode_states, marker_files=marker_files)
    logging.info("Wrote benchmark plan manifest rows=%d -> %s", plan_count, dataset_root / "benchmark_manifest_plan.jsonl")

    seqgen.generate_from_scratch(seq_args)
    summary = _rebuild_benchmark_indexes(dataset_root)
    logging.info(
        "Cross-band benchmark complete | manifest=%d val=%d test=%d 16to23=%d 23to16=%d bidirectional=%d",
        int(summary["manifest_state_count"]),
        int(summary["pair_index_val_count"]),
        int(summary["pair_index_test_count"]),
        int(summary["cross_band_16_to_23_count"]),
        int(summary["cross_band_23_to_16_count"]),
        int(summary["cross_band_16_23_bidirectional_count"]),
    )
    return summary


def main() -> None:
    args = _parse_args()
    legacy.setup_logging(args.log_level)
    if args.mode == "inspect":
        inspect_existing_benchmark(args.dataset_root.resolve())
        return
    summary = generate_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
