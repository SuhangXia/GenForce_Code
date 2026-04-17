#!/usr/bin/env python3
"""Sequence-oriented multiscale tactile dataset generator for 4:3 gels.

This script is a new standalone generator built on top of the existing
`generate_multiscale_dataset.py` logic. It preserves the proven simulation
pipeline while upgrading the data model from static multi-frame samples to
full press/dwell/release sequences, plus:

- Profile A: full sequences + cross-scale pair index for adapter training
- Profile B: full sequences + clip index for sequence downstream models
- float / continuous scale support
- explicit scale split semantics, including held-out exact scales, bands,
  and explicit intervals
- metadata compatibility with TPS calibration `adapter_coord_map.npy`
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import csv
import datetime as dt
import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import random
import shlex
import shutil
import struct
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_multiscale_dataset as legacy
import rectgel_common as rectgel


DEFAULT_TRAIN_INDENTERS = [
    "cone",
    "cylinder",
    "cylinder_sh",
    "cylinder_si",
    "dotin",
    "dots",
    "hemisphere",
    "line",
    "moon",
    "prism",
    "random",
    "sphere",
]
DEFAULT_VAL_INDENTERS = ["sphere_s", "triangle"]
DEFAULT_TEST_INDENTERS = ["hexagon", "pacman", "torus", "wave"]

DEFAULT_PRECONTACT_FRAMES = (1, 1)
DEFAULT_PRESS_FRAMES = (8, 8)
DEFAULT_DWELL_FRAMES = (2, 2)
DEFAULT_RELEASE_FRAMES = (7, 7)

SEQUENCE_VARIANT = rectgel.RECTGEL_VARIANT
COORDINATE_CONVENTION = "X right positive, Y down positive, row-major"

IMAGE_INDEX_COLUMNS = [
    "dataset_root",
    "dataset_variant",
    "dataset_profile",
    "episode_id",
    "episode_dir",
    "indenter_name",
    "indenter_split",
    "is_unseen_indenter",
    "scale_key",
    "scale_requested_mm",
    "scale_simulated_mm",
    "gel_width_mm",
    "gel_height_mm",
    "scale_split",
    "is_unseen_scale",
    "global_seq_index",
    "phase_name",
    "phase_index",
    "phase_progress",
    "frame_name",
    "frame_actual_max_down_mm",
    "command_x_mm",
    "command_y_mm",
    "command_x_norm",
    "command_y_norm",
    "marker_name",
    "image_relpath",
    "image_abspath",
    "adapter_coord_map_relpath",
    "adapter_coord_map_abspath",
]


@dataclass
class TemporalPlan:
    precontact_frame_count: int
    press_frame_count: int
    dwell_frame_count: int
    release_frame_count: int
    press_sampling_mode: str
    release_mode_requested: str
    release_mode_actual: str
    final_depth_mm: float
    include_dwell: bool
    include_release: bool


@dataclass
class SequenceFrameRecord:
    global_seq_index: int
    frame_name: str
    phase_name: str
    phase_index: int
    phase_progress: float
    source_physics_frame_index: int
    frame_fraction_requested: float | None
    frame_fraction_actual: float | None
    frame_target_max_down_mm: float | None
    frame_actual_max_down_mm: float
    deformation_stats: Dict[str, float]
    rendered_markers: List[str] = field(default_factory=list)
    is_synthetic_frame: bool = False
    is_synthetic_precontact: bool = False
    is_synthetic_release: bool = False
    is_dwell_repeat: bool = False

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "global_seq_index": int(self.global_seq_index),
            "frame_name": self.frame_name,
            "phase_name": self.phase_name,
            "phase_index": int(self.phase_index),
            "phase_progress": float(self.phase_progress),
            "source_physics_frame_index": int(self.source_physics_frame_index),
            "frame_fraction_requested": None if self.frame_fraction_requested is None else float(self.frame_fraction_requested),
            "frame_fraction_actual": None if self.frame_fraction_actual is None else float(self.frame_fraction_actual),
            "frame_target_max_down_mm": None if self.frame_target_max_down_mm is None else float(self.frame_target_max_down_mm),
            "frame_actual_max_down_mm": float(self.frame_actual_max_down_mm),
            "deformation_stats": legacy._json_safe(self.deformation_stats),
            "rendered_markers": [str(v) for v in self.rendered_markers],
            "is_synthetic_frame": bool(self.is_synthetic_frame),
            "is_synthetic_precontact": bool(self.is_synthetic_precontact),
            "is_synthetic_release": bool(self.is_synthetic_release),
            "is_dwell_repeat": bool(self.is_dwell_repeat),
        }


@dataclass
class SequenceScaleState:
    episode_id: int
    scale_requested_mm: float
    scale_simulated_mm: float
    scale_quantized: bool
    scale_key: str
    scale_dir: Path
    temp_scale_dir: Path
    contact_x_mm: float
    contact_y_mm: float
    contact_x_norm: float
    contact_y_norm: float
    indenter_split: str
    scale_split: str
    is_unseen_indenter: bool
    is_unseen_scale: bool
    trajectory_length: int
    deformation_stats_final: Dict[str, float]
    camera_mode: str
    camera_distance_m: float
    camera_fov_deg: float
    adapter_coord_map_path: Path
    adapter_coord_map_shape: List[int]
    frames: List[SequenceFrameRecord]
    physics_npz_path: Path
    physics_reused: bool = False
    physics_npz_deleted: bool = False
    physics_npz_cleanup_policy: str = "keep"
    failed: bool = False
    sealed: bool = False


@dataclass
class SequenceEpisodeState:
    episode_id: int
    indenter: str
    indenter_split: str
    is_unseen_indenter: bool
    contact_x_norm: float
    contact_y_norm: float
    final_depth_mm: float
    position_sample_type_requested: str
    position_sample_type_actual: str
    contact_margin_constraint_mm: float
    episode_dir: Path
    temporal_plan: TemporalPlan
    scales: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def format_scale_key(scale_mm: float) -> str:
    value = 0.0 if abs(float(scale_mm)) < 5e-8 else float(scale_mm)
    return f"scale_{value:.4f}".replace("-", "m").replace(".", "p") + "mm"


def make_scale_progress_key(episode_id: int, scale_key: str) -> str:
    return f"episode_{int(episode_id):06d}/{scale_key}"


def parse_scale_intervals_mm(values: Sequence[str] | None, arg_name: str) -> List[Tuple[float, float]]:
    if not values:
        return []
    intervals: List[Tuple[float, float]] = []
    for raw in values:
        text = str(raw).strip()
        if ":" not in text:
            raise ValueError(f"{arg_name} expects interval specs like 15:18.5, got {raw}")
        start_s, end_s = text.split(":", 1)
        start = float(start_s)
        end = float(end_s)
        if end <= start:
            raise ValueError(f"{arg_name} interval must satisfy end > start, got {raw}")
        intervals.append((start, end))

    intervals_sorted = sorted(intervals)
    for idx in range(1, len(intervals_sorted)):
        prev_start, prev_end = intervals_sorted[idx - 1]
        cur_start, cur_end = intervals_sorted[idx]
        if cur_start < prev_end - 1e-9:
            raise ValueError(
                f"{arg_name} intervals overlap: ({prev_start}, {prev_end}) and ({cur_start}, {cur_end})"
            )
    return intervals_sorted


def scale_in_interval(scale_mm: float, interval: Tuple[float, float]) -> bool:
    lo, hi = interval
    value = float(scale_mm)
    return (lo - 1e-9) <= value < (hi - 1e-9)


def quantize_scale_if_needed(scale_requested_mm: float, quantization_mm: float | None) -> Tuple[float, bool]:
    requested = float(scale_requested_mm)
    if requested <= 0:
        raise ValueError(f"scale must be > 0, got {scale_requested_mm}")
    if quantization_mm is None or quantization_mm <= 0:
        return requested, False
    quant = float(quantization_mm)
    simulated = round(requested / quant) * quant
    simulated = round(float(simulated), 4)
    return simulated, abs(simulated - requested) > 5e-8


def compute_fixed_camera_distance_m(reference_scale_mm: float, base_fov_deg: float, distance_safety: float) -> float:
    gel_dims = rectgel.gel_dims_from_width_mm(reference_scale_mm)
    fit_width_m = rectgel.compute_full_sensor_fit_width_m(
        gel_width_m=float(gel_dims.width_mm) / 1000.0,
        gel_height_m=float(gel_dims.height_mm) / 1000.0,
        render_width_px=int(legacy.DEFAULT_IMAGE_RES[0]),
        render_height_px=int(legacy.DEFAULT_IMAGE_RES[1]),
    )
    ref_fov_rad = math.radians(float(base_fov_deg))
    distance = (fit_width_m / 2.0) / math.tan(ref_fov_rad / 2.0)
    return float(distance * float(distance_safety))


def compute_camera_params_for_scale(
    *,
    camera_mode: str,
    scale_mm: float,
    base_fov_deg: float,
    fixed_distance_m: float,
    distance_safety: float,
) -> Tuple[float, float]:
    gel_dims = rectgel.gel_dims_from_width_mm(scale_mm)
    fit_width_m = rectgel.compute_full_sensor_fit_width_m(
        gel_width_m=float(gel_dims.width_mm) / 1000.0,
        gel_height_m=float(gel_dims.height_mm) / 1000.0,
        render_width_px=int(legacy.DEFAULT_IMAGE_RES[0]),
        render_height_px=int(legacy.DEFAULT_IMAGE_RES[1]),
    )
    if camera_mode == "fixed_distance_variable_fov":
        distance_m = float(fixed_distance_m)
        fov_deg = math.degrees(2.0 * math.atan((fit_width_m / 2.0) / distance_m))
        return distance_m, float(fov_deg)
    if camera_mode == "fixed_fov_variable_distance":
        fov_deg = float(base_fov_deg)
        distance_m = (fit_width_m / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        distance_m *= float(distance_safety)
        return float(distance_m), fov_deg
    raise ValueError(f"Unsupported camera_mode: {camera_mode}")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def _progress_eta_text(stage_start_ts: float, done: int, total: int) -> str:
    elapsed = max(0.0, time.time() - float(stage_start_ts))
    if total <= 0:
        return f"progress=0/0 elapsed={_format_duration(elapsed)} eta=-- eta_at=--"
    done = max(0, min(int(done), int(total)))
    if done == 0:
        return f"progress=0/{total} elapsed={_format_duration(elapsed)} eta=-- eta_at=--"
    avg_sec = elapsed / float(done)
    remaining = max(int(total) - done, 0)
    eta_sec = avg_sec * float(remaining)
    eta_at = (dt.datetime.now() + dt.timedelta(seconds=eta_sec)).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"progress={done}/{total} elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta_sec)} eta_at={eta_at}"
    )


def _throughput_text(stage_start_ts: float, done: int) -> str:
    elapsed = max(1e-6, time.time() - float(stage_start_ts))
    rate = float(done) / elapsed
    return f"throughput={rate:.3f}/s"


def resolve_render_backlog_thresholds(args: argparse.Namespace) -> Tuple[int, int]:
    high = int(args.render_backlog_high_watermark)
    low = int(args.render_backlog_low_watermark)
    if high <= 0:
        high = max(8, int(args.max_render_workers) * 4)
    if low <= 0:
        low = max(int(args.max_render_workers), high // 2)
    if low > high:
        low = high
    return high, low


def inspect_binary_stl(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        size = int(path.stat().st_size)
        if size < 84:
            return False, f"too_small:{size}"
        with open(path, "rb") as f:
            header = f.read(84)
        if len(header) < 84:
            return False, f"short_header:{len(header)}"
        tri_count = int(struct.unpack("<I", header[80:84])[0])
        if tri_count <= 0:
            return False, f"zero_triangles:{tri_count}"
        expected_size = 84 + 50 * tri_count
        if size != expected_size:
            return False, f"size_mismatch:{size}!={expected_size}"
        return True, "ok"
    except Exception as exc:
        return False, f"inspect_error:{exc}"


def build_sequence_blender_script(*, render_device: str, render_gpu_backend: str) -> str:
    script = str(legacy.BLENDER_TEMP_SCRIPT)
    script = script.replace(
        "DEFAULT_IMAGE_RES = (568, 568)",
        f"DEFAULT_IMAGE_RES = ({int(legacy.DEFAULT_IMAGE_RES[0])}, {int(legacy.DEFAULT_IMAGE_RES[1])})",
    )
    script = script.replace(
        '    p.add_argument("--scale-mm", type=float, required=True)\n',
        '    p.add_argument("--scale-mm", type=float, required=True)\n'
        '    p.add_argument("--gel-height-mm", type=float, default=None)\n',
    )
    script = script.replace(
        "def apply_uv_math(obj, cx: float, cy: float, scale_m: float, uv_inset_ratio: float):\n",
        "def apply_uv_math(obj, cx: float, cy: float, scale_m: float, uv_inset_ratio: float, gel_height_m=None):\n",
    )
    script = script.replace(
        '    if scale_m <= 0:\n'
        '        raise ValueError("scale_m must be > 0")\n'
        '    if uv_inset_ratio < 0.0 or uv_inset_ratio >= 0.49:\n',
        '    if scale_m <= 0:\n'
        '        raise ValueError("scale_m must be > 0")\n'
        '    if gel_height_m is None:\n'
        '        gel_height_m = scale_m\n'
        '    if gel_height_m <= 0:\n'
        '        raise ValueError("gel_height_m must be > 0")\n'
        '    if uv_inset_ratio < 0.0 or uv_inset_ratio >= 0.49:\n',
    )
    script = script.replace(
        '            u = (w.x - cx) / scale_m + 0.5\n'
        '            v = (w.y - cy) / scale_m + 0.5\n',
        '            u = (w.x - cx) / scale_m + 0.5\n'
        '            v = (w.y - cy) / gel_height_m + 0.5\n',
    )
    script = script.replace(
        "def compute_full_sensor_fit_scale_m(scale_m: float, res_x: int, res_y: int) -> float:\n",
        "def compute_full_sensor_fit_scale_m(scale_m: float, res_x: int, res_y: int, gel_height_m=None) -> float:\n",
    )
    script = script.replace(
        '    # The gel is square, but the render is 4:3. With a horizontal-fit camera,\n'
        '    # matching the square width would crop the square vertically. Inflate the\n'
        '    # fitted span by the render aspect so the full square stays visible.\n'
        '    aspect = float(res_x) / float(res_y)\n'
        '    return float(scale_m) * max(1.0, aspect)\n',
        '    if gel_height_m is None:\n'
        '        gel_height_m = scale_m\n'
        '    if gel_height_m <= 0:\n'
        '        raise ValueError("gel_height_m must be > 0")\n'
        '    aspect = float(res_x) / float(res_y)\n'
        '    return max(float(scale_m), float(gel_height_m) * aspect)\n',
    )
    script = script.replace(
        "def setup_camera(\n    cx: float,\n    cy: float,\n    z_top: float,\n    scale_m: float,\n",
        "def setup_camera(\n    cx: float,\n    cy: float,\n    z_top: float,\n    scale_m: float,\n    gel_height_m,\n",
    )
    script = script.replace(
        '    fit_scale_m = compute_full_sensor_fit_scale_m(\n'
        '        scale_m,\n'
        '        DEFAULT_IMAGE_RES[0],\n'
        '        DEFAULT_IMAGE_RES[1],\n'
        '    )\n',
        '    if gel_height_m is None:\n'
        '        gel_height_m = scale_m\n'
        '    fit_scale_m = compute_full_sensor_fit_scale_m(\n'
        '        scale_m,\n'
        '        DEFAULT_IMAGE_RES[0],\n'
        '        DEFAULT_IMAGE_RES[1],\n'
        '        gel_height_m,\n'
        '    )\n',
    )
    script = script.replace(
        '        f"Camera setup | mode={camera_mode} scale_mm={scale_m*1000.0:.1f} "\n',
        '        f"Camera setup | mode={camera_mode} gel_width_mm={scale_m*1000.0:.1f} gel_height_mm={gel_height_m*1000.0:.1f} "\n',
    )
    script = script.replace(
        'apply_uv_math(obj, cx, cy, scale_m, args.uv_inset_ratio)\n',
        'apply_uv_math(\n'
        '            obj,\n'
        '            cx,\n'
        '            cy,\n'
        '            scale_m,\n'
        '            args.uv_inset_ratio,\n'
        '            None if args.gel_height_mm is None else args.gel_height_mm / 1000.0,\n'
        '        )\n',
    )
    script = script.replace(
        '    setup_camera(\n'
        '        cx,\n'
        '        cy,\n'
        '        z_top,\n'
        '        scale_m,\n'
        '        args.camera_mode,\n'
        '        args.base_fov_deg,\n'
        '        args.fixed_distance_m,\n'
        '        args.distance_safety,\n'
        '    )\n',
        '    setup_camera(\n'
        '        cx,\n'
        '        cy,\n'
        '        z_top,\n'
        '        scale_m,\n'
        '        None if args.gel_height_mm is None else args.gel_height_mm / 1000.0,\n'
        '        args.camera_mode,\n'
        '        args.base_fov_deg,\n'
        '        args.fixed_distance_m,\n'
        '        args.distance_safety,\n'
        '    )\n',
    )
    script = script.replace(
        '    p.add_argument("--render-samples", type=int, default=32)\n',
        '    p.add_argument("--render-samples", type=int, default=32)\n'
        '    p.add_argument("--render-device", choices=["cpu", "gpu"], default="cpu")\n'
        '    p.add_argument(\n'
        '        "--render-gpu-backend",\n'
        '        choices=["auto", "optix", "cuda", "hip", "oneapi", "metal"],\n'
        '        default="auto",\n'
        '    )\n',
    )
    script = script.replace(
        "def configure_render(out_dir: Path, render_samples: int):\n",
        "def configure_render(out_dir: Path, render_samples: int, render_device: str, render_gpu_backend: str):\n",
    )
    script = script.replace(
        '    if hasattr(scene, "cycles"):\n'
        '        scene.cycles.samples = max(1, int(render_samples))\n'
        '        scene.cycles.use_adaptive_sampling = True\n'
        '        scene.cycles.max_bounces = 4\n'
        '        scene.cycles.device = "CPU"\n',
        '    if hasattr(scene, "cycles"):\n'
        '        scene.cycles.samples = max(1, int(render_samples))\n'
        '        scene.cycles.use_adaptive_sampling = True\n'
        '        scene.cycles.max_bounces = 4\n'
        '        scene.cycles.device = "CPU"\n'
        '        if str(render_device).lower() == "gpu":\n'
        '            selected_backend = None\n'
        '            backend_arg = str(render_gpu_backend).strip().upper()\n'
        '            backend_candidates = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if backend_arg in {"", "AUTO"} else [backend_arg]\n'
        '            cycles_addon = bpy.context.preferences.addons.get("cycles")\n'
        '            if cycles_addon is None:\n'
        '                print("Cycles addon not available; falling back to CPU render")\n'
        '            else:\n'
        '                prefs = cycles_addon.preferences\n'
        '                for backend_name in backend_candidates:\n'
        '                    try:\n'
        '                        prefs.compute_device_type = backend_name\n'
        '                        if hasattr(prefs, "refresh_devices"):\n'
        '                            prefs.refresh_devices()\n'
        '                        elif hasattr(prefs, "get_devices"):\n'
        '                            prefs.get_devices()\n'
        '                        devices = list(getattr(prefs, "devices", []))\n'
        '                        gpu_devices = [dev for dev in devices if str(getattr(dev, "type", "")).upper() != "CPU"]\n'
        '                        if not gpu_devices:\n'
        '                            continue\n'
        '                        for dev in devices:\n'
        '                            dev.use = str(getattr(dev, "type", "")).upper() != "CPU"\n'
        '                        scene.cycles.device = "GPU"\n'
        '                        selected_backend = backend_name\n'
        '                        break\n'
        '                    except Exception as exc:\n'
        '                        print(f"GPU backend unavailable: {backend_name} ({exc})")\n'
        '                if selected_backend is None:\n'
        '                    print("No usable GPU backend found; falling back to CPU render")\n'
        '                else:\n'
        '                    print(f"Cycles render device=GPU backend={selected_backend}")\n'
        '        else:\n'
        '            print("Cycles render device=CPU")\n',
    )
    script = script.replace(
        "    configure_render(output_dir, int(args.render_samples))\n",
        "    configure_render(output_dir, int(args.render_samples), str(args.render_device), str(args.render_gpu_backend))\n",
    )
    return script


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multiscale tactile sequence datasets for 4:3 rectangular gels.")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--dataset-profile", choices=["A", "B"], required=True)
    p.add_argument("--particle", type=str, default="100000")
    p.add_argument("--episodes-per-indenter", type=int, default=1)
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--patch-grid", type=str, default=legacy.DEFAULT_PATCH_GRID)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--calibration-bundle-hint-path", type=str, default="")
    p.add_argument("--reference-throughput-images-per-hour", type=float, default=5714.0)
    p.add_argument("--reference-image-kb", type=float, default=159.0)
    p.add_argument("--reference-npz-mb", type=float, default=189.0)
    p.add_argument("--reference-stl-mb", type=float, default=0.95)
    p.add_argument(
        "--physics-npz-cleanup",
        choices=["keep", "delete_after_scale_complete"],
        default="keep",
    )

    p.add_argument("--x-min", type=float, default=legacy.DEFAULT_X_RANGE[0])
    p.add_argument("--x-max", type=float, default=legacy.DEFAULT_X_RANGE[1])
    p.add_argument("--y-min", type=float, default=legacy.DEFAULT_Y_RANGE[0])
    p.add_argument("--y-max", type=float, default=legacy.DEFAULT_Y_RANGE[1])
    p.add_argument("--position-clean-ratio", type=float, default=legacy.DEFAULT_POSITION_SAMPLE_RATIOS["clean"])
    p.add_argument(
        "--position-near-boundary-ratio",
        type=float,
        default=legacy.DEFAULT_POSITION_SAMPLE_RATIOS["near_boundary"],
    )
    p.add_argument(
        "--position-partial-crop-ratio",
        type=float,
        default=legacy.DEFAULT_POSITION_SAMPLE_RATIOS["partial_crop"],
    )
    p.add_argument("--near-boundary-max-margin-mm", type=float, default=legacy.DEFAULT_NEAR_BOUNDARY_MAX_MARGIN_MM)
    p.add_argument("--partial-crop-min-overhang-mm", type=float, default=legacy.DEFAULT_PARTIAL_CROP_MIN_OVERHANG_MM)
    p.add_argument("--partial-crop-max-overhang-mm", type=float, default=legacy.DEFAULT_PARTIAL_CROP_MAX_OVERHANG_MM)
    p.add_argument("--depth-min", type=float, default=legacy.DEFAULT_DEPTH_RANGE[0])
    p.add_argument("--depth-max", type=float, default=legacy.DEFAULT_DEPTH_RANGE[1])

    p.add_argument("--scale-mode", choices=["discrete", "continuous_uniform", "continuous_lhs"], default="discrete")
    p.add_argument("--gel-widths-mm", type=float, nargs="*", default=[float(v) for v in legacy.DEFAULT_SCALES_MM])
    p.add_argument("--gel-width-min-mm", type=float, default=None)
    p.add_argument("--gel-width-max-mm", type=float, default=None)
    p.add_argument("--scales-per-episode", type=int, default=0)
    p.add_argument("--continuous-scale-quantization-mm", type=float, default=None)
    p.add_argument("--heldout-gel-widths-mm", type=float, nargs="*", default=None)
    p.add_argument("--heldout-gel-width-band-mm", type=float, default=0.0)
    p.add_argument("--heldout-val-gel-widths-mm", type=float, nargs="*", default=None)
    p.add_argument("--heldout-test-gel-widths-mm", type=float, nargs="*", default=None)
    p.add_argument("--heldout-val-gel-width-band-mm", type=float, default=0.0)
    p.add_argument("--heldout-test-gel-width-band-mm", type=float, default=0.0)
    p.add_argument("--train-gel-width-intervals-mm", nargs="*", default=None)
    p.add_argument("--val-gel-width-intervals-mm", nargs="*", default=None)
    p.add_argument("--test-gel-width-intervals-mm", nargs="*", default=None)
    p.add_argument("--allowed-scale-splits", nargs="*", choices=["train", "val", "test"], default=None)

    p.add_argument("--precontact-frames-min", type=int, default=DEFAULT_PRECONTACT_FRAMES[0])
    p.add_argument("--precontact-frames-max", type=int, default=DEFAULT_PRECONTACT_FRAMES[1])
    p.add_argument("--press-frames-min", type=int, default=DEFAULT_PRESS_FRAMES[0])
    p.add_argument("--press-frames-max", type=int, default=DEFAULT_PRESS_FRAMES[1])
    p.add_argument("--dwell-frames-min", type=int, default=DEFAULT_DWELL_FRAMES[0])
    p.add_argument("--dwell-frames-max", type=int, default=DEFAULT_DWELL_FRAMES[1])
    p.add_argument("--release-frames-min", type=int, default=DEFAULT_RELEASE_FRAMES[0])
    p.add_argument("--release-frames-max", type=int, default=DEFAULT_RELEASE_FRAMES[1])
    p.add_argument("--press-sampling-mode", choices=["uniform_index", "uniform_max_down"], default="uniform_index")
    p.add_argument("--release-mode", choices=["auto", "mirror_loading"], default="auto")
    p.set_defaults(include_dwell=True, include_release=True)
    p.add_argument("--include-dwell", dest="include_dwell", action="store_true")
    p.add_argument("--no-include-dwell", dest="include_dwell", action="store_false")
    p.add_argument("--include-release", dest="include_release", action="store_true")
    p.add_argument("--no-include-release", dest="include_release", action="store_false")

    p.add_argument("--derive-clips-from-existing-dataset-root", type=Path, default=None)
    p.add_argument("--derive-storage-mode", choices=["index_only", "symlink"], default="index_only")
    p.add_argument("--clip-length", type=int, default=3)
    p.add_argument("--clip-window-stride", type=int, default=1)
    p.add_argument("--clip-gap-min", type=int, default=0)
    p.add_argument("--clip-gap-max", type=int, default=0)
    p.add_argument(
        "--clip-phase-policy",
        choices=["all", "press_only", "press_dwell", "balanced"],
        default="all",
    )
    p.add_argument("--clip-balance-seed", type=int, default=None)
    p.add_argument("--max-clips-per-scale", type=int, default=0)

    p.add_argument("--pair-marker-policy", choices=["same_texture_only", "all_texture_pairs"], default="same_texture_only")
    p.add_argument("--include-identity-pairs", action="store_true")
    p.add_argument("--pair-split-policy", choices=["same_split_only", "all"], default="same_split_only")

    p.add_argument("--train-indenters", nargs="*", default=DEFAULT_TRAIN_INDENTERS)
    p.add_argument("--val-indenters", nargs="*", default=DEFAULT_VAL_INDENTERS)
    p.add_argument("--test-indenters", nargs="*", default=DEFAULT_TEST_INDENTERS)
    p.add_argument("--marker-texture-names", nargs="*", default=None)
    p.add_argument("--marker-texture-count", type=int, default=0)
    p.add_argument("--marker-texture-seed", type=int, default=None)

    p.add_argument(
        "--camera-mode",
        choices=["fixed_distance_variable_fov", "fixed_fov_variable_distance"],
        default="fixed_distance_variable_fov",
    )
    p.add_argument("--fov-deg", type=float, default=legacy.DEFAULT_FOV_DEG)
    p.add_argument("--reference-gel-width-mm", type=float, default=legacy.DEFAULT_REFERENCE_SCALE_MM)
    p.add_argument("--camera-distance-m", type=float, default=None)
    p.add_argument("--distance-safety", type=float, default=legacy.DEFAULT_DISTANCE_SAFETY)
    p.add_argument("--uv-inset-ratio", type=float, default=0.01)
    p.add_argument("--uv-mode", choices=["unwrap_genforce", "physical_math"], default="physical_math")
    p.add_argument("--force-black-side-border-px", type=int, default=2)
    p.add_argument("--render-width", type=int, default=rectgel.DEFAULT_RENDER_RES[0])
    p.add_argument("--render-height", type=int, default=rectgel.DEFAULT_RENDER_RES[1])

    # Conservative defaults for a single-GPU workstation:
    # keep GPU-bound physics saturated, but avoid letting CPU-side Blender workers
    # starve physics and trigger false timeouts under the pipelined scheduler.
    p.add_argument("--max-physics-workers", type=int, default=6)
    p.add_argument("--max-meshing-workers", type=int, default=2)
    p.add_argument("--max-render-workers", type=int, default=4)
    p.add_argument("--physics-timeout-sec", type=int, default=900)
    p.add_argument("--meshing-timeout-sec", type=int, default=600)
    p.add_argument("--render-timeout-sec", type=int, default=900)
    p.add_argument("--render-samples", type=int, default=32)
    p.add_argument("--render-device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument(
        "--render-gpu-backend",
        choices=["auto", "optix", "cuda", "hip", "oneapi", "metal"],
        default="auto",
    )
    p.add_argument("--auto-balance-pipeline", action="store_true")
    p.add_argument("--render-backlog-high-watermark", type=int, default=0)
    p.add_argument("--render-backlog-low-watermark", type=int, default=0)
    p.add_argument("--python-cmd", type=str, default=sys.executable)
    p.add_argument("--blender-cmd", type=str, default="blender")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p.parse_args()


def normalize_runtime_defaults(args: argparse.Namespace) -> None:
    args.scales_mm = [float(v) for v in (args.gel_widths_mm or [])]
    args.scale_min_mm = None if args.gel_width_min_mm is None else float(args.gel_width_min_mm)
    args.scale_max_mm = None if args.gel_width_max_mm is None else float(args.gel_width_max_mm)
    args.reference_scale_mm = float(args.reference_gel_width_mm)
    args.heldout_scales_mm = None if args.heldout_gel_widths_mm is None else [float(v) for v in args.heldout_gel_widths_mm]
    args.heldout_scale_band_mm = float(args.heldout_gel_width_band_mm)
    args.heldout_val_scales_mm = None if args.heldout_val_gel_widths_mm is None else [float(v) for v in args.heldout_val_gel_widths_mm]
    args.heldout_test_scales_mm = None if args.heldout_test_gel_widths_mm is None else [float(v) for v in args.heldout_test_gel_widths_mm]
    args.heldout_val_scale_band_mm = float(args.heldout_val_gel_width_band_mm)
    args.heldout_test_scale_band_mm = float(args.heldout_test_gel_width_band_mm)
    args.train_scale_intervals_mm = list(args.train_gel_width_intervals_mm or [])
    args.val_scale_intervals_mm = list(args.val_gel_width_intervals_mm or [])
    args.test_scale_intervals_mm = list(args.test_gel_width_intervals_mm or [])
    if args.allowed_scale_splits is None:
        args.allowed_scale_splits = ["train"] if args.dataset_profile == "A" else ["train", "val", "test"]
    else:
        args.allowed_scale_splits = [str(v) for v in args.allowed_scale_splits]
    if args.clip_balance_seed is None:
        args.clip_balance_seed = int(args.seed)
    if args.marker_texture_seed is None:
        args.marker_texture_seed = int(args.seed)


def validate_args(args: argparse.Namespace) -> None:
    normalize_runtime_defaults(args)
    if args.resume and args.clean_output:
        raise ValueError("--resume and --clean-output cannot be used together")
    if args.episodes_per_indenter <= 0:
        raise ValueError("--episodes-per-indenter must be > 0")
    if args.depth_min <= 0 or args.depth_max <= 0 or args.depth_min > args.depth_max:
        raise ValueError("Invalid depth range")
    if args.position_clean_ratio < 0 or args.position_near_boundary_ratio < 0 or args.position_partial_crop_ratio < 0:
        raise ValueError("Position sampling ratios must be >= 0")
    if args.position_clean_ratio + args.position_near_boundary_ratio + args.position_partial_crop_ratio <= 0:
        raise ValueError("At least one position sampling ratio must be > 0")
    if args.near_boundary_max_margin_mm < 0:
        raise ValueError("--near-boundary-max-margin-mm must be >= 0")
    if args.partial_crop_min_overhang_mm < 0:
        raise ValueError("--partial-crop-min-overhang-mm must be >= 0")
    if args.partial_crop_max_overhang_mm < args.partial_crop_min_overhang_mm:
        raise ValueError("--partial-crop-max-overhang-mm must be >= --partial-crop-min-overhang-mm")
    if args.max_physics_workers < 0 or args.max_meshing_workers <= 0 or args.max_render_workers <= 0:
        raise ValueError("Worker counts must satisfy: physics >= 0, meshing > 0, render > 0")
    if args.max_physics_workers == 0 and not args.resume:
        raise ValueError("--max-physics-workers 0 is only supported together with --resume (backlog-drain mode)")
    if args.render_backlog_high_watermark < 0 or args.render_backlog_low_watermark < 0:
        raise ValueError("Render backlog watermarks must be >= 0")
    if (
        args.render_backlog_high_watermark > 0
        and args.render_backlog_low_watermark > 0
        and args.render_backlog_low_watermark > args.render_backlog_high_watermark
    ):
        raise ValueError("--render-backlog-low-watermark must be <= --render-backlog-high-watermark")
    if args.physics_timeout_sec <= 0 or args.meshing_timeout_sec <= 0 or args.render_timeout_sec <= 0:
        raise ValueError("Timeouts must be > 0")
    if args.render_samples <= 0:
        raise ValueError("--render-samples must be > 0")
    if args.reference_scale_mm <= 0:
        raise ValueError("--reference-gel-width-mm must be > 0")
    if args.fov_deg <= 0 or args.fov_deg >= 179:
        raise ValueError("--fov-deg must be in (0, 179)")
    if args.distance_safety <= 0:
        raise ValueError("--distance-safety must be > 0")
    if args.uv_inset_ratio < 0.0 or args.uv_inset_ratio >= 0.49:
        raise ValueError("--uv-inset-ratio must be in [0.0, 0.49)")
    if args.force_black_side_border_px < 0:
        raise ValueError("--force-black-side-border-px must be >= 0")
    if args.render_width <= 0 or args.render_height <= 0:
        raise ValueError("--render-width and --render-height must be > 0")
    if args.clip_length <= 0:
        raise ValueError("--clip-length must be > 0")
    if args.clip_window_stride <= 0:
        raise ValueError("--clip-window-stride must be > 0")
    if args.clip_gap_min < 0 or args.clip_gap_max < args.clip_gap_min:
        raise ValueError("Invalid clip gap range")
    if args.max_clips_per_scale < 0:
        raise ValueError("--max-clips-per-scale must be >= 0")
    if args.marker_texture_count < 0:
        raise ValueError("--marker-texture-count must be >= 0")
    if args.reference_throughput_images_per_hour <= 0:
        raise ValueError("--reference-throughput-images-per-hour must be > 0")
    if args.reference_image_kb <= 0:
        raise ValueError("--reference-image-kb must be > 0")
    if args.reference_npz_mb <= 0:
        raise ValueError("--reference-npz-mb must be > 0")
    if args.reference_stl_mb <= 0:
        raise ValueError("--reference-stl-mb must be > 0")
    for name in ("precontact", "press", "dwell", "release"):
        lo = getattr(args, f"{name}_frames_min")
        hi = getattr(args, f"{name}_frames_max")
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid {name} frame count range")

    if args.dataset_profile == "A" and args.derive_clips_from_existing_dataset_root is not None:
        raise ValueError("--derive-clips-from-existing-dataset-root is only valid for --dataset-profile B")

    if args.scale_mode == "discrete":
        if not args.scales_mm:
            raise ValueError("--gel-widths-mm is required in discrete mode")
    else:
        if args.scale_min_mm is None or args.scale_max_mm is None:
            raise ValueError("--gel-width-min-mm and --gel-width-max-mm are required in continuous modes")
        if args.scale_max_mm <= args.scale_min_mm:
            raise ValueError("--gel-width-max-mm must be > --gel-width-min-mm")
        if args.scales_per_episode <= 0:
            raise ValueError("--scales-per-episode must be > 0 in continuous modes")
    if args.continuous_scale_quantization_mm is not None and args.continuous_scale_quantization_mm <= 0:
        raise ValueError("--continuous-scale-quantization-mm must be > 0")
    if args.heldout_scale_band_mm < 0 or args.heldout_val_scale_band_mm < 0 or args.heldout_test_scale_band_mm < 0:
        raise ValueError("Held-out scale band widths must be >= 0")
    if not args.allowed_scale_splits:
        raise ValueError("--allowed-scale-splits must not be empty")

    all_split_indenters = set(args.train_indenters) | set(args.val_indenters) | set(args.test_indenters)
    if len(all_split_indenters) != (len(set(args.train_indenters)) + len(set(args.val_indenters)) + len(set(args.test_indenters))):
        raise ValueError("train/val/test indenter sets must be disjoint")

    explicit_interval_mode = bool(args.train_scale_intervals_mm or args.val_scale_intervals_mm or args.test_scale_intervals_mm)
    heldout_old_mode = bool(args.heldout_scales_mm or args.heldout_scale_band_mm > 0)
    heldout_new_mode = bool(
        args.heldout_val_scales_mm
        or args.heldout_test_scales_mm
        or args.heldout_val_scale_band_mm > 0
        or args.heldout_test_scale_band_mm > 0
    )
    if explicit_interval_mode and (heldout_old_mode or heldout_new_mode):
        raise ValueError("Do not mix explicit scale intervals with held-out exact/band arguments")
    if heldout_old_mode and heldout_new_mode:
        logging.warning("Both legacy held-out args and new val/test held-out args were provided; new val/test held-out settings take priority")

    val_scales = [float(v) for v in (args.heldout_val_scales_mm or [])]
    test_scales = [float(v) for v in (args.heldout_test_scales_mm or [])]
    for val_center in val_scales:
        for test_center in test_scales:
            val_lo = float(val_center) - float(args.heldout_val_scale_band_mm)
            val_hi = float(val_center) + float(args.heldout_val_scale_band_mm)
            test_lo = float(test_center) - float(args.heldout_test_scale_band_mm)
            test_hi = float(test_center) + float(args.heldout_test_scale_band_mm)
            if min(val_hi, test_hi) >= max(val_lo, test_lo) - 1e-9:
                raise ValueError("Held-out val/test scale bands overlap")


def determine_indenter_split(indenter_name: str, args: argparse.Namespace) -> Tuple[str, bool]:
    if indenter_name in set(args.train_indenters):
        return "train", False
    if indenter_name in set(args.val_indenters):
        return "val", False
    if indenter_name in set(args.test_indenters):
        return "test", True
    raise ValueError(
        f"Indenter '{indenter_name}' is not assigned to any split. "
        "Pass --train-indenters/--val-indenters/--test-indenters explicitly."
    )


def determine_scale_split(
    scale_simulated_mm: float,
    *,
    args: argparse.Namespace,
) -> Tuple[str, bool]:
    value = float(scale_simulated_mm)
    explicit_intervals = {
        "train": list(getattr(args, "train_scale_intervals_mm_parsed", [])),
        "val": list(getattr(args, "val_scale_intervals_mm_parsed", [])),
        "test": list(getattr(args, "test_scale_intervals_mm_parsed", [])),
    }
    if any(explicit_intervals.values()):
        matches = [
            split_name
            for split_name, intervals in explicit_intervals.items()
            for interval in intervals
            if scale_in_interval(value, interval)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Scale {value:.4f}mm must match exactly one explicit scale interval; got matches={matches}"
            )
        split_name = matches[0]
        return split_name, split_name != "train"

    if args.heldout_val_scales_mm is not None or args.heldout_test_scales_mm is not None:
        heldout_val_scales = [float(v) for v in (args.heldout_val_scales_mm or [])]
        heldout_test_scales = [float(v) for v in (args.heldout_test_scales_mm or [])]
        heldout_val_band = float(args.heldout_val_scale_band_mm)
        heldout_test_band = float(args.heldout_test_scale_band_mm)
    else:
        heldout_val_scales = []
        heldout_test_scales = [float(v) for v in (args.heldout_scales_mm or [])]
        heldout_val_band = 0.0
        heldout_test_band = float(args.heldout_scale_band_mm)

    for heldout in heldout_val_scales:
        center = float(heldout)
        if heldout_val_band > 0:
            if abs(value - center) <= heldout_val_band + 1e-9:
                return "val", True
        elif abs(value - center) <= 1e-9:
            return "val", True
    for heldout in heldout_test_scales:
        center = float(heldout)
        if heldout_test_band > 0:
            if abs(value - center) <= heldout_test_band + 1e-9:
                return "test", True
        elif abs(value - center) <= 1e-9:
            return "test", True
    return "train", False


def sample_episode_scale_values(
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if args.scale_mode == "discrete":
        requested_values = [float(v) for v in args.scales_mm]
    else:
        count = int(args.scales_per_episode)
        lo = float(args.scale_min_mm)
        hi = float(args.scale_max_mm)
        if args.scale_mode == "continuous_uniform":
            requested_values = [rng.uniform(lo, hi) for _ in range(count)]
        elif args.scale_mode == "continuous_lhs":
            bin_edges = np.linspace(lo, hi, count + 1, dtype=np.float64)
            requested_values = [rng.uniform(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(count)]
            rng.shuffle(requested_values)
        else:
            raise ValueError(f"Unsupported scale-mode: {args.scale_mode}")

    quant = args.continuous_scale_quantization_mm
    seen_keys: set[str] = set()
    scale_entries: List[Dict[str, Any]] = []
    max_attempts = 256
    attempts = 0
    while requested_values:
        requested = float(requested_values.pop(0))
        simulated, quantized = quantize_scale_if_needed(requested, quant)
        scale_key = format_scale_key(simulated)
        if scale_key in seen_keys:
            if args.scale_mode == "discrete":
                raise ValueError(
                    f"Duplicate scale key after quantization in discrete mode: {requested} -> {scale_key}"
                )
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Failed to sample enough unique scales after quantization")
            lo = float(args.scale_min_mm)
            hi = float(args.scale_max_mm)
            if args.scale_mode == "continuous_uniform":
                requested_values.append(random.Random(rng.random()).uniform(lo, hi))
            else:
                requested_values.append(rng.uniform(lo, hi))
            continue
        seen_keys.add(scale_key)
        scale_entries.append(
            {
                "scale_requested_mm": round(float(requested), 4),
                "scale_simulated_mm": round(float(simulated), 4),
                "scale_quantized": bool(quantized),
                "scale_key": scale_key,
            }
        )
    return scale_entries


def _continuous_scale_candidate_batch(args: argparse.Namespace, rng: random.Random, batch_size: int) -> List[float]:
    lo = float(args.scale_min_mm)
    hi = float(args.scale_max_mm)
    if args.scale_mode == "continuous_uniform":
        return [rng.uniform(lo, hi) for _ in range(batch_size)]
    if args.scale_mode == "continuous_lhs":
        bin_edges = np.linspace(lo, hi, batch_size + 1, dtype=np.float64)
        samples = [rng.uniform(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(batch_size)]
        rng.shuffle(samples)
        return samples
    raise ValueError(f"Unsupported continuous scale mode: {args.scale_mode}")


def sample_episode_scale_values_with_split_filter(
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    allowed_splits = set(str(v) for v in args.allowed_scale_splits)
    if args.scale_mode == "discrete":
        filtered_entries: List[Dict[str, Any]] = []
        for scale_entry in sample_episode_scale_values(args=args, rng=rng):
            scale_split, is_unseen_scale = determine_scale_split(
                float(scale_entry["scale_simulated_mm"]),
                args=args,
            )
            if scale_split not in allowed_splits:
                continue
            filtered_entries.append(
                {
                    **scale_entry,
                    "scale_split": scale_split,
                    "is_unseen_scale": bool(is_unseen_scale),
                }
            )
        if not filtered_entries:
            raise ValueError(
                f"No discrete scales remain after filtering by allowed scale splits {sorted(allowed_splits)}"
            )
        return filtered_entries

    requested_count = int(args.scales_per_episode)
    quant = args.continuous_scale_quantization_mm
    accepted: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    attempts = 0
    max_attempts = max(512, requested_count * 2048)
    batch_size = max(8, requested_count * 8)
    while len(accepted) < requested_count and attempts < max_attempts:
        for requested in _continuous_scale_candidate_batch(args=args, rng=rng, batch_size=batch_size):
            attempts += 1
            simulated, quantized = quantize_scale_if_needed(float(requested), quant)
            scale_key = format_scale_key(simulated)
            if scale_key in seen_keys:
                continue
            scale_split, is_unseen_scale = determine_scale_split(simulated, args=args)
            if scale_split not in allowed_splits:
                continue
            seen_keys.add(scale_key)
            accepted.append(
                {
                    "scale_requested_mm": round(float(requested), 4),
                    "scale_simulated_mm": round(float(simulated), 4),
                    "scale_quantized": bool(quantized),
                    "scale_key": scale_key,
                    "scale_split": scale_split,
                    "is_unseen_scale": bool(is_unseen_scale),
                }
            )
            if len(accepted) >= requested_count:
                break

    if len(accepted) < requested_count:
        raise RuntimeError(
            "Could not sample enough unique scales for allowed splits "
            f"{sorted(allowed_splits)} under current held-out configuration. "
            "The available continuous train/val/test region may be too narrow."
        )
    return accepted


def select_marker_textures(
    *,
    texture_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[Path], str]:
    all_textures = legacy.discover_textures(texture_dir)
    by_stem = {path.stem: path for path in all_textures}
    if args.marker_texture_names:
        missing = [name for name in args.marker_texture_names if name not in by_stem]
        if missing:
            raise ValueError(f"Unknown marker texture names: {missing}")
        selected = list({name: by_stem[name] for name in args.marker_texture_names}.values())
        mode = "explicit"
    elif int(args.marker_texture_count) > 0:
        count = int(args.marker_texture_count)
        if count > len(all_textures):
            raise ValueError(
                f"--marker-texture-count={count} exceeds available textures={len(all_textures)}"
            )
        rng = random.Random(int(args.marker_texture_seed))
        selected = sorted(rng.sample(all_textures, count), key=lambda path: path.stem)
        mode = "sampled"
    else:
        selected = list(all_textures)
        mode = "all"
    selected = sorted(selected, key=lambda path: path.stem)
    return selected, mode


def prepare_selected_texture_dir(
    *,
    selected_textures: Sequence[Path],
    selection_mode: str,
    source_texture_dir: Path,
    staging_root: Path,
) -> Path:
    if selection_mode == "all":
        return source_texture_dir
    out_dir = staging_root / "selected_marker_textures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)
    for texture_path in selected_textures:
        (out_dir / texture_path.name).symlink_to(texture_path)
    return out_dir


def build_episode_temporal_plan(args: argparse.Namespace, rng: random.Random, final_depth_mm: float) -> TemporalPlan:
    def _sample_count(name: str, enabled: bool = True) -> int:
        if not enabled:
            return 0
        lo = int(getattr(args, f"{name}_frames_min"))
        hi = int(getattr(args, f"{name}_frames_max"))
        return int(rng.randint(lo, hi))

    precontact_count = _sample_count("precontact")
    press_count = _sample_count("press")
    dwell_count = _sample_count("dwell", enabled=bool(args.include_dwell))
    release_count = _sample_count("release", enabled=bool(args.include_release))
    if precontact_count + press_count <= 0:
        raise ValueError("Temporal plan must contain at least one loading frame")

    release_mode_actual = "mirrored_loading" if (release_count > 0 and args.release_mode in {"auto", "mirror_loading"}) else "none"
    return TemporalPlan(
        precontact_frame_count=precontact_count,
        press_frame_count=press_count,
        dwell_frame_count=dwell_count,
        release_frame_count=release_count,
        press_sampling_mode=str(args.press_sampling_mode),
        release_mode_requested=str(args.release_mode),
        release_mode_actual=release_mode_actual,
        final_depth_mm=round(float(final_depth_mm), 4),
        include_dwell=bool(args.include_dwell),
        include_release=bool(args.include_release),
    )


def build_loading_samples_uniform_index(trajectory_length: int, count: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    if trajectory_length <= 0:
        raise ValueError("trajectory_length must be > 0")
    last = max(trajectory_length - 1, 0)
    if count == 1:
        positions = [0.0]
    else:
        positions = np.linspace(0.0, float(last), num=count, dtype=np.float64).tolist()
    samples: List[Dict[str, Any]] = []
    for pos in positions:
        idx = int(round(float(pos)))
        frac_req = 1.0 if last == 0 else float(pos) / float(last)
        frac_act = 1.0 if last == 0 else float(idx) / float(last)
        samples.append(
            {
                "source_physics_frame_index": idx,
                "frame_fraction_requested": float(frac_req),
                "frame_fraction_actual": float(frac_act),
                "frame_target_max_down_mm": None,
            }
        )
    return samples


def build_loading_samples_uniform_max_down(z_trajectory: np.ndarray, count: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    max_down = legacy.max_down_trajectory_mm(z_trajectory)
    deepest = float(np.max(max_down))
    t_last = max(int(max_down.shape[0]) - 1, 0)
    if count == 1:
        targets = [0.0]
    else:
        targets = np.linspace(0.0, deepest, num=count, dtype=np.float64).tolist()
    samples: List[Dict[str, Any]] = []
    for target in targets:
        idx = int(np.argmin(np.abs(max_down - float(target))))
        frac_act = 1.0 if t_last == 0 else float(idx) / float(t_last)
        samples.append(
            {
                "source_physics_frame_index": idx,
                "frame_fraction_requested": None,
                "frame_fraction_actual": float(frac_act),
                "frame_target_max_down_mm": float(target),
            }
        )
    return samples


def _phase_progress(phase_index: int, phase_length: int) -> float:
    if phase_length <= 1:
        return 1.0
    return float(phase_index) / float(phase_length - 1)


def build_precontact_samples(count: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    return [
        {
            "source_physics_frame_index": 0,
            "frame_fraction_requested": 0.0,
            "frame_fraction_actual": 0.0,
            "frame_target_max_down_mm": 0.0,
        }
        for _ in range(count)
    ]


def build_press_samples(z_trajectory: np.ndarray, temporal_plan: TemporalPlan) -> List[Dict[str, Any]]:
    press_count = int(temporal_plan.press_frame_count)
    if press_count <= 0:
        return []
    if temporal_plan.press_sampling_mode == "uniform_index":
        return build_loading_samples_uniform_index(int(z_trajectory.shape[0]), press_count)
    if temporal_plan.press_sampling_mode == "uniform_max_down":
        return build_loading_samples_uniform_max_down(z_trajectory, press_count)
    raise ValueError(f"Unsupported press_sampling_mode: {temporal_plan.press_sampling_mode}")


def build_phase_aware_frame_records(
    *,
    z_trajectory: np.ndarray,
    temporal_plan: TemporalPlan,
) -> Tuple[List[SequenceFrameRecord], str]:
    if int(temporal_plan.precontact_frame_count) + int(temporal_plan.press_frame_count) <= 0:
        raise ValueError("Temporal plan must contain at least one loading frame")
    pre_samples = build_precontact_samples(int(temporal_plan.precontact_frame_count))
    press_samples = build_press_samples(z_trajectory, temporal_plan)
    loading_anchor = press_samples[-1] if press_samples else (pre_samples[-1] if pre_samples else None)

    records: List[SequenceFrameRecord] = []
    next_global_idx = 0

    def _append_phase(
        phase_name: str,
        phase_samples: Sequence[Dict[str, Any]],
        *,
        synthetic: bool,
        synthetic_precontact: bool,
        synthetic_release: bool,
        dwell_repeat: bool,
    ) -> None:
        nonlocal next_global_idx
        phase_length = len(phase_samples)
        for phase_index, sample in enumerate(phase_samples):
            frame_index = int(sample["source_physics_frame_index"])
            frame_record = SequenceFrameRecord(
                global_seq_index=next_global_idx,
                frame_name=f"frame_{next_global_idx:06d}",
                phase_name=phase_name,
                phase_index=phase_index,
                phase_progress=_phase_progress(phase_index, phase_length),
                source_physics_frame_index=frame_index,
                frame_fraction_requested=sample.get("frame_fraction_requested"),
                frame_fraction_actual=sample.get("frame_fraction_actual"),
                frame_target_max_down_mm=sample.get("frame_target_max_down_mm"),
                frame_actual_max_down_mm=float(
                    legacy.deformation_stats_for_frame(z_trajectory, frame_index)["surface_max_down_mm"]
                ),
                deformation_stats=legacy.deformation_stats_for_frame(z_trajectory, frame_index),
                is_synthetic_frame=bool(synthetic),
                is_synthetic_precontact=bool(synthetic_precontact),
                is_synthetic_release=bool(synthetic_release),
                is_dwell_repeat=bool(dwell_repeat),
            )
            records.append(frame_record)
            next_global_idx += 1

    _append_phase("precontact", pre_samples, synthetic=True, synthetic_precontact=True, synthetic_release=False, dwell_repeat=False)
    _append_phase("press", press_samples, synthetic=False, synthetic_precontact=False, synthetic_release=False, dwell_repeat=False)

    if temporal_plan.dwell_frame_count > 0 and loading_anchor is not None:
        dwell_samples = [loading_anchor.copy() for _ in range(int(temporal_plan.dwell_frame_count))]
        _append_phase("dwell", dwell_samples, synthetic=True, synthetic_precontact=False, synthetic_release=False, dwell_repeat=True)

    release_mode_actual = "none"
    if temporal_plan.release_frame_count > 0 and loading_anchor is not None:
        mirrored_source = list(reversed(press_samples[:-1] if len(press_samples) > 1 else press_samples))
        if not mirrored_source:
            mirrored_source = [loading_anchor]
        if len(mirrored_source) == 1:
            release_samples = [mirrored_source[0].copy() for _ in range(int(temporal_plan.release_frame_count))]
        else:
            positions = np.linspace(0.0, float(len(mirrored_source) - 1), num=int(temporal_plan.release_frame_count))
            release_samples = [mirrored_source[int(round(float(pos)))].copy() for pos in positions]
        _append_phase("release", release_samples, synthetic=True, synthetic_precontact=False, synthetic_release=True, dwell_repeat=False)
        release_mode_actual = "mirrored_loading"

    return records, release_mode_actual


def extract_or_construct_sequence_from_npz(
    npz_path: Path,
    temporal_plan: TemporalPlan,
) -> Tuple[int, Dict[str, float], List[SequenceFrameRecord], str]:
    z_trajectory = legacy._load_z_trajectory(npz_path)
    trajectory_length = int(z_trajectory.shape[0])
    deform_final = legacy.deformation_stats_for_frame(z_trajectory, trajectory_length - 1)
    records, release_mode_actual = build_phase_aware_frame_records(z_trajectory=z_trajectory, temporal_plan=temporal_plan)
    return trajectory_length, deform_final, records, release_mode_actual


def validate_existing_sequence_scale_metadata(
    *,
    episode_dir: Path,
    scale_key: str,
    expected_profile: str,
    expected_scale_requested_mm: float,
    expected_scale_simulated_mm: float,
    expected_scale_quantized: bool,
    expected_temporal_plan: TemporalPlan,
    expected_marker_files: Sequence[str],
    expected_contact_x_mm: float,
    expected_contact_y_mm: float,
    expected_contact_x_norm: float,
    expected_contact_y_norm: float,
    expected_scale_split: str,
    expected_is_unseen_scale: bool,
) -> bool:
    scale_dir = episode_dir / scale_key
    meta_path = scale_dir / "sequence_metadata.json"
    coord_map_path = scale_dir / "adapter_coord_map.npy"
    if not meta_path.exists() or not coord_map_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    try:
        coord_map = np.load(coord_map_path)
    except Exception:
        return False
    if coord_map.ndim != 3 or coord_map.shape[-1] != 2:
        return False

    def _float_close(key: str, expected: float, tol: float = 1e-4) -> bool:
        got = meta.get(key)
        return got is not None and abs(float(got) - float(expected)) <= tol

    if meta.get("dataset_profile") != expected_profile:
        return False
    if not _float_close("scale_requested_mm", expected_scale_requested_mm):
        return False
    if not _float_close("scale_simulated_mm", expected_scale_simulated_mm):
        return False
    expected_dims = rectgel.gel_dims_from_width_mm(expected_scale_simulated_mm)
    if not _float_close("gel_width_mm", expected_dims.width_mm):
        return False
    if not _float_close("gel_height_mm", expected_dims.height_mm):
        return False
    if bool(meta.get("scale_quantized", False)) != bool(expected_scale_quantized):
        return False
    if not _float_close("contact_x_mm", expected_contact_x_mm):
        return False
    if not _float_close("contact_y_mm", expected_contact_y_mm):
        return False
    if not _float_close("contact_x_norm", expected_contact_x_norm, tol=1e-6):
        return False
    if not _float_close("contact_y_norm", expected_contact_y_norm, tol=1e-6):
        return False
    if str(meta.get("scale_split", "")) != str(expected_scale_split):
        return False
    if bool(meta.get("is_unseen_scale", False)) != bool(expected_is_unseen_scale):
        return False

    phase_lengths = meta.get("phase_lengths", {})
    expected_phase_lengths = {
        "precontact": int(expected_temporal_plan.precontact_frame_count),
        "press": int(expected_temporal_plan.press_frame_count),
        "dwell": int(expected_temporal_plan.dwell_frame_count),
        "release": int(expected_temporal_plan.release_frame_count),
    }
    if phase_lengths != expected_phase_lengths:
        return False
    expected_length = sum(expected_phase_lengths.values())
    if int(meta.get("sequence_length", -1)) != expected_length:
        return False
    meta_marker_files = [str(v) for v in meta.get("marker_files_selected", [])]
    if sorted(meta_marker_files) != sorted(str(v) for v in expected_marker_files):
        return False

    frames = meta.get("frames", [])
    if not isinstance(frames, list) or len(frames) != expected_length:
        return False
    phase_sequence = (
        [("precontact", int(expected_temporal_plan.precontact_frame_count))]
        + [("press", int(expected_temporal_plan.press_frame_count))]
        + [("dwell", int(expected_temporal_plan.dwell_frame_count))]
        + [("release", int(expected_temporal_plan.release_frame_count))]
    )
    expected_phase_names: List[str] = []
    for phase_name, phase_count in phase_sequence:
        expected_phase_names.extend([phase_name] * max(0, phase_count))
    expected_markers = set(str(v) for v in expected_marker_files)
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            return False
        frame_name = str(frame.get("frame_name", ""))
        if frame_idx < len(expected_phase_names) and str(frame.get("phase_name", "")) != expected_phase_names[frame_idx]:
            return False
        if frame_idx < int(expected_temporal_plan.precontact_frame_count):
            if int(frame.get("source_physics_frame_index", -1)) != 0:
                return False
        rendered_markers = set(str(v) for v in frame.get("rendered_markers", []))
        if rendered_markers != expected_markers:
            return False
        frame_dir = scale_dir / frame_name
        actual_frame_markers = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        if actual_frame_markers != sorted(expected_markers):
            return False
    return True


def estimate_generation_plan(
    *,
    dataset_profile: str,
    episode_states: Mapping[int, SequenceEpisodeState],
    physics_tasks: Sequence[Dict[str, Any]],
    marker_count: int,
    pair_marker_policy: str,
    include_identity_pairs: bool,
    pair_split_policy: str,
    clip_length: int,
    clip_window_stride: int,
    clip_gap_min: int,
    clip_gap_max: int,
    keep_intermediates: bool,
    physics_npz_cleanup: str,
    reference_image_kb: float,
    reference_npz_mb: float,
    reference_stl_mb: float,
    reference_throughput_images_per_hour: float,
) -> Dict[str, Any]:
    episodes_total = int(len(episode_states))
    scale_slots_total = 0
    sequence_frames_total = 0
    pair_rows_estimate = 0
    clip_rows_estimate = 0

    for episode_state in episode_states.values():
        scale_summaries = list(episode_state.scales.values())
        scale_slots_total += len(scale_summaries)
        split_groups: Dict[str, List[Dict[str, Any]]] = {}
        for scale_summary in scale_summaries:
            sequence_length = int(scale_summary.get("sequence_length", 0))
            sequence_frames_total += sequence_length
            split_groups.setdefault(str(scale_summary.get("scale_split", "")), []).append(scale_summary)

            if dataset_profile == "B":
                windows = 0
                for window_idx, _ in enumerate(
                    _clip_window_indices(sequence_length, clip_length, clip_gap_min, clip_gap_max)
                ):
                    if clip_window_stride > 1 and window_idx % clip_window_stride != 0:
                        continue
                    windows += 1
                clip_rows_estimate += windows * int(marker_count)

        if dataset_profile == "A":
            if pair_split_policy == "same_split_only":
                group_iter = split_groups.values()
            else:
                group_iter = [scale_summaries]
            for scale_group in group_iter:
                if not scale_group:
                    continue
                ordered_pairs = 0
                for source_scale in scale_group:
                    for target_scale in scale_group:
                        if (
                            str(source_scale.get("scale_key", "")) == str(target_scale.get("scale_key", ""))
                            and not include_identity_pairs
                        ):
                            continue
                        ordered_pairs += 1
                if not ordered_pairs:
                    continue
                pair_factor = int(marker_count) if pair_marker_policy == "same_texture_only" else int(marker_count) ** 2
                reference_sequence_length = int(scale_group[0].get("sequence_length", 0))
                pair_rows_estimate += ordered_pairs * reference_sequence_length * pair_factor

    physics_tasks_planned = int(len(physics_tasks))
    pending_sequence_frames = 0
    for task in physics_tasks:
        temporal_plan = task["temporal_plan"]
        pending_sequence_frames += int(
            temporal_plan["precontact_frame_count"]
            + temporal_plan["press_frame_count"]
            + temporal_plan["dwell_frame_count"]
            + temporal_plan["release_frame_count"]
        )
    meshing_tasks_planned = int(pending_sequence_frames)
    render_tasks_planned = int(pending_sequence_frames)
    expected_images_total = int(sequence_frames_total * marker_count)
    expected_images_planned = int(pending_sequence_frames * marker_count)
    rough_eta_hours = float(expected_images_planned) / float(reference_throughput_images_per_hour)
    estimated_image_gb_total = (float(expected_images_total) * float(reference_image_kb)) / (1024.0 * 1024.0)
    estimated_image_gb_planned = (float(expected_images_planned) * float(reference_image_kb)) / (1024.0 * 1024.0)
    estimated_npz_gb_total = (float(scale_slots_total) * float(reference_npz_mb)) / 1024.0
    estimated_npz_gb_planned = (float(physics_tasks_planned) * float(reference_npz_mb)) / 1024.0
    estimated_stl_gb_total = (float(sequence_frames_total) * float(reference_stl_mb)) / 1024.0
    estimated_stl_gb_planned = (float(pending_sequence_frames) * float(reference_stl_mb)) / 1024.0
    persistent_stl_gb_total = estimated_stl_gb_total if keep_intermediates else 0.0
    persistent_stl_gb_planned = estimated_stl_gb_planned if keep_intermediates else 0.0
    estimated_total_gb_if_keep_npz = estimated_image_gb_total + estimated_npz_gb_total + persistent_stl_gb_total
    estimated_total_gb_if_delete_npz_after_scale_complete = estimated_image_gb_total + persistent_stl_gb_total

    return {
        "totals": {
            "episodes": int(episodes_total),
            "scale_slots": int(scale_slots_total),
            "physics_tasks_planned": physics_tasks_planned,
            "meshing_tasks_planned": meshing_tasks_planned,
            "render_tasks_planned": render_tasks_planned,
            "sequence_frames_total": int(sequence_frames_total),
            "sequence_frames_planned": int(pending_sequence_frames),
            "expected_images_total": int(expected_images_total),
            "expected_images_planned": int(expected_images_planned),
            "textures": int(marker_count),
            "expected_pair_rows": int(pair_rows_estimate),
            "expected_clip_rows": int(clip_rows_estimate),
        },
        "estimate": {
            "reference_throughput_images_per_hour": float(reference_throughput_images_per_hour),
            "rough_eta_hours": float(rough_eta_hours),
            "notes": (
                "Rough estimate based on historical average images/hour. "
                "Actual runtime depends on workers, CPU/GPU, marker count, and resume hit rate."
            ),
        },
        "disk_estimate": {
            "avg_image_kb": round(float(reference_image_kb), 2),
            "avg_npz_mb": round(float(reference_npz_mb), 2),
            "avg_stl_mb": round(float(reference_stl_mb), 2),
            "expected_images_total": int(expected_images_total),
            "expected_images_planned": int(expected_images_planned),
            "estimated_image_gb_total": round(float(estimated_image_gb_total), 2),
            "estimated_image_gb_planned": round(float(estimated_image_gb_planned), 2),
            "estimated_npz_gb_total": round(float(estimated_npz_gb_total), 2),
            "estimated_npz_gb_planned": round(float(estimated_npz_gb_planned), 2),
            "estimated_stl_gb_total": round(float(estimated_stl_gb_total), 2),
            "estimated_stl_gb_planned": round(float(estimated_stl_gb_planned), 2),
            "estimated_total_gb_if_keep_npz": round(float(estimated_total_gb_if_keep_npz), 2),
            "estimated_total_gb_if_delete_npz_after_scale_complete": round(
                float(estimated_total_gb_if_delete_npz_after_scale_complete),
                2,
            ),
            "estimated_peak_gb_note": (
                "Peak disk usage is only a rough estimate. Actual peak depends on concurrent workers, "
                "resume hit rate, temporary in-flight STL/NPZ files, and whether keep_intermediates is enabled."
            ),
            "keep_intermediates": bool(keep_intermediates),
            "physics_npz_cleanup": str(physics_npz_cleanup),
        },
    }


def log_planning_summary(plan: Mapping[str, Any], dataset_profile: str) -> None:
    totals = dict(plan.get("totals", {}))
    estimate = dict(plan.get("estimate", {}))
    disk_estimate = dict(plan.get("disk_estimate", {}))
    logging.info(
        "Planning summary | profile=%s episodes=%d scale_slots=%d physics=%d meshing=%d render=%d "
        "seq_frames_total=%d seq_frames_planned=%d textures=%d expected_images_total=%d expected_images_planned=%d "
        "expected_pairs=%d expected_clips=%d throughput_ref=%.1f img/h rough_eta=%.2f h",
        dataset_profile,
        int(totals.get("episodes", 0)),
        int(totals.get("scale_slots", 0)),
        int(totals.get("physics_tasks_planned", 0)),
        int(totals.get("meshing_tasks_planned", 0)),
        int(totals.get("render_tasks_planned", 0)),
        int(totals.get("sequence_frames_total", 0)),
        int(totals.get("sequence_frames_planned", 0)),
        int(totals.get("textures", 0)),
        int(totals.get("expected_images_total", 0)),
        int(totals.get("expected_images_planned", 0)),
        int(totals.get("expected_pair_rows", 0)),
        int(totals.get("expected_clip_rows", 0)),
        float(estimate.get("reference_throughput_images_per_hour", 0.0)),
        float(estimate.get("rough_eta_hours", 0.0)),
    )
    logging.info(
        "Disk estimate | image_gb_total=%.2f image_gb_planned=%.2f npz_gb_total=%.2f npz_gb_planned=%.2f "
        "stl_gb_total=%.2f stl_gb_planned=%.2f total_gb_keep_npz=%.2f total_gb_delete_npz=%.2f "
        "keep_intermediates=%s npz_cleanup=%s",
        float(disk_estimate.get("estimated_image_gb_total", 0.0)),
        float(disk_estimate.get("estimated_image_gb_planned", 0.0)),
        float(disk_estimate.get("estimated_npz_gb_total", 0.0)),
        float(disk_estimate.get("estimated_npz_gb_planned", 0.0)),
        float(disk_estimate.get("estimated_stl_gb_total", 0.0)),
        float(disk_estimate.get("estimated_stl_gb_planned", 0.0)),
        float(disk_estimate.get("estimated_total_gb_if_keep_npz", 0.0)),
        float(disk_estimate.get("estimated_total_gb_if_delete_npz_after_scale_complete", 0.0)),
        bool(disk_estimate.get("keep_intermediates", False)),
        str(disk_estimate.get("physics_npz_cleanup", "")),
    )


def maybe_log_stage_progress(
    *,
    stage_name: str,
    stage_start_ts: float,
    done: int,
    total: int,
    reused: int,
    failed: int,
    force: bool = False,
) -> None:
    if total <= 0:
        return
    if not force and done <= 0:
        return
    logging.info(
        "%s progress | completed=%d total=%d reused=%d failed=%d | %s | %s",
        stage_name,
        int(done),
        int(total),
        int(reused),
        int(failed),
        _progress_eta_text(stage_start_ts, done, total),
        _throughput_text(stage_start_ts, done),
    )


def maybe_log_overall_progress(
    *,
    run_start_ts: float,
    physics_done: int,
    physics_total: int,
    meshing_done: int,
    meshing_total: int,
    render_done: int,
    render_total: int,
    marker_count: int,
    expected_images: int,
    force: bool = False,
) -> None:
    total_units = int(physics_total) + int(meshing_total) + int(render_total)
    done_units = int(physics_done) + int(meshing_done) + int(render_done)
    if total_units <= 0:
        return
    now_ts = time.time()
    last_log_ts = float(getattr(maybe_log_overall_progress, "_last_log_ts", 0.0))
    if not force and done_units > 1 and done_units < total_units and (now_ts - last_log_ts) < 60.0:
        return
    elapsed = max(0.0, time.time() - float(run_start_ts))
    if done_units <= 0:
        eta_text = "eta=-- eta_at=--"
    else:
        avg_sec = elapsed / float(done_units)
        remaining_units = max(total_units - done_units, 0)
        eta_sec = avg_sec * float(remaining_units)
        eta_at = (dt.datetime.now() + dt.timedelta(seconds=eta_sec)).strftime("%Y-%m-%d %H:%M:%S")
        eta_text = f"eta={_format_duration(eta_sec)} eta_at={eta_at}"
    rendered_images = int(render_done) * int(marker_count)
    logging.info(
        "Overall progress | tasks=%d/%d physics=%d/%d meshing=%d/%d render=%d/%d rendered_images=%d/%d "
        "elapsed=%s %s",
        done_units,
        total_units,
        int(physics_done),
        int(physics_total),
        int(meshing_done),
        int(meshing_total),
        int(render_done),
        int(render_total),
        rendered_images,
        int(expected_images),
        _format_duration(elapsed),
        eta_text,
    )
    maybe_log_overall_progress._last_log_ts = now_ts


def write_scaled_config_float(base_cfg: Path, out_cfg: Path, scale_mm: float) -> None:
    rectgel.write_rectangular_config_from_width(base_cfg, out_cfg, scale_mm)


def run_physics_sequence_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_requested_mm = float(task["scale_requested_mm"])
    scale_simulated_mm = float(task["scale_simulated_mm"])
    indenter = str(task["indenter"])
    x_mm = float(task["x_mm"])
    y_mm = float(task["y_mm"])
    depth_mm = float(task["depth_mm"])

    repo_root = Path(task["repo_root"])
    temp_scale_dir = Path(task["temp_scale_dir"])
    temp_scale_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Path(task["base_parameters"])
    temp_cfg = temp_scale_dir / "parameters_scaled.yml"
    npz_root = temp_scale_dir / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)
    resume_mode = bool(task.get("resume", False))

    try:
        npz_path = legacy.expected_npz_path(npz_root, indenter, x_mm, y_mm, depth_mm)
        physics_reused = False
        if resume_mode and npz_path.exists():
            physics_reused = True
        else:
            write_scaled_config_float(base_cfg, temp_cfg, scale_simulated_mm)
            gel_cmd = [
                str(task["python_cmd"]),
                "sim/deformation/gel_press.py",
                "--config",
                str(temp_cfg),
                "--particle",
                str(task["particle"]),
                "--dir_output",
                str(npz_root),
                "--dataset",
                "sim/assets/indenters/input",
                "--object",
                indenter,
                "--x",
                f"{x_mm:.4f}",
                "--y",
                f"{y_mm:.4f}",
                "--depth",
                f"{depth_mm:.1f}",
            ]
            legacy.run_cmd_checked(
                gel_cmd,
                cwd=repo_root,
                timeout_sec=int(task["physics_timeout_sec"]),
                stage="gel_press",
            )

        if not npz_path.exists():
            x_tag = legacy.format_coord_suffix(x_mm)
            y_tag = legacy.format_coord_suffix(y_mm)
            candidates = sorted((npz_root / indenter).glob(f"{x_tag}_{y_tag}_*.npz"))
            if not candidates:
                raise FileNotFoundError(f"No deformation npz found for {indenter} at {npz_root}")
            npz_path = candidates[-1]
            if resume_mode:
                physics_reused = True

        temporal_plan_payload = TemporalPlan(**task["temporal_plan"])
        trajectory_length, deform_final, frames, release_mode_actual = extract_or_construct_sequence_from_npz(
            npz_path=npz_path,
            temporal_plan=temporal_plan_payload,
        )
        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_requested_mm": scale_requested_mm,
            "scale_simulated_mm": scale_simulated_mm,
            "scale_quantized": bool(task["scale_quantized"]),
            "scale_key": str(task["scale_key"]),
            "indenter": indenter,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "depth_mm": depth_mm,
            "temp_scale_dir": str(temp_scale_dir),
            "npz_path": str(npz_path),
            "trajectory_length": trajectory_length,
            "deformation_stats_final": deform_final,
            "physics_reused": physics_reused,
            "frames": [frame.to_metadata_dict() for frame in frames],
            "release_mode_actual": release_mode_actual,
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_requested_mm": scale_requested_mm,
            "scale_simulated_mm": scale_simulated_mm,
            "scale_key": str(task["scale_key"]),
            "indenter": indenter,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "depth_mm": depth_mm,
            "temp_scale_dir": str(temp_scale_dir),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def build_scale_state_from_task_and_npz(
    *,
    episode_state: SequenceEpisodeState,
    task: Dict[str, Any],
    npz_path: Path,
    patch_h: int,
    patch_w: int,
    physics_reused: bool,
    physics_npz_cleanup_policy: str,
) -> SequenceScaleState:
    camera_distance_m, camera_fov_deg = task["camera_params"]
    scale_dir = episode_state.episode_dir / str(task["scale_key"])
    scale_dir.mkdir(parents=True, exist_ok=True)
    gel_dims = rectgel.gel_dims_from_width_mm(float(task["scale_simulated_mm"]))
    coord_map = rectgel.make_adapter_coord_map(
        gel_width_mm=float(gel_dims.width_mm),
        gel_height_mm=float(gel_dims.height_mm),
        patch_h=patch_h,
        patch_w=patch_w,
    )
    adapter_coord_map_path = scale_dir / "adapter_coord_map.npy"
    np.save(adapter_coord_map_path, coord_map)
    temporal_plan_payload = TemporalPlan(**task["temporal_plan"])
    trajectory_length, deform_final, frames, _ = extract_or_construct_sequence_from_npz(
        npz_path=npz_path,
        temporal_plan=temporal_plan_payload,
    )
    return SequenceScaleState(
        episode_id=int(task["episode_id"]),
        scale_requested_mm=float(task["scale_requested_mm"]),
        scale_simulated_mm=float(task["scale_simulated_mm"]),
        scale_quantized=bool(task["scale_quantized"]),
        scale_key=str(task["scale_key"]),
        scale_dir=scale_dir,
        temp_scale_dir=Path(task["temp_scale_dir"]),
        contact_x_mm=float(task["x_mm"]),
        contact_y_mm=float(task["y_mm"]),
        contact_x_norm=float(task["contact_x_norm"]),
        contact_y_norm=float(task["contact_y_norm"]),
        indenter_split=episode_state.indenter_split,
        scale_split=str(task["scale_split"]),
        is_unseen_indenter=bool(episode_state.is_unseen_indenter),
        is_unseen_scale=bool(task["is_unseen_scale"]),
        trajectory_length=int(trajectory_length),
        deformation_stats_final=deform_final,
        camera_mode=str(task["camera_mode"]),
        camera_distance_m=float(camera_distance_m),
        camera_fov_deg=float(camera_fov_deg),
        adapter_coord_map_path=adapter_coord_map_path,
        adapter_coord_map_shape=[int(v) for v in coord_map.shape],
        frames=frames,
        physics_npz_path=npz_path,
        physics_reused=bool(physics_reused),
        physics_npz_cleanup_policy=str(physics_npz_cleanup_policy),
    )


def build_meshing_tasks_for_scale_state(
    *,
    scale_state: SequenceScaleState,
    episode_id: int,
    repo_root: Path,
    args: argparse.Namespace,
    open3d_script: Path,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for frame in scale_state.frames:
        stl_path = scale_state.temp_scale_dir / "stl" / f"{frame.frame_name}.stl"
        tasks.append(
            {
                "episode_id": episode_id,
                "scale_simulated_mm": scale_state.scale_simulated_mm,
                "frame_name": frame.frame_name,
                "source_physics_frame_index": frame.source_physics_frame_index,
                "repo_root": repo_root,
                "stl_path": stl_path,
                "python_cmd": args.python_cmd,
                "open3d_script": open3d_script,
                "npz_path": scale_state.physics_npz_path,
                "meshing_timeout_sec": args.meshing_timeout_sec,
                "sample": {"frame_name": frame.frame_name},
                "resume": bool(args.resume),
            }
        )
    return tasks


def run_meshing_sequence_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_simulated_mm = float(task["scale_simulated_mm"])
    frame_name = str(task["frame_name"])
    frame_index = int(task["source_physics_frame_index"])
    repo_root = Path(task["repo_root"])
    stl_path = Path(task["stl_path"])
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    resume_mode = bool(task.get("resume", False))

    try:
        if resume_mode and stl_path.exists():
            valid, reason = inspect_binary_stl(stl_path)
            if valid:
                return {
                    "status": "ok",
                    "episode_id": episode_id,
                    "scale_simulated_mm": scale_simulated_mm,
                    "frame_name": frame_name,
                    "frame_index": frame_index,
                    "stl_path": str(stl_path),
                    "sample": task["sample"],
                    "reused": True,
                }
            logging.warning(
                "Invalid existing STL detected; regenerating | ep=%d scale=%.4f frame=%s stl=%s reason=%s",
                episode_id,
                scale_simulated_mm,
                frame_name,
                stl_path,
                reason,
            )
            stl_path.unlink(missing_ok=True)
        mesh_cmd = [
            str(task["python_cmd"]),
            str(task["open3d_script"]),
            "--input-npz",
            str(task["npz_path"]),
            "--output-stl",
            str(stl_path),
            "--frame-index",
            str(frame_index),
        ]
        legacy.run_cmd_checked(
            mesh_cmd,
            cwd=repo_root,
            timeout_sec=int(task["meshing_timeout_sec"]),
            stage=f"npz_to_stl[{frame_name}]",
        )
        valid, reason = inspect_binary_stl(stl_path)
        if not valid:
            stl_path.unlink(missing_ok=True)
            raise RuntimeError(f"Invalid STL generated at {stl_path}: {reason}")
        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_simulated_mm": scale_simulated_mm,
            "frame_name": frame_name,
            "frame_index": frame_index,
            "stl_path": str(stl_path),
            "sample": task["sample"],
            "reused": False,
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_simulated_mm": scale_simulated_mm,
            "frame_name": frame_name,
            "frame_index": frame_index,
            "stl_path": str(stl_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_render_sequence_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_simulated_mm = float(task["scale_simulated_mm"])
    frame_name = str(task["frame_name"])
    frame_dir = Path(task["frame_dir"])
    frame_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(task["repo_root"])
    stl_path = Path(task["stl_path"])
    keep_intermediates = bool(task["keep_intermediates"])
    resume_mode = bool(task.get("resume", False))
    expected_marker_files = [str(v) for v in task.get("expected_marker_files", [])]
    if not expected_marker_files:
        raise ValueError("run_render_sequence_task requires non-empty expected_marker_files")

    try:
        existing_marker_files = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        if resume_mode and existing_marker_files == sorted(expected_marker_files):
            return {
                "status": "ok",
                "episode_id": episode_id,
                "scale_simulated_mm": scale_simulated_mm,
                "frame_name": frame_name,
                "rendered_markers": sorted(expected_marker_files),
                "reused": True,
            }
        if existing_marker_files:
            for img_path in frame_dir.glob("marker_*.jpg"):
                img_path.unlink(missing_ok=True)

        if not stl_path.exists():
            raise FileNotFoundError(f"Missing STL for render: {stl_path}")
        valid_stl, stl_reason = inspect_binary_stl(stl_path)
        if not valid_stl:
            raise RuntimeError(f"Invalid STL for render: {stl_path} ({stl_reason})")

        render_cmd = [
            str(task["blender_cmd"]),
            "-b",
            "--python",
            str(task["blender_script"]),
            "--",
            "--stl",
            str(stl_path),
            "--textures-dir",
            str(task["textures_dir"]),
            "--output-dir",
            str(frame_dir),
            "--scale-mm",
            f"{scale_simulated_mm:.4f}",
            "--gel-height-mm",
            f"{rectgel.gel_dims_from_width_mm(scale_simulated_mm).height_mm:.4f}",
            "--camera-mode",
            str(task["camera_mode"]),
            "--base-fov-deg",
            str(task["base_fov_deg"]),
            "--fixed-distance-m",
            str(task["fixed_distance_m"]),
            "--distance-safety",
            str(task["distance_safety"]),
            "--uv-mode",
            str(task["uv_mode"]),
            "--uv-inset-ratio",
            str(task["uv_inset_ratio"]),
            "--render-samples",
            str(task["render_samples"]),
            "--render-device",
            str(task.get("render_device", "cpu")),
            "--render-gpu-backend",
            str(task.get("render_gpu_backend", "auto")),
        ]
        legacy.run_cmd_checked(
            render_cmd,
            cwd=repo_root,
            timeout_sec=int(task["render_timeout_sec"]),
            stage=f"blender_render[{frame_name}]",
        )

        rendered = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        if not rendered:
            raise RuntimeError(f"No marker renders produced in {frame_dir}")

        side_border_px = int(task.get("force_black_side_border_px", 0))
        if side_border_px > 0:
            for name in rendered:
                img_path = frame_dir / name
                with legacy.Image.open(img_path) as im:
                    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
                px = min(side_border_px, max(1, rgb.shape[1] // 2))
                rgb[:, :px, :] = 0
                rgb[:, -px:, :] = 0
                legacy.Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=100)

        missing_after_render = [name for name in expected_marker_files if not (frame_dir / name).exists()]
        if missing_after_render:
            raise RuntimeError(f"Missing marker renders after render for {frame_name}: {missing_after_render[:3]}")

        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_simulated_mm": scale_simulated_mm,
            "frame_name": frame_name,
            "rendered_markers": sorted(p.name for p in frame_dir.glob("marker_*.jpg")),
            "reused": False,
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_simulated_mm": scale_simulated_mm,
            "frame_name": frame_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if (not keep_intermediates) and stl_path.exists():
            stl_path.unlink(missing_ok=True)


def persist_episode_metadata(state: SequenceEpisodeState, particle: str, dataset_profile: str) -> None:
    payload = {
        "episode_id": int(state.episode_id),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_profile": str(dataset_profile),
        "indenter": state.indenter,
        "particle": str(particle),
        "latent_contact": {
            "contact_x_norm": float(state.contact_x_norm),
            "contact_y_norm": float(state.contact_y_norm),
            "final_depth_mm": float(state.final_depth_mm),
        },
        "position_sample_type_requested": state.position_sample_type_requested,
        "position_sample_type_actual": state.position_sample_type_actual,
        "contact_margin_constraint_mm": float(state.contact_margin_constraint_mm),
        "indenter_split": state.indenter_split,
        "is_unseen_indenter": bool(state.is_unseen_indenter),
        "temporal_plan": {
            "precontact_frame_count": int(state.temporal_plan.precontact_frame_count),
            "press_frame_count": int(state.temporal_plan.press_frame_count),
            "dwell_frame_count": int(state.temporal_plan.dwell_frame_count),
            "release_frame_count": int(state.temporal_plan.release_frame_count),
            "release_mode": state.temporal_plan.release_mode_actual,
            "press_sampling_mode": state.temporal_plan.press_sampling_mode,
            "final_depth_mm": float(state.temporal_plan.final_depth_mm),
            "precontact_is_repeated_zero_frame": bool(state.temporal_plan.precontact_frame_count > 0),
            "is_synthetic_release": bool(state.temporal_plan.release_mode_actual == "mirrored_loading"),
            "is_dwell_repeated": bool(state.temporal_plan.dwell_frame_count > 0),
        },
        "scales": legacy._json_safe(state.scales),
        "errors": [str(v) for v in state.errors],
    }
    legacy.write_json_atomic(state.episode_dir / "metadata.json", payload)


def persist_scale_sequence_metadata(
    *,
    episode_state: SequenceEpisodeState,
    scale_state: SequenceScaleState,
    dataset_profile: str,
    image_resolution: Tuple[int, int],
    marker_files_selected: Sequence[str],
) -> Dict[str, Any]:
    gel_dims = rectgel.gel_dims_from_width_mm(scale_state.scale_simulated_mm)
    phase_lengths = {
        "precontact": int(episode_state.temporal_plan.precontact_frame_count),
        "press": int(episode_state.temporal_plan.press_frame_count),
        "dwell": int(episode_state.temporal_plan.dwell_frame_count),
        "release": int(episode_state.temporal_plan.release_frame_count),
    }
    sequence_meta = {
        "dataset_profile": dataset_profile,
        "episode_id": int(scale_state.episode_id),
        "indenter": episode_state.indenter,
        "indenter_split": episode_state.indenter_split,
        "is_unseen_indenter": bool(episode_state.is_unseen_indenter),
        "scale_requested_mm": float(scale_state.scale_requested_mm),
        "scale_simulated_mm": float(scale_state.scale_simulated_mm),
        "scale_mm": float(scale_state.scale_simulated_mm),
        "gel_width_mm": float(gel_dims.width_mm),
        "gel_height_mm": float(gel_dims.height_mm),
        "scale_quantized": bool(scale_state.scale_quantized),
        "scale_key": scale_state.scale_key,
        "contact_x_mm": float(scale_state.contact_x_mm),
        "contact_y_mm": float(scale_state.contact_y_mm),
        "contact_x_norm": float(scale_state.contact_x_norm),
        "contact_y_norm": float(scale_state.contact_y_norm),
        "sequence_length": int(len(scale_state.frames)),
        "phase_lengths": phase_lengths,
        "camera_mode": scale_state.camera_mode,
        "camera_distance_m": float(scale_state.camera_distance_m),
        "camera_fov_deg": float(scale_state.camera_fov_deg),
        "adapter_coord_map": str(scale_state.adapter_coord_map_path.relative_to(scale_state.scale_dir)),
        "adapter_coord_map_shape": [int(v) for v in scale_state.adapter_coord_map_shape],
        "scale_split": scale_state.scale_split,
        "is_unseen_scale": bool(scale_state.is_unseen_scale),
        "trajectory_length": int(scale_state.trajectory_length),
        "deformation_stats_final": legacy._json_safe(scale_state.deformation_stats_final),
        "physics_npz_relpath": legacy._safe_relpath(scale_state.physics_npz_path, episode_state.episode_dir),
        "physics_npz_deleted": bool(scale_state.physics_npz_deleted),
        "physics_npz_cleanup_policy": str(scale_state.physics_npz_cleanup_policy),
        "physics_reused": bool(scale_state.physics_reused),
        "marker_files_selected": [str(v) for v in marker_files_selected],
        "image_resolution": {"width": int(image_resolution[0]), "height": int(image_resolution[1])},
        "coordinate_convention": COORDINATE_CONVENTION,
        "frames": [frame.to_metadata_dict() for frame in scale_state.frames],
    }
    legacy.write_json_atomic(scale_state.scale_dir / "sequence_metadata.json", sequence_meta)
    episode_state.scales[scale_state.scale_key] = {
        "scale_requested_mm": float(scale_state.scale_requested_mm),
        "scale_simulated_mm": float(scale_state.scale_simulated_mm),
        "scale_mm": float(scale_state.scale_simulated_mm),
        "gel_width_mm": float(gel_dims.width_mm),
        "gel_height_mm": float(gel_dims.height_mm),
        "scale_quantized": bool(scale_state.scale_quantized),
        "scale_key": scale_state.scale_key,
        "sequence_metadata": str((scale_state.scale_dir / "sequence_metadata.json").relative_to(episode_state.episode_dir)),
        "contact_x_mm": float(scale_state.contact_x_mm),
        "contact_y_mm": float(scale_state.contact_y_mm),
        "contact_x_norm": float(scale_state.contact_x_norm),
        "contact_y_norm": float(scale_state.contact_y_norm),
        "sequence_length": int(len(scale_state.frames)),
        "phase_lengths": phase_lengths,
        "camera_mode": scale_state.camera_mode,
        "camera_distance_m": float(scale_state.camera_distance_m),
        "camera_fov_deg": float(scale_state.camera_fov_deg),
        "adapter_coord_map": str(scale_state.adapter_coord_map_path.relative_to(episode_state.episode_dir)),
        "adapter_coord_map_shape": [int(v) for v in scale_state.adapter_coord_map_shape],
        "marker_files_selected": [str(v) for v in marker_files_selected],
        "physics_npz_relpath": legacy._safe_relpath(scale_state.physics_npz_path, episode_state.episode_dir),
        "physics_npz_deleted": bool(scale_state.physics_npz_deleted),
        "physics_npz_cleanup_policy": str(scale_state.physics_npz_cleanup_policy),
        "scale_split": scale_state.scale_split,
        "is_unseen_scale": bool(scale_state.is_unseen_scale),
        "frames": {frame.frame_name: frame.to_metadata_dict() for frame in scale_state.frames},
    }
    return sequence_meta


def maybe_cleanup_completed_scale_npz(
    *,
    episode_state: SequenceEpisodeState,
    scale_state: SequenceScaleState,
    marker_files_selected: Sequence[str],
    keep_intermediates: bool,
) -> bool:
    if str(scale_state.physics_npz_cleanup_policy) != "delete_after_scale_complete":
        return False
    if not scale_state.adapter_coord_map_path.exists():
        return False
    expected_markers = sorted(str(v) for v in marker_files_selected)
    for frame in scale_state.frames:
        rendered = sorted(str(v) for v in frame.rendered_markers)
        if rendered != expected_markers:
            return False
        frame_dir = scale_state.scale_dir / str(frame.frame_name)
        actual = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        if actual != expected_markers:
            return False
    if scale_state.physics_npz_deleted:
        return True
    if not scale_state.physics_npz_path.exists():
        scale_state.physics_npz_deleted = True
        return True
    try:
        scale_state.physics_npz_path.unlink()
        scale_state.physics_npz_deleted = True
        parent = scale_state.physics_npz_path.parent
        grandparent = parent.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        if (not keep_intermediates) and grandparent.exists() and grandparent.name == "npz" and not any(grandparent.iterdir()):
            grandparent.rmdir()
        return True
    except Exception as exc:
        logging.warning(
            "NPZ cleanup skipped | ep=%06d scale=%s npz=%s err=%s",
            int(scale_state.episode_id),
            str(scale_state.scale_key),
            scale_state.physics_npz_path,
            exc,
        )
        return False


def scale_has_complete_render_outputs(
    *,
    scale_state: SequenceScaleState,
    marker_files_selected: Sequence[str],
) -> bool:
    expected_markers = sorted(str(v) for v in marker_files_selected)
    if not expected_markers:
        return False
    if not scale_state.adapter_coord_map_path.exists():
        return False
    for frame in scale_state.frames:
        rendered = sorted(str(v) for v in frame.rendered_markers)
        if rendered != expected_markers:
            return False
        frame_dir = scale_state.scale_dir / str(frame.frame_name)
        actual = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        if actual != expected_markers:
            return False
    return True


def maybe_finalize_completed_scale(
    *,
    episode_state: SequenceEpisodeState,
    scale_state: SequenceScaleState,
    dataset_profile: str,
    image_resolution: Tuple[int, int],
    marker_files_selected: Sequence[str],
    particle: str,
    keep_intermediates: bool,
) -> bool:
    if scale_state.sealed:
        return True
    if not scale_has_complete_render_outputs(scale_state=scale_state, marker_files_selected=marker_files_selected):
        return False
    try:
        persist_scale_sequence_metadata(
            episode_state=episode_state,
            scale_state=scale_state,
            dataset_profile=dataset_profile,
            image_resolution=image_resolution,
            marker_files_selected=marker_files_selected,
        )
        npz_deleted = maybe_cleanup_completed_scale_npz(
            episode_state=episode_state,
            scale_state=scale_state,
            marker_files_selected=marker_files_selected,
            keep_intermediates=keep_intermediates,
        )
        if npz_deleted:
            persist_scale_sequence_metadata(
                episode_state=episode_state,
                scale_state=scale_state,
                dataset_profile=dataset_profile,
                image_resolution=image_resolution,
                marker_files_selected=marker_files_selected,
            )
        persist_episode_metadata(episode_state, particle, dataset_profile)
        scale_state.sealed = True
        return True
    except Exception as exc:
        msg = f"Finalize scale failed | ep={episode_state.episode_id} scale={scale_state.scale_key} err={exc}"
        episode_state.errors.append(msg)
        logging.error(msg)
        return False


def build_run_status_payload(
    *,
    dataset_root: Path,
    dataset_profile: str,
    args: argparse.Namespace,
    phase: str,
    episode_states: Mapping[int, SequenceEpisodeState],
    message: str = "",
    source_sequence_dataset_root: Path | None = None,
) -> Dict[str, Any]:
    completed_episodes = sum(1 for state in episode_states.values() if not state.errors and state.scales)
    failed_episodes = sum(1 for state in episode_states.values() if state.errors)
    planning = getattr(args, "planning_summary", {})
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_profile": dataset_profile,
        "dataset_variant": SEQUENCE_VARIANT,
        "command": shlex.join([str(sys.executable), *sys.argv]),
        "phase": phase,
        "message": str(message),
        "scale_mode": str(args.scale_mode),
        "source_sequence_dataset_root": str(source_sequence_dataset_root) if source_sequence_dataset_root is not None else "",
        "episodes_per_indenter": int(args.episodes_per_indenter),
        "episode_count_total": int(len(episode_states)),
        "episode_count_completed": int(completed_episodes),
        "episode_count_failed": int(failed_episodes),
        "clip_index_expected": bool(dataset_profile == "B"),
        "pair_index_expected": bool(dataset_profile == "A"),
        "marker_textures_selected": [str(v) for v in getattr(args, "marker_textures_selected", [])],
        "marker_texture_count": int(len(getattr(args, "marker_textures_selected", []))),
        "marker_texture_selection_mode": str(getattr(args, "marker_texture_selection_mode", "")),
        "marker_texture_seed": int(getattr(args, "marker_texture_seed", args.seed)),
        "render_device": str(args.render_device),
        "render_gpu_backend": str(args.render_gpu_backend),
        "auto_balance_pipeline": bool(args.auto_balance_pipeline),
        "render_backlog_thresholds": {
            "high": int(resolve_render_backlog_thresholds(args)[0]),
            "low": int(resolve_render_backlog_thresholds(args)[1]),
        },
        "physics_npz_cleanup": str(args.physics_npz_cleanup),
        "keep_intermediates": bool(args.keep_intermediates),
        "cleanup_effective": bool(str(args.physics_npz_cleanup) == "delete_after_scale_complete"),
        "planning": legacy._json_safe(planning),
    }


def write_run_status(
    *,
    dataset_root: Path,
    dataset_profile: str,
    args: argparse.Namespace,
    phase: str,
    episode_states: Mapping[int, SequenceEpisodeState],
    message: str = "",
    source_sequence_dataset_root: Path | None = None,
) -> None:
    payload = build_run_status_payload(
        dataset_root=dataset_root,
        dataset_profile=dataset_profile,
        args=args,
        phase=phase,
        episode_states=episode_states,
        message=message,
        source_sequence_dataset_root=source_sequence_dataset_root,
    )
    legacy.write_json_atomic(dataset_root / "run_status.json", payload)


def build_manifest(
    *,
    dataset_root: Path,
    dataset_profile: str,
    args: argparse.Namespace,
    episode_states: Mapping[int, SequenceEpisodeState],
    patch_h: int,
    patch_w: int,
    source_sequence_dataset_root: Path | None = None,
    storage_mode: str = "full",
) -> Dict[str, Any]:
    manifest_episodes: List[Dict[str, Any]] = []
    requested_counts = {"clean": 0, "near_boundary": 0, "partial_crop": 0}
    actual_counts: Dict[str, int] = {}
    sequence_lengths: List[int] = []

    for episode_id in sorted(episode_states):
        state = episode_states[episode_id]
        if state.errors:
            continue
        scale_summaries = list(state.scales.values())
        if not scale_summaries:
            continue
        for scale_summary in scale_summaries:
            sequence_lengths.append(int(scale_summary["sequence_length"]))
        requested_counts[state.position_sample_type_requested] = requested_counts.get(state.position_sample_type_requested, 0) + 1
        actual_counts[state.position_sample_type_actual] = actual_counts.get(state.position_sample_type_actual, 0) + 1

        manifest_episodes.append(
            {
                "episode_id": int(state.episode_id),
                "path": state.episode_dir.name,
                "local_path": state.episode_dir.name,
                "source_path": state.episode_dir.name,
                "source_metadata_path": f"{state.episode_dir.name}/metadata.json",
                "indenter": state.indenter,
                "indenter_split": state.indenter_split,
                "is_unseen_indenter": bool(state.is_unseen_indenter),
                "latent_contact": {
                    "contact_x_norm": float(state.contact_x_norm),
                    "contact_y_norm": float(state.contact_y_norm),
                    "final_depth_mm": float(state.final_depth_mm),
                },
                "scales": [
                    {
                        "scale_key": str(scale_summary["scale_key"]),
                        "scale_requested_mm": float(scale_summary["scale_requested_mm"]),
                        "scale_simulated_mm": float(scale_summary["scale_simulated_mm"]),
                        "gel_width_mm": float(scale_summary.get("gel_width_mm", scale_summary["scale_simulated_mm"])),
                        "gel_height_mm": float(
                            scale_summary.get(
                                "gel_height_mm",
                                rectgel.gel_dims_from_width_mm(scale_summary["scale_simulated_mm"]).height_mm,
                            )
                        ),
                        "scale_split": str(scale_summary["scale_split"]),
                        "is_unseen_scale": bool(scale_summary["is_unseen_scale"]),
                        "sequence_metadata": str(scale_summary["sequence_metadata"]),
                    }
                    for scale_summary in scale_summaries
                ],
            }
        )

    unique_lengths = sorted(set(sequence_lengths))
    if not unique_lengths:
        sequence_length_summary: int | Dict[str, Any] = 0
    elif len(unique_lengths) == 1:
        sequence_length_summary = int(unique_lengths[0])
    else:
        sequence_length_summary = {
            "min": int(min(unique_lengths)),
            "max": int(max(unique_lengths)),
            "unique_values": [int(v) for v in unique_lengths],
        }

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_variant": SEQUENCE_VARIANT,
        "dataset_profile": dataset_profile,
        "storage_mode": str(storage_mode),
        "derived_from_sequence_root": str(source_sequence_dataset_root) if source_sequence_dataset_root is not None else "",
        "run_status_json": "run_status.json",
        "command": shlex.join([str(sys.executable), *sys.argv]),
        "particle": str(args.particle),
        "scale_mode": str(args.scale_mode),
        "discrete_gel_widths_mm": [float(v) for v in args.scales_mm] if args.scale_mode == "discrete" else None,
        "continuous_gel_width_min_mm": None if args.scale_min_mm is None else float(args.scale_min_mm),
        "continuous_gel_width_max_mm": None if args.scale_max_mm is None else float(args.scale_max_mm),
        "discrete_scales_mm": [float(v) for v in args.scales_mm] if args.scale_mode == "discrete" else None,
        "continuous_scale_min_mm": None if args.scale_min_mm is None else float(args.scale_min_mm),
        "continuous_scale_max_mm": None if args.scale_max_mm is None else float(args.scale_max_mm),
        "scales_per_episode": int(args.scales_per_episode) if args.scales_per_episode else None,
        "continuous_scale_quantization_mm": None
        if args.continuous_scale_quantization_mm is None
        else float(args.continuous_scale_quantization_mm),
        "heldout_scales_mm": [float(v) for v in (args.heldout_scales_mm or [])],
        "heldout_scale_band_mm": float(args.heldout_scale_band_mm),
        "heldout_val_scales_mm": [float(v) for v in (args.heldout_val_scales_mm or [])],
        "heldout_test_scales_mm": [float(v) for v in (args.heldout_test_scales_mm or [])],
        "heldout_val_scale_band_mm": float(args.heldout_val_scale_band_mm),
        "heldout_test_scale_band_mm": float(args.heldout_test_scale_band_mm),
        "train_scale_intervals_mm": [list(interval) for interval in getattr(args, "train_scale_intervals_mm_parsed", [])],
        "val_scale_intervals_mm": [list(interval) for interval in getattr(args, "val_scale_intervals_mm_parsed", [])],
        "test_scale_intervals_mm": [list(interval) for interval in getattr(args, "test_scale_intervals_mm_parsed", [])],
        "allowed_scale_splits": [str(v) for v in args.allowed_scale_splits],
        "pair_split_policy": str(args.pair_split_policy),
        "marker_textures_selected": [str(v) for v in getattr(args, "marker_textures_selected", [])],
        "marker_texture_count": int(len(getattr(args, "marker_textures_selected", []))),
        "marker_texture_selection_mode": str(getattr(args, "marker_texture_selection_mode", "all")),
        "marker_texture_seed": int(getattr(args, "marker_texture_seed", args.seed)),
        "render_device": str(args.render_device),
        "render_gpu_backend": str(args.render_gpu_backend),
        "auto_balance_pipeline": bool(args.auto_balance_pipeline),
        "render_backlog_thresholds": {
            "high": int(resolve_render_backlog_thresholds(args)[0]),
            "low": int(resolve_render_backlog_thresholds(args)[1]),
        },
        "physics_npz_cleanup": str(args.physics_npz_cleanup),
        "keep_intermediates": bool(args.keep_intermediates),
        "cleanup_effective": bool(str(args.physics_npz_cleanup) == "delete_after_scale_complete"),
        "patch_grid": [int(patch_h), int(patch_w)],
        "image_resolution": [int(legacy.DEFAULT_IMAGE_RES[0]), int(legacy.DEFAULT_IMAGE_RES[1])],
        "coordinate_convention": COORDINATE_CONVENTION,
        "gel_aspect_ratio": "4:3",
        "camera_mode": str(args.camera_mode),
        "base_fov_deg": float(args.fov_deg),
        "reference_gel_width_mm": float(args.reference_scale_mm),
        "reference_gel_height_mm": float(rectgel.gel_dims_from_width_mm(args.reference_scale_mm).height_mm),
        "reference_scale_mm": float(args.reference_scale_mm),
        "distance_safety": float(args.distance_safety),
        "uv_mode": str(args.uv_mode),
        "uv_inset_ratio": float(args.uv_inset_ratio),
        "frame_sampling_mode": "sequence_temporal_plan",
        "position_sample_ratios": {
            "clean": float(args.position_clean_ratio),
            "near_boundary": float(args.position_near_boundary_ratio),
            "partial_crop": float(args.position_partial_crop_ratio),
        },
        "near_boundary_max_margin_mm": float(args.near_boundary_max_margin_mm),
        "partial_crop_min_overhang_mm": float(args.partial_crop_min_overhang_mm),
        "partial_crop_max_overhang_mm": float(args.partial_crop_max_overhang_mm),
        "position_sample_counts_requested": {k: int(v) for k, v in requested_counts.items()},
        "position_sample_counts_actual": {str(k): int(v) for k, v in actual_counts.items()},
        "sequence_length_summary": sequence_length_summary,
        "episodes_per_indenter": int(args.episodes_per_indenter),
        "indenter_splits": {
            "train": [str(v) for v in args.train_indenters],
            "val": [str(v) for v in args.val_indenters],
            "test": [str(v) for v in args.test_indenters],
        },
        "calibration_bundle_hint_path": str(args.calibration_bundle_hint_path) if args.calibration_bundle_hint_path else "",
        "source_sequence_dataset_root": str(source_sequence_dataset_root) if source_sequence_dataset_root is not None else "",
        "planning": legacy._json_safe(getattr(args, "planning_summary", {})),
        "disk_estimate": legacy._json_safe(getattr(args, "planning_summary", {}).get("disk_estimate", {})),
        "pair_index_files": {
            "all": "pair_index.jsonl",
            "train": "pair_index_train.jsonl",
            "val": "pair_index_val.jsonl",
            "test": "pair_index_test.jsonl",
        }
        if dataset_profile == "A"
        else {},
        "clip_index_files": {"all": "clip_index.jsonl"} if dataset_profile == "B" else {},
        "episodes": manifest_episodes,
    }
    return manifest


def ensure_episode_symlink(local_path: Path, source_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_symlink() or local_path.exists():
        if local_path.is_symlink() and local_path.resolve() == source_path.resolve():
            return
        if local_path.is_dir() and not local_path.is_symlink():
            raise FileExistsError(f"Cannot replace existing directory with symlink: {local_path}")
        local_path.unlink()
    local_path.symlink_to(source_path, target_is_directory=True)


def build_derived_profile_b_manifest(
    *,
    dataset_root: Path,
    args: argparse.Namespace,
    source_root: Path,
    source_manifest: Mapping[str, Any],
    storage_mode: str,
) -> Dict[str, Any]:
    manifest = dict(source_manifest)
    manifest["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["dataset_root"] = str(dataset_root)
    manifest["dataset_profile"] = "B"
    manifest["dataset_variant"] = SEQUENCE_VARIANT
    manifest["run_status_json"] = "run_status.json"
    manifest["command"] = shlex.join([str(sys.executable), *sys.argv])
    manifest["storage_mode"] = str(storage_mode)
    manifest["derived_from_sequence_root"] = str(source_root)
    manifest["source_sequence_dataset_root"] = str(source_root)
    manifest["derive_storage_mode"] = str(storage_mode)
    manifest["allowed_scale_splits"] = [str(v) for v in args.allowed_scale_splits]
    manifest["pair_split_policy"] = str(args.pair_split_policy)
    manifest["pair_index_files"] = {}
    manifest["clip_index_files"] = {"all": "clip_index.jsonl"}
    manifest["clip_length"] = int(args.clip_length)
    manifest["clip_window_stride"] = int(args.clip_window_stride)
    manifest["clip_gap_min"] = int(args.clip_gap_min)
    manifest["clip_gap_max"] = int(args.clip_gap_max)
    manifest["clip_phase_policy"] = str(args.clip_phase_policy)
    manifest["clip_balance_seed"] = int(args.clip_balance_seed)
    manifest["max_clips_per_scale"] = int(args.max_clips_per_scale)

    episodes_out: List[Dict[str, Any]] = []
    for ep_entry in source_manifest.get("episodes", []):
        episode_path = str(ep_entry.get("path", ""))
        episode_copy = dict(ep_entry)
        episode_copy["source_path"] = episode_path
        episode_copy["source_metadata_path"] = f"{episode_path}/metadata.json" if episode_path else ""
        if storage_mode == "symlink":
            episode_copy["path"] = episode_path
            episode_copy["local_path"] = episode_path
        else:
            episode_copy["path"] = None
            episode_copy["local_path"] = ""
        episodes_out.append(episode_copy)
    manifest["episodes"] = episodes_out
    return manifest


def load_scale_sequence_metadata(sequence_meta_path: Path) -> Dict[str, Any]:
    with open(sequence_meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_sequence_dataset_records(dataset_root: Path) -> List[Dict[str, Any]]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records: List[Dict[str, Any]] = []
    for ep_entry in manifest.get("episodes", []):
        episode_dir = dataset_root / str(ep_entry["path"])
        episode_meta_path = episode_dir / "metadata.json"
        if not episode_meta_path.exists():
            continue
        with open(episode_meta_path, "r", encoding="utf-8") as f:
            episode_meta = json.load(f)
        scale_payloads: Dict[str, Dict[str, Any]] = {}
        for scale_key, scale_summary in episode_meta.get("scales", {}).items():
            seq_rel = scale_summary.get("sequence_metadata")
            if not seq_rel:
                continue
            seq_path = episode_dir / seq_rel
            if not seq_path.exists():
                continue
            scale_payloads[str(scale_key)] = load_scale_sequence_metadata(seq_path)
        records.append(
            {
                "episode_dir": episode_dir,
                "episode_meta": episode_meta,
                "scale_payloads": scale_payloads,
            }
        )
    return records


def build_sequence_index_records(
    *,
    dataset_root: Path,
    dataset_profile: str,
    records: Sequence[Dict[str, Any]],
    source_dataset_root: Path | None = None,
) -> List[Dict[str, Any]]:
    source_root = source_dataset_root if source_dataset_root is not None else dataset_root
    out: List[Dict[str, Any]] = []
    for item in records:
        episode_dir = item["episode_dir"]
        episode_meta = item["episode_meta"]
        latent_contact = episode_meta.get("latent_contact", {})
        for scale_key, scale_meta in item["scale_payloads"].items():
            adapter_rel = str(scale_meta.get("adapter_coord_map", "adapter_coord_map.npy"))
            adapter_abs = str((episode_dir / scale_key / adapter_rel).resolve())
            scale_dir = episode_dir / scale_key
            for frame in scale_meta.get("frames", []):
                marker_relpaths = {
                    str(marker_name): legacy._safe_relpath(scale_dir / str(frame["frame_name"]) / str(marker_name), source_root)
                    for marker_name in frame.get("rendered_markers", [])
                }
                out.append(
                    {
                        "dataset_root": str(dataset_root),
                        "dataset_variant": SEQUENCE_VARIANT,
                        "dataset_profile": dataset_profile,
                        "source_dataset_root": str(source_root),
                        "episode_id": int(episode_meta["episode_id"]),
                        "episode_dir": legacy._safe_relpath(episode_dir, source_root),
                        "indenter_name": episode_meta["indenter"],
                        "indenter_split": episode_meta.get("indenter_split", ""),
                        "is_unseen_indenter": bool(episode_meta.get("is_unseen_indenter", False)),
                        "scale_key": scale_key,
                        "scale_requested_mm": float(scale_meta["scale_requested_mm"]),
                        "scale_simulated_mm": float(scale_meta["scale_simulated_mm"]),
                        "gel_width_mm": float(scale_meta.get("gel_width_mm", scale_meta["scale_simulated_mm"])),
                        "gel_height_mm": float(
                            scale_meta.get(
                                "gel_height_mm",
                                rectgel.gel_dims_from_width_mm(scale_meta["scale_simulated_mm"]).height_mm,
                            )
                        ),
                        "scale_split": scale_meta.get("scale_split", ""),
                        "is_unseen_scale": bool(scale_meta.get("is_unseen_scale", False)),
                        "global_seq_index": int(frame["global_seq_index"]),
                        "frame_name": frame["frame_name"],
                        "phase_name": frame["phase_name"],
                        "phase_index": int(frame["phase_index"]),
                        "phase_progress": float(frame["phase_progress"]),
                        "source_physics_frame_index": int(frame["source_physics_frame_index"]),
                        "frame_fraction_requested": frame.get("frame_fraction_requested"),
                        "frame_fraction_actual": frame.get("frame_fraction_actual"),
                        "frame_target_max_down_mm": frame.get("frame_target_max_down_mm"),
                        "frame_actual_max_down_mm": float(frame["frame_actual_max_down_mm"]),
                        "command_x_mm": float(scale_meta["contact_x_mm"]),
                        "command_y_mm": float(scale_meta["contact_y_mm"]),
                        "command_x_norm": float(scale_meta["contact_x_norm"]),
                        "command_y_norm": float(scale_meta["contact_y_norm"]),
                        "adapter_coord_map_relpath": legacy._safe_relpath(episode_dir / scale_key / adapter_rel, source_root),
                        "adapter_coord_map_abspath": adapter_abs,
                        "marker_image_relpaths": marker_relpaths,
                        "rendered_markers": [str(v) for v in frame.get("rendered_markers", [])],
                        "contact_x_norm": float(latent_contact.get("contact_x_norm", scale_meta["contact_x_norm"])),
                        "contact_y_norm": float(latent_contact.get("contact_y_norm", scale_meta["contact_y_norm"])),
                    }
                )
    return out


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(legacy._json_safe(dict(row)), ensure_ascii=False) + "\n")
            count += 1
    return count


def rebuild_sequence_image_index_csv(dataset_root: Path, sequence_index_records: Sequence[Mapping[str, Any]]) -> Tuple[Path, int]:
    csv_path = dataset_root / "image_index.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGE_INDEX_COLUMNS)
        writer.writeheader()
        rows_written = 0
        for record in sequence_index_records:
            for marker_name, image_relpath in sorted(record["marker_image_relpaths"].items()):
                image_abspath = str((Path(record["source_dataset_root"]) / image_relpath).resolve())
                row = {
                    "dataset_root": str(record["dataset_root"]),
                    "dataset_variant": str(record["dataset_variant"]),
                    "dataset_profile": str(record["dataset_profile"]),
                    "episode_id": int(record["episode_id"]),
                    "episode_dir": str(record["episode_dir"]),
                    "indenter_name": str(record["indenter_name"]),
                    "indenter_split": str(record["indenter_split"]),
                    "is_unseen_indenter": bool(record["is_unseen_indenter"]),
                    "scale_key": str(record["scale_key"]),
                    "scale_requested_mm": float(record["scale_requested_mm"]),
                    "scale_simulated_mm": float(record["scale_simulated_mm"]),
                    "gel_width_mm": float(record["gel_width_mm"]),
                    "gel_height_mm": float(record["gel_height_mm"]),
                    "scale_split": str(record["scale_split"]),
                    "is_unseen_scale": bool(record["is_unseen_scale"]),
                    "global_seq_index": int(record["global_seq_index"]),
                    "phase_name": str(record["phase_name"]),
                    "phase_index": int(record["phase_index"]),
                    "phase_progress": float(record["phase_progress"]),
                    "frame_name": str(record["frame_name"]),
                    "frame_actual_max_down_mm": float(record["frame_actual_max_down_mm"]),
                    "command_x_mm": float(record["command_x_mm"]),
                    "command_y_mm": float(record["command_y_mm"]),
                    "command_x_norm": float(record["command_x_norm"]),
                    "command_y_norm": float(record["command_y_norm"]),
                    "marker_name": str(marker_name),
                    "image_relpath": str(image_relpath),
                    "image_abspath": image_abspath,
                    "adapter_coord_map_relpath": str(record["adapter_coord_map_relpath"]),
                    "adapter_coord_map_abspath": str(record["adapter_coord_map_abspath"]),
                }
                writer.writerow({key: legacy._csv_cell(row.get(key)) for key in IMAGE_INDEX_COLUMNS})
                rows_written += 1
    return csv_path, rows_written


def build_pair_index_for_adapter_dataset(
    *,
    dataset_root: Path,
    records: Sequence[Dict[str, Any]],
    pair_marker_policy: str,
    include_identity_pairs: bool,
    pair_split_policy: str,
    source_dataset_root: Path | None = None,
) -> Dict[str, int]:
    source_root = source_dataset_root if source_dataset_root is not None else dataset_root
    paths = {
        "all": dataset_root / "pair_index.jsonl",
        "train": dataset_root / "pair_index_train.jsonl",
        "val": dataset_root / "pair_index_val.jsonl",
        "test": dataset_root / "pair_index_test.jsonl",
    }
    counts = {name: 0 for name in paths}

    with open(paths["all"], "w", encoding="utf-8") as f_all, \
        open(paths["train"], "w", encoding="utf-8") as f_train, \
        open(paths["val"], "w", encoding="utf-8") as f_val, \
        open(paths["test"], "w", encoding="utf-8") as f_test:
        split_writers = {"train": f_train, "val": f_val, "test": f_test}
        for item in records:
            episode_meta = item["episode_meta"]
            scale_payloads = item["scale_payloads"]
            scale_keys = sorted(scale_payloads.keys())
            if not scale_keys:
                continue

            for source_key in scale_keys:
                for target_key in scale_keys:
                    if source_key == target_key and not include_identity_pairs:
                        continue
                    if source_key not in scale_payloads or target_key not in scale_payloads:
                        continue
                    source_scale = scale_payloads[source_key]
                    target_scale = scale_payloads[target_key]
                    source_scale_split = str(source_scale.get("scale_split", ""))
                    target_scale_split = str(target_scale.get("scale_split", ""))
                    if source_scale_split == target_scale_split and source_scale_split in {"train", "val", "test"}:
                        pair_split = source_scale_split
                    else:
                        pair_split = "cross_split"
                    if pair_split_policy == "same_split_only" and pair_split == "cross_split":
                        continue
                    source_frames = {int(frame["global_seq_index"]): frame for frame in source_scale.get("frames", [])}
                    target_frames = {int(frame["global_seq_index"]): frame for frame in target_scale.get("frames", [])}
                    shared_seq_indices = sorted(set(source_frames.keys()) & set(target_frames.keys()))
                    for seq_idx in shared_seq_indices:
                        s_frame = source_frames[seq_idx]
                        t_frame = target_frames[seq_idx]
                        source_markers = [str(v) for v in s_frame.get("rendered_markers", [])]
                        target_markers = [str(v) for v in t_frame.get("rendered_markers", [])]
                        if pair_marker_policy == "same_texture_only":
                            marker_pairs = [(name, name) for name in sorted(set(source_markers) & set(target_markers))]
                        else:
                            marker_pairs = [(s, t) for s in source_markers for t in target_markers]

                        for source_marker_name, target_marker_name in marker_pairs:
                            row = {
                                "episode_id": int(episode_meta["episode_id"]),
                                "indenter": episode_meta["indenter"],
                                "marker_name": source_marker_name
                                if source_marker_name == target_marker_name
                                else f"{source_marker_name}__to__{target_marker_name}",
                                "source_marker_name": source_marker_name,
                                "target_marker_name": target_marker_name,
                                "source_scale_mm": float(source_scale["scale_simulated_mm"]),
                                "target_scale_mm": float(target_scale["scale_simulated_mm"]),
                                "source_gel_width_mm": float(source_scale.get("gel_width_mm", source_scale["scale_simulated_mm"])),
                                "source_gel_height_mm": float(
                                    source_scale.get(
                                        "gel_height_mm",
                                        rectgel.gel_dims_from_width_mm(source_scale["scale_simulated_mm"]).height_mm,
                                    )
                                ),
                                "target_gel_width_mm": float(target_scale.get("gel_width_mm", target_scale["scale_simulated_mm"])),
                                "target_gel_height_mm": float(
                                    target_scale.get(
                                        "gel_height_mm",
                                        rectgel.gel_dims_from_width_mm(target_scale["scale_simulated_mm"]).height_mm,
                                    )
                                ),
                                "source_scale_key": source_key,
                                "target_scale_key": target_key,
                                "global_seq_index": int(seq_idx),
                                "phase_name": s_frame["phase_name"],
                                "phase_progress": float(s_frame["phase_progress"]),
                                "source_frame_relpath": legacy._safe_relpath(
                                    item["episode_dir"] / source_key / s_frame["frame_name"] / source_marker_name,
                                    source_root,
                                ),
                                "target_frame_relpath": legacy._safe_relpath(
                                    item["episode_dir"] / target_key / t_frame["frame_name"] / target_marker_name,
                                    source_root,
                                ),
                                "source_adapter_coord_map_relpath": legacy._safe_relpath(
                                    item["episode_dir"] / source_key / source_scale.get("adapter_coord_map", "adapter_coord_map.npy"),
                                    source_root,
                                ),
                                "target_adapter_coord_map_relpath": legacy._safe_relpath(
                                    item["episode_dir"] / target_key / target_scale.get("adapter_coord_map", "adapter_coord_map.npy"),
                                    source_root,
                                ),
                                "contact_x_norm": float(episode_meta["latent_contact"]["contact_x_norm"]),
                                "contact_y_norm": float(episode_meta["latent_contact"]["contact_y_norm"]),
                                "source_contact_x_mm": float(source_scale["contact_x_mm"]),
                                "source_contact_y_mm": float(source_scale["contact_y_mm"]),
                                "target_contact_x_mm": float(target_scale["contact_x_mm"]),
                                "target_contact_y_mm": float(target_scale["contact_y_mm"]),
                                "frame_depth_mm": float(t_frame["frame_actual_max_down_mm"]),
                                "source_frame_depth_mm": float(s_frame["frame_actual_max_down_mm"]),
                                "target_frame_depth_mm": float(t_frame["frame_actual_max_down_mm"]),
                                "indenter_split": episode_meta.get("indenter_split", ""),
                                "source_scale_split": source_scale_split,
                                "target_scale_split": target_scale_split,
                                "scale_split_source": source_scale_split,
                                "scale_split_target": target_scale_split,
                                "pair_split": pair_split,
                                "is_unseen_indenter": bool(episode_meta.get("is_unseen_indenter", False)),
                                "is_unseen_scale_source": bool(source_scale.get("is_unseen_scale", False)),
                                "is_unseen_scale_target": bool(target_scale.get("is_unseen_scale", False)),
                            }
                            line = json.dumps(legacy._json_safe(row), ensure_ascii=False) + "\n"
                            f_all.write(line)
                            counts["all"] += 1
                            if pair_split in split_writers:
                                split_writers[pair_split].write(line)
                                counts[pair_split] += 1
    return counts


def _clip_window_indices(
    sequence_length: int,
    clip_length: int,
    gap_min: int,
    gap_max: int,
) -> Iterator[List[int]]:
    if clip_length <= 0:
        return
    if clip_length == 1:
        for idx in range(sequence_length):
            yield [idx]
        return

    def _dfs(prefix: List[int]) -> Iterator[List[int]]:
        if len(prefix) == clip_length:
            yield prefix.copy()
            return
        start = 0 if not prefix else prefix[-1] + gap_min + 1
        for idx in range(start, sequence_length):
            if prefix:
                gap = idx - prefix[-1] - 1
                if gap < gap_min or gap > gap_max:
                    if gap > gap_max:
                        break
                    continue
            prefix.append(idx)
            yield from _dfs(prefix)
            prefix.pop()

    yield from _dfs([])


def _clip_phase_policy_ok(phase_names: Sequence[str], policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "press_only":
        return all(name == "press" for name in phase_names)
    if policy == "press_dwell":
        return all(name in {"press", "dwell"} for name in phase_names)
    if policy == "balanced":
        return True
    raise ValueError(f"Unsupported clip-phase-policy: {policy}")


def build_clip_index_for_downstream_dataset(
    *,
    dataset_root: Path,
    records: Sequence[Dict[str, Any]],
    clip_length: int,
    clip_window_stride: int,
    clip_gap_min: int,
    clip_gap_max: int,
    clip_phase_policy: str,
    clip_balance_seed: int,
    max_clips_per_scale: int,
    storage_mode: str,
    source_dataset_root: Path | None = None,
) -> int:
    source_root = source_dataset_root if source_dataset_root is not None else dataset_root
    path = dataset_root / "clip_index.jsonl"
    row_count = 0

    def _candidate_row(
        *,
        item: Dict[str, Any],
        scale_key: str,
        scale_meta: Dict[str, Any],
        marker_name: str,
        window_frames: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        target_frame = window_frames[-1]
        phase_names = [str(frame["phase_name"]) for frame in window_frames]
        return {
            "clip_id": (
                f"clip_{int(item['episode_meta']['episode_id']):06d}_{scale_key}_"
                f"{Path(marker_name).stem}_{int(window_frames[0]['global_seq_index']):06d}_"
                f"{int(target_frame['global_seq_index']):06d}"
            ),
            "episode_id": int(item["episode_meta"]["episode_id"]),
            "indenter": item["episode_meta"]["indenter"],
            "scale_mm": float(scale_meta["scale_simulated_mm"]),
            "scale_key": scale_key,
            "marker_name": marker_name,
            "frame_count": int(clip_length),
            "frame_relpaths": [
                legacy._safe_relpath(item["episode_dir"] / scale_key / frame["frame_name"] / marker_name, source_root)
                for frame in window_frames
            ],
            "frame_indices": [int(frame["global_seq_index"]) for frame in window_frames],
            "phase_names": phase_names,
            "target_phase_name": phase_names[-1],
            "target_frame_index": int(target_frame["global_seq_index"]),
            "target_x_norm": float(scale_meta["contact_x_norm"]),
            "target_y_norm": float(scale_meta["contact_y_norm"]),
            "target_depth_mm": float(target_frame["frame_actual_max_down_mm"]),
            "target_contact_x_mm": float(scale_meta["contact_x_mm"]),
            "target_contact_y_mm": float(scale_meta["contact_y_mm"]),
            "indenter_split": item["episode_meta"].get("indenter_split", ""),
            "scale_split": scale_meta.get("scale_split", ""),
            "target_scale_split": scale_meta.get("scale_split", ""),
            "is_unseen_indenter": bool(item["episode_meta"].get("is_unseen_indenter", False)),
            "is_unseen_scale": bool(scale_meta.get("is_unseen_scale", False)),
            "target_is_unseen_scale": bool(scale_meta.get("is_unseen_scale", False)),
            "source_dataset_root": str(source_root),
            "storage_mode": str(storage_mode),
        }

    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            episode_meta = item["episode_meta"]
            for scale_key, scale_meta in item["scale_payloads"].items():
                frames = list(scale_meta.get("frames", []))
                if len(frames) < clip_length:
                    continue
                frames_sorted = sorted(frames, key=lambda frame: int(frame["global_seq_index"]))
                marker_names = [str(v) for v in frames_sorted[0].get("rendered_markers", [])]
                if not marker_names:
                    continue

                if clip_phase_policy == "balanced":
                    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
                    phase_cap = max_clips_per_scale if max_clips_per_scale > 0 else 10000
                    for window_idx, window_positions in enumerate(_clip_window_indices(len(frames_sorted), clip_length, clip_gap_min, clip_gap_max)):
                        if clip_window_stride > 1 and window_idx % clip_window_stride != 0:
                            continue
                        window_frames = [frames_sorted[pos] for pos in window_positions]
                        phase_names = [str(frame["phase_name"]) for frame in window_frames]
                        if not _clip_phase_policy_ok(phase_names, clip_phase_policy):
                            continue
                        target_phase = phase_names[-1]
                        bucket = grouped_rows.setdefault(target_phase, [])
                        if len(bucket) >= phase_cap:
                            continue
                        for marker_name in marker_names:
                            bucket.append(
                                _candidate_row(
                                    item=item,
                                    scale_key=scale_key,
                                    scale_meta=scale_meta,
                                    marker_name=marker_name,
                                    window_frames=window_frames,
                                )
                            )
                    if not grouped_rows:
                        continue
                    min_count = min(len(rows) for rows in grouped_rows.values())
                    phase_rng = random.Random(f"{clip_balance_seed}:{int(episode_meta['episode_id'])}:{scale_key}")
                    for phase_name in sorted(grouped_rows.keys()):
                        rows = grouped_rows[phase_name]
                        phase_rng.shuffle(rows)
                        write_count = min_count
                        if max_clips_per_scale > 0:
                            write_count = min(write_count, max_clips_per_scale)
                        for row in rows[:write_count]:
                            f.write(json.dumps(legacy._json_safe(row), ensure_ascii=False) + "\n")
                            row_count += 1
                    continue

                per_marker_counts = {marker_name: 0 for marker_name in marker_names}
                for window_idx, window_positions in enumerate(_clip_window_indices(len(frames_sorted), clip_length, clip_gap_min, clip_gap_max)):
                    if clip_window_stride > 1 and window_idx % clip_window_stride != 0:
                        continue
                    if max_clips_per_scale > 0 and all(count >= max_clips_per_scale for count in per_marker_counts.values()):
                        break
                    window_frames = [frames_sorted[pos] for pos in window_positions]
                    phase_names = [str(frame["phase_name"]) for frame in window_frames]
                    if not _clip_phase_policy_ok(phase_names, clip_phase_policy):
                        continue
                    for marker_name in marker_names:
                        if max_clips_per_scale > 0 and per_marker_counts[marker_name] >= max_clips_per_scale:
                            continue
                        row = _candidate_row(
                            item=item,
                            scale_key=scale_key,
                            scale_meta=scale_meta,
                            marker_name=marker_name,
                            window_frames=window_frames,
                        )
                        f.write(json.dumps(legacy._json_safe(row), ensure_ascii=False) + "\n")
                        row_count += 1
                        per_marker_counts[marker_name] += 1

    return row_count


def ensure_output_root(dataset_root: Path, args: argparse.Namespace) -> None:
    if args.clean_output and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)


def finalize_artifacts(
    *,
    dataset_root: Path,
    dataset_profile: str,
    args: argparse.Namespace,
    source_dataset_root: Path | None = None,
    storage_mode: str = "full",
) -> Dict[str, int]:
    records = scan_sequence_dataset_records(source_dataset_root if source_dataset_root is not None else dataset_root)
    sequence_index_records = build_sequence_index_records(
        dataset_root=dataset_root,
        dataset_profile=dataset_profile,
        records=records,
        source_dataset_root=source_dataset_root,
    )
    sequence_index_count = write_jsonl(dataset_root / "sequence_index.jsonl", sequence_index_records)
    _, image_index_count = rebuild_sequence_image_index_csv(dataset_root, sequence_index_records)

    pair_count = 0
    clip_count = 0
    if dataset_profile == "A":
        pair_counts = build_pair_index_for_adapter_dataset(
            dataset_root=dataset_root,
            records=records,
            pair_marker_policy=args.pair_marker_policy,
            include_identity_pairs=bool(args.include_identity_pairs),
            pair_split_policy=str(args.pair_split_policy),
            source_dataset_root=source_dataset_root,
        )
        pair_count = int(pair_counts["all"])
    else:
        clip_count = build_clip_index_for_downstream_dataset(
            dataset_root=dataset_root,
            records=records,
            clip_length=int(args.clip_length),
            clip_window_stride=int(args.clip_window_stride),
            clip_gap_min=int(args.clip_gap_min),
            clip_gap_max=int(args.clip_gap_max),
            clip_phase_policy=str(args.clip_phase_policy),
            clip_balance_seed=int(args.clip_balance_seed),
            max_clips_per_scale=int(args.max_clips_per_scale),
            storage_mode=str(storage_mode),
            source_dataset_root=source_dataset_root,
        )
    return {
        "sequence_index_count": int(sequence_index_count),
        "image_index_count": int(image_index_count),
        "pair_index_count": int(pair_count),
        "clip_index_count": int(clip_count),
    }


def build_episode_states_for_generation(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    indenter_names: Sequence[str],
    indenter_bboxes: Mapping[str, Dict[str, float]],
    expected_marker_files: Sequence[str],
    patch_h: int,
    patch_w: int,
    rng: random.Random,
    materialize_dirs: bool = True,
    persist_initial_metadata: bool = True,
) -> Tuple[Dict[int, SequenceEpisodeState], List[Dict[str, Any]]]:
    episode_states: Dict[int, SequenceEpisodeState] = {}
    physics_tasks: List[Dict[str, Any]] = []

    requested_sample_types_by_indenter = {
        indenter: legacy.assign_position_sample_types(
            episodes_per_indenter=int(args.episodes_per_indenter),
            clean_ratio=float(args.position_clean_ratio),
            near_boundary_ratio=float(args.position_near_boundary_ratio),
            partial_crop_ratio=float(args.position_partial_crop_ratio),
            rng=rng,
        )
        for indenter in indenter_names
    }

    episode_id = 0
    base_parameters = args.repo_root / "sim" / "parameters.yml"
    fixed_distance_m = (
        float(args.camera_distance_m)
        if args.camera_distance_m is not None
        else compute_fixed_camera_distance_m(
            reference_scale_mm=float(args.reference_scale_mm),
            base_fov_deg=float(args.fov_deg),
            distance_safety=float(args.distance_safety),
        )
    )

    for indenter in indenter_names:
        indenter_split, is_unseen_indenter = determine_indenter_split(indenter, args)
        for requested_sample_type in requested_sample_types_by_indenter[indenter]:
            scale_entries = sample_episode_scale_values_with_split_filter(args=args, rng=rng)
            scale_values = [float(entry["scale_simulated_mm"]) for entry in scale_entries]
            contact_constraint_scale_mm = float(min(scale_values))
            position_reference_scale_mm = float(max(scale_values))
            contact_constraint_dims = rectgel.gel_dims_from_width_mm(contact_constraint_scale_mm)
            position_reference_dims = rectgel.gel_dims_from_width_mm(position_reference_scale_mm)

            bbox_mm = indenter_bboxes[indenter]
            clean_bounds = rectgel.compute_margin_constrained_normalized_sampling_bounds(
                bbox_mm=bbox_mm,
                gel_width_mm=float(contact_constraint_dims.width_mm),
                gel_height_mm=float(contact_constraint_dims.height_mm),
                reference_width_mm=float(position_reference_dims.width_mm),
                reference_height_mm=float(position_reference_dims.height_mm),
                requested_x_min_mm=float(args.x_min),
                requested_x_max_mm=float(args.x_max),
                requested_y_min_mm=float(args.y_min),
                requested_y_max_mm=float(args.y_max),
                min_margin_mm=float(args.near_boundary_max_margin_mm),
            )
            inside_bounds_min_scale = rectgel.compute_margin_constrained_normalized_sampling_bounds(
                bbox_mm=bbox_mm,
                gel_width_mm=float(contact_constraint_dims.width_mm),
                gel_height_mm=float(contact_constraint_dims.height_mm),
                reference_width_mm=float(position_reference_dims.width_mm),
                reference_height_mm=float(position_reference_dims.height_mm),
                requested_x_min_mm=float(args.x_min),
                requested_x_max_mm=float(args.x_max),
                requested_y_min_mm=float(args.y_min),
                requested_y_max_mm=float(args.y_max),
                min_margin_mm=0.0,
            )
            inside_bounds_reference_scale = rectgel.compute_margin_constrained_normalized_sampling_bounds(
                bbox_mm=bbox_mm,
                gel_width_mm=float(position_reference_dims.width_mm),
                gel_height_mm=float(position_reference_dims.height_mm),
                reference_width_mm=float(position_reference_dims.width_mm),
                reference_height_mm=float(position_reference_dims.height_mm),
                requested_x_min_mm=float(args.x_min),
                requested_x_max_mm=float(args.x_max),
                requested_y_min_mm=float(args.y_min),
                requested_y_max_mm=float(args.y_max),
                min_margin_mm=0.0,
            )

            sampled_contact = rectgel.sample_stratified_contact_position(
                rng=rng,
                requested_sample_type=str(requested_sample_type),
                bbox_mm=bbox_mm,
                clean_bounds=clean_bounds,
                inside_bounds_min_scale=inside_bounds_min_scale,
                inside_bounds_reference_scale=inside_bounds_reference_scale,
                min_scale_width_mm=float(contact_constraint_dims.width_mm),
                min_scale_height_mm=float(contact_constraint_dims.height_mm),
                reference_width_mm=float(position_reference_dims.width_mm),
                reference_height_mm=float(position_reference_dims.height_mm),
                near_boundary_max_margin_mm=float(args.near_boundary_max_margin_mm),
                partial_crop_min_overhang_mm=float(args.partial_crop_min_overhang_mm),
                partial_crop_max_overhang_mm=float(args.partial_crop_max_overhang_mm),
            )

            contact_x_norm = float(sampled_contact["x_norm"])
            contact_y_norm = float(sampled_contact["y_norm"])
            final_depth_mm = round(float(rng.uniform(float(args.depth_min), float(args.depth_max))), 1)
            temporal_plan = build_episode_temporal_plan(args=args, rng=rng, final_depth_mm=final_depth_mm)

            episode_dir = dataset_root / f"episode_{episode_id:06d}"
            if materialize_dirs:
                episode_dir.mkdir(parents=True, exist_ok=True)

            episode_state = SequenceEpisodeState(
                episode_id=episode_id,
                indenter=indenter,
                indenter_split=indenter_split,
                is_unseen_indenter=bool(is_unseen_indenter),
                contact_x_norm=contact_x_norm,
                contact_y_norm=contact_y_norm,
                final_depth_mm=final_depth_mm,
                position_sample_type_requested=str(sampled_contact["requested_sample_type"]),
                position_sample_type_actual=str(sampled_contact["actual_sample_type"]),
                contact_margin_constraint_mm=float(sampled_contact["min_scale_margin_mm"]),
                episode_dir=episode_dir,
                temporal_plan=temporal_plan,
            )
            episode_states[episode_id] = episode_state

            for scale_entry in scale_entries:
                scale_key = str(scale_entry["scale_key"])
                requested_mm = float(scale_entry["scale_requested_mm"])
                simulated_mm = float(scale_entry["scale_simulated_mm"])
                gel_dims = rectgel.gel_dims_from_width_mm(simulated_mm)
                scale_split = str(scale_entry["scale_split"])
                is_unseen_scale = bool(scale_entry["is_unseen_scale"])
                contact_x_mm, contact_y_mm = rectgel.contact_mm_from_normalized(
                    contact_x_norm,
                    contact_y_norm,
                    gel_dims.width_mm,
                    gel_dims.height_mm,
                )
                contact_x_mm = round(float(contact_x_mm), 4)
                contact_y_mm = round(float(contact_y_mm), 4)
                scale_dir = episode_dir / scale_key
                temp_scale_dir = dataset_root / "_intermediates" / f"episode_{episode_id:06d}" / scale_key
                if materialize_dirs:
                    scale_dir.mkdir(parents=True, exist_ok=True)
                    temp_scale_dir.mkdir(parents=True, exist_ok=True)

                episode_state.scales[scale_key] = {
                    "scale_requested_mm": requested_mm,
                    "scale_simulated_mm": simulated_mm,
                    "scale_mm": simulated_mm,
                    "gel_width_mm": float(gel_dims.width_mm),
                    "gel_height_mm": float(gel_dims.height_mm),
                    "scale_quantized": bool(scale_entry["scale_quantized"]),
                    "scale_key": scale_key,
                    "sequence_metadata": str((scale_dir / "sequence_metadata.json").relative_to(episode_dir)),
                    "contact_x_mm": contact_x_mm,
                    "contact_y_mm": contact_y_mm,
                    "contact_x_norm": contact_x_norm,
                    "contact_y_norm": contact_y_norm,
                    "sequence_length": sum(
                        [
                            temporal_plan.precontact_frame_count,
                            temporal_plan.press_frame_count,
                            temporal_plan.dwell_frame_count,
                            temporal_plan.release_frame_count,
                        ]
                    ),
                    "phase_lengths": {
                        "precontact": temporal_plan.precontact_frame_count,
                        "press": temporal_plan.press_frame_count,
                        "dwell": temporal_plan.dwell_frame_count,
                        "release": temporal_plan.release_frame_count,
                    },
                    "camera_mode": args.camera_mode,
                    "adapter_coord_map": str((scale_dir / "adapter_coord_map.npy").relative_to(episode_dir)),
                    "adapter_coord_map_shape": [
                        int(v)
                        for v in rectgel.make_adapter_coord_map(
                            gel_width_mm=float(gel_dims.width_mm),
                            gel_height_mm=float(gel_dims.height_mm),
                            patch_h=patch_h,
                            patch_w=patch_w,
                        ).shape
                    ],
                    "marker_files_selected": [str(v) for v in expected_marker_files],
                    "scale_split": scale_split,
                    "is_unseen_scale": bool(is_unseen_scale),
                }

                if args.resume and validate_existing_sequence_scale_metadata(
                    episode_dir=episode_dir,
                    scale_key=scale_key,
                    expected_profile=args.dataset_profile,
                    expected_scale_requested_mm=requested_mm,
                    expected_scale_simulated_mm=simulated_mm,
                    expected_scale_quantized=bool(scale_entry["scale_quantized"]),
                    expected_temporal_plan=temporal_plan,
                    expected_marker_files=expected_marker_files,
                    expected_contact_x_mm=contact_x_mm,
                    expected_contact_y_mm=contact_y_mm,
                    expected_contact_x_norm=contact_x_norm,
                    expected_contact_y_norm=contact_y_norm,
                    expected_scale_split=scale_split,
                    expected_is_unseen_scale=bool(is_unseen_scale),
                ):
                    continue

                physics_tasks.append(
                    {
                        "episode_id": episode_id,
                        "scale_requested_mm": requested_mm,
                        "scale_simulated_mm": simulated_mm,
                        "scale_quantized": bool(scale_entry["scale_quantized"]),
                        "scale_key": scale_key,
                        "indenter": indenter,
                        "x_mm": contact_x_mm,
                        "y_mm": contact_y_mm,
                        "depth_mm": final_depth_mm,
                        "repo_root": args.repo_root,
                        "temp_scale_dir": temp_scale_dir,
                        "base_parameters": base_parameters,
                        "python_cmd": args.python_cmd,
                        "particle": args.particle,
                        "physics_timeout_sec": args.physics_timeout_sec,
                        "resume": bool(args.resume),
                        "temporal_plan": {
                            "precontact_frame_count": temporal_plan.precontact_frame_count,
                            "press_frame_count": temporal_plan.press_frame_count,
                            "dwell_frame_count": temporal_plan.dwell_frame_count,
                            "release_frame_count": temporal_plan.release_frame_count,
                            "press_sampling_mode": temporal_plan.press_sampling_mode,
                            "release_mode_requested": temporal_plan.release_mode_requested,
                            "release_mode_actual": temporal_plan.release_mode_actual,
                            "final_depth_mm": temporal_plan.final_depth_mm,
                            "include_dwell": temporal_plan.include_dwell,
                            "include_release": temporal_plan.include_release,
                        },
                        "contact_x_norm": contact_x_norm,
                        "contact_y_norm": contact_y_norm,
                        "scale_split": scale_split,
                        "is_unseen_scale": bool(is_unseen_scale),
                        "camera_mode": args.camera_mode,
                        "camera_params": compute_camera_params_for_scale(
                            camera_mode=str(args.camera_mode),
                            scale_mm=simulated_mm,
                            base_fov_deg=float(args.fov_deg),
                            fixed_distance_m=float(fixed_distance_m),
                            distance_safety=float(args.distance_safety),
                        ),
                    }
                )
            if persist_initial_metadata and materialize_dirs:
                persist_episode_metadata(episode_state, args.particle, args.dataset_profile)
            episode_id += 1

    return episode_states, physics_tasks


def generate_from_scratch(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    dataset_root = legacy.resolve_path(repo_root, args.dataset_root).resolve()

    parameters_path = repo_root / "sim" / "parameters.yml"
    texture_dir = repo_root / "sim" / "marker" / "marker_pattern"
    indenter_dir = repo_root / "sim" / "assets" / "indenters" / "input" / f"npy_{args.particle}"
    if not parameters_path.exists():
        raise FileNotFoundError(f"Missing {parameters_path}")
    if not texture_dir.exists():
        raise FileNotFoundError(f"Missing marker texture directory: {texture_dir}")
    if not indenter_dir.exists():
        raise FileNotFoundError(f"Missing indenter directory: {indenter_dir}")

    patch_h, patch_w = legacy.parse_patch_grid(args.patch_grid)
    indenter_names = legacy.discover_indenter_names(indenter_dir, args.objects)
    split_set = set(args.train_indenters) | set(args.val_indenters) | set(args.test_indenters)
    unknown_indenters = [name for name in indenter_names if name not in split_set]
    if unknown_indenters:
        raise ValueError(f"Indenters missing split assignment: {unknown_indenters}")

    with open(parameters_path, "r", encoding="utf-8") as f:
        parameters_cfg = yaml.safe_load(f)
    indenter_pose = parameters_cfg.get("indenter", {}).get("pose", {}) if isinstance(parameters_cfg, dict) else {}
    indenter_rotation = legacy.rotation_matrix_from_pose(indenter_pose, degrees=False)
    indenter_bboxes = {
        name: legacy.compute_indenter_contact_bbox_mm(indenter_dir / f"{name}.npy", indenter_rotation)
        for name in indenter_names
    }

    selected_textures, marker_selection_mode = select_marker_textures(texture_dir=texture_dir, args=args)
    args.marker_textures_selected = [str(path.stem) for path in selected_textures]
    args.marker_texture_selection_mode = str(marker_selection_mode)
    expected_marker_files = [f"marker_{path.stem}.jpg" for path in selected_textures]
    logging.info(
        "Marker textures | mode=%s count=%d seed=%d selected=%s",
        marker_selection_mode,
        len(selected_textures),
        int(args.marker_texture_seed),
        args.marker_textures_selected,
    )

    args.train_scale_intervals_mm_parsed = parse_scale_intervals_mm(args.train_scale_intervals_mm, "--train-scale-intervals-mm")
    args.val_scale_intervals_mm_parsed = parse_scale_intervals_mm(args.val_scale_intervals_mm, "--val-scale-intervals-mm")
    args.test_scale_intervals_mm_parsed = parse_scale_intervals_mm(args.test_scale_intervals_mm, "--test-scale-intervals-mm")
    if not args.estimate_only:
        ensure_output_root(dataset_root, args)
        (dataset_root / "_intermediates").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    episode_states, physics_tasks = build_episode_states_for_generation(
        args=args,
        dataset_root=dataset_root,
        indenter_names=indenter_names,
        indenter_bboxes=indenter_bboxes,
        expected_marker_files=expected_marker_files,
        patch_h=patch_h,
        patch_w=patch_w,
        rng=rng,
        materialize_dirs=not bool(args.estimate_only),
        persist_initial_metadata=not bool(args.estimate_only),
    )
    planning_summary = estimate_generation_plan(
        dataset_profile=args.dataset_profile,
        episode_states=episode_states,
        physics_tasks=physics_tasks,
        marker_count=len(expected_marker_files),
        pair_marker_policy=str(args.pair_marker_policy),
        include_identity_pairs=bool(args.include_identity_pairs),
        pair_split_policy=str(args.pair_split_policy),
        clip_length=int(args.clip_length),
        clip_window_stride=int(args.clip_window_stride),
        clip_gap_min=int(args.clip_gap_min),
        clip_gap_max=int(args.clip_gap_max),
        keep_intermediates=bool(args.keep_intermediates),
        physics_npz_cleanup=str(args.physics_npz_cleanup),
        reference_image_kb=float(args.reference_image_kb),
        reference_npz_mb=float(args.reference_npz_mb),
        reference_stl_mb=float(args.reference_stl_mb),
        reference_throughput_images_per_hour=float(args.reference_throughput_images_per_hour),
    )
    args.planning_summary = planning_summary
    log_planning_summary(planning_summary, args.dataset_profile)
    if args.estimate_only:
        return
    write_run_status(
        dataset_root=dataset_root,
        dataset_profile=args.dataset_profile,
        args=args,
        phase="prepared",
        episode_states=episode_states,
        message=f"Prepared {len(physics_tasks)} physics tasks",
    )

    script_tmp_dir = Path(tempfile.mkdtemp(prefix="multiscale_sequence_scripts_"))
    open3d_script = script_tmp_dir / "tmp_npz2stl.py"
    blender_script = script_tmp_dir / "tmp_blender_multirender.py"
    open3d_script.write_text(legacy.OPEN3D_TEMP_SCRIPT, encoding="utf-8")
    blender_script.write_text(
        build_sequence_blender_script(
            render_device=str(args.render_device),
            render_gpu_backend=str(args.render_gpu_backend),
        ),
        encoding="utf-8",
    )
    selected_texture_dir = prepare_selected_texture_dir(
        selected_textures=selected_textures,
        selection_mode=marker_selection_mode,
        source_texture_dir=texture_dir,
        staging_root=script_tmp_dir,
    )

    scale_states: Dict[Tuple[int, str], SequenceScaleState] = {}
    run_start_ts = time.time()
    total_physics_tasks = len(physics_tasks)
    total_meshing_tasks_planned = int(args.planning_summary["totals"]["meshing_tasks_planned"])
    total_render_tasks_planned = int(args.planning_summary["totals"]["render_tasks_planned"])
    expected_images_planned = int(args.planning_summary["totals"]["expected_images_planned"])

    try:
        backlog_drain_mode = bool(args.resume and int(args.max_physics_workers) == 0)
        if backlog_drain_mode:
            logging.info("Backlog drain mode enabled | resume=True max_physics_workers=0")
            total_physics_tasks = 0
        physics_stage_start = time.time()
        meshing_stage_start: float | None = None
        render_stage_start: float | None = None
        physics_done = 0
        physics_reused = 0
        physics_failed = 0
        meshing_done = 0
        meshing_reused = 0
        meshing_failed = 0
        meshing_submitted = 0
        render_done = 0
        render_reused = 0
        render_failed = 0
        render_submitted = 0
        physics_status_written = False
        meshing_status_written = False

        with contextlib.ExitStack() as stack:
            physics_pool = (
                stack.enter_context(cf.ProcessPoolExecutor(max_workers=int(args.max_physics_workers)))
                if int(args.max_physics_workers) > 0
                else None
            )
            meshing_pool = stack.enter_context(cf.ProcessPoolExecutor(max_workers=int(args.max_meshing_workers)))
            render_pool = stack.enter_context(cf.ProcessPoolExecutor(max_workers=int(args.max_render_workers)))
            physics_futures: Dict[cf.Future, Dict[str, Any]] = {}
            meshing_futures: Dict[cf.Future, Dict[str, Any]] = {}
            render_futures: Dict[cf.Future, Dict[str, Any]] = {}
            physics_pending_tasks: deque[Dict[str, Any]] = deque()
            meshing_pending_tasks: deque[Dict[str, Any]] = deque()
            render_pending_tasks: deque[Dict[str, Any]] = deque()
            render_backlog_high, render_backlog_low = resolve_render_backlog_thresholds(args)
            pipeline_paused = False

            def render_backlog_size() -> int:
                return int(len(render_futures) + len(render_pending_tasks))

            def maybe_update_balance_state(force_log: bool = False) -> None:
                nonlocal pipeline_paused
                if not bool(args.auto_balance_pipeline):
                    return
                backlog = render_backlog_size()
                if (not pipeline_paused) and backlog >= render_backlog_high:
                    pipeline_paused = True
                    logging.info(
                        "Pipeline auto-balance | pausing meshing/physics submissions render_backlog=%d high=%d low=%d",
                        backlog,
                        render_backlog_high,
                        render_backlog_low,
                    )
                elif pipeline_paused and backlog <= render_backlog_low:
                    pipeline_paused = False
                    logging.info(
                        "Pipeline auto-balance | resuming meshing/physics submissions render_backlog=%d high=%d low=%d",
                        backlog,
                        render_backlog_high,
                        render_backlog_low,
                    )
                elif force_log and bool(args.auto_balance_pipeline):
                    logging.info(
                        "Pipeline auto-balance | state=%s render_backlog=%d high=%d low=%d",
                        "paused" if pipeline_paused else "active",
                        backlog,
                        render_backlog_high,
                        render_backlog_low,
                    )

            def submit_ready_work() -> None:
                maybe_update_balance_state()
                while render_pending_tasks and len(render_futures) < int(args.max_render_workers):
                    render_task = render_pending_tasks.popleft()
                    render_futures[render_pool.submit(run_render_sequence_task, render_task)] = render_task
                if pipeline_paused:
                    return
                while meshing_pending_tasks and len(meshing_futures) < int(args.max_meshing_workers):
                    mesh_task = meshing_pending_tasks.popleft()
                    meshing_futures[meshing_pool.submit(run_meshing_sequence_task, mesh_task)] = mesh_task
                if physics_pool is not None:
                    while physics_pending_tasks and len(physics_futures) < int(args.max_physics_workers):
                        physics_task = physics_pending_tasks.popleft()
                        physics_futures[physics_pool.submit(run_physics_sequence_task, physics_task)] = physics_task

            if physics_pool is not None:
                physics_pending_tasks.extend(physics_tasks)
            else:
                backlog_seeded = 0
                backlog_missing_npz = 0
                backlog_invalid_npz = 0
                for task in physics_tasks:
                    npz_root = Path(task["temp_scale_dir"]) / "npz"
                    npz_path = legacy.expected_npz_path(
                        npz_root,
                        str(task["indenter"]),
                        float(task["x_mm"]),
                        float(task["y_mm"]),
                        float(task["depth_mm"]),
                    )
                    if not npz_path.exists():
                        x_tag = legacy.format_coord_suffix(float(task["x_mm"]))
                        y_tag = legacy.format_coord_suffix(float(task["y_mm"]))
                        candidates = sorted((npz_root / str(task["indenter"])).glob(f"{x_tag}_{y_tag}_*.npz"))
                        if not candidates:
                            backlog_missing_npz += 1
                            continue
                        npz_path = candidates[-1]
                    try:
                        episode_id = int(task["episode_id"])
                        scale_state = build_scale_state_from_task_and_npz(
                            episode_state=episode_states[episode_id],
                            task=task,
                            npz_path=npz_path,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            physics_reused=True,
                            physics_npz_cleanup_policy=str(args.physics_npz_cleanup),
                        )
                    except Exception as exc:
                        backlog_invalid_npz += 1
                        logging.warning(
                            "Backlog drain skipped invalid NPZ | ep=%d scale=%s npz=%s err=%s",
                            int(task["episode_id"]),
                            str(task["scale_key"]),
                            npz_path,
                            exc,
                        )
                        continue
                    scale_states[(episode_id, scale_state.scale_key)] = scale_state
                    for mesh_task in build_meshing_tasks_for_scale_state(
                        scale_state=scale_state,
                        episode_id=episode_id,
                        repo_root=repo_root,
                        args=args,
                        open3d_script=open3d_script,
                    ):
                        meshing_pending_tasks.append(mesh_task)
                        meshing_submitted += 1
                    if meshing_stage_start is None and meshing_pending_tasks:
                        meshing_stage_start = time.time()
                    backlog_seeded += 1
                logging.info(
                    "Backlog drain seeded scales | seeded=%d missing_npz=%d invalid_npz=%d",
                    backlog_seeded,
                    backlog_missing_npz,
                    backlog_invalid_npz,
                )
            maybe_update_balance_state(force_log=True)
            while (
                physics_pending_tasks
                or meshing_pending_tasks
                or render_pending_tasks
                or physics_futures
                or meshing_futures
                or render_futures
            ):
                submit_ready_work()
                pending_futures: List[cf.Future] = []
                pending_futures.extend(list(physics_futures.keys()))
                pending_futures.extend(list(meshing_futures.keys()))
                pending_futures.extend(list(render_futures.keys()))
                if not pending_futures:
                    maybe_update_balance_state(force_log=True)
                    time.sleep(0.1)
                    continue
                done_set, _ = cf.wait(pending_futures, return_when=cf.FIRST_COMPLETED)

                for future in done_set:
                    if future in physics_futures:
                        task = physics_futures.pop(future)
                        result = future.result()
                        episode_id = int(task["episode_id"])
                        episode_state = episode_states[episode_id]
                        physics_done += 1

                        if result["status"] != "ok":
                            episode_state.errors.append(str(result.get("error", "physics failed")))
                            physics_failed += 1
                            logging.error(
                                "Physics failed | ep=%d scale=%s err=%s",
                                episode_id,
                                task["scale_key"],
                                result.get("error", ""),
                            )
                        else:
                            scale_state = build_scale_state_from_task_and_npz(
                                episode_state=episode_state,
                                task=task,
                                npz_path=Path(result["npz_path"]),
                                patch_h=patch_h,
                                patch_w=patch_w,
                                physics_reused=bool(result["physics_reused"]),
                                physics_npz_cleanup_policy=str(args.physics_npz_cleanup),
                            )
                            if result["physics_reused"]:
                                physics_reused += 1
                            scale_states[(episode_id, scale_state.scale_key)] = scale_state

                            for mesh_task in build_meshing_tasks_for_scale_state(
                                scale_state=scale_state,
                                episode_id=episode_id,
                                repo_root=repo_root,
                                args=args,
                                open3d_script=open3d_script,
                            ):
                                meshing_pending_tasks.append(mesh_task)
                                meshing_submitted += 1
                            if meshing_stage_start is None and meshing_pending_tasks:
                                meshing_stage_start = time.time()

                        maybe_log_stage_progress(
                            stage_name="Physics",
                            stage_start_ts=physics_stage_start,
                            done=physics_done,
                            total=total_physics_tasks,
                            reused=physics_reused,
                            failed=physics_failed,
                        )
                        maybe_log_overall_progress(
                            run_start_ts=run_start_ts,
                            physics_done=physics_done,
                            physics_total=total_physics_tasks,
                            meshing_done=meshing_done,
                            meshing_total=total_meshing_tasks_planned,
                            render_done=render_done,
                            render_total=total_render_tasks_planned,
                            marker_count=len(expected_marker_files),
                            expected_images=expected_images_planned,
                        )

                    elif future in meshing_futures:
                        task = meshing_futures.pop(future)
                        result = future.result()
                        episode_id = int(task["episode_id"])
                        scale_key = format_scale_key(float(task["scale_simulated_mm"]))
                        meshing_done += 1

                        if result["status"] != "ok":
                            episode_states[episode_id].errors.append(str(result.get("error", "meshing failed")))
                            meshing_failed += 1
                            logging.error(
                                "Meshing failed | ep=%d scale=%s frame=%s err=%s",
                                episode_id,
                                scale_key,
                                task["frame_name"],
                                result.get("error", ""),
                            )
                        else:
                            scale_state = scale_states.get((episode_id, scale_key))
                            if scale_state is not None:
                                if result.get("reused"):
                                    meshing_reused += 1
                                render_task = {
                                    "episode_id": episode_id,
                                    "scale_simulated_mm": scale_state.scale_simulated_mm,
                                    "frame_name": str(result["frame_name"]),
                                    "frame_dir": scale_state.scale_dir / str(result["frame_name"]),
                                    "repo_root": repo_root,
                                    "stl_path": result["stl_path"],
                                    "keep_intermediates": bool(args.keep_intermediates),
                                    "resume": bool(args.resume),
                                    "expected_marker_files": expected_marker_files,
                                    "blender_cmd": args.blender_cmd,
                                    "blender_script": blender_script,
                                    "textures_dir": selected_texture_dir,
                                    "camera_mode": scale_state.camera_mode,
                                    "base_fov_deg": args.fov_deg,
                                    "fixed_distance_m": scale_state.camera_distance_m,
                                    "distance_safety": args.distance_safety,
                                    "uv_mode": args.uv_mode,
                                    "uv_inset_ratio": args.uv_inset_ratio,
                                    "render_samples": args.render_samples,
                                    "render_device": args.render_device,
                                    "render_gpu_backend": args.render_gpu_backend,
                                    "render_timeout_sec": args.render_timeout_sec,
                                    "force_black_side_border_px": args.force_black_side_border_px,
                                }
                                render_pending_tasks.append(render_task)
                                render_submitted += 1
                                if render_stage_start is None and render_pending_tasks:
                                    render_stage_start = time.time()

                        maybe_log_stage_progress(
                            stage_name="Meshing",
                            stage_start_ts=meshing_stage_start if meshing_stage_start is not None else run_start_ts,
                            done=meshing_done,
                            total=total_meshing_tasks_planned,
                            reused=meshing_reused,
                            failed=meshing_failed,
                        )
                        maybe_log_overall_progress(
                            run_start_ts=run_start_ts,
                            physics_done=physics_done,
                            physics_total=total_physics_tasks,
                            meshing_done=meshing_done,
                            meshing_total=total_meshing_tasks_planned,
                            render_done=render_done,
                            render_total=total_render_tasks_planned,
                            marker_count=len(expected_marker_files),
                            expected_images=expected_images_planned,
                        )

                    elif future in render_futures:
                        task = render_futures.pop(future)
                        result = future.result()
                        episode_id = int(task["episode_id"])
                        scale_key = format_scale_key(float(task["scale_simulated_mm"]))
                        render_done += 1

                        if result["status"] != "ok":
                            episode_states[episode_id].errors.append(str(result.get("error", "render failed")))
                            render_failed += 1
                            logging.error(
                                "Render failed | ep=%d scale=%s frame=%s err=%s",
                                episode_id,
                                scale_key,
                                task["frame_name"],
                                result.get("error", ""),
                            )
                        else:
                            scale_state = scale_states.get((episode_id, scale_key))
                            if scale_state is not None:
                                if result.get("reused"):
                                    render_reused += 1
                                rendered_lookup = {frame.frame_name: frame for frame in scale_state.frames}
                                frame_record = rendered_lookup.get(str(result["frame_name"]))
                                if frame_record is not None:
                                    frame_record.rendered_markers = [str(v) for v in result["rendered_markers"]]
                                maybe_finalize_completed_scale(
                                    episode_state=episode_states[episode_id],
                                    scale_state=scale_state,
                                    dataset_profile=args.dataset_profile,
                                    image_resolution=legacy.DEFAULT_IMAGE_RES,
                                    marker_files_selected=expected_marker_files,
                                    particle=args.particle,
                                    keep_intermediates=bool(args.keep_intermediates),
                                )

                        maybe_log_stage_progress(
                            stage_name="Render",
                            stage_start_ts=render_stage_start if render_stage_start is not None else run_start_ts,
                            done=render_done,
                            total=total_render_tasks_planned,
                            reused=render_reused,
                            failed=render_failed,
                        )
                        maybe_log_overall_progress(
                            run_start_ts=run_start_ts,
                            physics_done=physics_done,
                            physics_total=total_physics_tasks,
                            meshing_done=meshing_done,
                            meshing_total=total_meshing_tasks_planned,
                            render_done=render_done,
                            render_total=total_render_tasks_planned,
                            marker_count=len(expected_marker_files),
                            expected_images=expected_images_planned,
                        )

                submit_ready_work()

                if (not physics_status_written) and (not physics_futures) and (not physics_pending_tasks):
                    maybe_log_stage_progress(
                        stage_name="Physics",
                        stage_start_ts=physics_stage_start,
                        done=physics_done,
                        total=total_physics_tasks,
                        reused=physics_reused,
                        failed=physics_failed,
                        force=True,
                    )
                    write_run_status(
                        dataset_root=dataset_root,
                        dataset_profile=args.dataset_profile,
                        args=args,
                        phase="physics_complete",
                        episode_states=episode_states,
                        message=f"Physics complete for {len(scale_states)} scales",
                    )
                    physics_status_written = True

                if (
                    (not meshing_status_written)
                    and (not physics_futures)
                    and (not physics_pending_tasks)
                    and (not meshing_futures)
                    and (not meshing_pending_tasks)
                ):
                    maybe_log_stage_progress(
                        stage_name="Meshing",
                        stage_start_ts=meshing_stage_start if meshing_stage_start is not None else run_start_ts,
                        done=meshing_done,
                        total=total_meshing_tasks_planned,
                        reused=meshing_reused,
                        failed=meshing_failed,
                        force=True,
                    )
                    write_run_status(
                        dataset_root=dataset_root,
                        dataset_profile=args.dataset_profile,
                        args=args,
                        phase="meshing_complete",
                        episode_states=episode_states,
                        message=f"Meshing complete for {render_submitted} logical frames",
                    )
                    meshing_status_written = True

        maybe_log_stage_progress(
            stage_name="Physics",
            stage_start_ts=physics_stage_start,
            done=physics_done,
            total=total_physics_tasks,
            reused=physics_reused,
            failed=physics_failed,
            force=True,
        )
        maybe_log_stage_progress(
            stage_name="Meshing",
            stage_start_ts=meshing_stage_start if meshing_stage_start is not None else run_start_ts,
            done=meshing_done,
            total=total_meshing_tasks_planned,
            reused=meshing_reused,
            failed=meshing_failed,
            force=True,
        )
        maybe_log_stage_progress(
            stage_name="Render",
            stage_start_ts=render_stage_start if render_stage_start is not None else run_start_ts,
            done=render_done,
            total=total_render_tasks_planned,
            reused=render_reused,
            failed=render_failed,
            force=True,
        )
        maybe_log_overall_progress(
            run_start_ts=run_start_ts,
            physics_done=physics_done,
            physics_total=total_physics_tasks,
            meshing_done=meshing_done,
            meshing_total=total_meshing_tasks_planned,
            render_done=render_done,
            render_total=total_render_tasks_planned,
            marker_count=len(expected_marker_files),
            expected_images=expected_images_planned,
            force=True,
        )

        for (episode_id, scale_key), scale_state in sorted(scale_states.items()):
            maybe_finalize_completed_scale(
                episode_state=episode_states[episode_id],
                scale_state=scale_state,
                dataset_profile=args.dataset_profile,
                image_resolution=legacy.DEFAULT_IMAGE_RES,
                marker_files_selected=expected_marker_files,
                particle=args.particle,
                keep_intermediates=bool(args.keep_intermediates),
            )

        manifest = build_manifest(
            dataset_root=dataset_root,
            dataset_profile=args.dataset_profile,
            args=args,
            episode_states=episode_states,
            patch_h=patch_h,
            patch_w=patch_w,
            storage_mode="full",
        )
        legacy.write_json_atomic(dataset_root / "manifest.json", manifest)
        index_counts = finalize_artifacts(dataset_root=dataset_root, dataset_profile=args.dataset_profile, args=args, storage_mode="full")
        write_run_status(
            dataset_root=dataset_root,
            dataset_profile=args.dataset_profile,
            args=args,
            phase="complete",
            episode_states=episode_states,
            message=(
                f"Done | sequence_index={index_counts['sequence_index_count']} "
                f"image_index={index_counts['image_index_count']} "
                f"pair_index={index_counts['pair_index_count']} "
                f"clip_index={index_counts['clip_index_count']}"
            ),
        )
        logging.info(
            "Generation complete | dataset_profile=%s episodes=%d sequence_index=%d image_index=%d pair_index=%d clip_index=%d",
            args.dataset_profile,
            len(manifest.get("episodes", [])),
            index_counts["sequence_index_count"],
            index_counts["image_index_count"],
            index_counts["pair_index_count"],
            index_counts["clip_index_count"],
        )
    finally:
        shutil.rmtree(script_tmp_dir, ignore_errors=True)


def derive_profile_b_from_existing_root(args: argparse.Namespace) -> None:
    if args.dataset_profile != "B":
        raise ValueError("derive mode is only supported for dataset-profile B")
    repo_root = args.repo_root.resolve()
    dataset_root = legacy.resolve_path(repo_root, args.dataset_root).resolve()
    source_root = legacy.resolve_path(repo_root, args.derive_clips_from_existing_dataset_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source sequence dataset root: {source_root}")
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    with open(source_manifest_path, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)

    ensure_output_root(dataset_root, args)
    source_records = scan_sequence_dataset_records(source_root)

    episode_states: Dict[int, SequenceEpisodeState] = {}
    for item in source_records:
        episode_meta = item["episode_meta"]
        temporal_plan_payload = episode_meta.get("temporal_plan", {})
        episode_state = SequenceEpisodeState(
            episode_id=int(episode_meta["episode_id"]),
            indenter=str(episode_meta["indenter"]),
            indenter_split=str(episode_meta.get("indenter_split", "")),
            is_unseen_indenter=bool(episode_meta.get("is_unseen_indenter", False)),
            contact_x_norm=float(episode_meta["latent_contact"]["contact_x_norm"]),
            contact_y_norm=float(episode_meta["latent_contact"]["contact_y_norm"]),
            final_depth_mm=float(episode_meta["latent_contact"]["final_depth_mm"]),
            position_sample_type_requested=str(episode_meta.get("position_sample_type_requested", "")),
            position_sample_type_actual=str(episode_meta.get("position_sample_type_actual", "")),
            contact_margin_constraint_mm=float(episode_meta.get("contact_margin_constraint_mm", 0.0)),
            episode_dir=source_root / f"episode_{int(episode_meta['episode_id']):06d}",
            temporal_plan=TemporalPlan(
                precontact_frame_count=int(temporal_plan_payload.get("precontact_frame_count", 0)),
                press_frame_count=int(temporal_plan_payload.get("press_frame_count", 0)),
                dwell_frame_count=int(temporal_plan_payload.get("dwell_frame_count", 0)),
                release_frame_count=int(temporal_plan_payload.get("release_frame_count", 0)),
                press_sampling_mode=str(temporal_plan_payload.get("press_sampling_mode", "uniform_index")),
                release_mode_requested=str(temporal_plan_payload.get("release_mode", "mirrored_loading")),
                release_mode_actual=str(temporal_plan_payload.get("release_mode", "mirrored_loading")),
                final_depth_mm=float(temporal_plan_payload.get("final_depth_mm", episode_meta["latent_contact"]["final_depth_mm"])),
                include_dwell=bool(int(temporal_plan_payload.get("dwell_frame_count", 0)) > 0),
                include_release=bool(int(temporal_plan_payload.get("release_frame_count", 0)) > 0),
            ),
            scales=episode_meta.get("scales", {}),
        )
        episode_states[episode_state.episode_id] = episode_state

    if args.derive_storage_mode == "symlink":
        for ep_entry in source_manifest.get("episodes", []):
            episode_rel = str(ep_entry.get("path", ""))
            if not episode_rel:
                continue
            ensure_episode_symlink(dataset_root / episode_rel, source_root / episode_rel)

    manifest = build_derived_profile_b_manifest(
        dataset_root=dataset_root,
        args=args,
        source_root=source_root,
        source_manifest=source_manifest,
        storage_mode=str(args.derive_storage_mode),
    )
    legacy.write_json_atomic(dataset_root / "manifest.json", manifest)
    index_counts = finalize_artifacts(
        dataset_root=dataset_root,
        dataset_profile="B",
        args=args,
        source_dataset_root=(source_root if args.derive_storage_mode == "index_only" else None),
        storage_mode=str(args.derive_storage_mode),
    )
    write_run_status(
        dataset_root=dataset_root,
        dataset_profile="B",
        args=args,
        phase="complete",
        episode_states=episode_states,
        message=(
            f"Derived profile B from existing root | sequence_index={index_counts['sequence_index_count']} "
            f"image_index={index_counts['image_index_count']} clip_index={index_counts['clip_index_count']}"
        ),
        source_sequence_dataset_root=source_root,
    )
    logging.info(
        "Derived dataset B complete | source=%s clip_index=%d",
        source_root,
        index_counts["clip_index_count"],
    )


def main() -> None:
    args = parse_args()
    legacy.setup_logging(args.log_level)
    validate_args(args)
    legacy.DEFAULT_IMAGE_RES = (int(args.render_width), int(args.render_height))
    logging.info(
        "Starting rectangular-gel sequence generation | profile=%s scale_mode=%s dataset_root=%s render=%dx%d",
        args.dataset_profile,
        args.scale_mode,
        args.dataset_root,
        int(args.render_width),
        int(args.render_height),
    )
    if args.derive_clips_from_existing_dataset_root is not None:
        derive_profile_b_from_existing_root(args)
    else:
        generate_from_scratch(args)


if __name__ == "__main__":
    main()
