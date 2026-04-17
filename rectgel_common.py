#!/usr/bin/env python3
"""Shared helpers for 4:3 rectangular-gel dataset generation.

This module is additive-only. It does not modify the existing square-gel
pipeline; it centralizes the few geometric rules that a rectangular-aware
generator/exporter must agree on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import random

import math

import numpy as np
import yaml


RECTGEL_VARIANT = "static_multiscale_rectgel_sequence_v1"
DEFAULT_RENDER_RES = (640, 480)
HEIGHT_OVER_WIDTH = 3.0 / 4.0


@dataclass(frozen=True)
class GelDims:
    width_mm: float
    height_mm: float


@dataclass(frozen=True)
class ParticleSpacing:
    spacing_l_mm: float
    spacing_w_mm: float


def gel_dims_from_width_mm(gel_width_mm: float) -> GelDims:
    width = float(gel_width_mm)
    if width <= 0:
        raise ValueError(f"gel_width_mm must be > 0, got {gel_width_mm}")
    return GelDims(width_mm=width, height_mm=round(width * HEIGHT_OVER_WIDTH, 4))


def format_width_key(width_mm: float) -> str:
    value = 0.0 if abs(float(width_mm)) < 5e-8 else float(width_mm)
    return f"scale_{value:.4f}".replace("-", "m").replace(".", "p") + "mm"


def format_dim_suffix(value_mm: float) -> str:
    rounded = round(float(value_mm), 4)
    if abs(rounded - int(round(rounded))) <= 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def make_sensor_size_slug(sensor_name: str, gel_width_mm: float) -> str:
    dims = gel_dims_from_width_mm(gel_width_mm)
    return f"{sensor_name}{format_dim_suffix(dims.width_mm)}x{format_dim_suffix(dims.height_mm)}"


def normalized_contact_from_reference_mm(
    x_mm: float,
    y_mm: float,
    reference_width_mm: float,
    reference_height_mm: float,
) -> Tuple[float, float]:
    if reference_width_mm <= 0 or reference_height_mm <= 0:
        raise ValueError("reference_width_mm and reference_height_mm must be > 0")
    return (
        float(x_mm) / (float(reference_width_mm) / 2.0),
        float(y_mm) / (float(reference_height_mm) / 2.0),
    )


def contact_mm_from_normalized(
    x_norm: float,
    y_norm: float,
    gel_width_mm: float,
    gel_height_mm: float,
) -> Tuple[float, float]:
    if gel_width_mm <= 0 or gel_height_mm <= 0:
        raise ValueError("gel_width_mm and gel_height_mm must be > 0")
    return (
        float(x_norm) * (float(gel_width_mm) / 2.0),
        float(y_norm) * (float(gel_height_mm) / 2.0),
    )


def compute_margin_constrained_normalized_sampling_bounds(
    *,
    bbox_mm: Dict[str, float],
    gel_width_mm: float,
    gel_height_mm: float,
    reference_width_mm: float,
    reference_height_mm: float,
    requested_x_min_mm: float,
    requested_x_max_mm: float,
    requested_y_min_mm: float,
    requested_y_max_mm: float,
    min_margin_mm: float = 0.0,
) -> Dict[str, float]:
    if gel_width_mm <= 0 or gel_height_mm <= 0:
        raise ValueError("gel_width_mm and gel_height_mm must be > 0")
    if reference_width_mm <= 0 or reference_height_mm <= 0:
        raise ValueError("reference_width_mm and reference_height_mm must be > 0")
    if min_margin_mm < 0:
        raise ValueError(f"min_margin_mm must be >= 0, got {min_margin_mm}")

    half_width = float(gel_width_mm) / 2.0
    half_height = float(gel_height_mm) / 2.0
    ref_half_width = float(reference_width_mm) / 2.0
    ref_half_height = float(reference_height_mm) / 2.0

    feasible_x_min_norm = (-half_width - float(bbox_mm["x_min_mm"]) + float(min_margin_mm)) / half_width
    feasible_x_max_norm = (half_width - float(bbox_mm["x_max_mm"]) - float(min_margin_mm)) / half_width
    feasible_y_min_norm = (-half_height - float(bbox_mm["y_min_mm"]) + float(min_margin_mm)) / half_height
    feasible_y_max_norm = (half_height - float(bbox_mm["y_max_mm"]) - float(min_margin_mm)) / half_height

    requested_x_min_norm = float(requested_x_min_mm) / ref_half_width
    requested_x_max_norm = float(requested_x_max_mm) / ref_half_width
    requested_y_min_norm = float(requested_y_min_mm) / ref_half_height
    requested_y_max_norm = float(requested_y_max_mm) / ref_half_height

    x_min_norm = max(feasible_x_min_norm, requested_x_min_norm)
    x_max_norm = min(feasible_x_max_norm, requested_x_max_norm)
    y_min_norm = max(feasible_y_min_norm, requested_y_min_norm)
    y_max_norm = min(feasible_y_max_norm, requested_y_max_norm)

    return {
        "x_min_norm": x_min_norm,
        "x_max_norm": x_max_norm,
        "y_min_norm": y_min_norm,
        "y_max_norm": y_max_norm,
        "feasible_x_min_norm": feasible_x_min_norm,
        "feasible_x_max_norm": feasible_x_max_norm,
        "feasible_y_min_norm": feasible_y_min_norm,
        "feasible_y_max_norm": feasible_y_max_norm,
        "requested_x_min_norm": requested_x_min_norm,
        "requested_x_max_norm": requested_x_max_norm,
        "requested_y_min_norm": requested_y_min_norm,
        "requested_y_max_norm": requested_y_max_norm,
        "min_margin_mm": float(min_margin_mm),
    }


def compute_contact_margins_mm(
    *,
    bbox_mm: Dict[str, float],
    x_norm: float,
    y_norm: float,
    gel_width_mm: float,
    gel_height_mm: float,
) -> Dict[str, float]:
    half_width = float(gel_width_mm) / 2.0
    half_height = float(gel_height_mm) / 2.0
    x_mm, y_mm = contact_mm_from_normalized(
        x_norm=x_norm,
        y_norm=y_norm,
        gel_width_mm=gel_width_mm,
        gel_height_mm=gel_height_mm,
    )
    left = x_mm + float(bbox_mm["x_min_mm"]) + half_width
    right = half_width - (x_mm + float(bbox_mm["x_max_mm"]))
    bottom = y_mm + float(bbox_mm["y_min_mm"]) + half_height
    top = half_height - (y_mm + float(bbox_mm["y_max_mm"]))
    return {
        "left_mm": float(left),
        "right_mm": float(right),
        "bottom_mm": float(bottom),
        "top_mm": float(top),
        "min_margin_mm": float(min(left, right, bottom, top)),
    }


def sample_normalized_contact_from_bounds(
    rng: random.Random,
    bounds: Dict[str, float],
) -> Tuple[float, float]:
    if bounds["x_min_norm"] > bounds["x_max_norm"] or bounds["y_min_norm"] > bounds["y_max_norm"]:
        raise ValueError(f"Invalid sampling bounds: {bounds}")
    return (
        round(rng.uniform(bounds["x_min_norm"], bounds["x_max_norm"]), 6),
        round(rng.uniform(bounds["y_min_norm"], bounds["y_max_norm"]), 6),
    )


def sample_stratified_contact_position(
    *,
    rng: random.Random,
    requested_sample_type: str,
    bbox_mm: Dict[str, float],
    clean_bounds: Dict[str, float],
    inside_bounds_min_scale: Dict[str, float],
    inside_bounds_reference_scale: Dict[str, float],
    min_scale_width_mm: float,
    min_scale_height_mm: float,
    reference_width_mm: float,
    reference_height_mm: float,
    near_boundary_max_margin_mm: float,
    partial_crop_min_overhang_mm: float,
    partial_crop_max_overhang_mm: float,
    max_attempts: int = 512,
) -> Dict[str, float | str]:
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be > 0, got {max_attempts}")

    def _bounds_valid(bounds: Dict[str, float]) -> bool:
        return bounds["x_min_norm"] <= bounds["x_max_norm"] and bounds["y_min_norm"] <= bounds["y_max_norm"]

    def _sample_clean() -> Tuple[float, float] | None:
        if not _bounds_valid(clean_bounds):
            return None
        return sample_normalized_contact_from_bounds(rng, clean_bounds)

    def _sample_near_boundary() -> Tuple[float, float] | None:
        if not _bounds_valid(inside_bounds_min_scale):
            return None
        for _ in range(max_attempts):
            x_norm, y_norm = sample_normalized_contact_from_bounds(rng, inside_bounds_min_scale)
            margin = compute_contact_margins_mm(
                bbox_mm=bbox_mm,
                x_norm=x_norm,
                y_norm=y_norm,
                gel_width_mm=min_scale_width_mm,
                gel_height_mm=min_scale_height_mm,
            )["min_margin_mm"]
            if 0.0 <= margin <= float(near_boundary_max_margin_mm):
                return x_norm, y_norm
        return None

    def _sample_partial_crop() -> Tuple[float, float] | None:
        if not _bounds_valid(inside_bounds_reference_scale):
            return None
        for _ in range(max_attempts):
            x_norm, y_norm = sample_normalized_contact_from_bounds(rng, inside_bounds_reference_scale)
            min_scale_margin = compute_contact_margins_mm(
                bbox_mm=bbox_mm,
                x_norm=x_norm,
                y_norm=y_norm,
                gel_width_mm=min_scale_width_mm,
                gel_height_mm=min_scale_height_mm,
            )["min_margin_mm"]
            if -float(partial_crop_max_overhang_mm) <= min_scale_margin <= -float(partial_crop_min_overhang_mm):
                return x_norm, y_norm
        return None

    def _sample_reference_safe() -> Tuple[float, float] | None:
        if not _bounds_valid(inside_bounds_reference_scale):
            return None
        return sample_normalized_contact_from_bounds(rng, inside_bounds_reference_scale)

    strategy_order = {
        "clean": ("clean", "near_boundary", "partial_crop", "reference_safe_fallback"),
        "near_boundary": ("near_boundary", "clean", "partial_crop", "reference_safe_fallback"),
        "partial_crop": ("partial_crop", "near_boundary", "clean", "reference_safe_fallback"),
    }
    if requested_sample_type not in strategy_order:
        raise ValueError(f"Unknown requested_sample_type: {requested_sample_type}")

    samplers = {
        "clean": _sample_clean,
        "near_boundary": _sample_near_boundary,
        "partial_crop": _sample_partial_crop,
        "reference_safe_fallback": _sample_reference_safe,
    }

    chosen: Tuple[float, float] | None = None
    actual_sample_type = requested_sample_type
    for actual_sample_type in strategy_order[requested_sample_type]:
        chosen = samplers[actual_sample_type]()
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError(
            f"Failed to sample position for {requested_sample_type} after trying {strategy_order[requested_sample_type]}"
        )

    x_norm, y_norm = chosen
    min_scale_margin = compute_contact_margins_mm(
        bbox_mm=bbox_mm,
        x_norm=x_norm,
        y_norm=y_norm,
        gel_width_mm=min_scale_width_mm,
        gel_height_mm=min_scale_height_mm,
    )["min_margin_mm"]
    reference_scale_margin = compute_contact_margins_mm(
        bbox_mm=bbox_mm,
        x_norm=x_norm,
        y_norm=y_norm,
        gel_width_mm=reference_width_mm,
        gel_height_mm=reference_height_mm,
    )["min_margin_mm"]
    return {
        "x_norm": float(x_norm),
        "y_norm": float(y_norm),
        "requested_sample_type": str(requested_sample_type),
        "actual_sample_type": str(actual_sample_type),
        "min_scale_margin_mm": float(min_scale_margin),
        "reference_scale_margin_mm": float(reference_scale_margin),
    }


def make_adapter_coord_map(gel_width_mm: float, gel_height_mm: float, patch_h: int, patch_w: int) -> np.ndarray:
    if gel_width_mm <= 0 or gel_height_mm <= 0:
        raise ValueError("gel_width_mm and gel_height_mm must be > 0")
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_h and patch_w must be > 0")

    step_x = float(gel_width_mm) / float(patch_w)
    step_y = float(gel_height_mm) / float(patch_h)
    x_axis = np.linspace(
        -float(gel_width_mm) / 2.0 + step_x / 2.0,
        float(gel_width_mm) / 2.0 - step_x / 2.0,
        patch_w,
        dtype=np.float64,
    )
    y_axis = np.linspace(
        -float(gel_height_mm) / 2.0 + step_y / 2.0,
        float(gel_height_mm) / 2.0 - step_y / 2.0,
        patch_h,
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    return np.stack([xx, yy], axis=-1).astype(np.float64, copy=False)


def compute_full_sensor_fit_width_m(
    gel_width_m: float,
    gel_height_m: float,
    render_width_px: int,
    render_height_px: int,
) -> float:
    if gel_width_m <= 0 or gel_height_m <= 0:
        raise ValueError("gel_width_m and gel_height_m must be > 0")
    if render_width_px <= 0 or render_height_px <= 0:
        raise ValueError("render resolution must be > 0")
    aspect = float(render_width_px) / float(render_height_px)
    return max(float(gel_width_m), float(gel_height_m) * aspect)


def load_base_particle_spacing(base_cfg: Path) -> ParticleSpacing:
    with open(base_cfg, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        size_l = float(data["elastomer"]["size"]["l"])
        size_w = float(data["elastomer"]["size"]["w"])
        num_l = int(data["elastomer"]["particle"]["num_l"])
        num_w = int(data["elastomer"]["particle"]["num_w"])
    except Exception as exc:
        raise KeyError("Missing elastomer.{size.l,size.w,particle.num_l,particle.num_w} in parameters yaml") from exc
    if num_l < 2 or num_w < 2:
        raise ValueError("Base num_l and num_w must be >= 2")
    return ParticleSpacing(
        spacing_l_mm=size_l / float(num_l - 1),
        spacing_w_mm=size_w / float(num_w - 1),
    )


def particle_counts_from_spacing(
    gel_width_mm: float,
    gel_height_mm: float,
    base_spacing: ParticleSpacing,
) -> Tuple[int, int]:
    num_l = max(2, int(round(float(gel_width_mm) / float(base_spacing.spacing_l_mm))) + 1)
    num_w = max(2, int(round(float(gel_height_mm) / float(base_spacing.spacing_w_mm))) + 1)
    return num_l, num_w


def write_rectangular_config_from_width(base_cfg: Path, out_cfg: Path, gel_width_mm: float) -> GelDims:
    dims = gel_dims_from_width_mm(gel_width_mm)
    base_spacing = load_base_particle_spacing(base_cfg)
    num_l, num_w = particle_counts_from_spacing(
        gel_width_mm=dims.width_mm,
        gel_height_mm=dims.height_mm,
        base_spacing=base_spacing,
    )
    with open(base_cfg, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        data["elastomer"]["size"]["l"] = float(dims.width_mm)
        data["elastomer"]["size"]["w"] = float(dims.height_mm)
        data["elastomer"]["particle"]["num_l"] = int(num_l)
        data["elastomer"]["particle"]["num_w"] = int(num_w)
    except Exception as exc:
        raise KeyError("Missing elastomer.size.l/w or elastomer.particle.num_l/num_w in parameters yaml") from exc
    with open(out_cfg, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return dims
