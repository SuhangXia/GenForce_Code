from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


def load_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload)}")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object in {path}:{line_no}, got {type(payload)}")
            rows.append(payload)
    return rows


def load_coord_map(path: str | Path) -> torch.Tensor:
    arr = np.load(Path(path))
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Expected coord map shape (H,W,2), got {arr.shape} at {path}")
    return torch.from_numpy(arr).to(torch.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {rgb.shape}")
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return gray.astype(np.float32) / 255.0


def build_temporal_observation(
    gray_t: np.ndarray,
    gray_0: np.ndarray,
    gray_prev: np.ndarray | None,
) -> torch.Tensor:
    if gray_t.shape != gray_0.shape:
        raise ValueError(f"gray_t / gray_0 shape mismatch: {gray_t.shape} vs {gray_0.shape}")
    if gray_prev is None:
        delta_prev = np.zeros_like(gray_t, dtype=np.float32)
    else:
        if gray_prev.shape != gray_t.shape:
            raise ValueError(f"gray_prev shape mismatch: {gray_prev.shape} vs {gray_t.shape}")
        delta_prev = gray_t - gray_prev
    stacked = np.stack([gray_t, gray_t - gray_0, delta_prev], axis=0)
    return torch.from_numpy(stacked.astype(np.float32, copy=False))


def infer_boundary_subset(position_sample_type_actual: str | None) -> str:
    raw = str(position_sample_type_actual or "").strip()
    if raw in {"clean", "near_boundary", "partial_crop"}:
        return raw
    return "unknown"


def infer_scale_dir_from_frame(frame_path: Path) -> Path:
    if frame_path.parent.name.startswith("frame_"):
        return frame_path.parent.parent
    raise ValueError(f"Could not infer scale dir from frame path: {frame_path}")


def infer_episode_dir_from_scale_dir(scale_dir: Path) -> Path:
    return scale_dir.parent


def infer_rest_frame_path(frame_path: Path) -> Path:
    scale_dir = infer_scale_dir_from_frame(frame_path)
    marker_name = frame_path.name
    candidate = scale_dir / "frame_000000" / marker_name
    return candidate if candidate.exists() else frame_path


def resolve_existing_path(root: Path, rel_or_abs: str | Path) -> Path:
    text = str(rel_or_abs)
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def clamp_window(center_index: int, size: int, total: int) -> list[int]:
    if size <= 0:
        raise ValueError(f"Window size must be positive, got {size}")
    if total <= 0:
        raise ValueError(f"Total item count must be positive, got {total}")
    radius = size // 2
    indices: list[int] = []
    for offset in range(size):
        idx = center_index - radius + offset
        idx = max(0, min(total - 1, idx))
        indices.append(idx)
    return indices


def safe_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Could not parse {name}={value!r} as float") from exc


def safe_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Could not parse {name}={value!r} as int") from exc


@dataclass(frozen=True)
class FrameMetadata:
    global_seq_index: int
    phase_name: str
    phase_index: int
    phase_progress: float
    source_physics_frame_index: int
    frame_actual_max_down_mm: float
    is_synthetic_precontact: bool
    is_synthetic_release: bool
    is_dwell_repeat: bool
    rendered_markers: tuple[str, ...]


def build_seq_valid_mask(anchor_position: int, window_positions: Iterable[int]) -> torch.Tensor:
    anchor = int(anchor_position)
    values = [1.0 if int(pos) == anchor else 0.0 for pos in window_positions]
    if not any(values):
        values[len(values) // 2] = 1.0
    return torch.tensor(values, dtype=torch.float32)
