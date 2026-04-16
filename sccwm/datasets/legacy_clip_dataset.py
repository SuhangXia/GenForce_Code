from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, get_worker_info

from .utils import (
    build_temporal_observation,
    infer_boundary_subset,
    infer_rest_frame_path,
    infer_scale_dir_from_frame,
    load_coord_map,
    load_json,
    load_rgb,
    read_jsonl,
    resolve_existing_path,
    rgb_to_gray,
    safe_float,
    safe_int,
)


class LegacyClipDataset(Dataset[dict[str, Any]]):
    """Dataset-B clip loader for SCCWM legacy regressor and plugin experiments."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str | None = None,
        clip_index_file: str | Path | None = None,
        gray_cache_max_items: int = 256,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = None if split is None else str(split)
        self.gray_cache_max_items = max(int(gray_cache_max_items), 0)
        self.index_path = resolve_existing_path(self.dataset_root, clip_index_file or "clip_index.jsonl")
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._gray_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._episode_meta_cache: dict[Path, dict[str, Any]] = {}
        self._rows = self._build_rows()
        if not self._rows:
            raise RuntimeError(f"No clip samples found in {self.index_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        frame_paths = row["frame_paths"]
        rest_path = row["rest_frame_path"]
        gray_rest = self._load_gray(rest_path)
        previous_gray = None
        obs_frames: list[torch.Tensor] = []
        for frame_path in frame_paths:
            gray = self._load_gray(frame_path)
            obs_frames.append(build_temporal_observation(gray, gray_rest, previous_gray))
            previous_gray = gray
        return {
            "clip_obs": torch.stack(obs_frames, dim=0),
            "rest_obs": build_temporal_observation(gray_rest, gray_rest, None),
            "coord_map": self._load_coord(row["coord_map_path"]),
            "scale_mm": torch.tensor(row["scale_mm"], dtype=torch.float32),
            "x_norm": torch.tensor(row["x_norm"], dtype=torch.float32),
            "y_norm": torch.tensor(row["y_norm"], dtype=torch.float32),
            "depth_mm": torch.tensor(row["depth_mm"], dtype=torch.float32),
            "phase_names": row["phase_names"],
            "target_phase_name": row["target_phase_name"],
            "indenter_split": row["indenter_split"],
            "scale_split": row["scale_split"],
            "target_scale_split": row["target_scale_split"],
            "boundary_subset": row["boundary_subset"],
            "is_unseen_indenter": torch.tensor(row["is_unseen_indenter"], dtype=torch.bool),
            "is_unseen_scale": torch.tensor(row["is_unseen_scale"], dtype=torch.bool),
            "metadata": {
                "clip_id": row["clip_id"],
                "episode_id": row["episode_id"],
                "marker_name": row["marker_name"],
                "frame_indices": row["frame_indices"],
                "storage_mode": row["storage_mode"],
            },
        }

    def _build_rows(self) -> list[dict[str, Any]]:
        rows = read_jsonl(self.index_path)
        samples: list[dict[str, Any]] = []
        for row in rows:
            scale_split = str(row.get("scale_split", ""))
            if self.split is not None and scale_split and self.split != scale_split:
                continue
            source_root = Path(str(row.get("source_dataset_root", self.dataset_root))).expanduser()
            if not source_root.is_absolute():
                source_root = (self.dataset_root / source_root).resolve()
            frame_paths = [resolve_existing_path(source_root, relpath) for relpath in row["frame_relpaths"]]
            scale_dir = infer_scale_dir_from_frame(frame_paths[0])
            coord_map_path = scale_dir / "adapter_coord_map.npy"
            if not coord_map_path.exists():
                raise FileNotFoundError(f"Missing coord map for clip sample: {coord_map_path}")
            episode_dir = scale_dir.parent
            episode_meta_path = episode_dir / "metadata.json"
            boundary_subset = "unknown"
            if episode_meta_path.exists():
                boundary_subset = infer_boundary_subset(self._load_episode_meta(episode_meta_path).get("position_sample_type_actual"))
            samples.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "episode_id": safe_int(row["episode_id"], "episode_id"),
                    "marker_name": str(row["marker_name"]),
                    "frame_paths": frame_paths,
                    "frame_indices": [safe_int(v, "frame_index") for v in row["frame_indices"]],
                    "coord_map_path": coord_map_path,
                    "rest_frame_path": infer_rest_frame_path(frame_paths[0]),
                    "scale_mm": safe_float(row["scale_mm"], "scale_mm"),
                    "x_norm": safe_float(row["target_x_norm"], "target_x_norm"),
                    "y_norm": safe_float(row["target_y_norm"], "target_y_norm"),
                    "depth_mm": safe_float(row["target_depth_mm"], "target_depth_mm"),
                    "phase_names": [str(v) for v in row.get("phase_names", [])],
                    "target_phase_name": str(row.get("target_phase_name", "")),
                    "indenter_split": str(row.get("indenter_split", "")),
                    "scale_split": scale_split,
                    "target_scale_split": str(row.get("target_scale_split", scale_split)),
                    "boundary_subset": boundary_subset,
                    "is_unseen_indenter": bool(row.get("is_unseen_indenter", False)),
                    "is_unseen_scale": bool(row.get("is_unseen_scale", False)),
                    "storage_mode": str(row.get("storage_mode", "")),
                }
            )
        return samples

    def _load_coord(self, path: Path) -> torch.Tensor:
        if path not in self._coord_cache:
            self._coord_cache[path] = load_coord_map(path)
        return self._coord_cache[path]

    def _load_gray(self, path: Path) -> torch.Tensor:
        cached = self._gray_cache.get(path)
        if cached is not None:
            self._gray_cache.move_to_end(path)
            return cached
        gray = torch.from_numpy(rgb_to_gray(load_rgb(path)))
        worker_info = get_worker_info()
        cache_limit = self.gray_cache_max_items
        if worker_info is not None and cache_limit > 0:
            cache_limit = min(cache_limit, 64)
        if cache_limit > 0:
            self._gray_cache[path] = gray
            self._gray_cache.move_to_end(path)
            while len(self._gray_cache) > cache_limit:
                self._gray_cache.popitem(last=False)
        return gray

    def _load_episode_meta(self, path: Path) -> dict[str, Any]:
        if path not in self._episode_meta_cache:
            self._episode_meta_cache[path] = load_json(path)
        return self._episode_meta_cache[path]
