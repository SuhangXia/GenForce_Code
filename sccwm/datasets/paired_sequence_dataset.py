from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, get_worker_info

from .utils import (
    FrameMetadata,
    build_seq_valid_mask,
    build_temporal_observation,
    clamp_window,
    infer_boundary_subset,
    infer_episode_dir_from_scale_dir,
    infer_scale_dir_from_frame,
    iter_jsonl,
    load_coord_map,
    load_json,
    load_rgb,
    resolve_existing_path,
    rgb_to_gray,
    safe_float,
    safe_int,
)


class PairedSequenceDataset(Dataset[dict[str, Any]]):
    """Paired source-target sequence dataset backed by dataset-A pair_index JSONL."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "train",
        sequence_length: int = 3,
        index_file: str | Path | None = None,
        allow_cross_split: bool = True,
        episode_id_subset: set[int] | None = None,
        gray_cache_max_items: int = 256,
    ) -> None:
        super().__init__()
        if sequence_length not in {1, 3, 5}:
            raise ValueError(f"sequence_length must be one of {{1,3,5}}, got {sequence_length}")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.allow_cross_split = bool(allow_cross_split)
        self.episode_id_subset = None if episode_id_subset is None else {int(v) for v in episode_id_subset}
        self.uses_streaming_index = True
        self.gray_cache_max_items = max(int(gray_cache_max_items), 0)
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._gray_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._episode_meta_cache: dict[Path, dict[str, Any]] = {}
        self._sequence_meta_cache: dict[Path, dict[str, Any]] = {}
        self._frame_meta_cache: dict[Path, dict[int, FrameMetadata]] = {}

        default_name = f"pair_index_{self.split}.jsonl" if self.split in {"train", "val", "test"} else "pair_index.jsonl"
        self.index_path = resolve_existing_path(self.dataset_root, index_file or default_name)
        self._groups = self._build_groups()
        self._cumulative_sizes = self._build_cumulative_sizes(self._groups)
        if not self._groups:
            raise RuntimeError(f"No paired SCCWM samples found in {self.index_path}")

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_index = bisect.bisect_right(self._cumulative_sizes, index)
        prev_size = 0 if group_index == 0 else self._cumulative_sizes[group_index - 1]
        anchor_position = index - prev_size
        sample = self._groups[group_index]
        source_window = self._build_window(sample, branch="source", anchor_position=anchor_position)
        target_window = self._build_window(sample, branch="target", anchor_position=anchor_position)
        seq_valid_mask = build_seq_valid_mask(anchor_position, source_window["window_positions"])
        absolute_contact = sample["absolute_contact_xy_mm"]
        world_origin = sample["world_origin_xy_mm"]
        return {
            "source_obs": source_window["obs"],
            "target_obs": target_window["obs"],
            "source_rest_obs": source_window["rest_obs"],
            "target_rest_obs": target_window["rest_obs"],
            "source_coord_map": source_window["coord_map"],
            "target_coord_map": target_window["coord_map"],
            "source_scale_mm": torch.tensor(sample["source_scale_mm"], dtype=torch.float32),
            "target_scale_mm": torch.tensor(sample["target_scale_mm"], dtype=torch.float32),
            "x_norm": torch.tensor(sample["x_norm_seq"][anchor_position], dtype=torch.float32),
            "y_norm": torch.tensor(sample["y_norm_seq"][anchor_position], dtype=torch.float32),
            "depth_mm": torch.tensor(sample["depth_mm_seq"][anchor_position], dtype=torch.float32),
            "source_depth_mm": torch.tensor(sample["source_depth_mm_seq"][anchor_position], dtype=torch.float32),
            "target_depth_mm": torch.tensor(sample["target_depth_mm_seq"][anchor_position], dtype=torch.float32),
            "phase_names": [sample["phase_name_seq"][pos] for pos in source_window["window_positions"]],
            "phase_indices": torch.tensor(source_window["phase_indices"], dtype=torch.long),
            "phase_progress": torch.tensor(source_window["phase_progress"], dtype=torch.float32),
            "seq_valid_mask": seq_valid_mask,
            "source_contact_mm": torch.tensor(sample["source_contact_mm_seq"][anchor_position], dtype=torch.float32),
            "target_contact_mm": torch.tensor(sample["target_contact_mm_seq"][anchor_position], dtype=torch.float32),
            "absolute_contact_xy_mm": torch.tensor(absolute_contact if absolute_contact is not None else [float("nan"), float("nan")], dtype=torch.float32),
            "world_origin_xy_mm": torch.tensor(world_origin if world_origin is not None else [0.0, 0.0], dtype=torch.float32),
            "has_absolute_contact_xy_mm": torch.tensor(absolute_contact is not None, dtype=torch.bool),
            "pair_split": sample["pair_split"],
            "source_scale_split": sample["source_scale_split"],
            "target_scale_split": sample["target_scale_split"],
            "indenter_split": sample["indenter_split"],
            "boundary_subset": sample["boundary_subset"],
            "is_unseen_indenter": torch.tensor(sample["is_unseen_indenter"], dtype=torch.bool),
            "is_unseen_scale_source": torch.tensor(sample["is_unseen_scale_source"], dtype=torch.bool),
            "is_unseen_scale_target": torch.tensor(sample["is_unseen_scale_target"], dtype=torch.bool),
            "episode_id": torch.tensor(sample["episode_id"], dtype=torch.long),
            "global_seq_index": torch.tensor(sample["global_seq_index_seq"][anchor_position], dtype=torch.long),
            "metadata": {
                "episode_id": sample["episode_id"],
                "source_scale_key": sample["source_scale_key"],
                "target_scale_key": sample["target_scale_key"],
                "source_marker_name": sample["source_marker_name"],
                "target_marker_name": sample["target_marker_name"],
                "phase_name": sample["phase_name_seq"][anchor_position],
                "frame_key": (
                    f"{sample['episode_id']}::{sample['source_scale_key']}::{sample['target_scale_key']}::"
                    f"{sample['source_marker_name']}::{sample['target_marker_name']}::{sample['global_seq_index_seq'][anchor_position]}"
                ),
            },
        }

    @staticmethod
    def _build_cumulative_sizes(groups: list[dict[str, Any]]) -> list[int]:
        total = 0
        sizes: list[int] = []
        for group in groups:
            total += len(group["global_seq_index_seq"])
            sizes.append(total)
        return sizes

    def _build_groups(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in iter_jsonl(self.index_path):
            episode_id = safe_int(row["episode_id"], "episode_id")
            if self.episode_id_subset is not None and episode_id not in self.episode_id_subset:
                continue
            pair_split = str(row.get("pair_split", ""))
            if (not self.allow_cross_split) and pair_split == "cross_split":
                continue
            key = (
                episode_id,
                str(row["source_scale_key"]),
                str(row["target_scale_key"]),
                str(row["source_marker_name"]),
                str(row["target_marker_name"]),
            )
            group = grouped.get(key)
            if group is None:
                source_frame_path = resolve_existing_path(self.dataset_root, row["source_frame_relpath"])
                target_frame_path = resolve_existing_path(self.dataset_root, row["target_frame_relpath"])
                source_coord_path = resolve_existing_path(self.dataset_root, row["source_adapter_coord_map_relpath"])
                target_coord_path = resolve_existing_path(self.dataset_root, row["target_adapter_coord_map_relpath"])
                source_scale_dir = infer_scale_dir_from_frame(source_frame_path)
                target_scale_dir = infer_scale_dir_from_frame(target_frame_path)
                episode_meta = self._load_episode_meta(infer_episode_dir_from_scale_dir(source_scale_dir) / "metadata.json")
                group = {
                    "episode_id": key[0],
                    "source_scale_key": key[1],
                    "target_scale_key": key[2],
                    "source_marker_name": key[3],
                    "target_marker_name": key[4],
                    "source_coord_path": source_coord_path,
                    "target_coord_path": target_coord_path,
                    "source_scale_dir": source_scale_dir,
                    "target_scale_dir": target_scale_dir,
                    "source_scale_mm": safe_float(row["source_scale_mm"], "source_scale_mm"),
                    "target_scale_mm": safe_float(row["target_scale_mm"], "target_scale_mm"),
                    "pair_split": str(row.get("pair_split", "")),
                    "source_scale_split": str(row.get("source_scale_split", "")),
                    "target_scale_split": str(row.get("target_scale_split", "")),
                    "indenter_split": str(row.get("indenter_split", "")),
                    "boundary_subset": infer_boundary_subset(episode_meta.get("position_sample_type_actual")),
                    "is_unseen_indenter": bool(row.get("is_unseen_indenter", False)),
                    "is_unseen_scale_source": bool(row.get("is_unseen_scale_source", False)),
                    "is_unseen_scale_target": bool(row.get("is_unseen_scale_target", False)),
                    "absolute_contact_xy_mm": episode_meta.get("absolute_contact_xy_mm"),
                    "world_origin_xy_mm": episode_meta.get("world_origin_xy_mm"),
                    "global_seq_index_seq": [],
                    "phase_name_seq": [],
                    "x_norm_seq": [],
                    "y_norm_seq": [],
                    "depth_mm_seq": [],
                    "source_depth_mm_seq": [],
                    "target_depth_mm_seq": [],
                    "source_contact_mm_seq": [],
                    "target_contact_mm_seq": [],
                }
                grouped[key] = group
            group["global_seq_index_seq"].append(safe_int(row["global_seq_index"], "global_seq_index"))
            group["phase_name_seq"].append(str(row["phase_name"]))
            group["x_norm_seq"].append(safe_float(row["contact_x_norm"], "contact_x_norm"))
            group["y_norm_seq"].append(safe_float(row["contact_y_norm"], "contact_y_norm"))
            group["depth_mm_seq"].append(safe_float(row["frame_depth_mm"], "frame_depth_mm"))
            group["source_depth_mm_seq"].append(safe_float(row.get("source_frame_depth_mm", row["frame_depth_mm"]), "source_frame_depth_mm"))
            group["target_depth_mm_seq"].append(safe_float(row.get("target_frame_depth_mm", row["frame_depth_mm"]), "target_frame_depth_mm"))
            group["source_contact_mm_seq"].append(
                (
                    safe_float(row["source_contact_x_mm"], "source_contact_x_mm"),
                    safe_float(row["source_contact_y_mm"], "source_contact_y_mm"),
                )
            )
            group["target_contact_mm_seq"].append(
                (
                    safe_float(row["target_contact_x_mm"], "target_contact_x_mm"),
                    safe_float(row["target_contact_y_mm"], "target_contact_y_mm"),
                )
            )
        groups = list(grouped.values())
        for group in groups:
            order = sorted(range(len(group["global_seq_index_seq"])), key=lambda idx: group["global_seq_index_seq"][idx])
            for key_name in (
                "global_seq_index_seq",
                "phase_name_seq",
                "x_norm_seq",
                "y_norm_seq",
                "depth_mm_seq",
                "source_depth_mm_seq",
                "target_depth_mm_seq",
                "source_contact_mm_seq",
                "target_contact_mm_seq",
            ):
                seq = group[key_name]
                group[key_name] = [seq[idx] for idx in order]
        return groups

    def _build_window(self, sample: dict[str, Any], *, branch: str, anchor_position: int) -> dict[str, Any]:
        assert branch in {"source", "target"}
        coord_map = self._load_coord(sample[f"{branch}_coord_path"])
        scale_dir: Path = sample[f"{branch}_scale_dir"]
        marker_name = sample[f"{branch}_marker_name"]
        frame_meta_map = self._load_frame_meta_map(scale_dir / "sequence_metadata.json")
        seq_idx_seq = sample["global_seq_index_seq"]
        window_positions = clamp_window(anchor_position, self.sequence_length, len(seq_idx_seq))
        obs_frames: list[torch.Tensor] = []
        phase_indices: list[int] = []
        phase_progress: list[float] = []
        anchor_seq_idx = seq_idx_seq[anchor_position]
        anchor_meta = frame_meta_map[anchor_seq_idx]
        anchor_rest_path = scale_dir / "frame_000000" / marker_name
        if not anchor_rest_path.exists():
            anchor_rest_path = scale_dir / anchor_meta.frame_name / marker_name
        gray_rest = self._load_gray(anchor_rest_path)
        previous_gray = None
        for pos in window_positions:
            seq_idx = seq_idx_seq[pos]
            meta = frame_meta_map[seq_idx]
            frame_path = scale_dir / meta.frame_name / marker_name
            gray = self._load_gray(frame_path)
            obs_frames.append(build_temporal_observation(gray, gray_rest, previous_gray))
            phase_indices.append(meta.phase_index)
            phase_progress.append(meta.phase_progress)
            previous_gray = gray
        rest_obs = build_temporal_observation(gray_rest, gray_rest, None)
        return {
            "obs": torch.stack(obs_frames, dim=0),
            "rest_obs": rest_obs,
            "coord_map": coord_map,
            "phase_indices": phase_indices,
            "phase_progress": phase_progress,
            "window_positions": window_positions,
        }

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

    def _load_scale_sequence_meta(self, path: Path) -> dict[str, Any]:
        if path not in self._sequence_meta_cache:
            self._sequence_meta_cache[path] = load_json(path)
        return self._sequence_meta_cache[path]

    def _load_frame_meta_map(self, path: Path) -> dict[int, FrameMetadata]:
        if path not in self._frame_meta_cache:
            self._frame_meta_cache[path] = self._frame_meta_map(self._load_scale_sequence_meta(path))
        return self._frame_meta_cache[path]

    @staticmethod
    def _frame_meta_map(sequence_meta: dict[str, Any]) -> dict[int, FrameMetadata]:
        frames = sequence_meta.get("frames", [])
        if not isinstance(frames, list):
            raise TypeError("sequence_metadata.frames must be a list")
        out: dict[int, FrameMetadata] = {}
        for frame in frames:
            seq_idx = safe_int(frame["global_seq_index"], "global_seq_index")
            out[seq_idx] = FrameMetadata(
                global_seq_index=seq_idx,
                frame_name=str(frame.get("frame_name", f"frame_{seq_idx:06d}")),
                phase_name=str(frame.get("phase_name", "")),
                phase_index=safe_int(frame.get("phase_index", 0), "phase_index"),
                phase_progress=safe_float(frame.get("phase_progress", 0.0), "phase_progress"),
                source_physics_frame_index=safe_int(frame.get("source_physics_frame_index", 0), "source_physics_frame_index"),
                frame_actual_max_down_mm=safe_float(frame.get("frame_actual_max_down_mm", 0.0), "frame_actual_max_down_mm"),
                is_synthetic_precontact=bool(frame.get("is_synthetic_precontact", False)),
                is_synthetic_release=bool(frame.get("is_synthetic_release", False)),
                is_dwell_repeat=bool(frame.get("is_dwell_repeat", False)),
                rendered_markers=tuple(str(v) for v in frame.get("rendered_markers", [])),
            )
        return out
