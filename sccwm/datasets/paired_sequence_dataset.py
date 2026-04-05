from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .utils import (
    FrameMetadata,
    build_seq_valid_mask,
    build_temporal_observation,
    clamp_window,
    infer_boundary_subset,
    infer_episode_dir_from_scale_dir,
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
    ) -> None:
        super().__init__()
        if sequence_length not in {1, 3, 5}:
            raise ValueError(f"sequence_length must be one of {{1,3,5}}, got {sequence_length}")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.allow_cross_split = bool(allow_cross_split)
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._gray_cache: dict[Path, torch.Tensor] = {}
        self._episode_meta_cache: dict[Path, dict[str, Any]] = {}
        self._sequence_meta_cache: dict[Path, dict[str, Any]] = {}

        default_name = f"pair_index_{self.split}.jsonl" if self.split in {"train", "val", "test"} else "pair_index.jsonl"
        self.index_path = resolve_existing_path(self.dataset_root, index_file or default_name)
        self._samples = self._build_samples()
        if not self._samples:
            raise RuntimeError(f"No paired SCCWM samples found in {self.index_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        source_window = self._build_window(sample, branch="source")
        target_window = self._build_window(sample, branch="target")
        seq_valid_mask = build_seq_valid_mask(sample["anchor_position"], source_window["window_positions"])
        absolute_contact = sample.get("absolute_contact_xy_mm")
        world_origin = sample.get("world_origin_xy_mm")
        return {
            "source_obs": source_window["obs"],
            "target_obs": target_window["obs"],
            "source_rest_obs": source_window["rest_obs"],
            "target_rest_obs": target_window["rest_obs"],
            "source_coord_map": source_window["coord_map"],
            "target_coord_map": target_window["coord_map"],
            "source_scale_mm": torch.tensor(sample["source_scale_mm"], dtype=torch.float32),
            "target_scale_mm": torch.tensor(sample["target_scale_mm"], dtype=torch.float32),
            "x_norm": torch.tensor(sample["x_norm"], dtype=torch.float32),
            "y_norm": torch.tensor(sample["y_norm"], dtype=torch.float32),
            "depth_mm": torch.tensor(sample["depth_mm"], dtype=torch.float32),
            "source_depth_mm": torch.tensor(sample["source_depth_mm"], dtype=torch.float32),
            "target_depth_mm": torch.tensor(sample["target_depth_mm"], dtype=torch.float32),
            "phase_names": sample["phase_names"],
            "phase_indices": torch.tensor(source_window["phase_indices"], dtype=torch.long),
            "phase_progress": torch.tensor(source_window["phase_progress"], dtype=torch.float32),
            "seq_valid_mask": seq_valid_mask,
            "source_contact_mm": torch.tensor(sample["source_contact_mm"], dtype=torch.float32),
            "target_contact_mm": torch.tensor(sample["target_contact_mm"], dtype=torch.float32),
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
            "global_seq_index": torch.tensor(sample["global_seq_index"], dtype=torch.long),
            "metadata": {
                "episode_id": sample["episode_id"],
                "source_scale_key": sample["source_scale_key"],
                "target_scale_key": sample["target_scale_key"],
                "source_marker_name": sample["source_marker_name"],
                "target_marker_name": sample["target_marker_name"],
                "phase_name": sample["phase_name"],
                "frame_key": sample["frame_key"],
            },
        }

    def _build_samples(self) -> list[dict[str, Any]]:
        rows = read_jsonl(self.index_path)
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pair_split = str(row.get("pair_split", ""))
            if (not self.allow_cross_split) and pair_split == "cross_split":
                continue
            key = (
                safe_int(row["episode_id"], "episode_id"),
                str(row["source_scale_key"]),
                str(row["target_scale_key"]),
                str(row["source_marker_name"]),
                str(row["target_marker_name"]),
            )
            grouped[key].append(row)

        samples: list[dict[str, Any]] = []
        for key, items in grouped.items():
            items_sorted = sorted(items, key=lambda row: safe_int(row["global_seq_index"], "global_seq_index"))
            source_frame_paths = [resolve_existing_path(self.dataset_root, row["source_frame_relpath"]) for row in items_sorted]
            target_frame_paths = [resolve_existing_path(self.dataset_root, row["target_frame_relpath"]) for row in items_sorted]
            source_coord_path = resolve_existing_path(self.dataset_root, items_sorted[0]["source_adapter_coord_map_relpath"])
            target_coord_path = resolve_existing_path(self.dataset_root, items_sorted[0]["target_adapter_coord_map_relpath"])
            source_scale_dir = infer_scale_dir_from_frame(source_frame_paths[0])
            target_scale_dir = infer_scale_dir_from_frame(target_frame_paths[0])
            source_sequence_meta = self._load_scale_sequence_meta(source_scale_dir / "sequence_metadata.json")
            target_sequence_meta = self._load_scale_sequence_meta(target_scale_dir / "sequence_metadata.json")
            episode_meta = self._load_episode_meta(infer_episode_dir_from_scale_dir(source_scale_dir) / "metadata.json")
            source_frame_meta = self._frame_meta_map(source_sequence_meta)
            target_frame_meta = self._frame_meta_map(target_sequence_meta)
            boundary_subset = infer_boundary_subset(episode_meta.get("position_sample_type_actual"))

            for anchor_pos, row in enumerate(items_sorted):
                seq_idx = safe_int(row["global_seq_index"], "global_seq_index")
                src_meta = source_frame_meta.get(seq_idx)
                tgt_meta = target_frame_meta.get(seq_idx)
                if src_meta is None or tgt_meta is None:
                    raise KeyError(
                        f"Missing frame metadata for episode={key[0]} source_scale={key[1]} target_scale={key[2]} seq_idx={seq_idx}"
                    )
                samples.append(
                    {
                        "episode_id": key[0],
                        "source_scale_key": key[1],
                        "target_scale_key": key[2],
                        "source_marker_name": key[3],
                        "target_marker_name": key[4],
                        "frame_key": f"{key[0]}::{key[1]}::{key[2]}::{key[3]}::{key[4]}::{seq_idx}",
                        "items_sorted": items_sorted,
                        "source_frame_paths": source_frame_paths,
                        "target_frame_paths": target_frame_paths,
                        "source_coord_path": source_coord_path,
                        "target_coord_path": target_coord_path,
                        "source_frame_meta": source_frame_meta,
                        "target_frame_meta": target_frame_meta,
                        "anchor_position": anchor_pos,
                        "global_seq_index": seq_idx,
                        "source_scale_mm": safe_float(row["source_scale_mm"], "source_scale_mm"),
                        "target_scale_mm": safe_float(row["target_scale_mm"], "target_scale_mm"),
                        "x_norm": safe_float(row["contact_x_norm"], "contact_x_norm"),
                        "y_norm": safe_float(row["contact_y_norm"], "contact_y_norm"),
                        "depth_mm": safe_float(row["frame_depth_mm"], "frame_depth_mm"),
                        "source_depth_mm": safe_float(row.get("source_frame_depth_mm", row["frame_depth_mm"]), "source_frame_depth_mm"),
                        "target_depth_mm": safe_float(row.get("target_frame_depth_mm", row["frame_depth_mm"]), "target_frame_depth_mm"),
                        "source_contact_mm": (
                            safe_float(row["source_contact_x_mm"], "source_contact_x_mm"),
                            safe_float(row["source_contact_y_mm"], "source_contact_y_mm"),
                        ),
                        "target_contact_mm": (
                            safe_float(row["target_contact_x_mm"], "target_contact_x_mm"),
                            safe_float(row["target_contact_y_mm"], "target_contact_y_mm"),
                        ),
                        "pair_split": str(row.get("pair_split", "")),
                        "source_scale_split": str(row.get("source_scale_split", "")),
                        "target_scale_split": str(row.get("target_scale_split", "")),
                        "indenter_split": str(row.get("indenter_split", "")),
                        "boundary_subset": boundary_subset,
                        "is_unseen_indenter": bool(row.get("is_unseen_indenter", False)),
                        "is_unseen_scale_source": bool(row.get("is_unseen_scale_source", False)),
                        "is_unseen_scale_target": bool(row.get("is_unseen_scale_target", False)),
                        "phase_name": str(row["phase_name"]),
                        "phase_names": [str(items_sorted[pos]["phase_name"]) for pos in clamp_window(anchor_pos, self.sequence_length, len(items_sorted))],
                        "absolute_contact_xy_mm": episode_meta.get("absolute_contact_xy_mm"),
                        "world_origin_xy_mm": episode_meta.get("world_origin_xy_mm"),
                    }
                )
        return samples

    def _build_window(self, sample: dict[str, Any], *, branch: str) -> dict[str, Any]:
        assert branch in {"source", "target"}
        items = sample["items_sorted"]
        frame_paths = sample[f"{branch}_frame_paths"]
        coord_map = self._load_coord(sample[f"{branch}_coord_path"])
        frame_meta_map: dict[int, FrameMetadata] = sample[f"{branch}_frame_meta"]
        window_positions = clamp_window(sample["anchor_position"], self.sequence_length, len(items))
        obs_frames: list[torch.Tensor] = []
        phase_indices: list[int] = []
        phase_progress: list[float] = []
        anchor_rest_path = infer_rest_frame_path(frame_paths[sample["anchor_position"]])
        gray_rest = self._load_gray(anchor_rest_path)
        previous_gray = None
        for pos in window_positions:
            frame_path = frame_paths[pos]
            gray = self._load_gray(frame_path)
            obs_frames.append(build_temporal_observation(gray, gray_rest, previous_gray))
            row = items[pos]
            seq_idx = safe_int(row["global_seq_index"], "global_seq_index")
            meta = frame_meta_map[seq_idx]
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
        if path not in self._gray_cache:
            self._gray_cache[path] = torch.from_numpy(rgb_to_gray(load_rgb(path)))
        return self._gray_cache[path]

    def _load_episode_meta(self, path: Path) -> dict[str, Any]:
        if path not in self._episode_meta_cache:
            self._episode_meta_cache[path] = load_json(path)
        return self._episode_meta_cache[path]

    def _load_scale_sequence_meta(self, path: Path) -> dict[str, Any]:
        if path not in self._sequence_meta_cache:
            self._sequence_meta_cache[path] = load_json(path)
        return self._sequence_meta_cache[path]

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
