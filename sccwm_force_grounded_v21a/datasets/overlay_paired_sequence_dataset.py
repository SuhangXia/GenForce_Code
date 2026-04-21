from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sccwm.datasets.paired_sequence_dataset import PairedSequenceDataset
from sccwm.datasets.utils import build_temporal_observation


class OverlayPairedSequenceDataset(PairedSequenceDataset):
    """Paired sequence dataset with split image and metadata roots.

    Important anchor semantics:
    - `anchor_position` is the sample's true anchor in the full temporal group.
    - `window_positions` is the clamped local window around that anchor.
    - `seq_valid_mask` inherited from the original SCCWM pipeline is a one-hot
      anchor marker inside the window, not a generic "all valid frames" mask.

    v2.1a surfaces the anchor explicitly so losses do not need to infer it from
    `seq_valid_mask` or assume center alignment.
    """

    def __init__(
        self,
        image_root: str | Path,
        asset_root: str | Path,
        *,
        split: str = "train",
        sequence_length: int = 3,
        index_file: str | Path | None = None,
        allow_cross_split: bool = True,
        episode_id_subset: set[int] | None = None,
        gray_cache_max_items: int = 256,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        self.asset_root = Path(asset_root).expanduser().resolve()
        super().__init__(
            self.asset_root,
            split=split,
            sequence_length=sequence_length,
            index_file=index_file,
            allow_cross_split=allow_cross_split,
            episode_id_subset=episode_id_subset,
            gray_cache_max_items=gray_cache_max_items,
        )

    def _map_asset_path_to_image_path(self, asset_path: Path) -> Path:
        asset_path = asset_path.expanduser().resolve()
        try:
            relative = asset_path.relative_to(self.asset_root)
        except ValueError as exc:
            raise ValueError(f"Asset path {asset_path} is not under asset_root={self.asset_root}") from exc
        mapped = (self.image_root / relative).resolve()
        if mapped.exists():
            return mapped
        if self.image_root == self.asset_root and asset_path.exists():
            return asset_path
        raise FileNotFoundError(
            f"Overlay image file is missing for asset path {asset_path}. Expected mapped path: {mapped}"
        )

    def _build_window(self, sample: dict[str, Any], *, branch: str, anchor_position: int) -> dict[str, Any]:
        assert branch in {"source", "target"}
        coord_map = self._load_coord(sample[f"{branch}_coord_path"])
        scale_dir: Path = sample[f"{branch}_scale_dir"]
        marker_name = sample[f"{branch}_marker_name"]
        frame_meta_map = self._load_frame_meta_map(scale_dir / "sequence_metadata.json")
        seq_idx_seq = sample["global_seq_index_seq"]
        window_positions = self._window_positions(anchor_position, len(seq_idx_seq))
        obs_frames: list[torch.Tensor] = []
        phase_indices: list[int] = []
        phase_progress: list[float] = []
        anchor_seq_idx = seq_idx_seq[anchor_position]
        anchor_meta = frame_meta_map[anchor_seq_idx]
        anchor_rest_asset_path = scale_dir / "frame_000000" / marker_name
        if not anchor_rest_asset_path.exists():
            anchor_rest_asset_path = scale_dir / anchor_meta.frame_name / marker_name
        gray_rest = self._load_gray(self._map_asset_path_to_image_path(anchor_rest_asset_path))
        previous_gray = None
        for pos in window_positions:
            seq_idx = seq_idx_seq[pos]
            meta = frame_meta_map[seq_idx]
            frame_asset_path = scale_dir / meta.frame_name / marker_name
            gray = self._load_gray(self._map_asset_path_to_image_path(frame_asset_path))
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

    def _window_positions(self, anchor_position: int, total: int) -> list[int]:
        from sccwm.datasets.utils import clamp_window

        return clamp_window(anchor_position, self.sequence_length, total)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        group_index = index
        if self._cumulative_sizes:
            import bisect

            group_index = bisect.bisect_right(self._cumulative_sizes, index)
        prev_size = 0 if group_index == 0 else self._cumulative_sizes[group_index - 1]
        anchor_position = index - prev_size
        group = self._groups[group_index]
        window_positions = self._window_positions(anchor_position, len(group["global_seq_index_seq"]))
        try:
            anchor_index_in_window = window_positions.index(anchor_position)
        except ValueError:
            anchor_index_in_window = len(window_positions) // 2
        sample["anchor_index_in_window"] = torch.tensor(anchor_index_in_window, dtype=torch.long)
        sample["anchor_position_in_sequence"] = torch.tensor(anchor_position, dtype=torch.long)
        sample["window_center_index"] = torch.tensor(len(window_positions) // 2, dtype=torch.long)
        sample["anchor_selection_mode"] = "explicit_window_anchor"
        sample["anchor_phase_name"] = group["phase_name_seq"][anchor_position]
        sample["anchor_phase_progress"] = sample["phase_progress"][anchor_index_in_window].to(torch.float32).clone()
        return sample
