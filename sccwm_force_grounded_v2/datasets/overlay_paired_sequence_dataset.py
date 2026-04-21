from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sccwm.datasets.paired_sequence_dataset import PairedSequenceDataset
from sccwm.datasets.utils import build_temporal_observation


class OverlayPairedSequenceDataset(PairedSequenceDataset):
    """Paired sequence dataset with split image and metadata roots.

    Images are read from ``image_root`` while pair indices, metadata, sequence metadata,
    and coord maps are resolved from ``asset_root``. This is intended for black-border
    image overlays where non-image assets still live under the original dataset root.
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
            f"Overlay image file is missing for asset path {asset_path}. "
            f"Expected mapped image path: {mapped}"
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

