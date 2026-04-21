#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.datasets.utils import (
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
from sccwm.eval.common import (
    _batch_item,
    _format_elapsed,
    _move_eval_batch,
    _progress_write,
    _summarize_direct_records,
    save_eval_result,
)
from sccwm.eval.eval_dwl_tr import load_dwl_tr_for_eval
from sccwm.eval.eval_no_projector_tr import load_no_projector_tr_for_eval
from sccwm.losses import build_state_embedding
from sccwm.train.common import default_device
from sccwm.utils.config import load_config_with_overrides
from sccwm.eval.common import load_sccwm_for_eval

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


BAND_16 = (15.8, 16.2)
BAND_23 = (22.8, 23.2)
PROTOCOL_CHOICES = [
    "cross_band_16_to_23",
    "cross_band_23_to_16",
    "cross_band_16_23_bidirectional",
]
MODEL_TYPES = ["sccwm", "dwl_tr", "no_projector_tr"]
DEFAULT_CONFIGS = {
    "sccwm": "sccwm/configs/sccwm_stage3.yaml",
    "dwl_tr": "sccwm/configs/dwl_tr.yaml",
    "no_projector_tr": "sccwm/configs/no_projector_tr.yaml",
}


@dataclass(frozen=True)
class BranchCandidate:
    root_label: str
    dataset_root: Path
    frame_path: Path
    coord_path: Path
    scale_mm: float
    scale_split: str
    marker_name: str
    indenter: str
    indenter_split: str
    phase_name: str
    phase_progress: float
    x_norm: float
    y_norm: float
    depth_mm: float
    episode_id: int
    global_seq_index: int
    pair_split: str
    is_unseen_indenter: bool
    is_unseen_scale_target: bool

    @property
    def state_key(self) -> tuple[Any, ...]:
        # Strict same-state key across bands. This intentionally requires the
        # same indenter, marker, phase, and normalized contact/depth state.
        return (
            self.indenter,
            self.marker_name,
            self.phase_name,
            round(self.phase_progress, 6),
            round(self.x_norm, 6),
            round(self.y_norm, 6),
            round(self.depth_mm, 6),
        )

    @property
    def unique_id(self) -> tuple[str, str, int]:
        return (str(self.frame_path), str(self.coord_path), int(self.global_seq_index))


@dataclass(frozen=True)
class CrossBandPairSpec:
    pair_id: str
    source: BranchCandidate
    target: BranchCandidate
    x_norm: float
    y_norm: float
    depth_mm: float
    boundary_subset: str
    is_unseen_indenter: bool
    is_unseen_scale_target: bool


class CrossBandPairDataset(Dataset[dict[str, Any]]):
    def __init__(self, pair_specs: list[CrossBandPairSpec], *, sequence_length: int, gray_cache_max_items: int) -> None:
        self.pair_specs = pair_specs
        self.sequence_length = int(sequence_length)
        self.gray_cache_max_items = max(int(gray_cache_max_items), 0)
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._gray_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._sequence_meta_cache: dict[Path, dict[str, Any]] = {}
        self._frame_meta_cache: dict[Path, dict[int, dict[str, Any]]] = {}
        self._sequence_index_cache: dict[Path, list[int]] = {}
        self._episode_meta_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.pair_specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pair_specs[index]
        source_window = self._build_window(pair.source)
        target_window = self._build_window(pair.target)
        absolute_contact = self._select_vector(pair, key="absolute_contact_xy_mm")
        world_origin = self._select_vector(pair, key="world_origin_xy_mm", default=[0.0, 0.0])
        seq_valid_mask = source_window["seq_valid_mask"]
        return {
            "pair_id": pair.pair_id,
            "source_obs": source_window["obs"],
            "target_obs": target_window["obs"],
            "source_coord_map": source_window["coord_map"],
            "target_coord_map": target_window["coord_map"],
            "source_scale_mm": torch.tensor(pair.source.scale_mm, dtype=torch.float32),
            "target_scale_mm": torch.tensor(pair.target.scale_mm, dtype=torch.float32),
            "x_norm": torch.tensor(pair.x_norm, dtype=torch.float32),
            "y_norm": torch.tensor(pair.y_norm, dtype=torch.float32),
            "depth_mm": torch.tensor(pair.depth_mm, dtype=torch.float32),
            "seq_valid_mask": seq_valid_mask,
            "absolute_contact_xy_mm": torch.tensor(absolute_contact if absolute_contact is not None else [float("nan"), float("nan")], dtype=torch.float32),
            "world_origin_xy_mm": torch.tensor(world_origin, dtype=torch.float32),
            "has_absolute_contact_xy_mm": torch.tensor(absolute_contact is not None, dtype=torch.bool),
            "source_scale_split": pair.source.scale_split,
            "target_scale_split": pair.target.scale_split,
            "boundary_subset": pair.boundary_subset,
            "is_unseen_indenter": torch.tensor(pair.is_unseen_indenter, dtype=torch.bool),
            "is_unseen_scale_target": torch.tensor(pair.is_unseen_scale_target, dtype=torch.bool),
            "source_episode_id": torch.tensor(pair.source.episode_id, dtype=torch.long),
            "target_episode_id": torch.tensor(pair.target.episode_id, dtype=torch.long),
            "source_global_seq_index": torch.tensor(pair.source.global_seq_index, dtype=torch.long),
            "target_global_seq_index": torch.tensor(pair.target.global_seq_index, dtype=torch.long),
            "source_marker_name": pair.source.marker_name,
            "target_marker_name": pair.target.marker_name,
            "source_indenter": pair.source.indenter,
            "target_indenter": pair.target.indenter,
        }

    def _select_vector(self, pair: CrossBandPairSpec, *, key: str, default: list[float] | None = None) -> list[float] | None:
        source_meta = self._load_episode_meta_for_candidate(pair.source)
        target_meta = self._load_episode_meta_for_candidate(pair.target)
        source_value = source_meta.get(key)
        target_value = target_meta.get(key)
        if source_value is not None:
            return [float(source_value[0]), float(source_value[1])]
        if target_value is not None:
            return [float(target_value[0]), float(target_value[1])]
        return None if default is None else [float(default[0]), float(default[1])]

    def _build_window(self, candidate: BranchCandidate) -> dict[str, Any]:
        coord_map = self._load_coord(candidate.coord_path)
        scale_dir = infer_scale_dir_from_frame(candidate.frame_path)
        sequence_meta_path = scale_dir / "sequence_metadata.json"
        frame_meta_map = self._load_frame_meta_map(sequence_meta_path)
        seq_indices = self._load_sequence_indices(sequence_meta_path)
        position_lookup = {seq_idx: idx for idx, seq_idx in enumerate(seq_indices)}
        if candidate.global_seq_index not in position_lookup:
            raise KeyError(f"Global sequence index {candidate.global_seq_index} not found in {sequence_meta_path}")
        anchor_position = position_lookup[candidate.global_seq_index]
        window_positions = clamp_window(anchor_position, self.sequence_length, len(seq_indices))
        anchor_rest_path = scale_dir / "frame_000000" / candidate.marker_name
        if not anchor_rest_path.exists():
            anchor_rest_path = candidate.frame_path
        gray_rest = self._load_gray(anchor_rest_path)
        previous_gray = None
        obs_frames: list[torch.Tensor] = []
        for pos in window_positions:
            seq_idx = seq_indices[pos]
            frame_meta = frame_meta_map[seq_idx]
            frame_path = scale_dir / str(frame_meta["frame_name"]) / candidate.marker_name
            gray = self._load_gray(frame_path)
            obs_frames.append(build_temporal_observation(gray.numpy(), gray_rest.numpy(), None if previous_gray is None else previous_gray.numpy()))
            previous_gray = gray
        return {
            "obs": torch.stack(obs_frames, dim=0),
            "coord_map": coord_map,
            "seq_valid_mask": build_seq_valid_mask(anchor_position, window_positions),
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
        if self.gray_cache_max_items > 0:
            self._gray_cache[path] = gray
            self._gray_cache.move_to_end(path)
            while len(self._gray_cache) > self.gray_cache_max_items:
                self._gray_cache.popitem(last=False)
        return gray

    def _load_sequence_indices(self, path: Path) -> list[int]:
        cached = self._sequence_index_cache.get(path)
        if cached is not None:
            return cached
        sequence_meta = self._load_sequence_meta(path)
        frames = sequence_meta.get("frames", [])
        if not isinstance(frames, list):
            raise TypeError(f"Expected frames list in {path}")
        indices = [safe_int(frame["global_seq_index"], "global_seq_index") for frame in frames]
        self._sequence_index_cache[path] = indices
        return indices

    def _load_sequence_meta(self, path: Path) -> dict[str, Any]:
        cached = self._sequence_meta_cache.get(path)
        if cached is not None:
            return cached
        payload = load_json(path)
        self._sequence_meta_cache[path] = payload
        return payload

    def _load_frame_meta_map(self, path: Path) -> dict[int, dict[str, Any]]:
        cached = self._frame_meta_cache.get(path)
        if cached is not None:
            return cached
        frames = self._load_sequence_meta(path).get("frames", [])
        if not isinstance(frames, list):
            raise TypeError(f"Expected frames list in {path}")
        mapped: dict[int, dict[str, Any]] = {}
        for frame in frames:
            seq_idx = safe_int(frame["global_seq_index"], "global_seq_index")
            mapped[seq_idx] = frame
        self._frame_meta_cache[path] = mapped
        return mapped

    def _load_episode_meta_for_candidate(self, candidate: BranchCandidate) -> dict[str, Any]:
        scale_dir = infer_scale_dir_from_frame(candidate.frame_path)
        path = infer_episode_dir_from_scale_dir(scale_dir) / "metadata.json"
        cached = self._episode_meta_cache.get(path)
        if cached is not None:
            return cached
        payload = load_json(path)
        self._episode_meta_cache[path] = payload
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strict 16mm<->23mm cross-band protocols across merged held-out pools.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--model-type", type=str, required=True, choices=MODEL_TYPES)
    parser.add_argument("--protocol", type=str, required=True, choices=PROTOCOL_CHOICES)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _in_band(scale_mm: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return float(lo) <= float(scale_mm) <= float(hi)


def _protocol_description(protocol: str) -> str:
    if protocol == "cross_band_16_to_23":
        return "source in [15.8,16.2], target in [22.8,23.2]"
    if protocol == "cross_band_23_to_16":
        return "source in [22.8,23.2], target in [15.8,16.2]"
    return "bidirectional union of 16->23 and 23->16"


def _load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config or DEFAULT_CONFIGS[args.model_type]
    return load_config_with_overrides(config_path, args.override)


def _iter_branch_candidates(root_label: str, dataset_root: Path, *, split: str) -> Iterable[BranchCandidate]:
    index_name = f"pair_index_{split}.jsonl" if split in {"train", "val", "test"} else "pair_index.jsonl"
    index_path = resolve_existing_path(dataset_root, index_name)
    seen: set[tuple[str, str, int]] = set()
    for row in iter_jsonl(index_path):
        for branch in ("source", "target"):
            scale_mm = safe_float(row[f"{branch}_scale_mm"], f"{branch}_scale_mm")
            if not (_in_band(scale_mm, BAND_16) or _in_band(scale_mm, BAND_23)):
                continue
            frame_path = resolve_existing_path(dataset_root, row[f"{branch}_frame_relpath"])
            coord_path = resolve_existing_path(dataset_root, row[f"{branch}_adapter_coord_map_relpath"])
            unique_id = (str(frame_path), str(coord_path), safe_int(row["global_seq_index"], "global_seq_index"))
            if unique_id in seen:
                continue
            seen.add(unique_id)
            yield BranchCandidate(
                root_label=root_label,
                dataset_root=dataset_root,
                frame_path=frame_path,
                coord_path=coord_path,
                scale_mm=scale_mm,
                scale_split=str(row.get(f"{branch}_scale_split", "")),
                marker_name=str(row[f"{branch}_marker_name"]),
                indenter=str(row.get("indenter", "")),
                indenter_split=str(row.get("indenter_split", "")),
                phase_name=str(row.get("phase_name", "")),
                phase_progress=safe_float(row.get("phase_progress", 0.0), "phase_progress"),
                x_norm=safe_float(row["contact_x_norm"], "contact_x_norm"),
                y_norm=safe_float(row["contact_y_norm"], "contact_y_norm"),
                depth_mm=safe_float(row["frame_depth_mm"], "frame_depth_mm"),
                episode_id=safe_int(row["episode_id"], "episode_id"),
                global_seq_index=safe_int(row["global_seq_index"], "global_seq_index"),
                pair_split=str(row.get("pair_split", "")),
                is_unseen_indenter=bool(row.get("is_unseen_indenter", False)),
                is_unseen_scale_target=bool(row.get("is_unseen_scale_target", False)),
            )


def _build_cross_band_pair_specs(cfg: dict[str, Any], *, protocol: str, limit: int) -> tuple[list[CrossBandPairSpec], dict[str, Any]]:
    ds_cfg = cfg.get("dataset", {})
    val_root = resolve_existing_path(Path.cwd(), ds_cfg["val_dataset_a_root"])
    test_root = resolve_existing_path(Path.cwd(), ds_cfg["test_dataset_a_root"])
    pool_16: dict[tuple[Any, ...], list[BranchCandidate]] = defaultdict(list)
    pool_23: dict[tuple[Any, ...], list[BranchCandidate]] = defaultdict(list)
    counts = {
        "val_root": str(val_root),
        "test_root": str(test_root),
        "candidate_16_count": 0,
        "candidate_23_count": 0,
        "val_16_count": 0,
        "test_23_count": 0,
    }
    for candidate in _iter_branch_candidates("val", val_root, split="val"):
        if _in_band(candidate.scale_mm, BAND_16):
            pool_16[candidate.state_key].append(candidate)
            counts["candidate_16_count"] += 1
            counts["val_16_count"] += 1
    for candidate in _iter_branch_candidates("test", test_root, split="test"):
        if _in_band(candidate.scale_mm, BAND_23):
            pool_23[candidate.state_key].append(candidate)
            counts["candidate_23_count"] += 1
            counts["test_23_count"] += 1

    shared_keys = sorted(set(pool_16.keys()) & set(pool_23.keys()))
    counts["matched_state_key_count"] = len(shared_keys)
    counts["protocol"] = protocol
    counts["protocol_description"] = _protocol_description(protocol)
    counts["limit_applied_after_filter"] = True
    counts["filter_stage"] = "metadata_pairing_before_inference"
    if not shared_keys:
        val_indenters = sorted({key[0] for key in pool_16.keys()})
        test_indenters = sorted({key[0] for key in pool_23.keys()})
        raise RuntimeError(
            "Strict 16<->23 cross-band eval found zero matched states across the supplied held-out roots. "
            f"candidate_16={counts['candidate_16_count']} candidate_23={counts['candidate_23_count']} "
            f"matched_state_keys=0. val_16_indenters={val_indenters} test_23_indenters={test_indenters}. "
            "This dataset cannot support a scientifically valid 16<->23 direct eval without regenerating "
            "cross-band paired indices or another shared-state pool."
        )

    pair_specs: list[CrossBandPairSpec] = []
    for state_idx, state_key in enumerate(shared_keys):
        left = sorted(pool_16[state_key], key=lambda item: item.unique_id)
        right = sorted(pool_23[state_key], key=lambda item: item.unique_id)
        for pair_idx, (cand_16, cand_23) in enumerate(zip(left, right), start=1):
            x_norm = 0.5 * (cand_16.x_norm + cand_23.x_norm)
            y_norm = 0.5 * (cand_16.y_norm + cand_23.y_norm)
            depth_mm = 0.5 * (cand_16.depth_mm + cand_23.depth_mm)
            if protocol in {"cross_band_16_to_23", "cross_band_16_23_bidirectional"}:
                pair_specs.append(
                    CrossBandPairSpec(
                        pair_id=f"16to23::{state_idx:06d}::{pair_idx:03d}",
                        source=cand_16,
                        target=cand_23,
                        x_norm=x_norm,
                        y_norm=y_norm,
                        depth_mm=depth_mm,
                        boundary_subset="cross_band_16_to_23",
                        is_unseen_indenter=bool(cand_16.is_unseen_indenter or cand_23.is_unseen_indenter),
                        is_unseen_scale_target=True,
                    )
                )
            if protocol in {"cross_band_23_to_16", "cross_band_16_23_bidirectional"}:
                pair_specs.append(
                    CrossBandPairSpec(
                        pair_id=f"23to16::{state_idx:06d}::{pair_idx:03d}",
                        source=cand_23,
                        target=cand_16,
                        x_norm=x_norm,
                        y_norm=y_norm,
                        depth_mm=depth_mm,
                        boundary_subset="cross_band_23_to_16",
                        is_unseen_indenter=bool(cand_16.is_unseen_indenter or cand_23.is_unseen_indenter),
                        is_unseen_scale_target=True,
                    )
                )
    counts["matched_pair_count_before_limit"] = len(pair_specs)
    if limit > 0:
        pair_specs = pair_specs[:limit]
    counts["matched_pair_count_after_limit"] = len(pair_specs)
    return pair_specs, counts


def _build_loader_from_pair_specs(cfg: dict[str, Any], pair_specs: list[CrossBandPairSpec], *, sequence_length: int | None) -> DataLoader[Any]:
    ds_cfg = cfg.get("dataset", {})
    seq_len = int(sequence_length or ds_cfg.get("sequence_length", 3))
    dataset = CrossBandPairDataset(
        pair_specs,
        sequence_length=seq_len,
        gray_cache_max_items=int(ds_cfg.get("gray_cache_max_items", 256)),
    )
    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("val_batch_size", 8)))
    workers = int(eval_cfg.get("workers", train_cfg.get("workers", 4)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


def _load_model_for_eval(cfg: dict[str, Any], *, model_type: str, checkpoint_path: str | Path, device: torch.device) -> Any:
    if model_type == "sccwm":
        return load_sccwm_for_eval(cfg, checkpoint_path, device)
    if model_type == "dwl_tr":
        return load_dwl_tr_for_eval(cfg, checkpoint_path, device)
    if model_type == "no_projector_tr":
        return load_no_projector_tr_for_eval(cfg, checkpoint_path, device)
    raise ValueError(f"Unsupported model type: {model_type}")


def _forward_single(model_type: str, model: Any, batch: dict[str, Any], *, branch: str) -> tuple[dict[str, Any], torch.Tensor]:
    assert branch in {"source", "target"}
    obs_key = f"{branch}_obs"
    coord_key = f"{branch}_coord_map"
    scale_key = f"{branch}_scale_mm"
    if model_type == "sccwm":
        outputs = model.forward_single(
            batch[obs_key],
            batch[coord_key],
            batch[scale_key],
            absolute_contact_xy_mm=batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None,
            world_origin_xy_mm=batch["world_origin_xy_mm"],
            valid_mask=batch["seq_valid_mask"],
        )
        state = build_state_embedding(outputs)
        return outputs, state
    if model_type == "dwl_tr":
        outputs = model.forward_single(
            batch[obs_key],
            batch[coord_key],
            batch[scale_key],
            valid_mask=batch["seq_valid_mask"],
            absolute_contact_xy_mm=batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None,
            world_origin_xy_mm=batch["world_origin_xy_mm"],
        )
        return outputs, outputs["state_embedding"]
    if model_type == "no_projector_tr":
        outputs = model.forward_single(
            batch[obs_key],
            batch[coord_key],
            batch[scale_key],
            valid_mask=batch["seq_valid_mask"],
        )
        return outputs, outputs["state_embedding"]
    raise ValueError(f"Unsupported model type: {model_type}")


def run_cross_band_eval(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    model_type: str,
    protocol: str,
    sequence_length: int | None,
    limit: int = 0,
) -> dict[str, Any]:
    pair_specs, matching_stats = _build_cross_band_pair_specs(cfg, protocol=protocol, limit=limit)
    device = default_device(cfg)
    model = _load_model_for_eval(cfg, model_type=model_type, checkpoint_path=checkpoint_path, device=device)
    loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    records: list[dict[str, Any]] = []
    eval_cfg = cfg.get("eval", {})
    log_interval_seconds = max(float(eval_cfg.get("timestamp_log_interval_seconds", 600.0)), 1.0)
    started_at = time.time()
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Cross-band eval {model_type} {protocol}", total=len(loader), leave=False)
        _progress_write(
            iterator,
            f"cross_band_eval started model_type={model_type} protocol={protocol} checkpoint={checkpoint_path} "
            f"matched_pairs={len(pair_specs)} total_batches={len(loader)} limit={limit if limit > 0 else 'none'}",
        )
        next_log_time = started_at + log_interval_seconds
        for batch_idx, batch in enumerate(iterator, start=1):
            batch = _move_eval_batch(batch, device)
            source_out, source_state = _forward_single(model_type, model, batch, branch="source")
            target_out, target_state = _forward_single(model_type, model, batch, branch="target")
            occ = (batch["source_obs"][:, -1, 0] - batch["source_obs"][:, 0, 0]).abs().mean(dim=(1, 2))
            pred_source = torch.stack([source_out["pred_x_norm"], source_out["pred_y_norm"], source_out["pred_depth_mm"]], dim=1)
            pred_target = torch.stack([target_out["pred_x_norm"], target_out["pred_y_norm"], target_out["pred_depth_mm"]], dim=1)
            target_gt = torch.stack([batch["x_norm"], batch["y_norm"], batch["depth_mm"]], dim=1)
            pos = torch.nn.functional.cosine_similarity(source_state, target_state, dim=1)
            pair_ids = batch["pair_id"]
            for sample_idx in range(pred_source.shape[0]):
                records.append(
                    {
                        "event_key": str(pair_ids[sample_idx]),
                        "pred_source": pred_source[sample_idx].cpu().tolist(),
                        "pred_target": pred_target[sample_idx].cpu().tolist(),
                        "target": target_gt[sample_idx].cpu().tolist(),
                        "occupancy": float(occ[sample_idx].item()),
                        "positive_score": float(pos[sample_idx].item()),
                        "state_source": source_state[sample_idx].cpu().tolist(),
                        "state_target": target_state[sample_idx].cpu().tolist(),
                        "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
                        "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
                        "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
                        "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
                        "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
                        "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
                        "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
                    }
                )
            iterator.set_postfix({"records": len(records)})
            now = time.time()
            if now >= next_log_time:
                _progress_write(
                    iterator,
                    f"cross_band_eval heartbeat model_type={model_type} protocol={protocol} "
                    f"batch={batch_idx}/{len(loader)} records={len(records)} elapsed={_format_elapsed(now - started_at)}",
                )
                while next_log_time <= now:
                    next_log_time += log_interval_seconds
    result = _summarize_direct_records(records, protocol=protocol, cfg=cfg)
    result["matching_stats"] = matching_stats
    result["model_type"] = model_type
    _progress_write(
        iterator,
        f"cross_band_eval finished model_type={model_type} protocol={protocol} records={len(records)} "
        f"elapsed={_format_elapsed(time.time() - started_at)}",
    )
    return result


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args)
    result = run_cross_band_eval(
        cfg,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        protocol=args.protocol,
        sequence_length=args.sequence_length,
        limit=args.limit,
    )
    output = args.output or f"sccwm/eval_outputs/{args.model_type}_{args.protocol}.json"
    save_eval_result(result, output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
