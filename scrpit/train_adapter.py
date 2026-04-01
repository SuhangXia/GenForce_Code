#!/usr/bin/env python3
"""
Training script for the Universal Scale Adapter (USA).

Reads datasets/usa_static_v1/manifest.json with physically isolated Train/Val/Test splits:
  - Train: 12 seen indenters
  - Val: 2 seen indenters
  - Test: 4 unseen indenters (pacman, wave, torus, hexagon) — zero-shot generalization

Pairing modes:
  - train: balanced random ordered scale pairs grouped by scale gap
  - val/test: target fixed to 15mm, source from [18,20,22,25]

Usage:
    python scrpit/train_adapter.py --dataset datasets/usa_static_v1
    python scrpit/train_adapter.py --sanity-check   # overfit 1 batch
    python scrpit/train_adapter.py --epochs 200 --lr 5e-5
"""

import argparse
import csv
import itertools
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import transforms
from tqdm import tqdm

from usa_adapter import UniversalScaleAdapter


def resolve_checkpoint_reference(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    root_candidate = (DEFAULT_CHECKPOINT_DIR.parent / path).resolve()
    legacy_candidate = (PROJECT_ROOT / path).resolve()
    if root_candidate.exists() or not legacy_candidate.exists():
        return root_candidate
    return legacy_candidate

# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "usa_static_v1"
DEFAULT_HOST_CHECKPOINT_ROOT = Path('/home/suhang/datasets/checkpoints')
DEFAULT_DOCKER_CHECKPOINT_ROOT = Path('/datasets/checkpoints')
DEFAULT_CHECKPOINT_DIR = (
    DEFAULT_DOCKER_CHECKPOINT_ROOT / 'usa_adapter'
    if DEFAULT_DOCKER_CHECKPOINT_ROOT.parent.exists()
    else DEFAULT_HOST_CHECKPOINT_ROOT / 'usa_adapter'
)

# Fixed indenter-level split for USA training/evaluation.
TRAIN_INDENTERS = frozenset(
    {
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
    }
)
VAL_INDENTERS = frozenset({"sphere_s", "triangle"})
TEST_INDENTERS = frozenset({"pacman", "wave", "torus", "hexagon"})

# Anchor scale for val/test (stable metrics)
ANCHOR_SCALE_MM = 15
SOURCE_SCALES_MM = [18, 20, 22, 25]  # scales used as source when target=15mm
NCE_TEMPERATURE = 0.07

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
for noisy_logger_name in ("httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(noisy_logger_name).setLevel(logging.ERROR)

METRICS_CSV_COLUMNS = [
    "epoch",
    "global_step",
    "lr",
    "train_loss",
    "train_mse",
    "train_cos_loss",
    "train_nce",
    "train_cos",
    "val_loss",
    "val_mse",
    "val_cos_loss",
    "val_nce",
    "val_cos",
    "best_val_loss",
    "best_val_cos",
    "epoch_seconds",
    "elapsed_seconds",
    "eta_seconds",
    "alpha_interp",
    "alpha_adapter",
    "train_valid_ratio",
    "val_valid_ratio",
    "train_interp_norm",
    "train_adapter_norm",
    "train_output_norm",
    "val_interp_norm",
    "val_adapter_norm",
    "val_output_norm",
]


# ---------------------------------------------------------------------------
# MultiscaleTactileDataset — datasets/usa_static_v1 format
# ---------------------------------------------------------------------------

class MultiscaleTactileDataset(Dataset):
    """
    Yields (img_context, coord_context, img_query, coord_query, scale_context_mm, scale_query_mm)
    for USA training.

    Uses real per-scale adapter_coord_map.npy exported by the generator.
    """

    def __init__(
        self,
        dataset_root: Path,
        episode_ids: list[int],
        episode_meta: dict[int, dict],
        scales_mm: list[int],
        mode: str,
        augment: bool = True,
        pairs_per_epoch: int | None = None,
        seed: int = 42,
        expected_token_count: int | None = None,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.episode_ids = episode_ids
        self.episode_meta = episode_meta
        self.scales_mm = sorted(scales_mm)
        self.mode = mode
        self.augment = augment
        self.seed = seed
        self.expected_token_count = expected_token_count

        self.scale_pairs = [(a, b) for a, b in itertools.permutations(self.scales_mm, 2)]
        self.scale_pairs_by_gap: dict[int, list[tuple[int, int]]] = {}
        for scale_context, scale_query in self.scale_pairs:
            gap = abs(int(scale_context) - int(scale_query))
            self.scale_pairs_by_gap.setdefault(gap, []).append((scale_context, scale_query))
        self.scale_gap_values = sorted(self.scale_pairs_by_gap.keys())

        self.source_scales = [s for s in self.scales_mm if s != ANCHOR_SCALE_MM]
        if not self.source_scales:
            self.source_scales = [s for s in SOURCE_SCALES_MM if s in self.scales_mm]

        self._len = pairs_per_epoch if pairs_per_epoch else len(episode_ids)
        self._rng = random.Random(seed)
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._warned_no_common: set[tuple[int, str, str]] = set()
        self._log_samples_done = 0
        self._max_sample_logs = 0
        self._coord_logs_done = 0
        self._max_coord_logs = 0

        img_size = 224
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return self._len

    @staticmethod
    def _parse_scale_key(scale_key: str) -> int:
        head = scale_key.removeprefix("scale_")
        value = head.removesuffix("mm")
        return int(value)

    @staticmethod
    def _should_log_from_worker() -> bool:
        info = get_worker_info()
        return info is None or info.id == 0

    def _sample_available_train_pair(self, scales: dict) -> tuple[str, str]:
        valid_keys = sorted([k for k in scales if k.startswith("scale_")])
        if len(valid_keys) < 2:
            raise RuntimeError(f"Episode has < 2 valid scales: {list(scales)}")

        valid_scales = [self._parse_scale_key(k) for k in valid_keys]
        valid_pairs_by_gap: dict[int, list[tuple[int, int]]] = {}
        for scale_context, scale_query in itertools.permutations(valid_scales, 2):
            gap = abs(int(scale_context) - int(scale_query))
            valid_pairs_by_gap.setdefault(gap, []).append((scale_context, scale_query))

        gap = self._rng.choice(sorted(valid_pairs_by_gap.keys()))
        scale_context, scale_query = self._rng.choice(valid_pairs_by_gap[gap])
        return f"scale_{scale_context}mm", f"scale_{scale_query}mm"

    def _choose_scale_pair(self, ep_id: int, scales: dict) -> tuple[str, str]:
        if self.mode == "train":
            gap = self._rng.choice(self.scale_gap_values)
            scale_context, scale_query = self._rng.choice(self.scale_pairs_by_gap[gap])
            key_context = f"scale_{scale_context}mm"
            key_query = f"scale_{scale_query}mm"
            if key_context in scales and key_query in scales and key_context != key_query:
                return key_context, key_query
            return self._sample_available_train_pair(scales)

        scale_query = ANCHOR_SCALE_MM
        scale_context = self._rng.choice(self.source_scales)
        if scale_context == scale_query:
            candidates = [s for s in self.source_scales if s != scale_query]
            if candidates:
                scale_context = self._rng.choice(candidates)
        key_context = f"scale_{scale_context}mm"
        key_query = f"scale_{scale_query}mm"

        if key_context in scales and key_query in scales and key_context != key_query:
            return key_context, key_query

        valid = sorted([k for k in scales if k.startswith("scale_")])
        if len(valid) < 2:
            raise RuntimeError(f"Episode {ep_id} has < 2 valid scales: {list(scales)}")
        return tuple(self._rng.sample(valid, 2))

    @staticmethod
    def _valid_frames(scale_meta: dict) -> dict[str, dict]:
        frames = scale_meta.get("frames", {})
        if not isinstance(frames, dict):
            return {}
        return {k: v for k, v in frames.items() if isinstance(v, dict)}

    def _pick_markers(
        self,
        ep_id: int,
        key_context: str,
        key_query: str,
        markers_context: list[str],
        markers_query: list[str],
        frame_context: str | None = None,
        frame_query: str | None = None,
    ) -> tuple[str, str]:
        if not markers_context or not markers_query:
            raise RuntimeError(
                f"Missing rendered_markers for episode {ep_id}: "
                f"{key_context}/{frame_context}={len(markers_context)} "
                f"{key_query}/{frame_query}={len(markers_query)}"
            )

        common = sorted(set(markers_context).intersection(markers_query))
        if common:
            marker_name = self._rng.choice(common)
            return marker_name, marker_name

        warn_key = (
            ep_id,
            key_context,
            key_query,
            frame_context if frame_context else "-",
            frame_query if frame_query else "-",
        )
        if warn_key not in self._warned_no_common:
            self._warned_no_common.add(warn_key)
            log.warning(
                "No common marker file for ep=%06d (%s/%s,%s/%s). Using fallback context=%s query=%s",
                ep_id,
                key_context,
                frame_context if frame_context else "-",
                key_query,
                frame_query if frame_query else "-",
                markers_context[0],
                markers_query[0],
            )
        return markers_context[0], markers_query[0]

    def _choose_frame_and_marker_pair(
        self,
        ep_id: int,
        key_context: str,
        key_query: str,
        meta_context: dict,
        meta_query: dict,
    ) -> tuple[str | None, str | None, str, str]:
        frames_context = self._valid_frames(meta_context)
        frames_query = self._valid_frames(meta_query)

        if frames_context and frames_query:
            names_context = sorted(frames_context.keys())
            names_query = sorted(frames_query.keys())
            common_frames = sorted(set(names_context).intersection(names_query))
            if common_frames:
                frame_name = self._rng.choice(common_frames)
                frame_context = frame_name
                frame_query = frame_name
            else:
                frame_context = self._rng.choice(names_context)
                frame_query = self._rng.choice(names_query)

            markers_context = sorted(frames_context[frame_context].get("rendered_markers", []))
            markers_query = sorted(frames_query[frame_query].get("rendered_markers", []))
            marker_context, marker_query = self._pick_markers(
                ep_id,
                key_context,
                key_query,
                markers_context,
                markers_query,
                frame_context=frame_context,
                frame_query=frame_query,
            )
            return frame_context, frame_query, marker_context, marker_query

        markers_context = sorted(meta_context.get("rendered_markers", []))
        markers_query = sorted(meta_query.get("rendered_markers", []))
        marker_context, marker_query = self._pick_markers(
            ep_id,
            key_context,
            key_query,
            markers_context,
            markers_query,
        )
        return None, None, marker_context, marker_query

    def _load_coord_map(self, episode_dir: Path, ep_id: int, scale_key: str, scale_meta: dict) -> torch.Tensor:
        rel = scale_meta.get("adapter_coord_map")
        if not rel:
            raise RuntimeError(f"Missing adapter_coord_map for episode {ep_id} {scale_key}")

        abs_path = episode_dir / rel
        if abs_path not in self._coord_cache:
            if not abs_path.exists():
                raise FileNotFoundError(f"adapter_coord_map not found: {abs_path}")

            arr = np.load(abs_path)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                raise RuntimeError(
                    f"Invalid adapter_coord_map shape at {abs_path}: expected (H,W,2), got {arr.shape}"
                )

            declared_shape = scale_meta.get("adapter_coord_map_shape")
            if declared_shape and list(arr.shape) != list(declared_shape):
                raise RuntimeError(
                    f"adapter_coord_map_shape mismatch at {abs_path}: "
                    f"declared={declared_shape} actual={list(arr.shape)}"
                )

            coords = torch.from_numpy(arr).to(torch.float32).reshape(-1, 2)
            if self.expected_token_count is not None and coords.shape[0] != self.expected_token_count:
                raise RuntimeError(
                    f"Token mismatch for {abs_path}: coord tokens={coords.shape[0]} "
                    f"but backbone expects {self.expected_token_count}. "
                    "Regenerate dataset with matching patch_grid or switch backbone tokenization."
                )

            self._coord_cache[abs_path] = coords
            if self._should_log_from_worker() and self._coord_logs_done < self._max_coord_logs:
                self._coord_logs_done += 1
                log.info(
                    "Loaded coord map: %s shape=%s flattened=%s",
                    abs_path,
                    tuple(arr.shape),
                    tuple(coords.shape),
                )

        return self._coord_cache[abs_path]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_id = self._rng.choice(self.episode_ids)
        meta = self.episode_meta[ep_id]
        scales = meta.get("scales", {})
        episode_path = meta["__episode_path"]
        episode_dir = self.dataset_root / episode_path

        key_context, key_query = self._choose_scale_pair(ep_id, scales)
        scale_meta_context = scales[key_context]
        scale_meta_query = scales[key_query]

        frame_context, frame_query, marker_context, marker_query = self._choose_frame_and_marker_pair(
            ep_id,
            key_context,
            key_query,
            scale_meta_context,
            scale_meta_query,
        )
        if frame_context is None:
            img_path_context = episode_dir / key_context / marker_context
            img_path_query = episode_dir / key_query / marker_query
        else:
            img_path_context = episode_dir / key_context / frame_context / marker_context
            img_path_query = episode_dir / key_query / frame_query / marker_query
        if not img_path_context.exists():
            raise FileNotFoundError(f"Missing context image: {img_path_context}")
        if not img_path_query.exists():
            raise FileNotFoundError(f"Missing query image: {img_path_query}")

        img_context = self.transform(Image.open(img_path_context).convert("RGB"))
        img_query = self.transform(Image.open(img_path_query).convert("RGB"))

        coord_context = self._load_coord_map(episode_dir, ep_id, key_context, scale_meta_context)
        coord_query = self._load_coord_map(episode_dir, ep_id, key_query, scale_meta_query)
        scale_context_mm = torch.tensor(float(self._parse_scale_key(key_context)), dtype=torch.float32)
        scale_query_mm = torch.tensor(float(self._parse_scale_key(key_query)), dtype=torch.float32)

        if self._should_log_from_worker() and self._log_samples_done < self._max_sample_logs:
            self._log_samples_done += 1
            log.info(
                "Sample ep=%06d pair=(%s -> %s) scales=(%.1f -> %.1f) frames=(%s, %s) "
                "markers=(%s, %s) coord_shapes=(%s, %s)",
                ep_id,
                key_context,
                key_query,
                float(scale_context_mm.item()),
                float(scale_query_mm.item()),
                frame_context if frame_context else "-",
                frame_query if frame_query else "-",
                marker_context,
                marker_query,
                tuple(coord_context.shape),
                tuple(coord_query.shape),
            )

        return (
            img_context,
            coord_context,
            img_query,
            coord_query,
            scale_context_mm,
            scale_query_mm,
        )


def load_manifest_and_split(
    manifest_path: Path,
    dataset_root: Path,
    train_indenters: frozenset[str],
    val_indenters: frozenset[str],
    test_indenters: frozenset[str],
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], dict[int, dict], dict]:
    """
    Load manifest, split by explicit indenter groups. Load full metadata (incl. scales) from each episode.
    Returns (train_ids, val_ids, test_ids, episode_meta, manifest_info).
    """
    with open(manifest_path) as f:
        data = json.load(f)

    manifest_info = {
        "patch_grid": data.get("patch_grid"),
        "coordinate_convention": data.get("coordinate_convention"),
    }

    episodes = data.get("episodes", [])
    episode_meta: dict[int, dict] = {}
    for ep in episodes:
        eid = ep["episode_id"]
        ep_path = ep.get("path", f"episode_{eid:06d}")
        meta_path = Path(dataset_root) / ep_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as mf:
                full_meta = json.load(mf)
            full_meta["__episode_path"] = ep_path
            episode_meta[eid] = full_meta
        else:
            fallback = dict(ep)
            fallback["__episode_path"] = ep_path
            episode_meta[eid] = fallback

    by_indenter: dict[str, list[int]] = {}
    for ep in episodes:
        eid = ep["episode_id"]
        ind = episode_meta[eid].get("indenter", ep.get("indenter", "unknown"))
        by_indenter.setdefault(ind, []).append(eid)

    if train_indenters & val_indenters:
        raise ValueError(f"Train/val indenter overlap: {sorted(train_indenters & val_indenters)}")
    if train_indenters & test_indenters:
        raise ValueError(f"Train/test indenter overlap: {sorted(train_indenters & test_indenters)}")
    if val_indenters & test_indenters:
        raise ValueError(f"Val/test indenter overlap: {sorted(val_indenters & test_indenters)}")

    assigned_indenters = set(train_indenters) | set(val_indenters) | set(test_indenters)
    actual_indenters = set(by_indenter.keys())
    missing_indenters = sorted(assigned_indenters - actual_indenters)
    unexpected_indenters = sorted(actual_indenters - assigned_indenters)
    if missing_indenters:
        raise RuntimeError(f"Manifest is missing expected indenters for split: {missing_indenters}")
    if unexpected_indenters:
        raise RuntimeError(f"Manifest contains indenters not assigned to a split: {unexpected_indenters}")

    train_ids = []
    val_ids = []
    test_ids = []

    rng = random.Random(seed)
    for ind, ids in sorted(by_indenter.items()):
        rng.shuffle(ids)
        if ind in train_indenters:
            train_ids.extend(ids)
        elif ind in val_indenters:
            val_ids.extend(ids)
        elif ind in test_indenters:
            test_ids.extend(ids)
        else:
            raise RuntimeError(f"Indenter {ind!r} is not assigned to any split")

    return train_ids, val_ids, test_ids, episode_meta, manifest_info


# ---------------------------------------------------------------------------
# ViT feature extractor
# ---------------------------------------------------------------------------

class FrozenViTFeatureExtractor(nn.Module):
    """Frozen ViT returning patch-level features (no CLS token)."""

    def __init__(self, model_name: str = "vit_base_patch16_224", device="cuda"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False
        self.to(device)
        self._device = device
        self.embed_dim = self.vit.embed_dim
        grid_size = getattr(self.vit.patch_embed, "grid_size", None)
        if grid_size is not None:
            self.patch_grid = (int(grid_size[0]), int(grid_size[1]))
            self.patch_token_count = int(self.patch_grid[0] * self.patch_grid[1])
        else:
            self.patch_token_count = int(getattr(self.vit.patch_embed, "num_patches"))
            self.patch_grid = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 196, 768)"""
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        return x[:, 1:, :]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity over (B, N, D)."""
    pred_flat = pred.reshape(-1, pred.size(-1))
    target_flat = target.reshape(-1, target.size(-1))
    pred_n = F.normalize(pred_flat, p=2, dim=1)
    target_n = F.normalize(target_flat, p=2, dim=1)
    return (pred_n * target_n).sum(dim=1).mean()


def feature_alignment_terms(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      mse_loss, cosine_loss(=1-cos_sim), cos_sim
    """
    mse = F.mse_loss(pred, target)
    cos_sim = cosine_similarity(pred, target)
    cos_loss = 1.0 - cos_sim
    return mse, cos_loss, cos_sim


def feature_alignment_terms_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred/target: (B, N, D)
    valid_mask:  (B, N), True means valid query token
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.dim() != 3:
        raise ValueError(f"pred and target must be (B,N,D), got {tuple(pred.shape)}")
    if valid_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"valid_mask shape mismatch: expected {tuple(pred.shape[:2])}, got {tuple(valid_mask.shape)}"
        )

    valid_mask = valid_mask.to(device=pred.device, dtype=torch.bool)
    if not valid_mask.any():
        zero = pred.new_zeros(())
        return zero, zero, zero

    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    mse = F.mse_loss(pred_valid, target_valid)

    pred_n = F.normalize(pred_valid, p=2, dim=-1)
    target_n = F.normalize(target_valid, p=2, dim=-1)
    cos_sim = (pred_n * target_n).sum(dim=-1).mean()
    cos_loss = 1.0 - cos_sim
    return mse, cos_loss, cos_sim


def masked_mean_pool(tokens: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    tokens:     (B, N, D)
    valid_mask: (B, N)
    returns:
      pooled:        (B, D)
      valid_example: (B,) bool
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be (B,N,D), got {tuple(tokens.shape)}")
    if valid_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"valid_mask shape mismatch: expected {tuple(tokens.shape[:2])}, got {tuple(valid_mask.shape)}"
        )

    valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
    weights = valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    counts = weights.sum(dim=1)
    valid_example = counts.squeeze(-1) > 0
    pooled = (tokens * weights).sum(dim=1) / counts.clamp_min(1.0)
    pooled = torch.where(valid_example.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, valid_example


def symmetric_info_nce(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = NCE_TEMPERATURE,
) -> tuple[torch.Tensor, bool]:
    pooled_pred, valid_pred = masked_mean_pool(pred_tokens, valid_mask)
    pooled_target, valid_target = masked_mean_pool(target_tokens, valid_mask)
    valid_examples = valid_pred & valid_target
    if int(valid_examples.sum().item()) < 2:
        return pred_tokens.new_zeros(()), False

    pred_sel = F.normalize(pooled_pred[valid_examples], p=2, dim=-1)
    target_sel = F.normalize(pooled_target[valid_examples], p=2, dim=-1)
    logits = pred_sel @ target_sel.transpose(0, 1)
    logits = logits / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.transpose(0, 1), labels)
    )
    return loss, True


def compute_bidirectional_alignment(
    *,
    adapter: UniversalScaleAdapter,
    feat_context: torch.Tensor,
    coord_context: torch.Tensor,
    scale_context_mm: torch.Tensor,
    feat_query: torch.Tensor,
    coord_query: torch.Tensor,
    scale_query_mm: torch.Tensor,
    lambda_mse: float,
    lambda_cos: float,
    lambda_nce: float,
) -> dict[str, torch.Tensor]:
    pred_query, query_valid_mask, debug_query = adapter(
        context_feat=feat_context,
        context_coord_map_mm=coord_context,
        query_coord_map_mm=coord_query,
        context_scale_mm=scale_context_mm,
        query_scale_mm=scale_query_mm,
        return_valid_mask=True,
        return_debug=True,
    )
    pred_context, context_valid_mask, debug_context = adapter(
        context_feat=feat_query,
        context_coord_map_mm=coord_query,
        query_coord_map_mm=coord_context,
        context_scale_mm=scale_query_mm,
        query_scale_mm=scale_context_mm,
        return_valid_mask=True,
        return_debug=True,
    )

    mse_context_to_query, cos_loss_context_to_query, cos_sim_context_to_query = (
        feature_alignment_terms_masked(pred_query, feat_query, query_valid_mask)
    )
    mse_query_to_context, cos_loss_query_to_context, cos_sim_query_to_context = (
        feature_alignment_terms_masked(pred_context, feat_context, context_valid_mask)
    )

    mse_component = mse_context_to_query + mse_query_to_context
    cos_component = cos_loss_context_to_query + cos_loss_query_to_context

    nce_terms: list[torch.Tensor] = []
    if lambda_nce > 0.0:
        nce_context_to_query, used_forward = symmetric_info_nce(
            pred_query, feat_query, query_valid_mask
        )
        if used_forward:
            nce_terms.append(nce_context_to_query)
        nce_query_to_context, used_reverse = symmetric_info_nce(
            pred_context, feat_context, context_valid_mask
        )
        if used_reverse:
            nce_terms.append(nce_query_to_context)

    if nce_terms:
        nce_component = torch.stack(nce_terms).mean()
    else:
        nce_component = feat_context.new_zeros(())

    loss = (
        lambda_mse * mse_component
        + lambda_cos * cos_component
        + lambda_nce * nce_component
    )
    cos_sim = 0.5 * (cos_sim_context_to_query + cos_sim_query_to_context)

    debug_keys = (
        "alpha_interp",
        "alpha_adapter",
        "valid_ratio",
        "interp_norm",
        "adapter_norm",
        "output_norm",
    )
    debug_metrics = {
        key: 0.5 * (debug_query[key] + debug_context[key])
        for key in debug_keys
    }

    return {
        "loss": loss,
        "mse_component": mse_component,
        "cos_component": cos_component,
        "nce_component": nce_component,
        "cos_sim": cos_sim,
        **debug_metrics,
    }


def evaluate_alignment(
    *,
    vit: FrozenViTFeatureExtractor,
    adapter: UniversalScaleAdapter,
    data_loader: DataLoader,
    device: torch.device,
    lambda_mse: float,
    lambda_cos: float,
    lambda_nce: float,
    desc: str,
    progress: bool,
) -> dict[str, float]:
    total_loss = 0.0
    total_mse = 0.0
    total_cos_component = 0.0
    total_nce_component = 0.0
    total_cos = 0.0
    total_alpha_interp = 0.0
    total_alpha_adapter = 0.0
    total_valid_ratio = 0.0
    total_interp_norm = 0.0
    total_adapter_norm = 0.0
    total_output_norm = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False, disable=not progress):
            (
                img_context,
                coord_context,
                img_query,
                coord_query,
                scale_context_mm,
                scale_query_mm,
            ) = [x.to(device) for x in batch]
            feat_context = vit(img_context)
            feat_query = vit(img_query)
            metrics = compute_bidirectional_alignment(
                adapter=adapter,
                feat_context=feat_context,
                coord_context=coord_context,
                scale_context_mm=scale_context_mm,
                feat_query=feat_query,
                coord_query=coord_query,
                scale_query_mm=scale_query_mm,
                lambda_mse=lambda_mse,
                lambda_cos=lambda_cos,
                lambda_nce=lambda_nce,
            )
            batch_size = img_context.size(0)
            total_loss += metrics["loss"].item() * batch_size
            total_mse += metrics["mse_component"].item() * batch_size
            total_cos_component += metrics["cos_component"].item() * batch_size
            total_nce_component += metrics["nce_component"].item() * batch_size
            total_cos += metrics["cos_sim"].item() * batch_size
            total_alpha_interp += metrics["alpha_interp"].item() * batch_size
            total_alpha_adapter += metrics["alpha_adapter"].item() * batch_size
            total_valid_ratio += metrics["valid_ratio"].item() * batch_size
            total_interp_norm += metrics["interp_norm"].item() * batch_size
            total_adapter_norm += metrics["adapter_norm"].item() * batch_size
            total_output_norm += metrics["output_norm"].item() * batch_size
            total_n += batch_size

    denom = max(total_n, 1)
    return {
        "loss": total_loss / denom,
        "mse_component": total_mse / denom,
        "cos_component": total_cos_component / denom,
        "nce_component": total_nce_component / denom,
        "cos_sim": total_cos / denom,
        "alpha_interp": total_alpha_interp / denom,
        "alpha_adapter": total_alpha_adapter / denom,
        "valid_ratio": total_valid_ratio / denom,
        "interp_norm": total_interp_norm / denom,
        "adapter_norm": total_adapter_norm / denom,
        "output_norm": total_output_norm / denom,
    }


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def initialize_metrics_csv(path: Path, overwrite: bool) -> None:
    mode = "w" if overwrite else "a"
    need_header = overwrite or (not path.exists())
    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        if need_header:
            writer.writeheader()


def append_metrics_row(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writerow({key: row.get(key, "") for key in METRICS_CSV_COLUMNS})


def make_adapter(embed_dim: int) -> UniversalScaleAdapter:
    return UniversalScaleAdapter(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        coord_num_frequencies=4,
        coord_scale_mm=10.0,
        interp_k=4,
        use_scale_token=True,
        use_final_norm=True,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if args.lambda_mse < 0 or args.lambda_cos < 0 or args.lambda_nce < 0:
        raise ValueError("lambda_mse, lambda_cos, and lambda_nce must be >= 0")
    if args.test_only and args.sanity_check:
        raise ValueError("--test-only cannot be used together with --sanity-check")

    dataset_root = Path(args.dataset)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    scales_mm = args.scales
    train_ids, val_ids, test_ids, episode_meta, manifest_info = load_manifest_and_split(
        manifest_path,
        dataset_root=dataset_root,
        train_indenters=TRAIN_INDENTERS,
        val_indenters=VAL_INDENTERS,
        test_indenters=TEST_INDENTERS,
        seed=args.seed,
    )

    log.info(
        "Split: train=%d val=%d test=%d | train indenters=%s | val indenters=%s | test indenters=%s",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        sorted(TRAIN_INDENTERS),
        sorted(VAL_INDENTERS),
        sorted(TEST_INDENTERS),
    )

    vit = FrozenViTFeatureExtractor(device=device)
    backbone_tokens = vit.patch_token_count
    dataset_patch_grid = manifest_info.get("patch_grid")
    coord_convention = manifest_info.get("coordinate_convention")
    log.info(
        "Coordinate map convention: %s",
        coord_convention if coord_convention else "(missing in manifest)",
    )
    if not dataset_patch_grid or len(dataset_patch_grid) != 2:
        raise RuntimeError(
            "Manifest is missing valid patch_grid=[H,W]. "
            "This dataset format requires patch_grid for strict backbone compatibility checks."
        )

    ds_h, ds_w = int(dataset_patch_grid[0]), int(dataset_patch_grid[1])
    dataset_tokens = ds_h * ds_w
    log.info(
        "Patch-grid check: dataset patch_grid=%dx%d (%d tokens), backbone tokens=%d%s",
        ds_h,
        ds_w,
        dataset_tokens,
        backbone_tokens,
        f" (grid={vit.patch_grid[0]}x{vit.patch_grid[1]})" if vit.patch_grid else "",
    )
    if dataset_tokens != backbone_tokens:
        raise RuntimeError(
            "Dataset patch_grid is incompatible with backbone patch token count: "
            f"dataset={ds_h}x{ds_w} ({dataset_tokens}) vs backbone_tokens={backbone_tokens}. "
            "Please regenerate dataset with matching patch_grid or change backbone."
        )

    def _log_example_pair():
        if not train_ids:
            return
        eid = train_ids[0]
        meta = episode_meta[eid]
        scales = sorted([k for k in meta.get("scales", {}).keys() if k.startswith("scale_")])
        if len(scales) < 2:
            return
        key_context, key_query = scales[0], scales[1]
        scale_meta_context = meta["scales"][key_context]
        scale_meta_query = meta["scales"][key_query]

        frames_context = (
            scale_meta_context.get("frames", {})
            if isinstance(scale_meta_context.get("frames", {}), dict)
            else {}
        )
        frames_query = (
            scale_meta_query.get("frames", {})
            if isinstance(scale_meta_query.get("frames", {}), dict)
            else {}
        )

        if frames_context and frames_query:
            frame_context = sorted(frames_context.keys())[0]
            frame_query = sorted(frames_query.keys())[0]
            markers_context = sorted(frames_context[frame_context].get("rendered_markers", []))
            markers_query = sorted(frames_query[frame_query].get("rendered_markers", []))
        else:
            frame_context = "-"
            frame_query = "-"
            markers_context = sorted(scale_meta_context.get("rendered_markers", []))
            markers_query = sorted(scale_meta_query.get("rendered_markers", []))

        common = sorted(set(markers_context).intersection(markers_query))
        chosen = common[0] if common else ((markers_context[0] if markers_context else "(none)"))
        log.info(
            "Example pair: ep=%06d path=%s pair=(%s -> %s) frame=(%s -> %s) marker=%s",
            eid,
            meta.get("__episode_path", f"episode_{eid:06d}"),
            key_context,
            key_query,
            frame_context,
            frame_query,
            chosen,
        )

    dataset_common_kwargs = dict(
        dataset_root=dataset_root,
        episode_meta=episode_meta,
        scales_mm=scales_mm,
        expected_token_count=backbone_tokens,
    )

    if args.test_only:
        ckpt_dir = resolve_checkpoint_reference(args.checkpoint_dir)
        adapter = make_adapter(vit.embed_dim).to(device)

        ckpt_path_raw = args.test_checkpoint or args.resume_from
        if ckpt_path_raw is None:
            default_candidates = [ckpt_dir / "best.pt", ckpt_dir / "best_loss.pt"]
            ckpt_path = next((candidate for candidate in default_candidates if candidate.exists()), None)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"Checkpoint not found for --test-only. Checked: {default_candidates[0]} and {default_candidates[1]}"
                )
        else:
            ckpt_path = resolve_checkpoint_reference(ckpt_path_raw)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found for --test-only: {ckpt_path}")

        log.info("TEST-ONLY mode: loading checkpoint %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" not in ckpt:
            raise RuntimeError(f"Invalid checkpoint (missing model_state_dict): {ckpt_path}")
        adapter.load_state_dict(ckpt["model_state_dict"])
        adapter.eval()

        if not test_ids:
            log.warning("TEST-ONLY mode: test split is empty. Nothing to evaluate.")
            return

        test_ds = MultiscaleTactileDataset(
            episode_ids=test_ids,
            mode="test",
            augment=False,
            pairs_per_epoch=min(500, len(test_ids) * 2),
            seed=args.seed + 2,
            **dataset_common_kwargs,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        metrics = evaluate_alignment(
            vit=vit,
            adapter=adapter,
            data_loader=test_loader,
            device=device,
            lambda_mse=args.lambda_mse,
            lambda_cos=args.lambda_cos,
            lambda_nce=args.lambda_nce,
            desc="Test",
            progress=args.progress,
        )
        if args.lambda_nce > 0:
            log.info(
                "TEST-ONLY [%s] (zero-shot indenters): loss=%.6f mse=%.6f cos_loss=%.6f nce=%.6f cos_sim=%.4f",
                ckpt_path.name,
                metrics["loss"],
                metrics["mse_component"],
                metrics["cos_component"],
                metrics["nce_component"],
                metrics["cos_sim"],
            )
        else:
            log.info(
                "TEST-ONLY [%s] (zero-shot indenters): loss=%.6f mse=%.6f cos_loss=%.6f cos_sim=%.4f",
                ckpt_path.name,
                metrics["loss"],
                metrics["mse_component"],
                metrics["cos_component"],
                metrics["cos_sim"],
            )
        return

    if args.sanity_check:
        log.info("SANITY CHECK: overfitting 1 batch for %d epochs", args.epochs)
        train_ds = MultiscaleTactileDataset(
            episode_ids=train_ids[: max(1, args.batch_size)],
            mode="train",
            augment=False,
            pairs_per_epoch=args.batch_size,
            seed=args.seed,
            **dataset_common_kwargs,
        )
        _log_example_pair()
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        single_batch = tuple(x.to(device) for x in next(iter(train_loader)))
        val_loader = None
    else:
        train_ds = MultiscaleTactileDataset(
            episode_ids=train_ids,
            mode="train",
            augment=True,
            pairs_per_epoch=args.pairs_per_epoch,
            seed=args.seed,
            **dataset_common_kwargs,
        )
        _log_example_pair()
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        if val_ids:
            val_ds = MultiscaleTactileDataset(
                episode_ids=val_ids,
                mode="val",
                augment=False,
                pairs_per_epoch=min(500, len(val_ids) * 2),
                seed=args.seed + 1,
                **dataset_common_kwargs,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
        else:
            val_loader = None

    adapter = make_adapter(vit.embed_dim).to(device)

    log.info(
        "USA parameters: %d",
        sum(p.numel() for p in adapter.parameters() if p.requires_grad),
    )
    log.info(
        "Loss weights: lambda_mse=%.4f lambda_cos=%.4f lambda_nce=%.4f",
        args.lambda_mse,
        args.lambda_cos,
        args.lambda_nce,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.sanity_check:
        batches_per_epoch = 1
    else:
        batches_per_epoch = max(1, args.pairs_per_epoch // args.batch_size)
    warmup_steps = args.warmup_epochs * batches_per_epoch
    total_steps = args.epochs * batches_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = resolve_checkpoint_reference(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoint dir: %s", ckpt_dir)
    metrics_csv_path = ckpt_dir / "metrics.csv"
    initialize_metrics_csv(metrics_csv_path, overwrite=(args.resume_from is None))
    log.info("Metrics CSV: %s", metrics_csv_path)

    best_val_loss = float("inf")
    best_val_cos = -1.0
    step = 0
    start_epoch = 1
    train_start_time = time.time()
    epoch_durations: list[float] = []

    def save_checkpoint(
        path: Path,
        *,
        epoch: int,
        loss: float,
        val_loss: float | None = None,
        val_cos: float | None = None,
    ) -> None:
        payload = {
            "epoch": int(epoch),
            "global_step": int(step),
            "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": float(loss),
            "best_val_loss": float(best_val_loss),
            "best_val_cos": float(best_val_cos),
        }
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        if val_cos is not None:
            payload["val_cos"] = float(val_cos)
        torch.save(payload, path)

    if args.resume_from is not None:
        resume_path = resolve_checkpoint_reference(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        if "model_state_dict" not in ckpt:
            raise RuntimeError(f"Invalid checkpoint (missing model_state_dict): {resume_path}")

        adapter.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            log.warning("Checkpoint missing optimizer_state_dict; optimizer state will be reinitialized.")

        step = int(ckpt.get("global_step", ckpt.get("step", 0)))
        ckpt_epoch = int(ckpt.get("epoch", 0))
        start_epoch = ckpt_epoch + 1

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif step > 0:
            sched_state = scheduler.state_dict()
            sched_state["last_epoch"] = step - 1
            scheduler.load_state_dict(sched_state)
            log.warning(
                "Checkpoint missing scheduler_state_dict; approximated scheduler resume from global_step=%d.",
                step,
            )

        if "best_val_loss" in ckpt:
            best_val_loss = float(ckpt["best_val_loss"])
        elif "val_loss" in ckpt:
            best_val_loss = float(ckpt["val_loss"])
        if "best_val_cos" in ckpt:
            best_val_cos = float(ckpt["best_val_cos"])
        elif "val_cos" in ckpt:
            best_val_cos = float(ckpt["val_cos"])

        log.info(
            "Resumed from %s | epoch=%d -> start_epoch=%d | global_step=%d | best_val_loss=%.6f best_val_cos=%.4f",
            resume_path,
            ckpt_epoch,
            start_epoch,
            step,
            best_val_loss,
            best_val_cos,
        )

    if start_epoch > args.epochs:
        log.warning(
            "Nothing to train: start_epoch=%d is greater than target epochs=%d.",
            start_epoch,
            args.epochs,
        )
        log.info("Checkpoints: %s", ckpt_dir)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        adapter.train()
        epoch_loss = 0.0
        epoch_mse_component = 0.0
        epoch_cos_component = 0.0
        epoch_nce_component = 0.0
        epoch_cos = 0.0
        epoch_alpha_interp = 0.0
        epoch_alpha_adapter = 0.0
        epoch_valid_ratio = 0.0
        epoch_interp_norm = 0.0
        epoch_adapter_norm = 0.0
        epoch_output_norm = 0.0
        n_batches = 0

        if args.sanity_check:
            batches = [single_batch]
        else:
            batches = train_loader

        pbar = tqdm(
            batches,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=True,
            disable=not args.progress,
        )

        for batch in pbar:
            if not args.sanity_check:
                batch = [x.to(device) for x in batch]
            (
                img_context,
                coord_context,
                img_query,
                coord_query,
                scale_context_mm,
                scale_query_mm,
            ) = batch

            with torch.no_grad():
                feat_context = vit(img_context)
                feat_query = vit(img_query)

            metrics = compute_bidirectional_alignment(
                adapter=adapter,
                feat_context=feat_context,
                coord_context=coord_context,
                scale_context_mm=scale_context_mm,
                feat_query=feat_query,
                coord_query=coord_query,
                scale_query_mm=scale_query_mm,
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
                lambda_nce=args.lambda_nce,
            )
            loss = metrics["loss"]
            mse_component = metrics["mse_component"]
            cos_component = metrics["cos_component"]
            nce_component = metrics["nce_component"]
            cos = metrics["cos_sim"].item()
            alpha_interp = metrics["alpha_interp"].item()
            alpha_adapter = metrics["alpha_adapter"].item()
            valid_ratio = metrics["valid_ratio"].item()
            interp_norm = metrics["interp_norm"].item()
            adapter_norm = metrics["adapter_norm"].item()
            output_norm = metrics["output_norm"].item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse_component += mse_component.item()
            epoch_cos_component += cos_component.item()
            epoch_nce_component += nce_component.item()
            epoch_cos += cos
            epoch_alpha_interp += alpha_interp
            epoch_alpha_adapter += alpha_adapter
            epoch_valid_ratio += valid_ratio
            epoch_interp_norm += interp_norm
            epoch_adapter_norm += adapter_norm
            epoch_output_norm += output_norm
            n_batches += 1
            step += 1

            if args.lambda_nce > 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{mse_component.item():.4f}",
                    cos_l=f"{cos_component.item():.4f}",
                    nce=f"{nce_component.item():.4f}",
                    cos=f"{cos:.4f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{mse_component.item():.4f}",
                    cos_l=f"{cos_component.item():.4f}",
                    cos=f"{cos:.4f}",
                )

            if not args.sanity_check:
                scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse_component = epoch_mse_component / max(n_batches, 1)
        avg_cos_component = epoch_cos_component / max(n_batches, 1)
        avg_nce_component = epoch_nce_component / max(n_batches, 1)
        avg_train_cos = epoch_cos / max(n_batches, 1)
        avg_alpha_interp = epoch_alpha_interp / max(n_batches, 1)
        avg_alpha_adapter = epoch_alpha_adapter / max(n_batches, 1)
        avg_valid_ratio = epoch_valid_ratio / max(n_batches, 1)
        avg_interp_norm = epoch_interp_norm / max(n_batches, 1)
        avg_adapter_norm = epoch_adapter_norm / max(n_batches, 1)
        avg_output_norm = epoch_output_norm / max(n_batches, 1)
        epoch_elapsed = time.time() - epoch_start_time
        epoch_durations.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_durations) / max(len(epoch_durations), 1)
        remaining_epochs = max(args.epochs - epoch, 0)
        eta_seconds = remaining_epochs * avg_epoch_time
        total_elapsed = time.time() - train_start_time
        lr_now = optimizer.param_groups[0]["lr"]
        time_suffix = (
            f"epoch_t={_format_duration(epoch_elapsed)} "
            f"elapsed={_format_duration(total_elapsed)} "
            f"eta={_format_duration(eta_seconds)}"
        )

        if val_loader and not args.sanity_check:
            adapter.eval()
            val_metrics = evaluate_alignment(
                vit=vit,
                adapter=adapter,
                data_loader=val_loader,
                device=device,
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
                lambda_nce=args.lambda_nce,
                desc="Val",
                progress=args.progress,
            )
            val_loss = val_metrics["loss"]
            val_mse_component = val_metrics["mse_component"]
            val_cos_component = val_metrics["cos_component"]
            val_nce_component = val_metrics["nce_component"]
            val_cos = val_metrics["cos_sim"]
            val_valid_ratio = val_metrics["valid_ratio"]
            val_interp_norm = val_metrics["interp_norm"]
            val_adapter_norm = val_metrics["adapter_norm"]
            val_output_norm = val_metrics["output_norm"]
            adapter.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "best_loss.pt",
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    val_cos=val_cos,
                )
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    val_cos=val_cos,
                )
            if val_cos > best_val_cos:
                best_val_cos = val_cos
                save_checkpoint(
                    ckpt_dir / "best_cos.pt",
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    val_cos=val_cos,
                )

            if args.lambda_nce > 0:
                log.info(
                    "Epoch %3d/%d  train_loss=%.6f train_mse=%.6f train_cos_loss=%.6f train_nce=%.6f train_cos=%.4f  "
                    "train_valid=%.4f train_interp=%.4f train_adapter=%.4f train_out=%.4f  "
                    "val_loss=%.6f val_mse=%.6f val_cos_loss=%.6f val_nce=%.6f val_cos=%.4f  "
                    "val_valid=%.4f val_interp=%.4f val_adapter=%.4f val_out=%.4f  "
                    "alpha_interp=%.4f alpha_adapter=%.4f  lr=%.2e  %s",
                    epoch,
                    args.epochs,
                    avg_loss,
                    avg_mse_component,
                    avg_cos_component,
                    avg_nce_component,
                    avg_train_cos,
                    avg_valid_ratio,
                    avg_interp_norm,
                    avg_adapter_norm,
                    avg_output_norm,
                    val_loss,
                    val_mse_component,
                    val_cos_component,
                    val_nce_component,
                    val_cos,
                    val_valid_ratio,
                    val_interp_norm,
                    val_adapter_norm,
                    val_output_norm,
                    avg_alpha_interp,
                    avg_alpha_adapter,
                    lr_now,
                    time_suffix,
                )
            else:
                log.info(
                    "Epoch %3d/%d  train_loss=%.6f train_mse=%.6f train_cos_loss=%.6f train_cos=%.4f  "
                    "train_valid=%.4f train_interp=%.4f train_adapter=%.4f train_out=%.4f  "
                    "val_loss=%.6f val_mse=%.6f val_cos_loss=%.6f val_cos=%.4f  "
                    "val_valid=%.4f val_interp=%.4f val_adapter=%.4f val_out=%.4f  "
                    "alpha_interp=%.4f alpha_adapter=%.4f  lr=%.2e  %s",
                    epoch,
                    args.epochs,
                    avg_loss,
                    avg_mse_component,
                    avg_cos_component,
                    avg_train_cos,
                    avg_valid_ratio,
                    avg_interp_norm,
                    avg_adapter_norm,
                    avg_output_norm,
                    val_loss,
                    val_mse_component,
                    val_cos_component,
                    val_cos,
                    val_valid_ratio,
                    val_interp_norm,
                    val_adapter_norm,
                    val_output_norm,
                    avg_alpha_interp,
                    avg_alpha_adapter,
                    lr_now,
                    time_suffix,
                )
            append_metrics_row(
                metrics_csv_path,
                {
                    "epoch": int(epoch),
                    "global_step": int(step),
                    "lr": float(lr_now),
                    "train_loss": float(avg_loss),
                    "train_mse": float(avg_mse_component),
                    "train_cos_loss": float(avg_cos_component),
                    "train_nce": float(avg_nce_component),
                    "train_cos": float(avg_train_cos),
                    "val_loss": float(val_loss),
                    "val_mse": float(val_mse_component),
                    "val_cos_loss": float(val_cos_component),
                    "val_nce": float(val_nce_component),
                    "val_cos": float(val_cos),
                    "best_val_loss": float(best_val_loss),
                    "best_val_cos": float(best_val_cos),
                    "epoch_seconds": float(epoch_elapsed),
                    "elapsed_seconds": float(total_elapsed),
                    "eta_seconds": float(eta_seconds),
                    "alpha_interp": float(avg_alpha_interp),
                    "alpha_adapter": float(avg_alpha_adapter),
                    "train_valid_ratio": float(avg_valid_ratio),
                    "val_valid_ratio": float(val_valid_ratio),
                    "train_interp_norm": float(avg_interp_norm),
                    "train_adapter_norm": float(avg_adapter_norm),
                    "train_output_norm": float(avg_output_norm),
                    "val_interp_norm": float(val_interp_norm),
                    "val_adapter_norm": float(val_adapter_norm),
                    "val_output_norm": float(val_output_norm),
                },
            )
        else:
            if args.lambda_nce > 0:
                log.info(
                    "Epoch %3d/%d  loss=%.6f mse=%.6f cos_loss=%.6f nce=%.6f cos=%.4f  "
                    "train_valid=%.4f train_interp=%.4f train_adapter=%.4f train_out=%.4f  "
                    "alpha_interp=%.4f alpha_adapter=%.4f  lr=%.2e  %s",
                    epoch,
                    args.epochs,
                    avg_loss,
                    avg_mse_component,
                    avg_cos_component,
                    avg_nce_component,
                    avg_train_cos,
                    avg_valid_ratio,
                    avg_interp_norm,
                    avg_adapter_norm,
                    avg_output_norm,
                    avg_alpha_interp,
                    avg_alpha_adapter,
                    lr_now,
                    time_suffix,
                )
            else:
                log.info(
                    "Epoch %3d/%d  loss=%.6f mse=%.6f cos_loss=%.6f cos=%.4f  "
                    "train_valid=%.4f train_interp=%.4f train_adapter=%.4f train_out=%.4f  "
                    "alpha_interp=%.4f alpha_adapter=%.4f  lr=%.2e  %s",
                    epoch,
                    args.epochs,
                    avg_loss,
                    avg_mse_component,
                    avg_cos_component,
                    avg_train_cos,
                    avg_valid_ratio,
                    avg_interp_norm,
                    avg_adapter_norm,
                    avg_output_norm,
                    avg_alpha_interp,
                    avg_alpha_adapter,
                    lr_now,
                    time_suffix,
                )

            if args.sanity_check and avg_loss < 0.001:
                log.info("Sanity check PASSED: loss -> 0")
                break

            if avg_loss < best_val_loss and args.sanity_check:
                best_val_loss = avg_loss
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    loss=avg_loss,
                )
            append_metrics_row(
                metrics_csv_path,
                {
                    "epoch": int(epoch),
                    "global_step": int(step),
                    "lr": float(lr_now),
                    "train_loss": float(avg_loss),
                    "train_mse": float(avg_mse_component),
                    "train_cos_loss": float(avg_cos_component),
                    "train_nce": float(avg_nce_component),
                    "train_cos": float(avg_train_cos),
                    "best_val_loss": float(best_val_loss),
                    "best_val_cos": float(best_val_cos),
                    "epoch_seconds": float(epoch_elapsed),
                    "elapsed_seconds": float(total_elapsed),
                    "eta_seconds": float(eta_seconds),
                    "alpha_interp": float(avg_alpha_interp),
                    "alpha_adapter": float(avg_alpha_adapter),
                    "train_valid_ratio": float(avg_valid_ratio),
                    "train_interp_norm": float(avg_interp_norm),
                    "train_adapter_norm": float(avg_adapter_norm),
                    "train_output_norm": float(avg_output_norm),
                },
            )

        if epoch % args.save_every == 0 and not args.sanity_check:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                loss=avg_loss,
            )

    log.info(
        "Training finished. Best val_loss: %.6f  Best val_cos: %.4f",
        best_val_loss,
        best_val_cos,
    )
    log.info("Checkpoints: %s", ckpt_dir)

    if args.eval_test and test_ids:
        test_ds = MultiscaleTactileDataset(
            episode_ids=test_ids,
            mode="test",
            augment=False,
            pairs_per_epoch=min(500, len(test_ids) * 2),
            seed=args.seed + 2,
            **dataset_common_kwargs,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        checkpoint_specs = [
            ("best_loss.pt", ckpt_dir / "best_loss.pt"),
            ("best_cos.pt", ckpt_dir / "best_cos.pt"),
        ]
        evaluated_any = False
        for checkpoint_label, ckpt_path in checkpoint_specs:
            if not ckpt_path.exists():
                continue
            evaluated_any = True
            log.info(
                "Evaluating on TEST set with %s (unseen indenters: %s)",
                checkpoint_label,
                sorted(TEST_INDENTERS),
            )
            ckpt = torch.load(ckpt_path, map_location=device)
            adapter.load_state_dict(ckpt["model_state_dict"])
            adapter.eval()
            test_metrics = evaluate_alignment(
                vit=vit,
                adapter=adapter,
                data_loader=test_loader,
                device=device,
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
                lambda_nce=args.lambda_nce,
                desc=f"Test[{checkpoint_label}]",
                progress=args.progress,
            )
            if args.lambda_nce > 0:
                log.info(
                    "TEST [%s] (zero-shot indenters): loss=%.6f mse=%.6f cos_loss=%.6f nce=%.6f cos_sim=%.4f",
                    checkpoint_label,
                    test_metrics["loss"],
                    test_metrics["mse_component"],
                    test_metrics["cos_component"],
                    test_metrics["nce_component"],
                    test_metrics["cos_sim"],
                )
            else:
                log.info(
                    "TEST [%s] (zero-shot indenters): loss=%.6f mse=%.6f cos_loss=%.6f cos_sim=%.4f",
                    checkpoint_label,
                    test_metrics["loss"],
                    test_metrics["mse_component"],
                    test_metrics["cos_component"],
                    test_metrics["cos_sim"],
                )
        if not evaluated_any:
            log.warning(
                "No best checkpoint found for --eval-test. Expected %s or %s.",
                ckpt_dir / "best_loss.pt",
                ckpt_dir / "best_cos.pt",
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train USA adapter (datasets/usa_static_v1)")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                   help="Path to datasets/usa_static_v1")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory for training checkpoints (best.pt and periodic epoch checkpoints).",
    )
    p.add_argument("--scales", nargs="+", type=int, default=[15, 18, 20, 22, 25])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--lambda_mse", type=float, default=1.0,
                   help="Weight for MSE feature-matching term.")
    p.add_argument("--lambda_cos", type=float, default=0.5,
                   help="Weight for cosine feature-alignment term (1-cosine_similarity).")
    p.add_argument("--lambda_nce", type=float, default=0.0,
                   help="Weight for optional symmetric InfoNCE global-alignment term.")
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.85,
        help="Deprecated and ignored. Train/val/test are now fixed by explicit indenter lists in the script.",
    )
    p.add_argument("--pairs_per_epoch", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Checkpoint path to resume training from (restores model/optimizer/scheduler/state).")
    p.add_argument("--test-only", action="store_true",
                   help="Skip training and run only held-out test evaluation from a checkpoint.")
    p.add_argument("--test-checkpoint", type=str, default=None,
                   help="Checkpoint path used by --test-only. Defaults to --resume-from, then best.pt, then best_loss.pt.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity-check", action="store_true",
                   help="Overfit 1 batch to verify model can reach loss->0")
    p.add_argument("--eval-test", action="store_true",
                   help="Evaluate on held-out test set (unseen indenters) after training")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.progress = not args.no_progress
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
