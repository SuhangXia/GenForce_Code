#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dd_usa_adapter import DirectDriveUniversalScaleAdapter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST_DATASET = Path('/home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix')
DEFAULT_DOCKER_DATASET = Path('/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix')
DEFAULT_HOST_CHECKPOINT_ROOT = Path('/home/suhang/datasets/checkpoints')
DEFAULT_DOCKER_CHECKPOINT_ROOT = Path('/datasets/checkpoints')
DEFAULT_CHECKPOINT_DIR = (
    DEFAULT_DOCKER_CHECKPOINT_ROOT / 'dd_usa_pretrain'
    if DEFAULT_DOCKER_CHECKPOINT_ROOT.parent.exists()
    else DEFAULT_HOST_CHECKPOINT_ROOT / 'dd_usa_pretrain'
)

TRAIN_INDENTERS = (
    'cone',
    'cylinder',
    'cylinder_sh',
    'cylinder_si',
    'dotin',
    'dots',
    'hemisphere',
    'line',
    'moon',
    'prism',
    'random',
    'sphere',
)
VAL_INDENTERS = ('sphere_s', 'triangle')
TEST_INDENTERS = ('hexagon', 'pacman', 'torus', 'wave')

METRICS_CSV_COLUMNS = [
    'epoch',
    'global_step',
    'lr',
    'train_loss',
    'train_overlap_mse',
    'train_overlap_cos',
    'train_id_mse',
    'train_id_cos',
    'train_anchor',
    'train_unsup',
    'train_norm',
    'train_cycle',
    'train_support_mean',
    'train_teacher_valid_mean',
    'train_overlap_mask_mean',
    'train_alpha_anchor',
    'train_alpha_corrected',
    'train_source_scale',
    'train_target_scale',
    'val_loss',
    'val_overlap_mse',
    'val_overlap_cos',
    'val_id_mse',
    'val_id_cos',
    'val_anchor',
    'val_unsup',
    'val_norm',
    'val_cycle',
    'val_support_mean',
    'val_teacher_valid_mean',
    'val_overlap_mask_mean',
    'val_alpha_anchor',
    'val_alpha_corrected',
    'val_source_scale',
    'val_target_scale',
    'best_val_loss',
    'epoch_seconds',
    'elapsed_seconds',
    'eta_seconds',
]


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)
for noisy_logger_name in ('httpx', 'huggingface_hub', 'urllib3'):
    logging.getLogger(noisy_logger_name).setLevel(logging.ERROR)


def resolve_default_dataset() -> Path:
    if DEFAULT_DOCKER_DATASET.exists():
        return DEFAULT_DOCKER_DATASET
    return DEFAULT_HOST_DATASET


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_checkpoint_reference(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    root_candidate = (DEFAULT_CHECKPOINT_DIR.parent / path).resolve()
    legacy_candidate = (PROJECT_ROOT / path).resolve()
    if root_candidate.exists() or not legacy_candidate.exists():
        return root_candidate
    return legacy_candidate


def initialize_metrics_csv(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writerow({key: row.get(key, '') for key in METRICS_CSV_COLUMNS})


def truncate_metrics_csv(path: Path, keep_through_epoch: int) -> None:
    if not path.exists():
        initialize_metrics_csv(path, overwrite=False)
        return

    rows_to_keep: list[dict[str, str]] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_str = str(row.get('epoch', '')).strip()
            if not epoch_str:
                continue
            try:
                epoch = int(float(epoch_str))
            except ValueError:
                continue
            if epoch <= keep_through_epoch:
                rows_to_keep.append(row)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()
        for row in rows_to_keep:
            writer.writerow({key: row.get(key, '') for key in METRICS_CSV_COLUMNS})


def format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f'{hrs:d}h{mins:02d}m{secs:02d}s'
    if mins > 0:
        return f'{mins:02d}m{secs:02d}s'
    return f'{secs:02d}s'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_scale_key(scale_key: str) -> float:
    return float(str(scale_key).removeprefix('scale_').removesuffix('mm'))


def infer_square_hw(num_tokens: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f'Cannot infer square grid from token count {num_tokens}')
    return side, side


class DDUSAPretrainPairDataset(Dataset):
    """Build ordered source->target pairs for the same physical event across scales."""

    def __init__(
        self,
        dataset_root: str | Path,
        indenters: Iterable[str],
        scales_mm: Iterable[float] | None,
        mode: str,
        transform: transforms.Compose | None = None,
        pairs_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.indenters = frozenset(str(ind) for ind in indenters)
        self.scales_mm = None if scales_mm is None else tuple(sorted(float(scale) for scale in scales_mm))
        self.mode = str(mode)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = int(seed)
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._manifest = self._load_manifest()
        self._events = self._load_events()
        self._all_pairs = self._build_pairs()
        if not self._all_pairs:
            raise RuntimeError(
                f'No DD-USA pretraining pairs found for indenters={sorted(self.indenters)} scales={self.scales_mm}'
            )
        if self.pairs_per_epoch is None:
            self._len = len(self._all_pairs)
        elif self.mode == 'train':
            self._len = int(self.pairs_per_epoch)
        else:
            self._len = min(int(self.pairs_per_epoch), len(self._all_pairs))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._all_pairs[index % len(self._all_pairs)]

        source = pair['source']
        target = pair['target']
        source_image = Image.open(source['image_path']).convert('RGB')
        target_image = Image.open(target['image_path']).convert('RGB')

        return {
            'source_image': self.transform(source_image),
            'target_image': self.transform(target_image),
            'source_coords': self._load_coord_map(source['coord_path']),
            'target_coords': self._load_coord_map(target['coord_path']),
            'source_valid_mask': self._load_valid_mask(source['coord_path']),
            'target_valid_mask': self._load_valid_mask(target['coord_path']),
            'source_scale_mm': torch.tensor(float(source['scale_mm']), dtype=torch.float32),
            'target_scale_mm': torch.tensor(float(target['scale_mm']), dtype=torch.float32),
            'indenter_name': source['indenter_name'],
            'episode_id': source['episode_id'],
            'frame_name': source['frame_name'],
            'marker_name': source['marker_name'],
        }

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.dataset_root / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _scale_allowed(self, scale_mm: float) -> bool:
        if self.scales_mm is None:
            return True
        return any(np.isclose(scale_mm, scale) for scale in self.scales_mm)

    def _load_events(self) -> dict[tuple[int, str, str], dict[float, dict[str, Any]]]:
        image_index_path = self.dataset_root / 'image_index.csv'
        if image_index_path.exists():
            rows = self._load_rows_from_image_index(image_index_path)
        else:
            rows = self._load_rows_from_manifest(self._manifest)

        events: dict[tuple[int, str, str], dict[float, dict[str, Any]]] = {}
        for row in rows:
            event_key = (row['episode_id'], row['frame_name'], row['marker_name'])
            scale_mm = float(row['scale_mm'])
            events.setdefault(event_key, {})[scale_mm] = row
        return {key: value for key, value in events.items() if value}

    def _build_pairs(self) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for scale_map in self._events.values():
            available_scales = sorted(scale_map.keys())
            for source_scale in available_scales:
                for target_scale in available_scales:
                    pairs.append(
                        {
                            'source': scale_map[source_scale],
                            'target': scale_map[target_scale],
                        }
                    )
        return pairs

    def _load_rows_from_image_index(self, index_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(index_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                indenter_name = str(row.get('indenter_name', '')).strip()
                if indenter_name not in self.indenters:
                    continue
                scale_mm = float(str(row.get('scale_mm', '')).strip())
                if not self._scale_allowed(scale_mm):
                    continue
                episode_dir = (self.dataset_root / str(row.get('episode_dir', '')).strip()).resolve()
                rows.append(
                    {
                        'episode_id': int(row['episode_id']),
                        'frame_name': str(row.get('frame_name', '')).strip(),
                        'marker_name': str(row.get('marker_name', '')).strip(),
                        'indenter_name': indenter_name,
                        'scale_mm': scale_mm,
                        'image_path': self._resolve_existing_path(
                            row.get('image_abspath'),
                            row.get('image_relpath'),
                            [self.dataset_root],
                        ),
                        'coord_path': self._resolve_existing_path(
                            row.get('adapter_coord_map_abspath'),
                            row.get('adapter_coord_map_relpath'),
                            [episode_dir, self.dataset_root],
                        ),
                    }
                )
        return rows

    def _load_rows_from_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode_entry in manifest.get('episodes', []):
            if not isinstance(episode_entry, dict):
                continue
            episode_id = int(episode_entry.get('episode_id', 0))
            episode_dir_name = str(episode_entry.get('path', f'episode_{episode_id:06d}'))
            episode_dir = (self.dataset_root / episode_dir_name).resolve()
            metadata_path = episode_dir / 'metadata.json'
            if not metadata_path.exists():
                continue
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            indenter_name = str(metadata.get('indenter', episode_entry.get('indenter', ''))).strip()
            if indenter_name not in self.indenters:
                continue
            scales = metadata.get('scales', {})
            if not isinstance(scales, dict):
                continue
            for scale_key, scale_meta in scales.items():
                if not isinstance(scale_meta, dict):
                    continue
                scale_mm = float(scale_meta.get('scale_mm', parse_scale_key(scale_key)))
                if not self._scale_allowed(scale_mm):
                    continue
                coord_rel = str(scale_meta.get('adapter_coord_map', '')).strip()
                if not coord_rel:
                    continue
                coord_path = (episode_dir / coord_rel).resolve()
                if not coord_path.exists():
                    continue
                frames = scale_meta.get('frames', {})
                if not isinstance(frames, dict):
                    continue
                for frame_name, frame_meta in frames.items():
                    if not isinstance(frame_meta, dict):
                        continue
                    rendered_markers = frame_meta.get('rendered_markers', [])
                    if not isinstance(rendered_markers, list):
                        continue
                    for marker_name in rendered_markers:
                        image_path = (episode_dir / str(scale_key) / str(frame_name) / str(marker_name)).resolve()
                        if not image_path.exists():
                            continue
                        rows.append(
                            {
                                'episode_id': episode_id,
                                'frame_name': str(frame_name),
                                'marker_name': str(marker_name),
                                'indenter_name': indenter_name,
                                'scale_mm': scale_mm,
                                'image_path': image_path,
                                'coord_path': coord_path,
                            }
                        )
        return rows

    @staticmethod
    def _resolve_existing_path(abs_path: object, rel_path: object, base_dirs: Iterable[Path]) -> Path:
        abs_text = str(abs_path or '').strip()
        rel_text = str(rel_path or '').strip()
        candidates: list[Path] = []
        if abs_text:
            candidates.append(Path(abs_text).expanduser())
        if rel_text:
            for base_dir in base_dirs:
                candidates.append((base_dir / rel_text).resolve())
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f'Could not resolve path. abs={abs_text!r} rel={rel_text!r}')

    def _load_coord_map(self, path: Path) -> torch.Tensor:
        if path not in self._coord_cache:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                raise ValueError(f'Expected coord map shape (H,W,2), got {arr.shape} at {path}')
            self._coord_cache[path] = torch.from_numpy(arr).to(torch.float32)
        return self._coord_cache[path]

    def _load_valid_mask(self, path: Path) -> torch.Tensor:
        coord = self._load_coord_map(path)
        return torch.isfinite(coord).all(dim=-1)


class FrozenViTFeatureExtractor(nn.Module):
    """Frozen ViT-B/16 returning patch tokens on a 2D grid."""

    def __init__(self, model_name: str = 'vit_base_patch16_224', device: torch.device | str = 'cuda') -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        self.to(device)
        self.embed_dim = int(self.vit.embed_dim)
        grid_size = getattr(self.vit.patch_embed, 'grid_size', None)
        if grid_size is not None:
            self.patch_grid = (int(grid_size[0]), int(grid_size[1]))
        else:
            self.patch_grid = infer_square_hw(int(self.vit.patch_embed.num_patches))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens)
        for block in self.vit.blocks:
            tokens = block(tokens)
        tokens = self.vit.norm(tokens)
        tokens = tokens[:, 1:, :]
        b = tokens.shape[0]
        h, w = self.patch_grid
        return tokens.reshape(b, h, w, self.embed_dim).contiguous()


def flatten_feat(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.dim() == 4:
        b, h, w, d = tokens.shape
        return tokens.reshape(b, h * w, d)
    if tokens.dim() == 3:
        return tokens
    raise ValueError(f'tokens must be (B,H,W,D) or (B,N,D), got {tuple(tokens.shape)}')


def flatten_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4 and mask.shape[-1] == 1:
        return mask.reshape(mask.shape[0], -1).to(dtype=torch.bool)
    if mask.dim() == 3:
        return mask.reshape(mask.shape[0], -1).to(dtype=torch.bool)
    if mask.dim() == 2:
        return mask.to(dtype=torch.bool)
    raise ValueError(f'mask must be (B,H,W), (B,H,W,1), or (B,N), got {tuple(mask.shape)}')


def flatten_weight_map(weights: torch.Tensor) -> torch.Tensor:
    if weights.dim() == 4 and weights.shape[-1] == 1:
        return weights.reshape(weights.shape[0], -1)
    if weights.dim() == 4 and weights.shape[1] == 1:
        return weights.reshape(weights.shape[0], -1)
    if weights.dim() == 3:
        return weights.reshape(weights.shape[0], -1)
    if weights.dim() == 2:
        return weights
    raise ValueError(f'weights must be 2D/3D/4D, got {tuple(weights.shape)}')


def bool_mask_to_grid(mask: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    mask_flat = flatten_mask(mask)
    h, w = hw
    if mask_flat.shape[1] != h * w:
        raise ValueError(f'Cannot reshape mask with {mask_flat.shape[1]} tokens to grid {hw}')
    return mask_flat.view(mask_flat.shape[0], h, w)


# Full-FoV universal pretraining cannot align the entire adapted image to a smaller target latent.
# DD-USA outputs live in a normalized direct-drive interface space, so raw target latent tokens are
# not token-aligned for scale-mismatched pairs. We therefore warp the teacher latent to the interface
# before overlap supervision, and use identity/cycle/support regularization outside direct overlap.
def compute_overlap_mask(
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    source_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    source_coords_flat = flatten_feat(source_coords)
    target_coords_flat = flatten_feat(target_coords)
    source_valid_flat = flatten_mask(source_valid_mask)
    if target_valid_mask is None:
        target_valid_flat = torch.ones(
            target_coords_flat.shape[:2],
            device=target_coords_flat.device,
            dtype=torch.bool,
        )
    else:
        target_valid_flat = flatten_mask(target_valid_mask)

    valid_source = source_valid_flat.unsqueeze(-1)
    large_pos = torch.full_like(source_coords_flat, torch.finfo(source_coords_flat.dtype).max)
    large_neg = torch.full_like(source_coords_flat, torch.finfo(source_coords_flat.dtype).min)
    min_bounds = torch.where(valid_source, source_coords_flat, large_pos).min(dim=1, keepdim=True).values
    max_bounds = torch.where(valid_source, source_coords_flat, large_neg).max(dim=1, keepdim=True).values
    has_source = source_valid_flat.any(dim=1, keepdim=True)

    inside = (target_coords_flat >= (min_bounds - 1e-6)) & (target_coords_flat <= (max_bounds + 1e-6))
    overlap_flat = inside[..., 0] & inside[..., 1] & target_valid_flat & has_source
    if target_coords.dim() == 4:
        b, h, w, _ = target_coords.shape
        return overlap_flat.view(b, h, w)
    return overlap_flat


def build_interface_coord_map_mm(
    target_coords_mm: torch.Tensor,
    source_scale_mm: torch.Tensor,
    target_scale_mm: torch.Tensor,
) -> torch.Tensor:
    if target_coords_mm.shape[-1] != 2:
        raise ValueError(f'target_coords_mm must end with dim=2, got {tuple(target_coords_mm.shape)}')
    b = target_coords_mm.shape[0]
    source_scale = source_scale_mm.to(device=target_coords_mm.device, dtype=target_coords_mm.dtype).reshape(b, 1, 1)
    target_scale = target_scale_mm.to(device=target_coords_mm.device, dtype=target_coords_mm.dtype).reshape(b, 1, 1)
    interface_flat = flatten_feat(target_coords_mm) / (target_scale / 2.0).clamp_min(1e-6)
    interface_flat = interface_flat * (source_scale / 2.0).clamp_min(1e-6)
    if target_coords_mm.dim() == 4:
        _, h, w, _ = target_coords_mm.shape
        return interface_flat.view(b, h, w, 2)
    return interface_flat


def metric_knn_resample_feat(
    source_feat_flat: torch.Tensor,
    source_coords_mm: torch.Tensor,
    query_coords_mm: torch.Tensor,
    source_valid_mask: torch.Tensor,
    k: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_feat_tokens = flatten_feat(source_feat_flat)
    source_coords_tokens = flatten_feat(source_coords_mm)
    query_coords_tokens = flatten_feat(query_coords_mm)
    source_valid_tokens = flatten_mask(source_valid_mask)

    b, ns, d = source_feat_tokens.shape
    nq = query_coords_tokens.shape[1]
    if source_coords_tokens.shape[:2] != (b, ns):
        raise ValueError(
            f'source_coords_mm must align with source_feat_flat: {tuple(source_coords_tokens.shape[:2])} vs {(b, ns)}'
        )
    if source_valid_tokens.shape != (b, ns):
        raise ValueError(f'source_valid_mask must be {(b, ns)}, got {tuple(source_valid_tokens.shape)}')

    if ns == 0 or not bool(source_valid_tokens.any()):
        zero_feat = source_feat_tokens.new_zeros((b, nq, d))
        zero_mask = source_valid_tokens.new_zeros((b, nq))
        zero_conf = source_feat_tokens.new_zeros((b, nq))
    else:
        k_eff = max(1, min(int(k), ns))
        dist = torch.cdist(query_coords_tokens, source_coords_tokens, p=2)
        large_val = torch.full_like(dist, torch.finfo(dist.dtype).max / 1024.0)
        dist = torch.where(source_valid_tokens[:, None, :], dist, large_val)

        knn_dist, knn_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)
        feat_index = knn_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        gathered_feat = torch.gather(
            source_feat_tokens[:, None, :, :].expand(-1, nq, -1, -1),
            2,
            feat_index,
        )
        gathered_valid = torch.gather(
            source_valid_tokens[:, None, :].expand(-1, nq, -1),
            2,
            knn_idx,
        )

        weights_raw = gathered_valid.to(dtype=source_feat_tokens.dtype) / knn_dist.clamp_min(1e-6)
        weight_mass = weights_raw.sum(dim=-1, keepdim=True)
        weights = weights_raw / weight_mass.clamp_min(1e-6)
        zero_feat = (weights.unsqueeze(-1) * gathered_feat).sum(dim=-2)

        nearest_dist = knn_dist[..., 0]
        zero_conf = torch.where(
            weight_mass.squeeze(-1) > 0.0,
            torch.exp(-nearest_dist),
            source_feat_tokens.new_zeros((b, nq)),
        )
        zero_mask = (weight_mass.squeeze(-1) > 0.0) & torch.isfinite(nearest_dist)

    if query_coords_mm.dim() == 4:
        _, h, w, _ = query_coords_mm.shape
        return (
            zero_feat.view(b, h, w, d),
            zero_mask.view(b, h, w),
            zero_conf.view(b, h, w),
        )
    return zero_feat, zero_mask, zero_conf


def warp_target_feat_to_interface(
    target_feat: torch.Tensor,
    target_coords: torch.Tensor,
    target_valid_mask: torch.Tensor,
    interface_coord_map_mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_feat_interface, teacher_valid_mask_resample, teacher_confidence = metric_knn_resample_feat(
        source_feat_flat=target_feat,
        source_coords_mm=target_coords,
        query_coords_mm=interface_coord_map_mm,
        source_valid_mask=target_valid_mask,
        k=4,
    )
    geom_overlap_mask_interface = compute_overlap_mask(
        source_coords=target_coords,
        target_coords=interface_coord_map_mm,
        source_valid_mask=target_valid_mask,
        target_valid_mask=None,
    )
    teacher_valid_flat = flatten_mask(teacher_valid_mask_resample) & flatten_mask(geom_overlap_mask_interface)
    if interface_coord_map_mm.dim() == 4:
        b, h, w, _ = interface_coord_map_mm.shape
        teacher_valid_mask_interface = teacher_valid_flat.view(b, h, w)
    else:
        teacher_valid_mask_interface = teacher_valid_flat
    return teacher_feat_interface, teacher_valid_mask_interface, teacher_confidence


def masked_latent_alignment(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_flat = flatten_feat(pred)
    target_flat = flatten_feat(target)
    mask_flat = flatten_mask(mask)
    if pred_flat.shape != target_flat.shape:
        raise ValueError(f'pred/target shape mismatch: {tuple(pred_flat.shape)} vs {tuple(target_flat.shape)}')
    if mask_flat.shape != pred_flat.shape[:2]:
        raise ValueError(f'mask shape mismatch: expected {tuple(pred_flat.shape[:2])}, got {tuple(mask_flat.shape)}')
    if not bool(mask_flat.any()):
        zero = pred_flat.new_zeros(())
        return zero, zero
    pred_valid = pred_flat[mask_flat]
    target_valid = target_flat[mask_flat]
    mse = F.mse_loss(pred_valid, target_valid)
    pred_norm = F.normalize(pred_valid, p=2, dim=-1)
    target_norm = F.normalize(target_valid, p=2, dim=-1)
    cos_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
    return mse, cos_loss


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_flat = flatten_feat(pred)
    target_flat = flatten_feat(target)
    mask_flat = flatten_mask(mask)
    if not bool(mask_flat.any()):
        return pred_flat.new_zeros(())
    return F.mse_loss(pred_flat[mask_flat], target_flat[mask_flat])


def masked_norm_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_flat = flatten_feat(pred)
    target_flat = flatten_feat(target)
    mask_flat = flatten_mask(mask)
    if not bool(mask_flat.any()):
        return pred_flat.new_zeros(())
    pred_norm = torch.linalg.vector_norm(pred_flat[mask_flat], dim=-1)
    target_norm = torch.linalg.vector_norm(target_flat[mask_flat], dim=-1)
    return (pred_norm - target_norm).abs().mean()


def compute_pretrain_losses(
    *,
    adapter: DirectDriveUniversalScaleAdapter,
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    adapted_feat: torch.Tensor,
    aux: dict[str, torch.Tensor],
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    source_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    source_scale_mm: torch.Tensor,
    target_scale_mm: torch.Tensor,
    enable_cycle: bool,
) -> dict[str, torch.Tensor]:
    interface_coord_map_mm = build_interface_coord_map_mm(
        target_coords_mm=target_coords,
        source_scale_mm=source_scale_mm,
        target_scale_mm=target_scale_mm,
    )
    teacher_feat_interface, teacher_valid_mask_interface, _ = warp_target_feat_to_interface(
        target_feat=target_feat,
        target_coords=target_coords,
        target_valid_mask=target_valid_mask,
        interface_coord_map_mm=interface_coord_map_mm,
    )
    geom_overlap_mask_interface = flatten_mask(
        compute_overlap_mask(
            source_coords=target_coords,
            target_coords=interface_coord_map_mm,
            source_valid_mask=target_valid_mask,
            target_valid_mask=None,
        )
    )
    model_support_mask = flatten_mask(aux['support_valid_mask'])
    teacher_valid_mask_flat = flatten_mask(teacher_valid_mask_interface)
    overlap_mask_interface = geom_overlap_mask_interface & model_support_mask & teacher_valid_mask_flat

    overlap_mse, overlap_cos = masked_latent_alignment(
        adapted_feat,
        teacher_feat_interface,
        overlap_mask_interface,
    )

    source_scale_flat = source_scale_mm.reshape(-1)
    target_scale_flat = target_scale_mm.reshape(-1)
    same_scale = torch.isclose(source_scale_flat, target_scale_flat, atol=1e-6)
    if bool(same_scale.any()):
        id_mse, id_cos = masked_latent_alignment(
            adapted_feat[same_scale],
            source_feat[same_scale],
            source_valid_mask[same_scale],
        )
    else:
        zero = adapted_feat.new_zeros(())
        id_mse = zero
        id_cos = zero

    anchor_loss = masked_mse(adapted_feat, aux['anchor_feat'], overlap_mask_interface)

    support_flat = flatten_weight_map(aux['support_map']).to(dtype=adapted_feat.dtype)
    residual_flat = flatten_feat(aux['residual_feat'])
    residual_energy = residual_flat.square().mean(dim=-1)
    unsup_loss = ((1.0 - support_flat).clamp_min(0.0) * residual_energy).mean()

    norm_loss = masked_norm_loss(adapted_feat, teacher_feat_interface, overlap_mask_interface)

    if enable_cycle:
        cycle_source_valid = bool_mask_to_grid(overlap_mask_interface, target_coords.shape[1:3])
        cycle_feat = adapter(
            source_feat=adapted_feat,
            source_coord_map_mm=target_coords,
            source_scale_mm=target_scale_mm,
            target_scale_mm=source_scale_mm,
            target_coord_map_mm=source_coords,
            source_valid_mask=cycle_source_valid,
            return_aux=False,
        )
        cycle_loss = masked_mse(cycle_feat, source_feat, source_valid_mask)
    else:
        cycle_loss = adapted_feat.new_zeros(())

    teacher_valid_flat = flatten_mask(teacher_valid_mask_interface)
    overlap_mask_flat = flatten_mask(overlap_mask_interface)
    return {
        'overlap_mse': overlap_mse,
        'overlap_cos': overlap_cos,
        'id_mse': id_mse,
        'id_cos': id_cos,
        'anchor': anchor_loss,
        'unsup': unsup_loss,
        'norm': norm_loss,
        'cycle': cycle_loss,
        'support_mean': aux['support_mean'],
        'teacher_valid_mean': teacher_valid_flat.to(dtype=adapted_feat.dtype).mean(),
        'overlap_mask_mean': overlap_mask_flat.to(dtype=adapted_feat.dtype).mean(),
        'alpha_anchor': aux['alpha_anchor'],
        'alpha_corrected': aux['alpha_corrected'],
        'source_scale': source_scale_flat.mean(),
        'target_scale': target_scale_flat.mean(),
    }

def combine_total_loss(loss_terms: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    return (
        args.lambda_overlap_mse * loss_terms['overlap_mse']
        + args.lambda_overlap_cos * loss_terms['overlap_cos']
        + args.lambda_id_mse * loss_terms['id_mse']
        + args.lambda_id_cos * loss_terms['id_cos']
        + args.lambda_anchor * loss_terms['anchor']
        + args.lambda_unsup * loss_terms['unsup']
        + args.lambda_norm * loss_terms['norm']
        + args.lambda_cycle * loss_terms['cycle']
    )


def build_datasets(args: argparse.Namespace) -> tuple[DDUSAPretrainPairDataset, DDUSAPretrainPairDataset]:
    dataset_root = resolve_path(args.dataset)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_dataset = DDUSAPretrainPairDataset(
        dataset_root=dataset_root,
        indenters=TRAIN_INDENTERS,
        scales_mm=args.train_scales,
        mode='train',
        transform=train_transform,
        pairs_per_epoch=args.pairs_per_epoch,
        seed=args.seed,
    )
    val_dataset = DDUSAPretrainPairDataset(
        dataset_root=dataset_root,
        indenters=VAL_INDENTERS,
        scales_mm=args.eval_scales or args.train_scales,
        mode='val',
        transform=train_transform,
        pairs_per_epoch=args.val_pairs_per_epoch,
        seed=args.seed,
    )
    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def make_adapter() -> DirectDriveUniversalScaleAdapter:
    return DirectDriveUniversalScaleAdapter(
        embed_dim=768,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        coord_num_frequencies=4,
        coord_include_input=True,
        convffn_ratio=4.0,
        rel_bias_hidden_dim=64,
        use_final_norm=True,
        learnable_mix=True,
    )


def save_checkpoint(
    path: Path,
    *,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': int(epoch),
            'global_step': int(global_step),
            'elapsed_seconds': float(elapsed_seconds),
            'best_val_loss': float(best_val_loss),
            'model_state_dict': adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'args': vars(args),
        },
        path,
    )


def resolve_resume_checkpoint(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = resolve_checkpoint_reference(path_str)
    if path.is_dir():
        path = path / 'latest.pt'
    if not path.exists():
        raise FileNotFoundError(f'Resume checkpoint not found: {path}')
    return path


def load_resume_state(
    path: Path,
    *,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if 'python_random_state' in checkpoint:
        random.setstate(checkpoint['python_random_state'])
    if 'numpy_random_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_random_state'])
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and checkpoint.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])

    scheduler_state = checkpoint.get('scheduler_state_dict', {})
    global_step = checkpoint.get('global_step')
    if global_step is None:
        global_step = int(scheduler_state.get('last_epoch', -1)) + 1

    resume_epoch = int(checkpoint.get('epoch', 0))
    best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
    elapsed_seconds = float(checkpoint.get('elapsed_seconds', 0.0))
    saved_args = checkpoint.get('args', {})
    return {
        'resume_epoch': resume_epoch,
        'start_epoch': resume_epoch + 1,
        'global_step': int(global_step),
        'best_val_loss': best_val_loss,
        'elapsed_seconds': elapsed_seconds,
        'saved_args': saved_args,
    }


def run_epoch(
    *,
    adapter: DirectDriveUniversalScaleAdapter,
    backbone: FrozenViTFeatureExtractor,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    adapter.train(is_train)
    backbone.eval()

    metric_names = [
        'loss',
        'overlap_mse',
        'overlap_cos',
        'id_mse',
        'id_cos',
        'anchor',
        'unsup',
        'norm',
        'cycle',
        'support_mean',
        'teacher_valid_mean',
        'overlap_mask_mean',
        'alpha_anchor',
        'alpha_corrected',
        'source_scale',
        'target_scale',
    ]
    sums = {name: 0.0 for name in metric_names}
    num_batches = 0

    iterator = tqdm(data_loader, desc=desc, leave=False, disable=args.no_progress)
    for batch in iterator:
        source_image = batch['source_image'].to(device, non_blocking=True)
        target_image = batch['target_image'].to(device, non_blocking=True)
        source_coords = batch['source_coords'].to(device, non_blocking=True)
        target_coords = batch['target_coords'].to(device, non_blocking=True)
        source_valid_mask = batch['source_valid_mask'].to(device, non_blocking=True)
        target_valid_mask = batch['target_valid_mask'].to(device, non_blocking=True)
        source_scale_mm = batch['source_scale_mm'].to(device, non_blocking=True)
        target_scale_mm = batch['target_scale_mm'].to(device, non_blocking=True)

        with torch.no_grad():
            source_feat = backbone(source_image)
            target_feat = backbone(target_image)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        adapted_feat, aux = adapter(
            source_feat=source_feat,
            source_coord_map_mm=source_coords,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
            target_coord_map_mm=target_coords,
            source_valid_mask=source_valid_mask,
            return_aux=True,
        )

        loss_terms = compute_pretrain_losses(
            adapter=adapter,
            source_feat=source_feat,
            target_feat=target_feat,
            adapted_feat=adapted_feat,
            aux=aux,
            source_coords=source_coords,
            target_coords=target_coords,
            source_valid_mask=source_valid_mask,
            target_valid_mask=target_valid_mask,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
            enable_cycle=args.enable_cycle if is_train else args.enable_cycle_in_val,
        )
        total_loss = combine_total_loss(loss_terms, args)

        if is_train:
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_metrics = {
            'loss': float(total_loss.detach().item()),
            'overlap_mse': float(loss_terms['overlap_mse'].detach().item()),
            'overlap_cos': float(loss_terms['overlap_cos'].detach().item()),
            'id_mse': float(loss_terms['id_mse'].detach().item()),
            'id_cos': float(loss_terms['id_cos'].detach().item()),
            'anchor': float(loss_terms['anchor'].detach().item()),
            'unsup': float(loss_terms['unsup'].detach().item()),
            'norm': float(loss_terms['norm'].detach().item()),
            'cycle': float(loss_terms['cycle'].detach().item()),
            'support_mean': float(loss_terms['support_mean'].detach().item()),
            'teacher_valid_mean': float(loss_terms['teacher_valid_mean'].detach().item()),
            'overlap_mask_mean': float(loss_terms['overlap_mask_mean'].detach().item()),
            'alpha_anchor': float(loss_terms['alpha_anchor'].detach().item()),
            'alpha_corrected': float(loss_terms['alpha_corrected'].detach().item()),
            'source_scale': float(loss_terms['source_scale'].detach().item()),
            'target_scale': float(loss_terms['target_scale'].detach().item()),
        }
        for key, value in batch_metrics.items():
            sums[key] += value
        num_batches += 1

        if not args.no_progress:
            iterator.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                ov_mse=f"{batch_metrics['overlap_mse']:.4f}",
                t_valid=f"{batch_metrics['teacher_valid_mean']:.3f}",
                ov_mask=f"{batch_metrics['overlap_mask_mean']:.3f}",
            )

    if num_batches == 0:
        return {name: 0.0 for name in metric_names}
    return {name: sums[name] / num_batches for name in metric_names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Full-FoV DD-USA universal latent pretraining.')
    parser.add_argument('--dataset', type=str, default=str(resolve_default_dataset()))
    parser.add_argument('--checkpoint-dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument('--train-scales', type=float, nargs='*', default=None)
    parser.add_argument('--eval-scales', type=float, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pairs-per-epoch', type=int, default=10000)
    parser.add_argument('--val-pairs-per-epoch', type=int, default=2000)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda-overlap-mse', type=float, default=1.0)
    parser.add_argument('--lambda-overlap-cos', type=float, default=0.5)
    parser.add_argument('--lambda-id-mse', type=float, default=1.0)
    parser.add_argument('--lambda-id-cos', type=float, default=0.5)
    parser.add_argument('--lambda-anchor', type=float, default=0.1)
    parser.add_argument('--lambda-unsup', type=float, default=0.05)
    parser.add_argument('--lambda-norm', type=float, default=0.02)
    parser.add_argument('--lambda-cycle', type=float, default=0.1)
    parser.add_argument('--enable-cycle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--enable-cycle-in-val', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_every <= 0:
        raise ValueError(f'val_every must be > 0, got {args.val_every}')
    if args.val_pairs_per_epoch <= 0:
        raise ValueError(f'val_pairs_per_epoch must be > 0, got {args.val_pairs_per_epoch}')
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_root = resolve_path(args.dataset)
    checkpoint_dir = resolve_checkpoint_reference(args.checkpoint_dir)
    resume_path = resolve_resume_checkpoint(args.resume_from)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = checkpoint_dir / 'metrics.csv'

    train_dataset, val_dataset = build_datasets(args)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, args)

    backbone = FrozenViTFeatureExtractor(device=device)
    adapter = make_adapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = max(args.warmup_epochs * steps_per_epoch, 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if resume_path is not None:
        resume_state = load_resume_state(
            resume_path,
            adapter=adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(resume_state['start_epoch'])
        best_val_loss = float(resume_state['best_val_loss'])
        global_step = int(resume_state['global_step'])
        resumed_elapsed_seconds = float(resume_state['elapsed_seconds'])
        truncate_metrics_csv(metrics_csv_path, keep_through_epoch=int(resume_state['resume_epoch']))
    else:
        resume_state = None
        start_epoch = 1
        best_val_loss = float('inf')
        global_step = 0
        resumed_elapsed_seconds = 0.0
        initialize_metrics_csv(metrics_csv_path, overwrite=True)

    log.info(
        'Device: %s | train_pairs_total=%d | train_pairs_epoch=%d | val_pairs_total=%d | val_pairs_epoch=%d | batch_size=%d | val_every=%d',
        device,
        len(train_dataset._all_pairs),
        len(train_dataset),
        len(val_dataset._all_pairs),
        len(val_dataset),
        args.batch_size,
        args.val_every,
    )
    log.info(
        'Train indenters=%s | Val indenters=%s | Held-out unseen=%s',
        list(TRAIN_INDENTERS),
        list(VAL_INDENTERS),
        list(TEST_INDENTERS),
    )
    log.info('Checkpoint dir: %s', checkpoint_dir)
    log.info('Metrics CSV: %s', metrics_csv_path)

    if resume_state is not None:
        log.info(
            'Resumed from %s | saved_epoch=%d | next_epoch=%d | global_step=%d | best_val_loss=%.6f',
            resume_path,
            resume_state['resume_epoch'],
            start_epoch,
            global_step,
            best_val_loss,
        )
        saved_args = resume_state.get('saved_args', {})
        if saved_args:
            log.info(
                'Resume checkpoint was created with batch_size=%s | pairs_per_epoch=%s | epochs=%s',
                saved_args.get('batch_size'),
                saved_args.get('pairs_per_epoch'),
                saved_args.get('epochs'),
            )

    if start_epoch > args.epochs:
        log.info(
            'Nothing to do: resume checkpoint is already at epoch %d, requested epochs=%d.',
            start_epoch - 1,
            args.epochs,
        )
        return

    training_start = time.time() - resumed_elapsed_seconds

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            adapter=adapter,
            backbone=backbone,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            desc=f'Train {epoch}/{args.epochs}',
        )
        global_step += len(train_loader)

        should_run_val = val_loader is not None and (epoch == 1 or epoch == args.epochs or epoch % args.val_every == 0)
        if should_run_val:
            with torch.no_grad():
                val_metrics = run_epoch(
                    adapter=adapter,
                    backbone=backbone,
                    data_loader=val_loader,
                    optimizer=None,
                    scheduler=None,
                    device=device,
                    args=args,
                    desc=f'Val {epoch}/{args.epochs}',
                )
        else:
            val_metrics = {name: float('nan') for name in train_metrics}

        lr_now = optimizer.param_groups[0]['lr']
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - training_start
        eta_seconds = (total_elapsed / epoch) * (args.epochs - epoch) if epoch > 0 else 0.0

        save_checkpoint(
            checkpoint_dir / 'latest.pt',
            adapter=adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            elapsed_seconds=total_elapsed,
            best_val_loss=(
                best_val_loss
                if math.isfinite(best_val_loss)
                else (val_metrics['loss'] if should_run_val else float('inf'))
            ),
            args=args,
        )

        if should_run_val and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                checkpoint_dir / 'best.pt',
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                elapsed_seconds=total_elapsed,
                best_val_loss=best_val_loss,
                args=args,
            )
            log.info('Saved new best checkpoint to %s (val_loss=%.6f)', checkpoint_dir / 'best.pt', best_val_loss)
        elif not should_run_val:
            log.info('Skipping validation at epoch %03d/%d (val_every=%d).', epoch, args.epochs, args.val_every)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f'epoch_{epoch:04d}.pt',
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                elapsed_seconds=total_elapsed,
                best_val_loss=best_val_loss,
                args=args,
            )

        append_metrics_row(
            metrics_csv_path,
            {
                'epoch': epoch,
                'global_step': global_step,
                'lr': lr_now,
                'train_loss': train_metrics['loss'],
                'train_overlap_mse': train_metrics['overlap_mse'],
                'train_overlap_cos': train_metrics['overlap_cos'],
                'train_id_mse': train_metrics['id_mse'],
                'train_id_cos': train_metrics['id_cos'],
                'train_anchor': train_metrics['anchor'],
                'train_unsup': train_metrics['unsup'],
                'train_norm': train_metrics['norm'],
                'train_cycle': train_metrics['cycle'],
                'train_support_mean': train_metrics['support_mean'],
                'train_teacher_valid_mean': train_metrics['teacher_valid_mean'],
                'train_overlap_mask_mean': train_metrics['overlap_mask_mean'],
                'train_alpha_anchor': train_metrics['alpha_anchor'],
                'train_alpha_corrected': train_metrics['alpha_corrected'],
                'train_source_scale': train_metrics['source_scale'],
                'train_target_scale': train_metrics['target_scale'],
                'val_loss': val_metrics['loss'],
                'val_overlap_mse': val_metrics['overlap_mse'],
                'val_overlap_cos': val_metrics['overlap_cos'],
                'val_id_mse': val_metrics['id_mse'],
                'val_id_cos': val_metrics['id_cos'],
                'val_anchor': val_metrics['anchor'],
                'val_unsup': val_metrics['unsup'],
                'val_norm': val_metrics['norm'],
                'val_cycle': val_metrics['cycle'],
                'val_support_mean': val_metrics['support_mean'],
                'val_teacher_valid_mean': val_metrics['teacher_valid_mean'],
                'val_overlap_mask_mean': val_metrics['overlap_mask_mean'],
                'val_alpha_anchor': val_metrics['alpha_anchor'],
                'val_alpha_corrected': val_metrics['alpha_corrected'],
                'val_source_scale': val_metrics['source_scale'],
                'val_target_scale': val_metrics['target_scale'],
                'best_val_loss': best_val_loss,
                'epoch_seconds': epoch_elapsed,
                'elapsed_seconds': total_elapsed,
                'eta_seconds': eta_seconds,
            },
        )

        log.info(
            'Epoch %03d/%d | train_loss=%.6f overlap_mse=%.6f overlap_cos=%.6f id_mse=%.6f id_cos=%.6f '
            'anchor=%.6f unsup=%.6f norm=%.6f cycle=%.6f support=%.4f teacher_valid=%.4f overlap_mask=%.4f '
            'alpha_anchor=%.4f alpha_corrected=%.4f src_scale=%.2f tgt_scale=%.2f | '
            'val_loss=%.6f overlap_mse=%.6f overlap_cos=%.6f id_mse=%.6f id_cos=%.6f '
            'anchor=%.6f unsup=%.6f norm=%.6f cycle=%.6f support=%.4f teacher_valid=%.4f overlap_mask=%.4f '
            'alpha_anchor=%.4f alpha_corrected=%.4f src_scale=%.2f tgt_scale=%.2f | lr=%.2e | epoch_t=%s | elapsed=%s | eta=%s',
            epoch,
            args.epochs,
            train_metrics['loss'],
            train_metrics['overlap_mse'],
            train_metrics['overlap_cos'],
            train_metrics['id_mse'],
            train_metrics['id_cos'],
            train_metrics['anchor'],
            train_metrics['unsup'],
            train_metrics['norm'],
            train_metrics['cycle'],
            train_metrics['support_mean'],
            train_metrics['teacher_valid_mean'],
            train_metrics['overlap_mask_mean'],
            train_metrics['alpha_anchor'],
            train_metrics['alpha_corrected'],
            train_metrics['source_scale'],
            train_metrics['target_scale'],
            val_metrics['loss'],
            val_metrics['overlap_mse'],
            val_metrics['overlap_cos'],
            val_metrics['id_mse'],
            val_metrics['id_cos'],
            val_metrics['anchor'],
            val_metrics['unsup'],
            val_metrics['norm'],
            val_metrics['cycle'],
            val_metrics['support_mean'],
            val_metrics['teacher_valid_mean'],
            val_metrics['overlap_mask_mean'],
            val_metrics['alpha_anchor'],
            val_metrics['alpha_corrected'],
            val_metrics['source_scale'],
            val_metrics['target_scale'],
            lr_now,
            format_duration(epoch_elapsed),
            format_duration(total_elapsed),
            format_duration(eta_seconds),
        )

    log.info('Training finished. Best val_loss: %.6f', best_val_loss)
    log.info('Checkpoints: %s', checkpoint_dir)


if __name__ == '__main__':
    main()
