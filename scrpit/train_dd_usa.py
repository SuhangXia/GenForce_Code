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
DEFAULT_HOST_DATASET = Path('/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
DEFAULT_DOCKER_DATASET = Path('/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'dd_usa'
DEFAULT_CANONICAL_SCALE_MM = 20.0
NCEPS = 1e-6

TRAIN_INDENTERS = frozenset(
    {
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
    }
)
VAL_INDENTERS = frozenset({'sphere_s', 'triangle'})
TEST_INDENTERS = frozenset({'hexagon', 'pacman', 'torus', 'wave'})
SEEN_INDENTERS = TRAIN_INDENTERS | VAL_INDENTERS

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)
for noisy_logger_name in ('httpx', 'huggingface_hub', 'urllib3'):
    logging.getLogger(noisy_logger_name).setLevel(logging.ERROR)

METRICS_CSV_COLUMNS = [
    'epoch',
    'global_step',
    'lr',
    'train_loss',
    'train_task_loss',
    'train_x_loss',
    'train_y_loss',
    'train_depth_loss',
    'train_id_loss',
    'train_anchor_loss',
    'train_unsup_loss',
    'train_x_mae',
    'train_y_mae',
    'train_depth_mae',
    'train_support_mean',
    'train_alpha_anchor',
    'train_alpha_corrected',
    'train_gamma_mean',
    'train_beta_mean',
    'val_loss',
    'val_task_loss',
    'val_x_loss',
    'val_y_loss',
    'val_depth_loss',
    'val_id_loss',
    'val_anchor_loss',
    'val_unsup_loss',
    'val_x_mae',
    'val_y_mae',
    'val_depth_mae',
    'val_support_mean',
    'val_alpha_anchor',
    'val_alpha_corrected',
    'val_gamma_mean',
    'val_beta_mean',
    'best_val_task_loss',
    'epoch_seconds',
    'elapsed_seconds',
    'eta_seconds',
]


def resolve_default_dataset() -> Path:
    if DEFAULT_DOCKER_DATASET.exists():
        return DEFAULT_DOCKER_DATASET
    return DEFAULT_HOST_DATASET


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



def format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f'{hrs:d}h{mins:02d}m{secs:02d}s'
    if mins > 0:
        return f'{mins:02d}m{secs:02d}s'
    return f'{secs:02d}s'



def parse_scale_key(scale_key: str) -> float:
    head = str(scale_key).removeprefix('scale_')
    tail = head.removesuffix('mm')
    return float(tail)



def infer_square_hw(num_tokens: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f'Cannot infer square grid from token count {num_tokens}')
    return side, side



def resolve_checkpoint_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path



def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


class DDUSATactileDataset(Dataset):
    """Per-image static tactile dataset for task-aware DD-USA training."""

    def __init__(
        self,
        dataset_root: str | Path,
        indenters: Iterable[str],
        scales_mm: Iterable[float] | None,
        canonical_scale_mm: float,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.indenters = frozenset(str(ind) for ind in indenters)
        self.scales_mm = None if scales_mm is None else sorted(float(scale) for scale in scales_mm)
        self.canonical_scale_mm = float(canonical_scale_mm)
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
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._manifest = self._load_manifest()
        self.patch_grid = self._read_patch_grid(self._manifest)
        self.synthetic_target_coords = self._make_uniform_coord_map(self.patch_grid, self.canonical_scale_mm)
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f'No samples found for indenters={sorted(self.indenters)} scales={self.scales_mm} '
                f'in dataset {self.dataset_root}'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        source_coords = self._load_coord_map(sample['source_coord_path'])
        if sample['target_coord_path'] is not None:
            target_coords = self._load_coord_map(sample['target_coord_path'])
        else:
            target_coords = self.synthetic_target_coords.clone()
        target = torch.tensor(sample['target'], dtype=torch.float32)
        return {
            'image': image_tensor,
            'source_coords': source_coords,
            'target_coords': target_coords,
            'source_scale_mm': torch.tensor(float(sample['source_scale_mm']), dtype=torch.float32),
            'target_scale_mm': torch.tensor(self.canonical_scale_mm, dtype=torch.float32),
            'target': target,
            'indenter_name': sample['indenter_name'],
            'episode_dir': sample['episode_dir'],
            'frame_name': sample['frame_name'],
            'marker_name': sample['marker_name'],
        }

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.dataset_root / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _read_patch_grid(manifest: dict[str, Any]) -> tuple[int, int]:
        patch_grid = manifest.get('patch_grid')
        if isinstance(patch_grid, (list, tuple)) and len(patch_grid) >= 2:
            return int(patch_grid[0]), int(patch_grid[1])
        return 14, 14

    def _load_samples(self) -> list[dict[str, Any]]:
        image_index_path = self.dataset_root / 'image_index.csv'
        if image_index_path.exists():
            return self._load_samples_from_image_index(image_index_path)
        return self._load_samples_from_manifest(self._manifest)

    def _scale_allowed(self, scale_mm: float) -> bool:
        if self.scales_mm is None:
            return True
        return any(np.isclose(scale_mm, candidate) for candidate in self.scales_mm)

    def _append_sample(
        self,
        samples: list[dict[str, Any]],
        *,
        image_path: Path,
        episode_dir: Path,
        indenter_name: str,
        source_scale_mm: float,
        source_coord_path: Path,
        target_coord_path: Path | None,
        command_x_mm: float,
        command_y_mm: float,
        depth_mm: float,
        frame_name: str,
        marker_name: str,
    ) -> None:
        target_radius = self.canonical_scale_mm / 2.0
        if target_radius <= 0:
            raise ValueError(f'Invalid canonical_scale_mm={self.canonical_scale_mm}')
        samples.append(
            {
                'image_path': image_path,
                'episode_dir': episode_dir.name,
                'indenter_name': indenter_name,
                'source_scale_mm': float(source_scale_mm),
                'source_coord_path': source_coord_path,
                'target_coord_path': target_coord_path,
                # Canonical normalization keeps labels in the frozen old-head convention.
                'target': [
                    float(command_x_mm) / target_radius,
                    float(command_y_mm) / target_radius,
                    float(depth_mm),
                ],
                'frame_name': frame_name,
                'marker_name': marker_name,
            }
        )

    def _load_samples_from_image_index(self, index_path: Path) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with open(index_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                indenter_name = str(row.get('indenter_name', '')).strip()
                if indenter_name not in self.indenters:
                    continue
                source_scale_mm = self._parse_float(row.get('scale_mm'), 'scale_mm')
                if not self._scale_allowed(source_scale_mm):
                    continue
                episode_dir = (self.dataset_root / str(row.get('episode_dir', '')).strip()).resolve()
                if not episode_dir.exists():
                    raise FileNotFoundError(f'Episode directory not found: {episode_dir}')
                image_path = self._resolve_path(
                    row.get('image_abspath'),
                    row.get('image_relpath'),
                    'image',
                    base_dirs=[self.dataset_root],
                )
                source_coord_path = self._resolve_path(
                    row.get('adapter_coord_map_abspath'),
                    row.get('adapter_coord_map_relpath'),
                    'adapter_coord_map',
                    base_dirs=[episode_dir, self.dataset_root],
                )
                target_coord_path = episode_dir / f'scale_{int(self.canonical_scale_mm)}mm' / 'adapter_coord_map.npy'
                if not target_coord_path.exists():
                    target_coord_path = None
                self._append_sample(
                    samples,
                    image_path=image_path,
                    episode_dir=episode_dir,
                    indenter_name=indenter_name,
                    source_scale_mm=source_scale_mm,
                    source_coord_path=source_coord_path,
                    target_coord_path=target_coord_path,
                    command_x_mm=self._parse_float(row.get('command_x_mm'), 'command_x_mm'),
                    command_y_mm=self._parse_float(row.get('command_y_mm'), 'command_y_mm'),
                    depth_mm=self._parse_float(row.get('frame_actual_max_down_mm'), 'frame_actual_max_down_mm'),
                    frame_name=str(row.get('frame_name', '')).strip(),
                    marker_name=str(row.get('marker_name', '')).strip(),
                )
        return samples

    def _load_samples_from_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for episode_entry in manifest.get('episodes', []):
            if not isinstance(episode_entry, dict):
                continue
            episode_id = int(episode_entry.get('episode_id', 0))
            episode_dir_name = str(episode_entry.get('path', f'episode_{episode_id:06d}'))
            episode_dir = self.dataset_root / episode_dir_name
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
            contact = metadata.get('contact', {})
            for scale_key, scale_meta in scales.items():
                if not isinstance(scale_meta, dict):
                    continue
                source_scale_mm = float(scale_meta.get('scale_mm', parse_scale_key(scale_key)))
                if not self._scale_allowed(source_scale_mm):
                    continue
                source_coord_rel = str(scale_meta.get('adapter_coord_map', '')).strip()
                if not source_coord_rel:
                    continue
                source_coord_path = (episode_dir / source_coord_rel).resolve()
                if not source_coord_path.exists():
                    continue
                target_coord_path = episode_dir / f'scale_{int(self.canonical_scale_mm)}mm' / 'adapter_coord_map.npy'
                if not target_coord_path.exists():
                    target_coord_path = None
                command_x_mm = float(scale_meta.get('contact_x_mm', contact.get('x_mm', 0.0)))
                command_y_mm = float(scale_meta.get('contact_y_mm', contact.get('y_mm', 0.0)))
                frames = scale_meta.get('frames', {})
                if not isinstance(frames, dict):
                    continue
                for frame_name, frame_meta in frames.items():
                    if not isinstance(frame_meta, dict):
                        continue
                    rendered_markers = frame_meta.get('rendered_markers', [])
                    if not isinstance(rendered_markers, list):
                        continue
                    depth_mm = float(frame_meta.get('frame_actual_max_down_mm', 0.0))
                    for marker_name in rendered_markers:
                        image_path = episode_dir / str(scale_key) / str(frame_name) / str(marker_name)
                        if not image_path.exists():
                            continue
                        self._append_sample(
                            samples,
                            image_path=image_path.resolve(),
                            episode_dir=episode_dir,
                            indenter_name=indenter_name,
                            source_scale_mm=source_scale_mm,
                            source_coord_path=source_coord_path,
                            target_coord_path=target_coord_path,
                            command_x_mm=command_x_mm,
                            command_y_mm=command_y_mm,
                            depth_mm=depth_mm,
                            frame_name=str(frame_name),
                            marker_name=str(marker_name),
                        )
        return samples

    @staticmethod
    def _resolve_path(
        abs_path: object,
        rel_path: object,
        field_name: str,
        base_dirs: Iterable[Path],
    ) -> Path:
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
        raise FileNotFoundError(
            f'Could not resolve {field_name} path. abs={abs_text!r} rel={rel_text!r}'
        )

    def _load_coord_map(self, path: Path) -> torch.Tensor:
        if path not in self._coord_cache:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                raise ValueError(f'Expected coord map shape (H,W,2), got {arr.shape} at {path}')
            self._coord_cache[path] = torch.from_numpy(arr).to(torch.float32)
        return self._coord_cache[path]

    @staticmethod
    def _make_uniform_coord_map(grid_hw: tuple[int, int], scale_mm: float) -> torch.Tensor:
        h, w = grid_hw
        ys = torch.linspace(-scale_mm / 2.0, scale_mm / 2.0, steps=h, dtype=torch.float32)
        xs = torch.linspace(-scale_mm / 2.0, scale_mm / 2.0, steps=w, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx, gy], dim=-1)

    @staticmethod
    def _parse_float(value: object, field_name: str) -> float:
        text = str(value).strip()
        if not text:
            raise ValueError(f'Missing required numeric field: {field_name}')
        return float(text)


class FrozenViTFeatureExtractor(nn.Module):
    """Frozen ViT-B/16 returning patch tokens on a 2D grid."""

    def __init__(self, model_name: str = 'vit_base_patch16_224', device: torch.device | str = 'cuda') -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        freeze_module(self.vit)
        self.to(device)
        self.embed_dim = int(self.vit.embed_dim)
        grid_size = getattr(self.vit.patch_embed, 'grid_size', None)
        if grid_size is not None:
            self.patch_grid = (int(grid_size[0]), int(grid_size[1]))
        else:
            self.patch_grid = infer_square_hw(int(self.vit.patch_embed.num_patches))
        self.patch_token_count = int(self.patch_grid[0] * self.patch_grid[1])

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


class CanonicalRegressionHead(nn.Module):
    """Simple canonical head: support-aware token pooling followed by a small MLP."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, out_dim: int = 3) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.out_dim = int(out_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    @staticmethod
    def _flatten_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if tokens.dim() == 4:
            b, h, w, d = tokens.shape
            return tokens.reshape(b, h * w, d), (h, w)
        if tokens.dim() == 3:
            return tokens, None
        raise ValueError(f'tokens must be (B,H,W,D) or (B,N,D), got {tuple(tokens.shape)}')

    def forward(
        self,
        tokens: torch.Tensor,
        support_map: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_tokens, hw = self._flatten_tokens(tokens)
        b, n, _ = flat_tokens.shape

        weights = torch.ones((b, n), device=flat_tokens.device, dtype=flat_tokens.dtype)
        if support_map is not None:
            if support_map.dim() == 4 and support_map.shape[-1] == 1:
                support_flat = support_map.reshape(b, -1)
            elif support_map.dim() == 3:
                support_flat = support_map.reshape(b, -1)
            elif support_map.dim() == 2:
                support_flat = support_map
            else:
                raise ValueError(f'Unsupported support_map shape: {tuple(support_map.shape)}')
            if support_flat.shape != (b, n):
                raise ValueError(
                    f'support_map shape mismatch: expected {(b, n)}, got {tuple(support_flat.shape)}'
                )
            weights = weights * support_flat.to(dtype=flat_tokens.dtype)
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_flat = valid_mask.reshape(b, -1)
            elif valid_mask.dim() == 2:
                valid_flat = valid_mask
            else:
                raise ValueError(f'Unsupported valid_mask shape: {tuple(valid_mask.shape)}')
            if valid_flat.shape != (b, n):
                raise ValueError(
                    f'valid_mask shape mismatch: expected {(b, n)}, got {tuple(valid_flat.shape)}'
                )
            weights = weights * valid_flat.to(dtype=flat_tokens.dtype)

        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(NCEPS)
        pooled = (flat_tokens * weights.unsqueeze(-1)).sum(dim=1) / weight_sum
        return self.mlp(pooled)



def discover_available_scales(dataset_root: Path) -> list[float]:
    image_index_path = dataset_root / 'image_index.csv'
    scales: set[float] = set()
    if image_index_path.exists():
        with open(image_index_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = str(row.get('scale_mm', '')).strip()
                if text:
                    scales.add(float(text))
        return sorted(scales)

    manifest_path = dataset_root / 'manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    for episode_entry in manifest.get('episodes', []):
        episode_id = int(episode_entry.get('episode_id', 0))
        episode_dir_name = str(episode_entry.get('path', f'episode_{episode_id:06d}'))
        metadata_path = dataset_root / episode_dir_name / 'metadata.json'
        if not metadata_path.exists():
            continue
        with open(metadata_path, 'r', encoding='utf-8') as mf:
            metadata = json.load(mf)
        scales_block = metadata.get('scales', {})
        if not isinstance(scales_block, dict):
            continue
        for scale_key, scale_meta in scales_block.items():
            if isinstance(scale_meta, dict) and 'scale_mm' in scale_meta:
                scales.add(float(scale_meta['scale_mm']))
            else:
                scales.add(parse_scale_key(str(scale_key)))
    return sorted(scales)



def build_head_from_checkpoint(
    head: CanonicalRegressionHead,
    checkpoint_path: Path | None,
    device: torch.device,
) -> CanonicalRegressionHead:
    if checkpoint_path is None:
        log.warning(
            'No --head-checkpoint provided. The canonical regression head will stay randomly initialized and frozen; '
            'this is only useful for code-path verification, not meaningful DD-USA training.'
        )
        freeze_module(head)
        return head

    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and 'head_state_dict' in payload:
        state_dict = payload['head_state_dict']
    elif isinstance(payload, dict) and 'model_state_dict' in payload:
        state_dict = payload['model_state_dict']
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f'Unsupported head checkpoint payload type: {type(payload)}')

    if not isinstance(state_dict, dict):
        raise TypeError('Resolved head state_dict is not a dictionary')

    try:
        missing, unexpected = head.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        stripped_state = {}
        prefixes = ('head.', 'canonical_head.', 'regression_head.')
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    stripped_state[key[len(prefix):]] = value
        missing, unexpected = head.load_state_dict(stripped_state, strict=False)

    if missing:
        log.warning('Head checkpoint missing keys: %s', sorted(missing))
    if unexpected:
        log.warning('Head checkpoint unexpected keys: %s', sorted(unexpected))

    freeze_module(head)
    return head



def flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.dim() == 4:
        b, h, w, d = tokens.shape
        return tokens.reshape(b, h * w, d)
    if tokens.dim() == 3:
        return tokens
    raise ValueError(f'tokens must be (B,H,W,D) or (B,N,D), got {tuple(tokens.shape)}')



def flatten_support(support_map: torch.Tensor) -> torch.Tensor:
    if support_map.dim() == 4 and support_map.shape[-1] == 1:
        b = support_map.shape[0]
        return support_map.reshape(b, -1)
    if support_map.dim() == 3:
        b = support_map.shape[0]
        return support_map.reshape(b, -1)
    if support_map.dim() == 2:
        return support_map
    raise ValueError(f'Unsupported support_map shape: {tuple(support_map.shape)}')



def compute_task_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_xy: float,
    lambda_depth: float,
) -> dict[str, torch.Tensor]:
    x_loss = F.smooth_l1_loss(pred[:, 0], target[:, 0])
    y_loss = F.smooth_l1_loss(pred[:, 1], target[:, 1])
    depth_loss = F.smooth_l1_loss(pred[:, 2], target[:, 2])
    task_loss = lambda_xy * (x_loss + y_loss) + lambda_depth * depth_loss

    x_mae = (pred[:, 0] - target[:, 0]).abs().mean()
    y_mae = (pred[:, 1] - target[:, 1]).abs().mean()
    depth_mae = (pred[:, 2] - target[:, 2]).abs().mean()
    return {
        'task_loss': task_loss,
        'x_loss': x_loss,
        'y_loss': y_loss,
        'depth_loss': depth_loss,
        'x_mae': x_mae,
        'y_mae': y_mae,
        'depth_mae': depth_mae,
    }



def compute_dd_usa_losses(
    *,
    head: CanonicalRegressionHead,
    pred: torch.Tensor,
    target: torch.Tensor,
    source_feat: torch.Tensor,
    adapted_feat: torch.Tensor,
    aux: dict[str, torch.Tensor],
    source_scale_mm: torch.Tensor,
    target_scale_mm: torch.Tensor,
    lambda_task: float,
    lambda_xy: float,
    lambda_depth: float,
    lambda_id: float,
    lambda_anchor: float,
    lambda_unsup: float,
) -> dict[str, torch.Tensor]:
    task_metrics = compute_task_components(pred, target, lambda_xy=lambda_xy, lambda_depth=lambda_depth)

    same_scale = torch.isclose(source_scale_mm, target_scale_mm, atol=1e-6)
    if bool(same_scale.any()):
        with torch.no_grad():
            baseline_pred = head(source_feat[same_scale])
        identity_loss = F.smooth_l1_loss(pred[same_scale], baseline_pred)
    else:
        identity_loss = pred.new_zeros(())

    anchor_loss = F.mse_loss(adapted_feat, aux['anchor_feat'])
    residual_tokens = flatten_tokens(aux['residual_feat'])
    support_flat = flatten_support(aux['support_map']).to(device=pred.device, dtype=pred.dtype)
    residual_energy = residual_tokens.square().mean(dim=-1)
    unsupported_residual_loss = ((1.0 - support_flat) * residual_energy).mean()

    total_loss = (
        lambda_task * task_metrics['task_loss']
        + lambda_id * identity_loss
        + lambda_anchor * anchor_loss
        + lambda_unsup * unsupported_residual_loss
    )

    metrics = {
        'loss': total_loss,
        'id_loss': identity_loss,
        'anchor_loss': anchor_loss,
        'unsup_loss': unsupported_residual_loss,
        'support_mean': aux['support_mean'],
        'alpha_anchor': aux['alpha_anchor'],
        'alpha_corrected': aux['alpha_corrected'],
        'gamma_mean': aux['gamma_mean'],
        'beta_mean': aux['beta_mean'],
    }
    metrics.update(task_metrics)
    return metrics



def evaluate_model(
    *,
    vit: FrozenViTFeatureExtractor,
    head: CanonicalRegressionHead,
    dd_usa: DirectDriveUniversalScaleAdapter,
    data_loader: DataLoader,
    device: torch.device,
    canonical_scale_mm: float,
    lambda_task: float,
    lambda_xy: float,
    lambda_depth: float,
    lambda_id: float,
    lambda_anchor: float,
    lambda_unsup: float,
    use_adapter: bool,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    totals = {
        'loss': 0.0,
        'task_loss': 0.0,
        'x_loss': 0.0,
        'y_loss': 0.0,
        'depth_loss': 0.0,
        'id_loss': 0.0,
        'anchor_loss': 0.0,
        'unsup_loss': 0.0,
        'x_mae': 0.0,
        'y_mae': 0.0,
        'depth_mae': 0.0,
        'support_mean': 0.0,
        'alpha_anchor': 0.0,
        'alpha_corrected': 0.0,
        'gamma_mean': 0.0,
        'beta_mean': 0.0,
    }
    total_n = 0

    dd_usa.eval()
    head.eval()
    vit.eval()

    with torch.no_grad():
        iterator = tqdm(data_loader, desc=desc, leave=False, disable=not progress)
        for batch in iterator:
            images = batch['image'].to(device)
            source_coords = batch['source_coords'].to(device)
            target_coords = batch['target_coords'].to(device)
            source_scale_mm = batch['source_scale_mm'].to(device)
            target_scale_mm = batch['target_scale_mm'].to(device)
            target = batch['target'].to(device)

            source_feat = vit(images)
            if use_adapter:
                adapted_feat, aux = dd_usa(
                    source_feat=source_feat,
                    source_coord_map_mm=source_coords,
                    source_scale_mm=source_scale_mm,
                    target_scale_mm=target_scale_mm,
                    target_coord_map_mm=target_coords,
                    return_aux=True,
                )
                pred = head(adapted_feat, support_map=aux['support_map'])
                metrics = compute_dd_usa_losses(
                    head=head,
                    pred=pred,
                    target=target,
                    source_feat=source_feat,
                    adapted_feat=adapted_feat,
                    aux=aux,
                    source_scale_mm=source_scale_mm,
                    target_scale_mm=target_scale_mm,
                    lambda_task=lambda_task,
                    lambda_xy=lambda_xy,
                    lambda_depth=lambda_depth,
                    lambda_id=lambda_id,
                    lambda_anchor=lambda_anchor,
                    lambda_unsup=lambda_unsup,
                )
            else:
                pred = head(source_feat)
                task_metrics = compute_task_components(
                    pred,
                    target,
                    lambda_xy=lambda_xy,
                    lambda_depth=lambda_depth,
                )
                zero = pred.new_zeros(())
                metrics = {
                    'loss': lambda_task * task_metrics['task_loss'],
                    'task_loss': task_metrics['task_loss'],
                    'x_loss': task_metrics['x_loss'],
                    'y_loss': task_metrics['y_loss'],
                    'depth_loss': task_metrics['depth_loss'],
                    'id_loss': zero,
                    'anchor_loss': zero,
                    'unsup_loss': zero,
                    'x_mae': task_metrics['x_mae'],
                    'y_mae': task_metrics['y_mae'],
                    'depth_mae': task_metrics['depth_mae'],
                    'support_mean': zero,
                    'alpha_anchor': zero,
                    'alpha_corrected': zero,
                    'gamma_mean': zero,
                    'beta_mean': zero,
                }

            batch_size = int(images.shape[0])
            for key in totals:
                totals[key] += float(metrics[key].item()) * batch_size
            total_n += batch_size

    denom = max(total_n, 1)
    return {key: value / denom for key, value in totals.items()}



def run_epoch(
    *,
    vit: FrozenViTFeatureExtractor,
    head: CanonicalRegressionHead,
    dd_usa: DirectDriveUniversalScaleAdapter,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: torch.device,
    lambda_task: float,
    lambda_xy: float,
    lambda_depth: float,
    lambda_id: float,
    lambda_anchor: float,
    lambda_unsup: float,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    dd_usa.train(training)
    vit.eval()
    head.eval()

    totals = {
        'loss': 0.0,
        'task_loss': 0.0,
        'x_loss': 0.0,
        'y_loss': 0.0,
        'depth_loss': 0.0,
        'id_loss': 0.0,
        'anchor_loss': 0.0,
        'unsup_loss': 0.0,
        'x_mae': 0.0,
        'y_mae': 0.0,
        'depth_mae': 0.0,
        'support_mean': 0.0,
        'alpha_anchor': 0.0,
        'alpha_corrected': 0.0,
        'gamma_mean': 0.0,
        'beta_mean': 0.0,
    }
    total_n = 0

    iterator = tqdm(data_loader, desc=desc, leave=False, disable=not progress)
    for batch in iterator:
        images = batch['image'].to(device)
        source_coords = batch['source_coords'].to(device)
        target_coords = batch['target_coords'].to(device)
        source_scale_mm = batch['source_scale_mm'].to(device)
        target_scale_mm = batch['target_scale_mm'].to(device)
        target = batch['target'].to(device)

        with torch.no_grad():
            source_feat = vit(images)

        adapted_feat, aux = dd_usa(
            source_feat=source_feat,
            source_coord_map_mm=source_coords,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
            target_coord_map_mm=target_coords,
            return_aux=True,
        )
        pred = head(adapted_feat, support_map=aux['support_map'])
        metrics = compute_dd_usa_losses(
            head=head,
            pred=pred,
            target=target,
            source_feat=source_feat,
            adapted_feat=adapted_feat,
            aux=aux,
            source_scale_mm=source_scale_mm,
            target_scale_mm=target_scale_mm,
            lambda_task=lambda_task,
            lambda_xy=lambda_xy,
            lambda_depth=lambda_depth,
            lambda_id=lambda_id,
            lambda_anchor=lambda_anchor,
            lambda_unsup=lambda_unsup,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            metrics['loss'].backward()
            torch.nn.utils.clip_grad_norm_(dd_usa.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = int(images.shape[0])
        for key in totals:
            totals[key] += float(metrics[key].item()) * batch_size
        total_n += batch_size

        iterator.set_postfix(
            loss=f"{metrics['loss'].item():.4f}",
            task=f"{metrics['task_loss'].item():.4f}",
            depth=f"{metrics['depth_loss'].item():.4f}",
            sup=f"{metrics['support_mean'].item():.3f}",
        )

    denom = max(total_n, 1)
    return {key: value / denom for key, value in totals.items()}



def log_comparison_suite(
    *,
    vit: FrozenViTFeatureExtractor,
    head: CanonicalRegressionHead,
    dd_usa: DirectDriveUniversalScaleAdapter,
    dataset_root: Path,
    canonical_scale_mm: float,
    eval_scales: list[float],
    device: torch.device,
    batch_size: int,
    workers: int,
    lambda_task: float,
    lambda_xy: float,
    lambda_depth: float,
    lambda_id: float,
    lambda_anchor: float,
    lambda_unsup: float,
    progress: bool,
) -> None:
    eval_specs = [
        ('seen', SEEN_INDENTERS),
        ('unseen', TEST_INDENTERS),
    ]
    for scale_mm in eval_scales:
        for split_name, indenters in eval_specs:
            try:
                dataset = DDUSATactileDataset(
                    dataset_root=dataset_root,
                    indenters=indenters,
                    scales_mm=[scale_mm],
                    canonical_scale_mm=canonical_scale_mm,
                )
            except RuntimeError:
                log.warning('Skipping eval for split=%s scale=%.1fmm because no samples were found.', split_name, scale_mm)
                continue
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
            baseline_metrics = evaluate_model(
                vit=vit,
                head=head,
                dd_usa=dd_usa,
                data_loader=loader,
                device=device,
                canonical_scale_mm=canonical_scale_mm,
                lambda_task=lambda_task,
                lambda_xy=lambda_xy,
                lambda_depth=lambda_depth,
                lambda_id=lambda_id,
                lambda_anchor=lambda_anchor,
                lambda_unsup=lambda_unsup,
                use_adapter=False,
                progress=progress,
                desc=f'Baseline[{split_name},{scale_mm:g}mm]',
            )
            adapted_metrics = evaluate_model(
                vit=vit,
                head=head,
                dd_usa=dd_usa,
                data_loader=loader,
                device=device,
                canonical_scale_mm=canonical_scale_mm,
                lambda_task=lambda_task,
                lambda_xy=lambda_xy,
                lambda_depth=lambda_depth,
                lambda_id=lambda_id,
                lambda_anchor=lambda_anchor,
                lambda_unsup=lambda_unsup,
                use_adapter=True,
                progress=progress,
                desc=f'DD-USA[{split_name},{scale_mm:g}mm]',
            )
            log.info(
                'EVAL split=%s scale=%.1fmm | without_dd_usa: task=%.6f x_mae=%.6f y_mae=%.6f depth_mae=%.6f | '
                'with_dd_usa: task=%.6f x_mae=%.6f y_mae=%.6f depth_mae=%.6f support=%.4f alpha_anchor=%.4f alpha_corrected=%.4f',
                split_name,
                scale_mm,
                baseline_metrics['task_loss'],
                baseline_metrics['x_mae'],
                baseline_metrics['y_mae'],
                baseline_metrics['depth_mae'],
                adapted_metrics['task_loss'],
                adapted_metrics['x_mae'],
                adapted_metrics['y_mae'],
                adapted_metrics['depth_mae'],
                adapted_metrics['support_mean'],
                adapted_metrics['alpha_anchor'],
                adapted_metrics['alpha_corrected'],
            )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train DD-USA against a frozen canonical downstream head')
    parser.add_argument('--dataset', type=str, default=str(resolve_default_dataset()))
    parser.add_argument('--checkpoint-dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument('--canonical-scale-mm', type=float, default=DEFAULT_CANONICAL_SCALE_MM)
    parser.add_argument('--train-scales', nargs='*', type=float, default=None)
    parser.add_argument('--eval-scales', nargs='*', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--head-checkpoint', type=str, default='')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--test-checkpoint', type=str, default='')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--eval-test', action='store_true')
    parser.add_argument('--lambda-task', type=float, default=1.0)
    parser.add_argument('--lambda-xy', type=float, default=1.0)
    parser.add_argument('--lambda-depth', type=float, default=1.0)
    parser.add_argument('--lambda-id', type=float, default=0.25)
    parser.add_argument('--lambda-anchor', type=float, default=0.10)
    parser.add_argument('--lambda-unsup', type=float, default=0.05)
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()



def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Device: %s', device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = Path(args.dataset).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (PROJECT_ROOT / dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset root not found: {dataset_root}')

    available_scales = discover_available_scales(dataset_root)
    train_scales = sorted(args.train_scales) if args.train_scales else available_scales
    eval_scales = sorted(args.eval_scales) if args.eval_scales else sorted(set(train_scales))
    if args.canonical_scale_mm not in eval_scales:
        eval_scales = sorted(set(eval_scales + [args.canonical_scale_mm]))

    log.info(
        'DD-USA dataset: %s | canonical_scale=%.1fmm | train_scales=%s | eval_scales=%s',
        dataset_root,
        args.canonical_scale_mm,
        train_scales,
        eval_scales,
    )
    log.info(
        'Split: train=%s | val=%s | test=%s',
        sorted(TRAIN_INDENTERS),
        sorted(VAL_INDENTERS),
        sorted(TEST_INDENTERS),
    )

    train_dataset = DDUSATactileDataset(
        dataset_root=dataset_root,
        indenters=TRAIN_INDENTERS,
        scales_mm=train_scales,
        canonical_scale_mm=args.canonical_scale_mm,
    )
    val_dataset = DDUSATactileDataset(
        dataset_root=dataset_root,
        indenters=VAL_INDENTERS,
        scales_mm=eval_scales,
        canonical_scale_mm=args.canonical_scale_mm,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    vit = FrozenViTFeatureExtractor(device=device)
    dd_usa = DirectDriveUniversalScaleAdapter(embed_dim=vit.embed_dim).to(device)
    head = CanonicalRegressionHead(embed_dim=vit.embed_dim).to(device)
    head_ckpt = resolve_checkpoint_path(args.head_checkpoint)
    head = build_head_from_checkpoint(head, head_ckpt, device)
    freeze_module(head)

    log.info('Frozen ViT patch grid: %s (%d tokens)', vit.patch_grid, vit.patch_token_count)
    log.info('DD-USA parameters: %d', sum(p.numel() for p in dd_usa.parameters() if p.requires_grad))
    log.info('Head checkpoint: %s', head_ckpt if head_ckpt is not None else '(none)')

    optimizer = torch.optim.AdamW(dd_usa.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches_per_epoch = max(len(train_loader), 1)
    warmup_steps = max(1, args.warmup_epochs * batches_per_epoch)
    total_steps = max(1, args.epochs * batches_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    if not ckpt_dir.is_absolute():
        ckpt_dir = (PROJECT_ROOT / ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = ckpt_dir / 'metrics.csv'
    initialize_metrics_csv(metrics_csv_path, overwrite=(not args.resume_from))

    best_val_task_loss = float('inf')
    global_step = 0
    start_epoch = 1
    train_start_time = time.time()
    epoch_durations: list[float] = []

    def save_checkpoint(path: Path, *, epoch: int, val_task_loss: float) -> None:
        payload = {
            'epoch': int(epoch),
            'global_step': int(global_step),
            'dd_usa_state_dict': dd_usa.state_dict(),
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_task_loss': float(best_val_task_loss),
            'val_task_loss': float(val_task_loss),
            'canonical_scale_mm': float(args.canonical_scale_mm),
            'train_scales': list(train_scales),
            'eval_scales': list(eval_scales),
        }
        torch.save(payload, path)

    resume_path = resolve_checkpoint_path(args.resume_from)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        payload = torch.load(resume_path, map_location=device)
        dd_usa.load_state_dict(payload['dd_usa_state_dict'])
        if 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        if 'scheduler_state_dict' in payload:
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        global_step = int(payload.get('global_step', 0))
        start_epoch = int(payload.get('epoch', 0)) + 1
        best_val_task_loss = float(payload.get('best_val_task_loss', payload.get('val_task_loss', float('inf'))))
        log.info(
            'Resumed from %s | start_epoch=%d | global_step=%d | best_val_task_loss=%.6f',
            resume_path,
            start_epoch,
            global_step,
            best_val_task_loss,
        )

    if args.test_only:
        test_path = resolve_checkpoint_path(args.test_checkpoint) or (ckpt_dir / 'best.pt')
        if not test_path.exists():
            raise FileNotFoundError(f'Test checkpoint not found: {test_path}')
        payload = torch.load(test_path, map_location=device)
        dd_usa.load_state_dict(payload['dd_usa_state_dict'])
        log.info('TEST-ONLY mode: loaded DD-USA checkpoint %s', test_path)
        log_comparison_suite(
            vit=vit,
            head=head,
            dd_usa=dd_usa,
            dataset_root=dataset_root,
            canonical_scale_mm=args.canonical_scale_mm,
            eval_scales=eval_scales,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            lambda_task=args.lambda_task,
            lambda_xy=args.lambda_xy,
            lambda_depth=args.lambda_depth,
            lambda_id=args.lambda_id,
            lambda_anchor=args.lambda_anchor,
            lambda_unsup=args.lambda_unsup,
            progress=not args.no_progress,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            vit=vit,
            head=head,
            dd_usa=dd_usa,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            lambda_task=args.lambda_task,
            lambda_xy=args.lambda_xy,
            lambda_depth=args.lambda_depth,
            lambda_id=args.lambda_id,
            lambda_anchor=args.lambda_anchor,
            lambda_unsup=args.lambda_unsup,
            progress=not args.no_progress,
            desc=f'Train {epoch}/{args.epochs}',
        )
        global_step += len(train_loader)

        val_metrics = evaluate_model(
            vit=vit,
            head=head,
            dd_usa=dd_usa,
            data_loader=val_loader,
            device=device,
            canonical_scale_mm=args.canonical_scale_mm,
            lambda_task=args.lambda_task,
            lambda_xy=args.lambda_xy,
            lambda_depth=args.lambda_depth,
            lambda_id=args.lambda_id,
            lambda_anchor=args.lambda_anchor,
            lambda_unsup=args.lambda_unsup,
            use_adapter=True,
            progress=not args.no_progress,
            desc='Val',
        )

        epoch_elapsed = time.time() - epoch_start
        epoch_durations.append(epoch_elapsed)
        total_elapsed = time.time() - train_start_time
        avg_epoch_time = sum(epoch_durations) / max(len(epoch_durations), 1)
        eta_seconds = max(args.epochs - epoch, 0) * avg_epoch_time
        lr_now = optimizer.param_groups[0]['lr']

        if val_metrics['task_loss'] < best_val_task_loss:
            best_val_task_loss = val_metrics['task_loss']
            save_checkpoint(ckpt_dir / 'best.pt', epoch=epoch, val_task_loss=val_metrics['task_loss'])
            log.info('Saved new best checkpoint to %s (val_task_loss=%.6f)', ckpt_dir / 'best.pt', best_val_task_loss)

        save_checkpoint(ckpt_dir / 'latest.pt', epoch=epoch, val_task_loss=val_metrics['task_loss'])
        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f'epoch_{epoch:04d}.pt'
            save_checkpoint(periodic_path, epoch=epoch, val_task_loss=val_metrics['task_loss'])
            log.info('Saved periodic checkpoint to %s', periodic_path)

        log.info(
            'Epoch %03d/%d  train_loss=%.6f task=%.6f x=%.6f y=%.6f depth=%.6f id=%.6f anchor=%.6f unsup=%.6f '
            'x_mae=%.6f y_mae=%.6f depth_mae=%.6f support=%.4f alpha_anchor=%.4f alpha_corrected=%.4f gamma=%.4f beta=%.4f  '
            'val_loss=%.6f val_task=%.6f val_x=%.6f val_y=%.6f val_depth=%.6f val_x_mae=%.6f val_y_mae=%.6f val_depth_mae=%.6f '
            'val_support=%.4f best_val_task=%.6f lr=%.2e epoch_t=%s elapsed=%s eta=%s',
            epoch,
            args.epochs,
            train_metrics['loss'],
            train_metrics['task_loss'],
            train_metrics['x_loss'],
            train_metrics['y_loss'],
            train_metrics['depth_loss'],
            train_metrics['id_loss'],
            train_metrics['anchor_loss'],
            train_metrics['unsup_loss'],
            train_metrics['x_mae'],
            train_metrics['y_mae'],
            train_metrics['depth_mae'],
            train_metrics['support_mean'],
            train_metrics['alpha_anchor'],
            train_metrics['alpha_corrected'],
            train_metrics['gamma_mean'],
            train_metrics['beta_mean'],
            val_metrics['loss'],
            val_metrics['task_loss'],
            val_metrics['x_loss'],
            val_metrics['y_loss'],
            val_metrics['depth_loss'],
            val_metrics['x_mae'],
            val_metrics['y_mae'],
            val_metrics['depth_mae'],
            val_metrics['support_mean'],
            best_val_task_loss,
            lr_now,
            format_duration(epoch_elapsed),
            format_duration(total_elapsed),
            format_duration(eta_seconds),
        )

        append_metrics_row(
            metrics_csv_path,
            {
                'epoch': epoch,
                'global_step': global_step,
                'lr': lr_now,
                'train_loss': train_metrics['loss'],
                'train_task_loss': train_metrics['task_loss'],
                'train_x_loss': train_metrics['x_loss'],
                'train_y_loss': train_metrics['y_loss'],
                'train_depth_loss': train_metrics['depth_loss'],
                'train_id_loss': train_metrics['id_loss'],
                'train_anchor_loss': train_metrics['anchor_loss'],
                'train_unsup_loss': train_metrics['unsup_loss'],
                'train_x_mae': train_metrics['x_mae'],
                'train_y_mae': train_metrics['y_mae'],
                'train_depth_mae': train_metrics['depth_mae'],
                'train_support_mean': train_metrics['support_mean'],
                'train_alpha_anchor': train_metrics['alpha_anchor'],
                'train_alpha_corrected': train_metrics['alpha_corrected'],
                'train_gamma_mean': train_metrics['gamma_mean'],
                'train_beta_mean': train_metrics['beta_mean'],
                'val_loss': val_metrics['loss'],
                'val_task_loss': val_metrics['task_loss'],
                'val_x_loss': val_metrics['x_loss'],
                'val_y_loss': val_metrics['y_loss'],
                'val_depth_loss': val_metrics['depth_loss'],
                'val_id_loss': val_metrics['id_loss'],
                'val_anchor_loss': val_metrics['anchor_loss'],
                'val_unsup_loss': val_metrics['unsup_loss'],
                'val_x_mae': val_metrics['x_mae'],
                'val_y_mae': val_metrics['y_mae'],
                'val_depth_mae': val_metrics['depth_mae'],
                'val_support_mean': val_metrics['support_mean'],
                'val_alpha_anchor': val_metrics['alpha_anchor'],
                'val_alpha_corrected': val_metrics['alpha_corrected'],
                'val_gamma_mean': val_metrics['gamma_mean'],
                'val_beta_mean': val_metrics['beta_mean'],
                'best_val_task_loss': best_val_task_loss,
                'epoch_seconds': epoch_elapsed,
                'elapsed_seconds': total_elapsed,
                'eta_seconds': eta_seconds,
            },
        )

    log.info('Training finished. Best val task loss: %.6f', best_val_task_loss)
    log.info('Checkpoints: %s', ckpt_dir)

    if args.eval_test:
        best_path = ckpt_dir / 'best.pt'
        if best_path.exists():
            payload = torch.load(best_path, map_location=device)
            dd_usa.load_state_dict(payload['dd_usa_state_dict'])
            log.info('Loaded best DD-USA checkpoint for final evaluation: %s', best_path)
        log_comparison_suite(
            vit=vit,
            head=head,
            dd_usa=dd_usa,
            dataset_root=dataset_root,
            canonical_scale_mm=args.canonical_scale_mm,
            eval_scales=eval_scales,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            lambda_task=args.lambda_task,
            lambda_xy=args.lambda_xy,
            lambda_depth=args.lambda_depth,
            lambda_id=args.lambda_id,
            lambda_anchor=args.lambda_anchor,
            lambda_unsup=args.lambda_unsup,
            progress=not args.no_progress,
        )


if __name__ == '__main__':
    train(parse_args())
