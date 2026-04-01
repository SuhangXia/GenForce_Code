from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCRPIT_DIR = PROJECT_ROOT / 'scrpit'
for candidate in (SCRIPT_DIR, PROJECT_ROOT, SCRPIT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from downstream_task.datasets import MultiscaleTactileDataset
from standalone_stea.utils import (
    CANONICAL_SCALE_MM,
    SEEN_INDENTERS,
    TRAIN_INDENTERS,
    UNSEEN_INDENTERS,
    VAL_INDENTERS,
    default_downstream_dataset_root,
    default_stea_train_dataset_root,
    resolve_path,
)

try:
    from train_dd_usa_pretrain import DDUSAPretrainPairDataset
except ImportError as exc:  # pragma: no cover - dependency is expected in the target runtime
    raise ImportError(
        'standalone_stea.data requires scrpit/train_dd_usa_pretrain.py to provide DDUSAPretrainPairDataset.'
    ) from exc


class STEAPairedEventDataset(Dataset[dict[str, Any]]):
    """Filter DD-USA event pairs to source->canonical target event matches."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        target_scale_mm: float,
        source_scales: Sequence[float],
        pairs_per_epoch: int | None,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.target_scale_mm = float(target_scale_mm)
        self.source_scales = tuple(float(scale) for scale in source_scales)
        self._pairs: list[dict[str, Any]] = []

        for pair in getattr(base_dataset, '_all_pairs', []):
            source_scale = float(pair['source']['scale_mm'])
            target_scale = float(pair['target']['scale_mm'])
            if not np.isclose(target_scale, self.target_scale_mm):
                continue
            if not any(np.isclose(source_scale, scale) for scale in self.source_scales):
                continue
            self._pairs.append(pair)

        if not self._pairs:
            raise RuntimeError(
                'No STEA source->target training pairs found for '
                f'target_scale_mm={self.target_scale_mm} and source_scales={self.source_scales}'
            )

        self._len = len(self._pairs) if pairs_per_epoch is None else int(pairs_per_epoch)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pairs[index % len(self._pairs)]
        source = pair['source']
        target = pair['target']
        source_image = self.base_dataset.transform(Image.open(source['image_path']).convert('RGB'))
        target_image = self.base_dataset.transform(Image.open(target['image_path']).convert('RGB'))
        return {
            'source_image': source_image,
            'target_image': target_image,
            'source_scale_mm': torch.tensor(float(source['scale_mm']), dtype=torch.float32),
            'target_scale_mm': torch.tensor(float(target['scale_mm']), dtype=torch.float32),
            'episode_id': int(source['episode_id']),
            'frame_name': str(source['frame_name']),
            'marker_name': str(source['marker_name']),
            'indenter_name': str(source['indenter_name']),
        }


def build_stea_pair_datasets(
    dataset_root: str | Path = default_stea_train_dataset_root(),
    *,
    target_scale_mm: float = CANONICAL_SCALE_MM,
    source_scales: Sequence[float] = (15.0, 18.0, 20.0, 22.0, 25.0),
    pairs_per_epoch: int | None = 24000,
    val_pairs_per_epoch: int | None = 2000,
    seed: int = 42,
) -> tuple[Dataset[Any], Dataset[Any]]:
    dataset_root = resolve_path(dataset_root)
    scales_mm = sorted({float(target_scale_mm), *[float(scale) for scale in source_scales]})
    train_base = DDUSAPretrainPairDataset(
        dataset_root=dataset_root,
        indenters=TRAIN_INDENTERS,
        scales_mm=scales_mm,
        mode='train',
        pairs_per_epoch=None,
        seed=seed,
    )
    val_base = DDUSAPretrainPairDataset(
        dataset_root=dataset_root,
        indenters=VAL_INDENTERS,
        scales_mm=scales_mm,
        mode='val',
        pairs_per_epoch=None,
        seed=seed + 1,
    )
    train_dataset = STEAPairedEventDataset(
        train_base,
        target_scale_mm=target_scale_mm,
        source_scales=source_scales,
        pairs_per_epoch=pairs_per_epoch,
    )
    val_dataset = STEAPairedEventDataset(
        val_base,
        target_scale_mm=target_scale_mm,
        source_scales=source_scales,
        pairs_per_epoch=val_pairs_per_epoch,
    )
    return train_dataset, val_dataset


class DownstreamScaleAnnotatedDataset(Dataset[dict[str, torch.Tensor]]):
    """Wrap the downstream dataset so each sample explicitly carries its source scale."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        scale_mm: float,
        indenters: Iterable[str],
        reference_scale_mm: float = CANONICAL_SCALE_MM,
    ) -> None:
        super().__init__()
        self.scale_mm = float(scale_mm)
        self.dataset = MultiscaleTactileDataset(
            root_dir=str(resolve_path(root_dir)),
            scale_mm=self.scale_mm,
            indenters=indenters,
            reference_scale_mm=float(reference_scale_mm),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, target, target_coords, source_coords = self.dataset[index]
        return {
            'image': image,
            'target': target,
            'target_coords': target_coords,
            'source_coords': source_coords,
            'source_scale_mm': torch.tensor(self.scale_mm, dtype=torch.float32),
        }


def build_downstream_split_dataset(
    root_dir: str | Path = default_downstream_dataset_root(),
    *,
    scales_mm: Sequence[float],
    split: str,
    reference_scale_mm: float = CANONICAL_SCALE_MM,
) -> Dataset[Any]:
    if split == 'train':
        indenters = TRAIN_INDENTERS
    elif split == 'val':
        indenters = VAL_INDENTERS
    elif split == 'seen':
        indenters = SEEN_INDENTERS
    elif split == 'unseen':
        indenters = UNSEEN_INDENTERS
    else:
        raise ValueError(f'Unsupported split: {split}')

    datasets = [
        DownstreamScaleAnnotatedDataset(
            root_dir=root_dir,
            scale_mm=float(scale_mm),
            indenters=indenters,
            reference_scale_mm=reference_scale_mm,
        )
        for scale_mm in scales_mm
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def create_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def unpack_downstream_batch(
    batch: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        image = batch['image']
        target = batch['target']
        target_coords = batch['target_coords']
        source_coords = batch['source_coords']
        source_scale_mm = batch['source_scale_mm']
    elif isinstance(batch, (tuple, list)) and len(batch) >= 5:
        image, target, target_coords, source_coords, source_scale_mm = batch[:5]
    else:
        raise TypeError('Expected downstream batch dict or tuple with five elements.')

    return (
        image.to(device=device, dtype=torch.float32),
        target.to(device=device, dtype=torch.float32),
        target_coords.to(device=device, dtype=torch.float32),
        source_coords.to(device=device, dtype=torch.float32),
        source_scale_mm.to(device=device, dtype=torch.float32),
    )
