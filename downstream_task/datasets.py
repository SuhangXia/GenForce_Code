from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_REFERENCE_SCALE_MM = 20.0


class MultiscaleTactileDataset(Dataset):
    """Static tactile dataset for downstream pose regression.

    The dataset reads the generator's ``image_index.csv`` and returns one sample per
    rendered marker image. Each sample includes the image tensor, the pose target
    ``[x_norm, y_norm, frame_actual_max_down_mm]``, the 20mm reference coordinate
    map, and the current sample-scale coordinate map.
    """

    def __init__(
        self,
        root_dir: str | Path,
        scale_mm: float,
        indenters: Iterable[str],
        reference_scale_mm: float = DEFAULT_REFERENCE_SCALE_MM,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.scale_mm = float(scale_mm)
        self.reference_scale_mm = float(reference_scale_mm)
        self.indenters = frozenset(str(name) for name in indenters)
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
        self._samples = self._load_samples()

        if not self._samples:
            raise RuntimeError(
                f"No downstream samples found in {self.root_dir} for scale={self.scale_mm} "
                f"and indenters={sorted(self.indenters)}"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self._samples[index]
        image = Image.open(sample['image_path']).convert('RGB')
        img_tensor = self.transform(image)

        target = torch.tensor(
            [sample['x_norm'], sample['y_norm'], sample['depth_mm']],
            dtype=torch.float32,
        )

        # target_coords = 20mm reference grid, source_coords = current sample grid.
        target_coords = self._load_coord_map(sample['reference_coord_path'])
        source_coords = self._load_coord_map(sample['current_coord_path'])
        return img_tensor, target, target_coords, source_coords

    def _load_samples(self) -> list[dict[str, object]]:
        index_path = self.root_dir / 'image_index.csv'
        if not index_path.exists():
            raise FileNotFoundError(
                f"Expected {index_path} to exist. Generate the dataset first so the downstream loader "
                'can read image_index.csv.'
            )

        samples: list[dict[str, object]] = []
        with open(index_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                indenter = str(row.get('indenter_name', '')).strip()
                if indenter not in self.indenters:
                    continue

                row_scale = self._parse_float(row.get('scale_mm'), 'scale_mm')
                if not np.isclose(row_scale, self.scale_mm):
                    continue

                image_path = self._resolve_path(
                    row.get('image_abspath'),
                    row.get('image_relpath'),
                    'image',
                )

                episode_dir = self.root_dir / str(row.get('episode_dir', '')).strip()
                if not episode_dir.exists():
                    raise FileNotFoundError(f"Episode directory not found for row: {episode_dir}")

                current_coord_path = self._resolve_path(
                    row.get('adapter_coord_map_abspath'),
                    row.get('adapter_coord_map_relpath'),
                    'adapter_coord_map',
                    base_dirs=[episode_dir],
                )

                reference_scale_dir = episode_dir / f"scale_{int(self.reference_scale_mm)}mm"
                reference_coord_path = reference_scale_dir / 'adapter_coord_map.npy'
                if not reference_coord_path.exists():
                    raise FileNotFoundError(
                        f"Missing reference coord map for episode {episode_dir.name}: {reference_coord_path}"
                    )

                command_x_mm = self._parse_float(row.get('command_x_mm'), 'command_x_mm')
                command_y_mm = self._parse_float(row.get('command_y_mm'), 'command_y_mm')
                depth_mm = self._parse_float(row.get('frame_actual_max_down_mm'), 'frame_actual_max_down_mm')

                scale_radius = row_scale / 2.0
                if scale_radius <= 0:
                    raise ValueError(f"Invalid scale_mm={row_scale} in {index_path}")

                samples.append(
                    {
                        'image_path': image_path,
                        'current_coord_path': current_coord_path,
                        'reference_coord_path': reference_coord_path,
                        'x_norm': command_x_mm / scale_radius,
                        'y_norm': command_y_mm / scale_radius,
                        'depth_mm': depth_mm,
                    }
                )

        return samples

    def _load_coord_map(self, path: Path) -> torch.Tensor:
        if path not in self._coord_cache:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                raise ValueError(f"Expected coord map shape (H, W, 2), got {arr.shape} at {path}")
            self._coord_cache[path] = torch.from_numpy(arr).to(torch.float32)
        return self._coord_cache[path]

    def _resolve_path(
        self,
        abs_path: object,
        rel_path: object,
        field_name: str,
        base_dirs: Iterable[Path] | None = None,
    ) -> Path:
        abs_text = str(abs_path or '').strip()
        rel_text = str(rel_path or '').strip()

        candidates: list[Path] = []
        if abs_text:
            candidates.append(Path(abs_text).expanduser())
        if rel_text:
            if base_dirs is not None:
                for base_dir in base_dirs:
                    candidates.append((base_dir / rel_text).resolve())
            candidates.append((self.root_dir / rel_text).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not resolve {field_name} path. abs={abs_text!r} rel={rel_text!r} root={self.root_dir}"
        )

    @staticmethod
    def _parse_float(value: object, field_name: str) -> float:
        text = str(value).strip()
        if not text:
            raise ValueError(f"Missing required numeric field: {field_name}")
        return float(text)
