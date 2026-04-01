from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

HOST_STEA_TRAIN_DATASET = Path('/home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix')
DOCKER_STEA_TRAIN_DATASET = Path('/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix')
HOST_DOWNSTREAM_DATASET = Path('/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
DOCKER_DOWNSTREAM_DATASET = Path('/datasets/usa_static_v1_large_run/downstream_test_16_20_23')
HOST_CHECKPOINT_ROOT = Path('/home/suhang/datasets/checkpoints')
DOCKER_CHECKPOINT_ROOT = Path('/datasets/checkpoints')

CANONICAL_SCALE_MM = 20.0
EMBED_DIM = 768
GRID_SIZE = 14
NUM_TOKENS = GRID_SIZE * GRID_SIZE
TARGET_NAMES = ['command_x_norm', 'command_y_norm', 'frame_actual_max_down_mm']
TRAIN_INDENTERS = [
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
]
VAL_INDENTERS = ['sphere_s', 'triangle']
SEEN_INDENTERS = TRAIN_INDENTERS + VAL_INDENTERS
UNSEEN_INDENTERS = ['pacman', 'wave', 'torus', 'hexagon']


class _WandbStub:
    @staticmethod
    def init(*args: Any, **kwargs: Any) -> None:
        print('wandb is not installed; continuing with local logging only.')
        return None

    @staticmethod
    def log(*args: Any, **kwargs: Any) -> None:
        return None

    @staticmethod
    def finish(*args: Any, **kwargs: Any) -> None:
        return None


try:
    import wandb as _wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency in lightweight runtimes
    wandb = _WandbStub()
else:
    if all(hasattr(_wandb, name) for name in ('init', 'log', 'finish')):
        wandb = _wandb
    else:  # pragma: no cover - defensive fallback for partial/local wandb shims
        print('wandb import is missing init/log/finish; falling back to local logging only.')
        wandb = _WandbStub()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_seconds(total_seconds: float) -> str:
    total = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}h{minutes:02d}m{seconds:02d}s'
    return f'{minutes:02d}m{seconds:02d}s'


def _swap_dataset_root(path: Path) -> Path:
    text = str(path)
    if text.startswith('/home/suhang/datasets/'):
        candidate = Path('/datasets') / text.removeprefix('/home/suhang/datasets/').lstrip('/')
        if candidate.exists():
            return candidate.resolve()
    if text.startswith('/datasets/'):
        candidate = Path('/home/suhang/datasets') / text.removeprefix('/datasets/').lstrip('/')
        if candidate.exists():
            return candidate.resolve()
    return path


def resolve_path(path_ref: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_ref).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        path = (base_dir / path).resolve()
    if path.exists():
        return path.resolve()
    swapped = _swap_dataset_root(path)
    if swapped.exists():
        return swapped
    return path.resolve()


def default_stea_train_dataset_root() -> Path:
    if HOST_STEA_TRAIN_DATASET.exists():
        return HOST_STEA_TRAIN_DATASET
    if DOCKER_STEA_TRAIN_DATASET.exists():
        return DOCKER_STEA_TRAIN_DATASET
    return HOST_STEA_TRAIN_DATASET


def default_downstream_dataset_root() -> Path:
    if HOST_DOWNSTREAM_DATASET.exists():
        return HOST_DOWNSTREAM_DATASET
    if DOCKER_DOWNSTREAM_DATASET.exists():
        return DOCKER_DOWNSTREAM_DATASET
    return HOST_DOWNSTREAM_DATASET


def default_checkpoint_dir(name: str) -> Path:
    root = HOST_CHECKPOINT_ROOT if HOST_CHECKPOINT_ROOT.exists() else DOCKER_CHECKPOINT_ROOT
    return root / name


def build_default_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_rgb_image(path: str | Path, *, transform: transforms.Compose | None = None) -> torch.Tensor:
    image = Image.open(resolve_path(path)).convert('RGB')
    transform = transform or build_default_image_transform()
    return transform(image)


def strip_cls_token_if_present(features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError(
            'Expected ViT features to have shape (B, N, D), '
            f'got {tuple(features.shape)}'
        )
    if features.shape[1] == NUM_TOKENS + 1:
        features = features[:, 1:, :]
    elif features.shape[1] != NUM_TOKENS:
        raise ValueError(
            f'Expected 197 tokens (with CLS) or {NUM_TOKENS} patch tokens, got {tuple(features.shape)}'
        )
    if features.shape[2] != EMBED_DIM:
        raise ValueError(f'Expected embed dim {EMBED_DIM}, got {features.shape[2]}')
    return features.contiguous()


class FrozenViTPatchExtractor(nn.Module):
    """Frozen ViT-B/16 returning patch tokens of shape (B, 196, 768)."""

    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True) -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return strip_cls_token_if_present(self.vit.forward_features(x))


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.dim() != 3 or tokens.shape[1:] != (NUM_TOKENS, EMBED_DIM):
        raise ValueError(f'tokens must be (B, {NUM_TOKENS}, {EMBED_DIM}), got {tuple(tokens.shape)}')
    batch = tokens.shape[0]
    return tokens.reshape(batch, GRID_SIZE, GRID_SIZE, EMBED_DIM).permute(0, 3, 1, 2).contiguous()


def map_to_tokens(token_map: torch.Tensor) -> torch.Tensor:
    if token_map.dim() != 4 or token_map.shape[1:] != (EMBED_DIM, GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f'token_map must be (B, {EMBED_DIM}, {GRID_SIZE}, {GRID_SIZE}), got {tuple(token_map.shape)}'
        )
    batch = token_map.shape[0]
    return token_map.permute(0, 2, 3, 1).reshape(batch, NUM_TOKENS, EMBED_DIM).contiguous()


class STEARegressorHead(nn.Module):
    """Standalone downstream head that mirrors the old spatial reducer + MLP regressor."""

    def __init__(self, out_dim: int = 3) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.flatten_dim = 128 * 7 * 7
        self.spatial_reducer = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.reg_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spatial = tokens_to_map(tokens)
        reduced = self.spatial_reducer(spatial)
        flat = reduced.reshape(reduced.shape[0], self.flatten_dim).contiguous()
        return self.reg_layer(flat)


def _load_checkpoint_payload(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if isinstance(payload, dict):
        return payload
    return {'model_state_dict': payload}


def load_canonical_head_from_regressor_ckpt(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    freeze: bool = True,
) -> STEARegressorHead:
    payload = _load_checkpoint_payload(checkpoint_path, device)
    state_dict = payload.get('model_state_dict', payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f'Checkpoint does not contain a valid model_state_dict: {checkpoint_path}')

    filtered = {
        key: value
        for key, value in state_dict.items()
        if key.startswith('spatial_reducer.') or key.startswith('reg_layer.')
    }
    if not filtered:
        raise RuntimeError(
            'Could not find spatial_reducer/reg_layer weights in canonical regressor checkpoint '
            f'{checkpoint_path}'
        )

    head = STEARegressorHead(out_dim=3).to(device)
    load_result = head.load_state_dict(filtered, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            'Failed to load canonical head cleanly. '
            f'missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}'
        )
    head.eval()
    if freeze:
        for param in head.parameters():
            param.requires_grad = False
    return head


def load_trained_head_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    freeze: bool = True,
) -> tuple[STEARegressorHead, dict[str, Any]]:
    payload = _load_checkpoint_payload(checkpoint_path, device)
    state_dict = payload.get('model_state_dict') or payload.get('head_state_dict') or payload
    if not isinstance(state_dict, dict):
        raise ValueError(f'Checkpoint does not contain a valid head state dict: {checkpoint_path}')
    head = STEARegressorHead(out_dim=int(payload.get('out_dim', 3))).to(device)
    load_result = head.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            'Failed to load STEA regressor head cleanly. '
            f'missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}'
        )
    head.eval()
    if freeze:
        for param in head.parameters():
            param.requires_grad = False
    return head, payload


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
