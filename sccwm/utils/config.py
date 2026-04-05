from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def resolve_path(path_ref: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path = Path(path_ref).expanduser()
    if not path.is_absolute():
        root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
        path = (root / path).resolve()
    if path.exists():
        return path.resolve()
    if str(path).startswith("/home/suhang/datasets/"):
        swapped = Path("/datasets") / str(path).removeprefix("/home/suhang/datasets/").lstrip("/")
        if swapped.exists():
            return swapped.resolve()
    if str(path).startswith("/datasets/"):
        swapped = Path("/home/suhang/datasets") / str(path).removeprefix("/datasets/").lstrip("/")
        if swapped.exists():
            return swapped.resolve()
    return path.resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML config to decode to dict, got {type(payload)} from {resolved}")
    base_refs: list[str] = []
    if "base_config" in payload:
        base_refs.append(str(payload.pop("base_config")))
    if "base_configs" in payload:
        raw = payload.pop("base_configs")
        if not isinstance(raw, list):
            raise TypeError(f"base_configs must be a list in {resolved}")
        base_refs.extend(str(v) for v in raw)
    merged: dict[str, Any] = {}
    for base_ref in base_refs:
        base_cfg = load_yaml_config(resolve_path(base_ref, base_dir=resolved.parent))
        merged = deep_update(merged, base_cfg)
    payload = deep_update(merged, payload)
    payload["_config_path"] = str(resolved)
    return payload


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_with_overrides(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    for raw in overrides or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override {raw!r}; expected key=value")
        key, value = raw.split("=", 1)
        parsed = yaml.safe_load(value)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise TypeError(f"Override path {key!r} collides with non-dict node")
        cursor[parts[-1]] = parsed
    return cfg


def maybe_override_from_args(cfg: dict[str, Any], args: argparse.Namespace, mapping: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for arg_name, path_parts in mapping.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        cursor = out
        for part in path_parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path_parts[-1]] = value
    return out


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_seconds(total_seconds: float) -> str:
    total = max(0, int(round(float(total_seconds))))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:02d}m{seconds:02d}s"
