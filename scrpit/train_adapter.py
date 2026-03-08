#!/usr/bin/env python3
"""
Training script for the Universal Scale Adapter (USA).

Reads adapter_dataset_ultimate/manifest.json with physically isolated Train/Val/Test splits:
  - Test: 4 indenters (pacman, wave, torus, hexagon) — zero-shot generalization
  - Train/Val: remaining indenters, 85% / 15% split

Pairing modes:
  - train: random (scale_A, scale_B) pairs
  - val/test: target fixed to 15mm, source from [18,20,22,25]

Usage:
    python scrpit/train_adapter.py --dataset adapter_dataset_ultimate
    python scrpit/train_adapter.py --sanity-check   # overfit 1 batch
    python scrpit/train_adapter.py --epochs 200 --lr 5e-5
"""

import argparse
import itertools
import json
import logging
import random
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

# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "adapter_dataset_ultimate"

# Indenters held out EXCLUSIVELY for test (zero-shot)
TEST_INDENTERS = frozenset({"pacman", "wave", "torus", "hexagon"})

# Anchor scale for val/test (stable metrics)
ANCHOR_SCALE_MM = 15
SOURCE_SCALES_MM = [18, 20, 22, 25]  # scales used as source when target=15mm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MultiscaleTactileDataset — adapter_dataset_ultimate format
# ---------------------------------------------------------------------------

class MultiscaleTactileDataset(Dataset):
    """
    Yields (img_context, coord_context, img_query, coord_query) for USA training.

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
        self.source_scales = [s for s in self.scales_mm if s != ANCHOR_SCALE_MM]
        if not self.source_scales:
            self.source_scales = [s for s in SOURCE_SCALES_MM if s in self.scales_mm]

        self._len = pairs_per_epoch if pairs_per_epoch else len(episode_ids)
        self._rng = random.Random(seed)
        self._coord_cache: dict[Path, torch.Tensor] = {}
        self._warned_no_common: set[tuple[int, str, str]] = set()
        self._log_samples_done = 0
        self._max_sample_logs = 8

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
        # scale_15mm -> 15
        head = scale_key.removeprefix("scale_")
        value = head.removesuffix("mm")
        return int(value)

    @staticmethod
    def _should_log_from_worker() -> bool:
        info = get_worker_info()
        return info is None or info.id == 0

    def _choose_scale_pair(self, ep_id: int, scales: dict) -> tuple[str, str]:
        if self.mode == "train":
            scale_context, scale_query = self._rng.choice(self.scale_pairs)
            key_context = f"scale_{scale_context}mm"
            key_query = f"scale_{scale_query}mm"
        else:
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

    def _choose_marker_pair(self, ep_id: int, key_context: str, key_query: str, meta_context: dict, meta_query: dict) -> tuple[str, str]:
        markers_context = sorted(meta_context.get("rendered_markers", []))
        markers_query = sorted(meta_query.get("rendered_markers", []))
        if not markers_context or not markers_query:
            raise RuntimeError(
                f"Missing rendered_markers for episode {ep_id}: "
                f"{key_context}={len(markers_context)} {key_query}={len(markers_query)}"
            )

        common = sorted(set(markers_context).intersection(markers_query))
        if common:
            marker_name = self._rng.choice(common)
            return marker_name, marker_name

        warn_key = (ep_id, key_context, key_query)
        if warn_key not in self._warned_no_common:
            self._warned_no_common.add(warn_key)
            log.warning(
                "No common marker file for ep=%06d (%s,%s). Using fallback context=%s query=%s",
                ep_id,
                key_context,
                key_query,
                markers_context[0],
                markers_query[0],
            )
        return markers_context[0], markers_query[0]

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
            if self._should_log_from_worker():
                log.info("Loaded coord map: %s shape=%s flattened=%s", abs_path, tuple(arr.shape), tuple(coords.shape))

        return self._coord_cache[abs_path]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_id = self._rng.choice(self.episode_ids)
        meta = self.episode_meta[ep_id]
        scales = meta.get("scales", {})
        episode_path = meta["__episode_path"]
        episode_dir = self.dataset_root / episode_path

        key_context, key_query = self._choose_scale_pair(ep_id, scales)
        scale_meta_context = scales[key_context]
        scale_meta_query = scales[key_query]

        marker_context, marker_query = self._choose_marker_pair(
            ep_id,
            key_context,
            key_query,
            scale_meta_context,
            scale_meta_query,
        )

        img_path_context = episode_dir / key_context / marker_context
        img_path_query = episode_dir / key_query / marker_query
        if not img_path_context.exists():
            raise FileNotFoundError(f"Missing context image: {img_path_context}")
        if not img_path_query.exists():
            raise FileNotFoundError(f"Missing query image: {img_path_query}")

        img_context = self.transform(Image.open(img_path_context).convert("RGB"))
        img_query = self.transform(Image.open(img_path_query).convert("RGB"))

        coord_context = self._load_coord_map(episode_dir, ep_id, key_context, scale_meta_context)
        coord_query = self._load_coord_map(episode_dir, ep_id, key_query, scale_meta_query)

        if self._should_log_from_worker() and self._log_samples_done < self._max_sample_logs:
            self._log_samples_done += 1
            log.info(
                "Sample ep=%06d pair=(%s -> %s) markers=(%s, %s) coord_shapes=(%s, %s)",
                ep_id,
                key_context,
                key_query,
                marker_context,
                marker_query,
                tuple(coord_context.shape),
                tuple(coord_query.shape),
            )

        return img_context, coord_context, img_query, coord_query


def load_manifest_and_split(
    manifest_path: Path,
    dataset_root: Path,
    test_indenters: frozenset[str],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], dict[int, dict], dict]:
    """
    Load manifest, split by indenter. Load full metadata (incl. scales) from each episode.
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

    train_ids = []
    val_ids = []
    test_ids = []

    rng = random.Random(seed)
    for ind, ids in by_indenter.items():
        rng.shuffle(ids)
        if ind in test_indenters:
            test_ids.extend(ids)
        else:
            n = len(ids)
            n_train = max(1, int(n * train_ratio))
            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train:])

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    dataset_root = Path(args.dataset)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    scales_mm = args.scales
    train_ids, val_ids, test_ids, episode_meta, manifest_info = load_manifest_and_split(
        manifest_path,
        dataset_root=dataset_root,
        test_indenters=TEST_INDENTERS,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    log.info("Split: train=%d val=%d test=%d (test indenters: %s)",
             len(train_ids), len(val_ids), len(test_ids), sorted(TEST_INDENTERS))

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
        markers_context = sorted(meta["scales"][key_context].get("rendered_markers", []))
        markers_query = sorted(meta["scales"][key_query].get("rendered_markers", []))
        common = sorted(set(markers_context).intersection(markers_query))
        chosen = common[0] if common else (markers_context[0] if markers_context else "(none)")
        log.info(
            "Example pair: ep=%06d path=%s pair=(%s -> %s) marker=%s",
            eid,
            meta.get("__episode_path", f"episode_{eid:06d}"),
            key_context,
            key_query,
            chosen,
        )

    # ---- Sanity check: single fixed batch ----
    dataset_common_kwargs = dict(
        dataset_root=dataset_root,
        episode_meta=episode_meta,
        scales_mm=scales_mm,
        expected_token_count=backbone_tokens,
    )

    # ---- Sanity check: single fixed batch ----
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
        # Force exactly 1 batch
        single_batch = next(iter(train_loader))
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

    # ---- Models ----
    adapter = UniversalScaleAdapter(
        embed_dim=vit.embed_dim, num_heads=8, num_layers=2
    ).to(device)

    log.info("USA parameters: %d",
             sum(p.numel() for p in adapter.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Warmup + Cosine annealing (step-based)
    import math
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

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "usa_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_cos = -1.0
    step = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        epoch_cos = 0.0
        n_batches = 0

        if args.sanity_check:
            img_context, coord_context, img_query, coord_query = [x.to(device) for x in single_batch]
            batches = [(img_context, coord_context, img_query, coord_query)] * 1
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
                img_context, coord_context, img_query, coord_query = [x.to(device) for x in batch]
            else:
                img_context, coord_context, img_query, coord_query = batch

            with torch.no_grad():
                feat_context = vit(img_context)
                feat_query = vit(img_query)

            pred_query = adapter(
                context_feat=feat_context,
                context_coord_map_mm=coord_context,
                query_coord_map_mm=coord_query,
            )
            loss_context_to_query = F.mse_loss(pred_query, feat_query)

            pred_context = adapter(
                context_feat=feat_query,
                context_coord_map_mm=coord_query,
                query_coord_map_mm=coord_context,
            )
            loss_query_to_context = F.mse_loss(pred_context, feat_context)

            loss = loss_context_to_query + loss_query_to_context
            cos = cosine_similarity(pred_query, feat_query).item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cos += cos
            n_batches += 1
            step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{cos:.4f}")

            if not args.sanity_check and n_batches % max(1, args.log_every * 10) == 0:
                log.info("  [%d/%d] loss=%.6f  cos=%.4f", n_batches, batches_per_epoch, loss.item(), cos)

            if not args.sanity_check:
                scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_train_cos = epoch_cos / max(n_batches, 1)

        # Validation
        if val_loader and not args.sanity_check:
            adapter.eval()
            val_loss = 0.0
            val_cos = 0.0
            val_n = 0
            with torch.no_grad():
                for img_context, coord_context, img_query, coord_query in tqdm(
                    val_loader, desc="Val", leave=False, disable=not args.progress
                ):
                    img_context, coord_context, img_query, coord_query = (
                        img_context.to(device), coord_context.to(device),
                        img_query.to(device), coord_query.to(device),
                    )
                    feat_context = vit(img_context)
                    feat_query = vit(img_query)
                    pred_query = adapter(
                        context_feat=feat_context,
                        context_coord_map_mm=coord_context,
                        query_coord_map_mm=coord_query,
                    )
                    pred_context = adapter(
                        context_feat=feat_query,
                        context_coord_map_mm=coord_query,
                        query_coord_map_mm=coord_context,
                    )
                    loss = F.mse_loss(pred_query, feat_query) + F.mse_loss(pred_context, feat_context)
                    cos = cosine_similarity(pred_query, feat_query)
                    val_loss += loss.item() * img_context.size(0)
                    val_cos += cos.item() * img_context.size(0)
                    val_n += img_context.size(0)
            val_loss /= max(val_n, 1)
            val_cos /= max(val_n, 1)
            adapter.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch, "model_state_dict": adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss, "val_loss": val_loss, "val_cos": val_cos,
                }, ckpt_dir / "best.pt")
            if val_cos > best_val_cos:
                best_val_cos = val_cos

            if epoch % args.log_every == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info("Epoch %3d/%d  train_loss=%.6f  train_cos=%.4f  val_loss=%.6f  val_cos=%.4f  lr=%.2e",
                         epoch, args.epochs, avg_loss, avg_train_cos, val_loss, val_cos, lr_now)
        else:
            if epoch % args.log_every == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info("Epoch %3d/%d  loss=%.6f  cos=%.4f  lr=%.2e", epoch, args.epochs, avg_loss, avg_train_cos, lr_now)

            if args.sanity_check and avg_loss < 0.001:
                log.info("Sanity check PASSED: loss -> 0")
                break

            if avg_loss < best_val_loss and args.sanity_check:
                best_val_loss = avg_loss
                torch.save({
                    "epoch": epoch, "model_state_dict": adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                }, ckpt_dir / "best.pt")

        if epoch % args.save_every == 0 and not args.sanity_check:
            torch.save({
                "epoch": epoch, "model_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")

    log.info("Training finished. Best val_loss: %.6f  Best val_cos: %.4f",
             best_val_loss, best_val_cos)
    log.info("Checkpoints: %s", ckpt_dir)

    # Optional: evaluate on held-out test set (zero-shot indenters)
    if args.eval_test and test_ids:
        ckpt_path = ckpt_dir / "best.pt"
        if ckpt_path.exists():
            log.info("Evaluating on TEST set (unseen indenters: %s)", sorted(TEST_INDENTERS))
            ckpt = torch.load(ckpt_path, map_location=device)
            adapter.load_state_dict(ckpt["model_state_dict"])
            adapter.eval()
            test_ds = MultiscaleTactileDataset(
                episode_ids=test_ids,
                mode="test",
                augment=False,
                pairs_per_epoch=min(500, len(test_ids) * 2),
                seed=args.seed + 2,
                **dataset_common_kwargs,
            )
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            test_loss = 0.0
            test_cos = 0.0
            test_n = 0
            with torch.no_grad():
                for img_context, coord_context, img_query, coord_query in tqdm(test_loader, desc="Test"):
                    img_context, coord_context = img_context.to(device), coord_context.to(device)
                    img_query, coord_query = img_query.to(device), coord_query.to(device)
                    feat_context, feat_query = vit(img_context), vit(img_query)
                    pred_query = adapter(
                        context_feat=feat_context,
                        context_coord_map_mm=coord_context,
                        query_coord_map_mm=coord_query,
                    )
                    pred_context = adapter(
                        context_feat=feat_query,
                        context_coord_map_mm=coord_query,
                        query_coord_map_mm=coord_context,
                    )
                    loss = F.mse_loss(pred_query, feat_query) + F.mse_loss(pred_context, feat_context)
                    cos = cosine_similarity(pred_query, feat_query)
                    test_loss += loss.item() * img_context.size(0)
                    test_cos += cos.item() * img_context.size(0)
                    test_n += img_context.size(0)
            test_loss /= max(test_n, 1)
            test_cos /= max(test_n, 1)
            log.info("TEST (zero-shot indenters): loss=%.6f  cos_sim=%.4f",
                     test_loss, test_cos)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train USA adapter (adapter_dataset_ultimate)")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                   help="Path to adapter_dataset_ultimate")
    p.add_argument("--scales", nargs="+", type=int, default=[15, 18, 20, 22, 25])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--train_ratio", type=float, default=0.85,
                   help="Train ratio for non-test indenters (val = 1 - train_ratio)")
    p.add_argument("--pairs_per_epoch", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=20)
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
