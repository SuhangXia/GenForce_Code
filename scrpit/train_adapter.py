#!/usr/bin/env python3
"""
Training script for the Universal Scale Adapter (USA).

Reads adapter_dataset_v2/manifest.json with physically isolated Train/Val/Test splits:
  - Test: 4 indenters (pacman, wave, torus, hexagon) — zero-shot generalization
  - Train/Val: remaining indenters, 85% / 15% split

Pairing modes:
  - train: random (scale_A, scale_B) pairs
  - val/test: target fixed to 15mm, source from [18,20,22,25]

Usage:
    python scrpit/train_adapter.py --dataset adapter_dataset_v2
    python scrpit/train_adapter.py --sanity-check   # overfit 1 batch
    python scrpit/train_adapter.py --epochs 200 --lr 5e-5
"""

import argparse
import itertools
import json
import logging
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from usa_adapter import UniversalScaleAdapter

# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "adapter_dataset_v2"

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
# Coordinate grid (16x16 → 196 tokens)
# ---------------------------------------------------------------------------

def _generate_coord_grid(
    width_mm: float, height_mm: float,
    grid_h: int = 14, grid_w: int = 14,
) -> torch.Tensor:
    """Create (grid_h*grid_w, 2) tensor of physical (x_mm, y_mm) coordinates."""
    xs = torch.linspace(0, width_mm, grid_w)
    ys = torch.linspace(0, height_mm, grid_h)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1)
    return coords.reshape(-1, 2)


# ---------------------------------------------------------------------------
# MultiscaleTactileDataset — adapter_dataset_v2 format
# ---------------------------------------------------------------------------

class MultiscaleTactileDataset(Dataset):
    """
    Yields (img_A, coords_A, img_B, coords_B) for USA training.

    Physically isolated splits by indenter:
      - test: episodes with indenter in TEST_INDENTERS
      - train/val: remaining episodes, 85% / 15%

    Pairing (mode):
      - train: random (scale_A, scale_B) with A != B
      - val/test: target B = 15mm, source A = random from [18,20,22,25]
    """

    def __init__(
        self,
        dataset_root: Path,
        episode_ids: list[int],
        episode_meta: dict[int, dict],
        scales_mm: list[int],
        mode: str,
        grid_h: int = 14,
        grid_w: int = 14,
        augment: bool = True,
        pairs_per_epoch: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.episode_ids = episode_ids
        self.episode_meta = episode_meta
        self.scales_mm = sorted(scales_mm)
        self.mode = mode
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.augment = augment
        self.seed = seed

        # Precompute coord grids per scale (all scales are square)
        self.scale_coords: dict[int, torch.Tensor] = {}
        for s in self.scales_mm:
            self.scale_coords[s] = _generate_coord_grid(s, s, grid_h, grid_w)

        # Scale pairs for train: all (A,B) with A != B
        self.scale_pairs = [(a, b) for a, b in itertools.permutations(self.scales_mm, 2)]

        # Val/test: source scales (excluding 15mm anchor)
        self.source_scales = [s for s in self.scales_mm if s != ANCHOR_SCALE_MM]
        if not self.source_scales:
            self.source_scales = [s for s in SOURCE_SCALES_MM if s in self.scales_mm]

        self._len = pairs_per_epoch if pairs_per_epoch else len(episode_ids)
        self._rng = random.Random(seed)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_id = self._rng.choice(self.episode_ids)
        meta = self.episode_meta[ep_id]
        scales = meta.get("scales", {})
        ep_dir = meta.get("dir", f"episode_{ep_id:06d}")

        if self.mode == "train":
            scale_a, scale_b = self._rng.choice(self.scale_pairs)
        else:
            # val/test: target B = 15mm, source A = random from source_scales
            scale_b = ANCHOR_SCALE_MM
            scale_a = self._rng.choice(self.source_scales)
            if scale_a == scale_b:
                scale_a = self._rng.choice([s for s in self.source_scales if s != scale_b])

        key_a = f"scale_{scale_a}mm"
        key_b = f"scale_{scale_b}mm"
        if key_a not in scales or key_b not in scales:
            # Fallback: pick any valid pair
            valid = [k for k in scales if k.startswith("scale_")]
            if len(valid) < 2:
                raise RuntimeError(f"Episode {ep_id} has < 2 scales: {list(scales)}")
            key_a, key_b = self._rng.sample(valid, 2)

        rel_a = scales[key_a]["image"]
        rel_b = scales[key_b]["image"]
        path_a = self.dataset_root / rel_a
        path_b = self.dataset_root / rel_b

        scale_a = scales[key_a]["physical_width_mm"]
        scale_b = scales[key_b]["physical_height_mm"]

        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        coords_a = self.scale_coords[scale_a]
        coords_b = self.scale_coords[scale_b]

        return img_a, coords_a, img_b, coords_b


def load_manifest_and_split(
    manifest_path: Path,
    dataset_root: Path,
    test_indenters: frozenset[str],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], dict[int, dict]]:
    """
    Load manifest, split by indenter. Load full metadata (incl. scales) from each episode.
    Returns (train_ids, val_ids, test_ids, episode_meta).
    """
    with open(manifest_path) as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    episode_meta: dict[int, dict] = {}
    for ep in episodes:
        eid = ep["episode_id"]
        ep_dir = ep.get("dir", f"episode_{eid:06d}")
        meta_path = Path(dataset_root) / ep_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as mf:
                full_meta = json.load(mf)
            episode_meta[eid] = full_meta
        else:
            episode_meta[eid] = ep

    by_indenter: dict[str, list[int]] = {}
    for ep in episodes:
        ind = ep.get("indenter", "unknown")
        by_indenter.setdefault(ind, []).append(ep["episode_id"])

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

    return train_ids, val_ids, test_ids, episode_meta


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
    train_ids, val_ids, test_ids, episode_meta = load_manifest_and_split(
        manifest_path,
        dataset_root=dataset_root,
        test_indenters=TEST_INDENTERS,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    log.info("Split: train=%d val=%d test=%d (test indenters: %s)",
             len(train_ids), len(val_ids), len(test_ids), sorted(TEST_INDENTERS))

    # ---- Sanity check: single fixed batch ----
    if args.sanity_check:
        log.info("SANITY CHECK: overfitting 1 batch for %d epochs", args.epochs)
        train_ds = MultiscaleTactileDataset(
            dataset_root=dataset_root,
            episode_ids=train_ids[: max(1, args.batch_size)],
            episode_meta=episode_meta,
            scales_mm=scales_mm,
            mode="train",
            augment=False,
            pairs_per_epoch=args.batch_size,
            seed=args.seed,
        )
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
            dataset_root=dataset_root,
            episode_ids=train_ids,
            episode_meta=episode_meta,
            scales_mm=scales_mm,
            mode="train",
            augment=True,
            pairs_per_epoch=args.pairs_per_epoch,
            seed=args.seed,
        )
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
                dataset_root=dataset_root,
                episode_ids=val_ids,
                episode_meta=episode_meta,
                scales_mm=scales_mm,
                mode="val",
                augment=False,
                pairs_per_epoch=min(500, len(val_ids) * 2),
                seed=args.seed + 1,
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
    vit = FrozenViTFeatureExtractor(device=device)
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
            img_a, coords_a, img_b, coords_b = [x.to(device) for x in single_batch]
            batches = [(img_a, coords_a, img_b, coords_b)] * 1
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
                img_a, coords_a, img_b, coords_b = [x.to(device) for x in batch]
            else:
                img_a, coords_a, img_b, coords_b = batch

            with torch.no_grad():
                feat_a = vit(img_a)
                feat_b = vit(img_b)

            pred_b = adapter(source_feat=feat_a, source_coords=coords_a, target_coords=coords_b)
            loss_a2b = F.mse_loss(pred_b, feat_b)

            pred_a = adapter(source_feat=feat_b, source_coords=coords_b, target_coords=coords_a)
            loss_b2a = F.mse_loss(pred_a, feat_a)

            loss = loss_a2b + loss_b2a
            cos = cosine_similarity(pred_b, feat_b).item()

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
                for img_a, coords_a, img_b, coords_b in tqdm(
                    val_loader, desc="Val", leave=False, disable=not args.progress
                ):
                    img_a, coords_a, img_b, coords_b = (
                        img_a.to(device), coords_a.to(device),
                        img_b.to(device), coords_b.to(device),
                    )
                    feat_a = vit(img_a)
                    feat_b = vit(img_b)
                    pred_b = adapter(feat_a, coords_a, coords_b)
                    pred_a = adapter(feat_b, coords_b, coords_a)
                    loss = F.mse_loss(pred_b, feat_b) + F.mse_loss(pred_a, feat_a)
                    cos = cosine_similarity(pred_b, feat_b)
                    val_loss += loss.item() * img_a.size(0)
                    val_cos += cos.item() * img_a.size(0)
                    val_n += img_a.size(0)
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
                dataset_root=dataset_root,
                episode_ids=test_ids,
                episode_meta=episode_meta,
                scales_mm=scales_mm,
                mode="test",
                augment=False,
                pairs_per_epoch=min(500, len(test_ids) * 2),
                seed=args.seed + 2,
            )
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            test_loss = 0.0
            test_cos = 0.0
            test_n = 0
            with torch.no_grad():
                for img_a, coords_a, img_b, coords_b in tqdm(test_loader, desc="Test"):
                    img_a, coords_a = img_a.to(device), coords_a.to(device)
                    img_b, coords_b = img_b.to(device), coords_b.to(device)
                    feat_a, feat_b = vit(img_a), vit(img_b)
                    pred_b = adapter(feat_a, coords_a, coords_b)
                    pred_a = adapter(feat_b, coords_b, coords_a)
                    loss = F.mse_loss(pred_b, feat_b) + F.mse_loss(pred_a, feat_a)
                    cos = cosine_similarity(pred_b, feat_b)
                    test_loss += loss.item() * img_a.size(0)
                    test_cos += cos.item() * img_a.size(0)
                    test_n += img_a.size(0)
            test_loss /= max(test_n, 1)
            test_cos /= max(test_n, 1)
            log.info("TEST (zero-shot indenters): loss=%.6f  cos_sim=%.4f",
                     test_loss, test_cos)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train USA adapter (adapter_dataset_v2)")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                   help="Path to adapter_dataset_v2")
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
