from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldLatticeConfig:
    lattice_size: int = 32
    x_min_mm: float = -15.0
    x_max_mm: float = 15.0
    y_min_mm: float = -15.0
    y_max_mm: float = 15.0
    eps: float = 1e-6


class WorldLatticeProjector(nn.Module):
    """Deterministic mm-space splat/gather between patch grids and a world lattice."""

    def __init__(self, config: WorldLatticeConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorldLatticeConfig()

    @property
    def lattice_size(self) -> int:
        return int(self.config.lattice_size)

    def _coord_to_index(self, coord_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        gx = (coord_map[..., 0] - float(cfg.x_min_mm)) / max(float(cfg.x_max_mm - cfg.x_min_mm), 1e-6) * (self.lattice_size - 1)
        gy = (coord_map[..., 1] - float(cfg.y_min_mm)) / max(float(cfg.y_max_mm - cfg.y_min_mm), 1e-6) * (self.lattice_size - 1)
        return gx, gy

    def lattice_normalized_to_mm(self, x_norm: torch.Tensor, y_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        x_mm = 0.5 * (x_norm + 1.0) * float(cfg.x_max_mm - cfg.x_min_mm) + float(cfg.x_min_mm)
        y_mm = 0.5 * (y_norm + 1.0) * float(cfg.y_max_mm - cfg.y_min_mm) + float(cfg.y_min_mm)
        return x_mm, y_mm

    def splat_to_world_lattice(
        self,
        patch_features: torch.Tensor,
        coord_map: torch.Tensor,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_features.dim() != 5:
            raise ValueError(f"Expected patch_features shape (B,T,D,H,W), got {tuple(patch_features.shape)}")
        if coord_map.dim() != 4 or coord_map.shape[-1] != 2:
            raise ValueError(f"Expected coord_map shape (B,H,W,2), got {tuple(coord_map.shape)}")
        b, t, d, hp, wp = patch_features.shape
        if coord_map.shape[1] != hp or coord_map.shape[2] != wp:
            raise ValueError(
                f"Feature grid / coord map mismatch: features {(hp, wp)} vs coord_map {tuple(coord_map.shape[1:3])}"
            )
        coords = coord_map
        if world_origin_xy_mm is not None:
            coords = coords + world_origin_xy_mm[:, None, None, :]
        if absolute_contact_xy_mm is not None:
            coords = coords + absolute_contact_xy_mm[:, None, None, :]
        gx, gy = self._coord_to_index(coords)
        x0 = torch.floor(gx)
        y0 = torch.floor(gy)
        x1 = x0 + 1
        y1 = y0 + 1
        wx1 = gx - x0
        wy1 = gy - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        neighbors = (
            (x0, y0, wx0 * wy0),
            (x1, y0, wx1 * wy0),
            (x0, y1, wx0 * wy1),
            (x1, y1, wx1 * wy1),
        )

        features = patch_features.reshape(b, t, d, hp * wp)
        world = patch_features.new_zeros((b * t, d, self.lattice_size * self.lattice_size))
        weights = patch_features.new_zeros((b * t, 1, self.lattice_size * self.lattice_size))
        for nx, ny, w in neighbors:
            valid = (nx >= 0) & (nx < self.lattice_size) & (ny >= 0) & (ny < self.lattice_size)
            idx = (ny.clamp(0, self.lattice_size - 1) * self.lattice_size + nx.clamp(0, self.lattice_size - 1)).to(torch.long).reshape(b, hp * wp)
            idx_bt = idx.unsqueeze(1).expand(b, t, hp * wp).reshape(b * t, 1, hp * wp)
            valid_bt = valid.reshape(b, hp * wp).unsqueeze(1).expand(b, t, hp * wp).reshape(b * t, 1, hp * wp).to(patch_features.dtype)
            w_bt = w.reshape(b, hp * wp).unsqueeze(1).expand(b, t, hp * wp).reshape(b * t, 1, hp * wp).to(patch_features.dtype) * valid_bt
            idx_expand = idx_bt.expand(-1, d, -1)
            src = features.reshape(b * t, d, hp * wp) * w_bt
            world.scatter_add_(2, idx_expand, src)
            weights.scatter_add_(2, idx_bt, w_bt)

        world = world / weights.clamp_min(float(self.config.eps))
        world = world.reshape(b, t, d, self.lattice_size, self.lattice_size)
        weights = weights.reshape(b, t, 1, self.lattice_size, self.lattice_size)
        return world, weights

    def gather_from_world_lattice(
        self,
        world_features: torch.Tensor,
        coord_map: torch.Tensor,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if world_features.dim() != 5:
            raise ValueError(f"Expected world_features shape (B,T,D,K,K), got {tuple(world_features.shape)}")
        coords = coord_map
        if world_origin_xy_mm is not None:
            coords = coords + world_origin_xy_mm[:, None, None, :]
        if absolute_contact_xy_mm is not None:
            coords = coords + absolute_contact_xy_mm[:, None, None, :]
        gx, gy = self._coord_to_index(coords)
        grid_x = 2.0 * gx / max(self.lattice_size - 1, 1) - 1.0
        grid_y = 2.0 * gy / max(self.lattice_size - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        b, t, d, k, _ = world_features.shape
        hp, wp = coord_map.shape[1], coord_map.shape[2]
        world_bt = world_features.reshape(b * t, d, k, k)
        grid_bt = grid.unsqueeze(1).expand(b, t, hp, wp, 2).reshape(b * t, hp, wp, 2)
        gathered = F.grid_sample(world_bt, grid_bt, mode="bilinear", padding_mode="zeros", align_corners=True)
        occupancy = F.grid_sample(
            torch.ones((b * t, 1, k, k), device=world_features.device, dtype=world_features.dtype),
            grid_bt,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return gathered.reshape(b, t, d, hp, wp), occupancy.reshape(b, t, 1, hp, wp)

    def forward(self, patch_features: torch.Tensor, coord_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.splat_to_world_lattice(patch_features, coord_map)
