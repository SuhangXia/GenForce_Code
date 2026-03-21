from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StaticPoseRegressor(nn.Module):
    """Frozen ViT + optional USA + spatial regression head for static tactile pose prediction."""

    def __init__(
        self,
        vit_backbone: nn.Module,
        usa_plugin: Optional[nn.Module] = None,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.vit_backbone = vit_backbone
        self.usa_plugin = usa_plugin
        self.out_dim = int(out_dim)
        self.patch_grid = (14, 14)
        self.embed_dim = 768
        self.flatten_dim = 128 * 7 * 7

        self._freeze_module(self.vit_backbone)
        if self.usa_plugin is not None:
            self._freeze_module(self.usa_plugin)

        self.spatial_reducer = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),
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

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> 'StaticPoseRegressor':
        super().train(mode)
        self.vit_backbone.eval()
        if self.usa_plugin is not None:
            self.usa_plugin.eval()
        return self

    def _reshape_to_grid(self, feats: torch.Tensor, module_name: str) -> torch.Tensor:
        if feats.dim() == 4:
            expected = (*self.patch_grid, self.embed_dim)
            if tuple(feats.shape[1:]) != expected:
                raise ValueError(
                    f'{module_name} returned {tuple(feats.shape)}; expected (B, {expected[0]}, {expected[1]}, {expected[2]})'
                )
            return feats.contiguous()

        if feats.dim() != 3:
            raise ValueError(
                f'{module_name} must return (B, N, D) or (B, H, W, D), got {tuple(feats.shape)}'
            )

        batch_size, num_tokens, embed_dim = feats.shape
        expected_tokens = self.patch_grid[0] * self.patch_grid[1]
        if num_tokens != expected_tokens or embed_dim != self.embed_dim:
            raise ValueError(
                f'{module_name} returned {tuple(feats.shape)}; expected (B, {expected_tokens}, {self.embed_dim})'
            )
        return feats.reshape(batch_size, self.patch_grid[0], self.patch_grid[1], self.embed_dim).contiguous()

    def forward(
        self,
        src_imgs: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
        source_coords: Optional[torch.Tensor] = None,
        target_scale: Optional[torch.Tensor] = None,
        source_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src_imgs:      (B, 3, 224, 224)
            target_coords: Optional USA query coordinate map
            source_coords: Optional USA context coordinate map
            target_scale:  Optional USA query scale
            source_scale:  Optional USA context scale

        Returns:
            Pose predictions of shape (B, out_dim)
        """
        if src_imgs.dim() != 4:
            raise ValueError(f'src_imgs must be (B, 3, 224, 224), got {tuple(src_imgs.shape)}')
        if src_imgs.shape[1] != 3:
            raise ValueError(f'src_imgs channel dim must be 3, got {src_imgs.shape[1]}')

        with torch.no_grad():
            vit_feats = self.vit_backbone(src_imgs)
            vit_grid = self._reshape_to_grid(vit_feats, 'vit_backbone')

            if self.usa_plugin is not None:
                if source_coords is None or target_coords is None:
                    raise ValueError('source_coords and target_coords are required when usa_plugin is provided')
                adapted_feats = self.usa_plugin(
                    context_feat=vit_grid,
                    context_coord_map_mm=source_coords,
                    query_coord_map_mm=target_coords,
                    context_scale_mm=source_scale,
                    query_scale_mm=target_scale,
                )
                vit_grid = self._reshape_to_grid(adapted_feats, 'usa_plugin')

        spatial_feats = vit_grid.permute(0, 3, 1, 2).contiguous()
        reduced_feats = self.spatial_reducer(spatial_feats)
        flat_feats = reduced_feats.reshape(reduced_feats.shape[0], self.flatten_dim).contiguous()
        preds = self.reg_layer(flat_feats)
        return preds


if __name__ == '__main__':
    class DummyViT(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.randn(batch_size, 14 * 14, 768, device=x.device, dtype=x.dtype)

    batch_size = 2
    model = StaticPoseRegressor(vit_backbone=DummyViT(), usa_plugin=None, out_dim=3)
    dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    outputs = model(dummy_imgs)
    print(outputs.shape)
