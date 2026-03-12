from __future__ import annotations

from typing import Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class TactilePhysicalTransform:
    """Resize a tactile mask by physical FOV and center it on a shared canvas."""

    def __init__(self, ppmm: int = 40, max_scale_mm: float = 30.0):
        if int(ppmm) <= 0:
            raise ValueError(f"ppmm must be > 0, got {ppmm}")
        if float(max_scale_mm) <= 0:
            raise ValueError(f"max_scale_mm must be > 0, got {max_scale_mm}")

        self.ppmm = int(ppmm)
        self.max_scale_mm = float(max_scale_mm)
        self.canvas_size = int(self.max_scale_mm * self.ppmm)
        if self.canvas_size <= 0:
            raise ValueError(
                f"canvas_size must be > 0, got {self.canvas_size} from "
                f"max_scale_mm={self.max_scale_mm} ppmm={self.ppmm}"
            )

    def _to_tensor(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            tensor = TF.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            raise TypeError(f"image must be a PIL.Image or torch.Tensor, got {type(image)!r}")

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(
                f"tensor image must have shape (H,W) or (C,H,W), got {tuple(tensor.shape)}"
            )

        return TF.convert_image_dtype(tensor, torch.float32)

    def __call__(self, image: Union[Image.Image, torch.Tensor], current_scale_mm: float) -> torch.Tensor:
        current_scale_mm = float(current_scale_mm)
        if current_scale_mm <= 0:
            raise ValueError(f"current_scale_mm must be > 0, got {current_scale_mm}")
        if current_scale_mm > self.max_scale_mm:
            raise ValueError(
                f"current_scale_mm={current_scale_mm} exceeds max_scale_mm={self.max_scale_mm}"
            )

        tensor = self._to_tensor(image)

        target_pixel_size = int(current_scale_mm * self.ppmm)
        if target_pixel_size <= 0:
            raise ValueError(
                f"target_pixel_size must be > 0, got {target_pixel_size} from "
                f"current_scale_mm={current_scale_mm} ppmm={self.ppmm}"
            )
        if target_pixel_size > self.canvas_size:
            raise ValueError(
                f"target_pixel_size={target_pixel_size} exceeds canvas_size={self.canvas_size}"
            )

        resized = TF.resize(
            tensor,
            [target_pixel_size, target_pixel_size],
            antialias=True,
        )

        pad_total = self.canvas_size - target_pixel_size
        left = pad_total // 2
        right = pad_total - left
        top = pad_total // 2
        bottom = pad_total - top

        padded = TF.pad(
            resized,
            [left, top, right, bottom],
            fill=0.0,
        )

        if padded.ndim != 3 or padded.shape[1] != self.canvas_size or padded.shape[2] != self.canvas_size:
            raise RuntimeError(
                f"expected output shape (C,{self.canvas_size},{self.canvas_size}), got {tuple(padded.shape)}"
            )

        return padded
