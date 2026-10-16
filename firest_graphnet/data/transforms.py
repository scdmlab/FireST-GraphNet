"""
transforms.py — Data augmentation and normalisation for fire progression data.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


class Compose:
    """Chain multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            sample["x_seq"] = torch.flip(sample["x_seq"], dims=[-1])
            sample["y"] = torch.flip(sample["y"], dims=[-1])
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            sample["x_seq"] = torch.flip(sample["x_seq"], dims=[-2])
            sample["y"] = torch.flip(sample["y"], dims=[-2])
        return sample


class RandomRotate90:
    """Random 90-degree rotation (k ∈ {0, 1, 2, 3})."""

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            sample["x_seq"] = torch.rot90(sample["x_seq"], k, dims=[-2, -1])
            sample["y"] = torch.rot90(sample["y"], k, dims=[-2, -1])
        return sample


class Normalize:
    """Per-band  z-score normalisation using provided mean / std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # mean, std: shape (C,)
        self.mean = mean.view(1, -1, 1, 1)  # (1, C, 1, 1) for broadcasting
        self.std = std.view(1, -1, 1, 1)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x_seq: (T, C, H, W)
        sample["x_seq"] = (sample["x_seq"] - self.mean) / (self.std + 1e-8)
        return sample


def get_train_transforms() -> Compose:
    return Compose([RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotate90()])


def get_val_transforms() -> Optional[Compose]:
    return None  # no augmentation at val time
