"""
cbam.py — Convolutional Block Attention Module (CBAM).

Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
Provides channel attention + spatial attention that can be optionally
attached to any conv block in the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → channel attention weights (B, C, 1, 1)."""
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=(2, 3))  # (B, C)
        max_pool = x.amax(dim=(2, 3))  # (B, C)
        attn = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool))
        return attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise pooling + conv."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → spatial attention map (B, 1, H, W)."""
        avg_out = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)        # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return torch.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """Full CBAM block: channel attention → spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x
