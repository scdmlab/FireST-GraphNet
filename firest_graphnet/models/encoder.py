"""
encoder.py — Multi-source CNN encoder with optional CBAM.

Processes a single-frame feature map:
    Input:  (B, C_in, H, W)    — multi-spectral + optional weather channels
    Output: (B, C_out, H, W)   — dense feature map (same spatial resolution)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from firest_graphnet.models.cbam import CBAM


class ConvBlock(nn.Module):
    """Conv → Norm → GELU → (optional CBAM)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        use_cbam: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.act(self.norm(self.conv(x))))


class MultiSourceEncoder(nn.Module):
    """3–5 layer CNN encoder with optional CBAM on each block.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 8 spectral bands).
    hidden_channels : list[int]
        Channel sizes for successive conv blocks (defines depth).
    use_cbam : bool
        Attach CBAM after each conv block.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: List[int] | None = None,
        use_cbam: bool = True,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]

        layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.append(ConvBlock(ch_in, ch_out, use_cbam=use_cbam))
            ch_in = ch_out
        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) → (B, C_out, H, W)."""
        return self.encoder(x)
