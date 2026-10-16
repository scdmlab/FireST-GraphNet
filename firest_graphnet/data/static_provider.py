"""
static_provider.py  --  Provider for static environmental features.

Provides per-pixel static features (land cover, elevation, etc.) that
remain constant across time but vary spatially.  These are concatenated
with the per-timestep fire + weather channels in the encoder.

Providers
─────────
  1. LULCStaticProvider        –  loads ESRI LULC from preprocessed npy
  2. PlaceholderStaticProvider  –  returns zeros (graceful fallback)

All providers expose:
    .channels   : int                     number of static feature channels
    .get()      → ndarray (C_s, H, W)    static feature map
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════════
class StaticProviderBase:
    """Abstract static feature provider."""

    def __init__(self, channels: int = 1, spatial_size: Tuple[int, int] = (256, 256)):
        self.channels = channels
        self.spatial_size = spatial_size

    def get(self, sample_idx: int = 0) -> np.ndarray:
        """Return static feature map.  shape (C_s, H, W)."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
#  1. LULCStaticProvider  —  ESRI Land-Use / Land-Cover from npy
# ═══════════════════════════════════════════════════════════════════════════
class LULCStaticProvider(StaticProviderBase):
    """Loads ESRI Land-Use / Land-Cover from a preprocessed npy file.

    Expected npy shape: (N, C_lulc, 1, H, W)  or  (N, C_lulc, H, W)
    where N = number of samples (aligned with af_*_img indices),
    C_lulc = land cover channels (typically 1 — class ID normalised to [0,1]).
    """

    def __init__(self, npy_path: str | Path, spatial_size: Tuple[int, int] = (256, 256)):
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"LULC npy not found: {npy_path}")
        self._data = np.load(str(npy_path), mmap_mode="r")
        # Handle both (N, C, 1, H, W) and (N, C, H, W)
        if self._data.ndim == 5:
            self._squeeze_t = True
            N, C = self._data.shape[0], self._data.shape[1]
        else:
            self._squeeze_t = False
            N, C = self._data.shape[0], self._data.shape[1]
        super().__init__(channels=C, spatial_size=spatial_size)
        self._N = N
        logger.info(f"LULCStaticProvider: {N} samples, {C} channels from {npy_path.name}")

    def get(self, sample_idx: int = 0) -> np.ndarray:
        """Return LULC features for a sample.  shape (C, H, W)."""
        idx = sample_idx % self._N
        if self._squeeze_t:
            return np.array(self._data[idx, :, 0, :, :])  # (C, H, W)
        return np.array(self._data[idx, :, :, :])  # (C, H, W)


# ═══════════════════════════════════════════════════════════════════════════
#  2. PlaceholderStaticProvider  —  zeros (graceful fallback)
# ═══════════════════════════════════════════════════════════════════════════
class PlaceholderStaticProvider(StaticProviderBase):
    """Fallback: returns zero static features.  Training never crashes."""

    _warned = False

    def get(self, sample_idx: int = 0) -> np.ndarray:
        if not PlaceholderStaticProvider._warned:
            logger.warning(
                "PlaceholderStaticProvider: returning zeros for static features. "
                "Re-run preprocessing with raw TS-SatFire data to generate LULC features."
            )
            PlaceholderStaticProvider._warned = True
        return np.zeros((self.channels, *self.spatial_size), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════
def build_static_provider(
    data_root: str | Path | None = None,
    channels: int = 1,
    spatial_size: Tuple[int, int] = (256, 256),
) -> StaticProviderBase:
    """Create a static feature provider, auto-detecting LULC npy files.

    Looks for ``lulc_*.npy`` or ``*_lulc_*.npy`` in dataset dirs.
    If not found, returns placeholder (zeros).
    """
    if data_root is not None:
        data_root = Path(data_root)
        for sub in ["dataset_train", "dataset_val", ""]:
            d = data_root / sub if sub else data_root
            if not d.exists():
                continue
            hits = sorted(d.glob("lulc_*.npy")) + sorted(d.glob("*_lulc_*.npy"))
            if hits:
                return LULCStaticProvider(hits[0], spatial_size=spatial_size)

    logger.info("No LULC npy found — using placeholder static features (zeros).")
    return PlaceholderStaticProvider(channels=channels, spatial_size=spatial_size)
