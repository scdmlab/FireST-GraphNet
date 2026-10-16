"""
firepred_provider.py  --  Provider for FirePred auxiliary data (weather + topo + fuel).

TS-SatFire FirePred 19-band GeoTIFF (per date):
  Band 0:  DEM elevation (m)
  Band 1:  Slope (°)
  Band 2:  Aspect (°)
  Band 3:  Canopy height (m)
  Band 4:  ERC (Energy Release Component)
  Band 5:  2m temperature (K)
  Band 6:  Dewpoint temperature (K)
  Band 7:  Relative humidity (%)
  Band 8:  Precipitation (mm)
  Band 9:  Wind speed (m/s)
  Band 10: Wind direction (°)
  Band 11: Fuel moisture (live)
  Band 12: PDSI (Palmer Drought Severity Index)
  Band 13: Fuel model (category, 1-16)
  Band 14: Dead fuel moisture (1h)
  Band 15: VPD (Vapor Pressure Deficit, kPa)
  Band 16: NDVI
  Band 17: Soil temperature (°C)
  Band 18: Snow cover (flag)

These 19 channels provide the canonical historical weather / topography / fuel
data for the TS-SatFire prediction task.  They are perfectly aligned with the
VIIRS imagery (same dates, same events).

Providers
─────────
  1. FirePredProvider         –  loads firepred_*.npy from preprocessed data
  2. PlaceholderFirePred      –  returns zeros (graceful fallback)

All providers expose:
    .channels   : int                         number of auxiliary channels
    .get(idx)   → ndarray (C_aux, H, W)      auxiliary features for one date
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
class FirePredProviderBase:
    """Abstract FirePred auxiliary data provider."""

    def __init__(self, channels: int = 19, spatial_size: Tuple[int, int] = (256, 256)):
        self.channels = channels
        self.spatial_size = spatial_size

    def get_window(self, indices: tuple) -> np.ndarray:
        """Return auxiliary data for a set of frame indices.
        shape (T, C_aux, H, W)."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
#  1. FirePredProvider  —  from preprocessed firepred_*.npy
# ═══════════════════════════════════════════════════════════════════════════
class FirePredProvider(FirePredProviderBase):
    """Loads FirePred auxiliary data from preprocessed firepred_*.npy.

    Expected npy shape: (N, 19, T, H, W)  —  matches af_*_img npy format.
    """

    def __init__(self, npy_path: str | Path, spatial_size: Tuple[int, int] = (256, 256)):
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"FirePred npy not found: {npy_path}")
        self._data = np.load(str(npy_path), mmap_mode="r")
        # (N, 19, T, H, W) or (N, 19, 1, H, W)
        N, C = self._data.shape[0], self._data.shape[1]
        super().__init__(channels=C, spatial_size=spatial_size)
        self._N = N
        logger.info(f"FirePredProvider: {N} samples, {C} channels from {npy_path.name}")

    def get_window(self, indices: tuple) -> np.ndarray:
        """Return (T, C_aux, H, W) for given frame indices."""
        frames = []
        for idx in indices:
            i = idx % self._N
            # shape: (C, T_stored, H, W) → take first T dim
            frame = self._data[i, :, 0, :, :]  # (C, H, W)
            frames.append(frame)
        return np.stack(frames, axis=0)  # (T, C, H, W)


# ═══════════════════════════════════════════════════════════════════════════
#  2. PlaceholderFirePred  —  zeros (graceful fallback)
# ═══════════════════════════════════════════════════════════════════════════
class PlaceholderFirePred(FirePredProviderBase):
    """Fallback: returns zero auxiliary features.  Training never crashes."""

    _warned = False

    def get_window(self, indices: tuple) -> np.ndarray:
        if not PlaceholderFirePred._warned:
            logger.warning(
                "PlaceholderFirePred: returning zeros — run preprocess_raw.py to "
                "generate firepred_*.npy from raw TS-SatFire TIFFs."
            )
            PlaceholderFirePred._warned = True
        T = len(indices)
        return np.zeros((T, self.channels, *self.spatial_size), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════
def build_firepred_provider(
    data_root: str | Path | None = None,
    channels: int = 19,
    spatial_size: Tuple[int, int] = (256, 256),
) -> FirePredProviderBase:
    """Create a FirePred provider, auto-detecting firepred_*.npy files.

    Looks for ``firepred_*.npy`` in dataset directories.
    If not found, returns placeholder (zeros).
    """
    if data_root is not None:
        data_root = Path(data_root)
        for sub in ["dataset_train", "dataset_val", ""]:
            d = data_root / sub if sub else data_root
            if not d.exists():
                continue
            hits = sorted(d.glob("firepred_*.npy"))
            if hits:
                return FirePredProvider(hits[0], spatial_size=spatial_size)

    logger.info(
        "No firepred_*.npy found — using zeros for auxiliary data. "
        "Run preprocess_raw.py to generate from raw TS-SatFire TIFFs."
    )
    return PlaceholderFirePred(channels=channels, spatial_size=spatial_size)
