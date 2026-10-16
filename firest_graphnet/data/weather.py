"""
weather.py — HRRR weather data provider.

两种工作模式
-----------
1. 占位符模式（默认）：无任何天气文件时返回零张量。
2. 气候态模式：当 climatology_dir 下存在 doy_*.npy 时，按 DOY 返回均值天气场。
   构建气候态的方法见 hrrr_alignment.py：
       python -m firest_graphnet.data.hrrr_alignment gen-download
       python download_hrrr.py
       python -m firest_graphnet.data.hrrr_alignment build-clim

天气通道顺序（4 通道）
---------------------
  0: TMP  2m 气温 (K)
  1: UGRD 10m U 风速 (m/s)
  2: VGRD 10m V 风速 (m/s)
  3: RH   2m 相对湿度 (%)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class WeatherProvider:
    """Interface for dynamic weather feature ingestion.

    Parameters
    ----------
    cache_dir : str | Path | None
        Directory containing pre-downloaded HRRR crops (legacy per-event files).
    climatology_dir : str | Path | None
        Directory containing DOY-averaged HRRR climatology files
        (``doy_001.npy`` … ``doy_365.npy``). Built by ``hrrr_alignment.py``.
        **Preferred mode** — automatically used when present.
    channels : int
        Number of weather variables (4 = TMP, UGRD, VGRD, RH).
    spatial_size : tuple
        (H, W) matching the fire imagery resolution.
    """

    PLACEHOLDER_MSG = "WeatherProvider: using placeholder (zero tensor)"

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        climatology_dir: Optional[str | Path] = None,
        channels: int = 4,
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        self.channels = channels
        self.spatial_size = spatial_size
        self._data_available = False
        self._clim_provider = None

        # 优先使用气候态提供器
        if climatology_dir is not None:
            clim_path = Path(climatology_dir)
            if clim_path.is_dir() and any(clim_path.glob("doy_*.npy")):
                from firest_graphnet.data.hrrr_alignment import HRRRClimatologyProvider
                self._clim_provider = HRRRClimatologyProvider(
                    clim_path, channels=channels, spatial_size=spatial_size
                )
                self._data_available = True
                logger.info(f"WeatherProvider: 已加载 HRRR 气候态 → {clim_path}")
            else:
                logger.warning(f"WeatherProvider: climatology_dir 为空 → {clim_path}")

        if not self._data_available and cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            if self.cache_dir.is_dir() and any(self.cache_dir.iterdir()):
                self._data_available = True
            else:
                logger.warning(self.PLACEHOLDER_MSG)
        else:
            self.cache_dir = None

        if not self._data_available:
            logger.warning(self.PLACEHOLDER_MSG)

    # ── public API ─────────────────────────────────────────────────────────
    def get_by_doy(self, doy: int) -> torch.Tensor:
        """Return climatological weather tensor for a given Day-of-Year.

        Parameters
        ----------
        doy : int  Day of Year (1-365); from ch12 of TS-SatFire raw files.

        Returns
        -------
        torch.Tensor  shape (channels, H, W)
        """
        if self._clim_provider is not None:
            arr = self._clim_provider.get(doy)
            return torch.from_numpy(arr).float()
        return torch.zeros(self.channels, *self.spatial_size, dtype=torch.float32)

    def __getitem__(
        self,
        event_id: str,
        timestamp: str = "",
        lead_time: int = 0,
    ) -> torch.Tensor:
        """Return weather tensor for a given event / time.

        Returns
        -------
        torch.Tensor  shape (channels, H, W)
        """
        if not self._data_available:
            return torch.zeros(
                self.channels, *self.spatial_size, dtype=torch.float32
            )

        # TODO: load real HRRR data from self.cache_dir
        # Expected file pattern: {cache_dir}/{event_id}/{timestamp}_lt{lead_time}.npy
        npy_path = (
            self.cache_dir / event_id / f"{timestamp}_lt{lead_time}.npy"
        )
        if npy_path.exists():
            arr = np.load(str(npy_path))
            return torch.from_numpy(arr).float()

        logger.warning(
            f"Weather file not found: {npy_path}  →  returning placeholder"
        )
        return torch.zeros(
            self.channels, *self.spatial_size, dtype=torch.float32
        )

    @property
    def available(self) -> bool:
        return self._data_available
