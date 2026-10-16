"""
weather_provider.py  --  Future weather providers for FiLM decoder conditioning.

In the corrected TS-SatFire pipeline:
  - HISTORICAL weather/topo/fuel comes from FirePred (19ch) via firepred_provider.py
  - FUTURE weather for the FiLM decoder comes from this module (HRRR 4ch)

Providers
─────────
  1. TSSatFireWeatherProvider     –  uses image bands as fallback proxy
  2. GoogleWeatherNext2Provider   –  pulls archived forecast from GCS
  3. PlaceholderForecastProvider  –  returns zeros (fallback)
  4. HRRRWeatherProvider          –  DOY climatology (TMP/UGRD/VGRD/RH)

All providers expose:
    .get_window(start, end)   → ndarray (T, Cw, H, W)    historical (legacy)
    .get_forecast(t0, horizon)→ ndarray (H, Cw, H, W)    future
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════════
class WeatherProviderBase:
    """Abstract weather provider."""

    def __init__(self, Cw: int = 8, spatial_size: Tuple[int, int] = (256, 256)):
        self.Cw = Cw
        self.spatial_size = spatial_size

    def get_window(self, start: int, end: int) -> np.ndarray:
        """Return historical weather window.  shape (T, Cw, H, W)."""
        raise NotImplementedError

    def get_forecast(self, t0: int, horizon: int) -> np.ndarray:
        """Return future weather forecast.  shape (H, Cw, H, W)."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
#  1. TSSatFireWeatherProvider  —  image bands  →  weather
# ═══════════════════════════════════════════════════════════════════════════
class TSSatFireWeatherProvider(WeatherProviderBase):
    """Uses image-band arrays from the TS-SatFire dataset as weather proxy.

    Parameters
    ----------
    img_npy_path : path to af_*_img_*.npy  (N, 8, 1, H, W)  memory-mapped
    """

    def __init__(self, img_npy_path: str | Path, **kw):
        self._data = np.load(str(img_npy_path), mmap_mode="r")
        N, C, _, H, W = self._data.shape
        super().__init__(Cw=C, spatial_size=(H, W))

    def get_window(self, start: int, end: int) -> np.ndarray:
        end = min(end, self._data.shape[0])
        start = max(start, 0)
        return np.array(self._data[start:end, :, 0, :, :])  # (T, C, H, W)

    def get_forecast(self, t0: int, horizon: int) -> np.ndarray:
        s = t0 + 1
        e = s + horizon
        if e > self._data.shape[0]:
            pad = e - self._data.shape[0]
            avail = np.array(self._data[s:self._data.shape[0], :, 0, :, :])
            zeros = np.zeros((pad, self.Cw, *self.spatial_size), dtype=np.float32)
            return np.concatenate([avail, zeros], axis=0)
        return np.array(self._data[s:e, :, 0, :, :])


# ═══════════════════════════════════════════════════════════════════════════
#  2. GoogleWeatherNext2Provider  —  GCS-archived forecast
# ═══════════════════════════════════════════════════════════════════════════
class GoogleWeatherNext2Provider(WeatherProviderBase):
    """Pulls time-sliced forecasts from Google WeatherNext2 archives.

    Supports:  BigQuery · Earth Engine · Zarr / Cloud storage

    If unavailable, falls back to PlaceholderForecastProvider automatically.
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache/weathernext2",
        variables: list[str] | None = None,
        Cw: int = 8,
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__(Cw=Cw, spatial_size=spatial_size)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.variables = variables or [
            "temperature_2m", "u_component_of_wind_10m",
            "v_component_of_wind_10m", "relative_humidity_2m",
        ]
        self._fallback = PlaceholderForecastProvider(Cw=Cw, spatial_size=spatial_size)
        self._available = False

        try:
            from firest_graphnet.data.weathernext2_downloader import WeatherNext2Downloader
            self._downloader = WeatherNext2Downloader(
                cache_dir=str(self.cache_dir),
                variables=self.variables,
            )
            self._available = True
            logger.info("GoogleWeatherNext2Provider: downloader ready")
        except Exception as e:
            logger.warning(f"GoogleWeatherNext2Provider: downloader unavailable ({e}), using placeholder")

    def get_window(self, start: int, end: int) -> np.ndarray:
        return self._fallback.get_window(start, end)

    def get_forecast(self, t0: int, horizon: int, **kw) -> np.ndarray:
        if not self._available:
            return self._fallback.get_forecast(t0, horizon)
        try:
            arr = self._downloader.fetch(t0=t0, horizon=horizon, **kw)
            if arr is not None:
                return arr
        except Exception as e:
            logger.warning(f"WeatherNext2 fetch failed: {e}")
        return self._fallback.get_forecast(t0, horizon)


# ═══════════════════════════════════════════════════════════════════════════
#  3. PlaceholderForecastProvider  —  zeros  (never crashes)
# ═══════════════════════════════════════════════════════════════════════════
class PlaceholderForecastProvider(WeatherProviderBase):
    """Fallback provider: returns zero tensors.  Training never crashes."""

    _warned = False

    def get_window(self, start: int, end: int) -> np.ndarray:
        T = max(end - start, 1)
        if not PlaceholderForecastProvider._warned:
            logger.warning("PlaceholderForecastProvider: returning zeros")
            PlaceholderForecastProvider._warned = True
        return np.zeros((T, self.Cw, *self.spatial_size), dtype=np.float32)

    def get_forecast(self, t0: int, horizon: int) -> np.ndarray:
        return np.zeros((horizon, self.Cw, *self.spatial_size), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  4. HRRRWeatherProvider  —  DOY climatology (TMP / UGRD / VGRD / RH)
# ═══════════════════════════════════════════════════════════════════════════
class HRRRWeatherProvider(WeatherProviderBase):
    """Real NWP weather from HRRR DOY climatology as future forecast conditioning.

    Loads pre-built doy_*.npy files (4 channels: TMP, UGRD, VGRD, RH, each
    (256, 256)) and serves them as future weather windows.  Since the TS-SatFire
    preprocessed arrays have no per-sample timestamps, we cycle through the
    DOY files based on sample index, giving varied weather conditioning.

    Parameters
    ----------
    clim_dir : path to hrrr_cache/climatology/  (contains doy_*.npy)
    spatial_size : spatial dims to crop/pad to; default (256, 256)
    """

    VARIABLES = ["TMP", "UGRD", "VGRD", "RH"]  # 4 channels

    def __init__(self, clim_dir: str | Path, spatial_size: Tuple[int, int] = (256, 256)):
        clim_dir = Path(clim_dir)
        files = sorted(clim_dir.glob("doy_*.npy"))
        if not files:
            raise FileNotFoundError(
                f"No doy_*.npy files in {clim_dir}.\n"
                "Run Step 1b in the notebook to download HRRR climatology first."
            )
        self._files = files
        # Load all DOY frames into memory (~1 MB total)
        self._data = np.stack(
            [self._load_patch(f, spatial_size) for f in files]
        )  # (D, 4, H, W)
        D, Cw = self._data.shape[:2]
        self._D = D
        super().__init__(Cw=Cw, spatial_size=spatial_size)
        logger.info(f"HRRRWeatherProvider: {D} DOY frames, {Cw} channels, {spatial_size}")

    @staticmethod
    def _load_patch(path: Path, size: Tuple[int, int]) -> np.ndarray:
        arr = np.load(str(path)).astype(np.float32)  # (4, Hp, Wp)
        _, H, W = arr.shape
        th, tw = size
        # Center-crop or zero-pad to target spatial size
        if H >= th and W >= tw:
            cy, cx = H // 2, W // 2
            arr = arr[:, cy - th // 2: cy + th // 2, cx - tw // 2: cx + tw // 2]
        else:
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)))
        return arr

    def _frame(self, idx: int) -> np.ndarray:
        """Return DOY frame at wrapped index.  (4, H, W)"""
        return self._data[idx % self._D]

    def get_window(self, start: int, end: int) -> np.ndarray:
        """Historical weather: cycles through DOY frames.  (T, 4, H, W)"""
        T = max(1, end - start)
        return np.stack([self._frame(start + t) for t in range(T)])

    def get_forecast(self, t0: int, horizon: int) -> np.ndarray:
        """Future weather forecast: next `horizon` DOY frames.  (H, 4, H, W)"""
        return np.stack([self._frame(t0 + 1 + t) for t in range(horizon)])


# ═══════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════
def build_weather_provider(
    mode: str = "dataset",
    data_root: str | Path | None = None,
    **kw,
) -> WeatherProviderBase:
    """
    Create a weather provider by name.

    Parameters
    ----------
    mode : 'hrrr' | 'dataset' | 'weathernext2' | 'placeholder'
    data_root : root directory for TS-SatFire (needed for 'dataset' and 'hrrr' modes)

    Note: When the encoder uses multi-source concatenation (fire + weather + static),
    'hrrr' mode falls back to 'placeholder' (zeros) instead of 'dataset' to avoid
    duplicating fire imagery bands as pseudo-weather channels.
    """
    if mode == "hrrr":
        clim_dir = Path(data_root) / "hrrr_cache" / "climatology" if data_root else None
        if clim_dir and clim_dir.exists() and list(clim_dir.glob("doy_*.npy")):
            return HRRRWeatherProvider(clim_dir, **kw)
        logger.warning(
            "HRRR climatology not found at %s — falling back to placeholder (zeros). "
            "Run Step 1b to download climatology, then re-run.",
            clim_dir,
        )
        # Fallback to placeholder to avoid duplicating fire imagery as weather
        return PlaceholderForecastProvider(Cw=4, **kw)

    if mode == "dataset":
        from firest_graphnet.utils.io import find_dataset_dirs
        train_dir, _ = find_dataset_dirs(data_root)
        npy = sorted(train_dir.glob("af_*_img_*.npy"))
        if npy:
            return TSSatFireWeatherProvider(npy[0], **kw)
        logger.warning("No img npy found — falling back to placeholder")
        return PlaceholderForecastProvider(**kw)

    if mode == "weathernext2":
        return GoogleWeatherNext2Provider(**kw)

    return PlaceholderForecastProvider(**kw)
