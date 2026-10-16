"""
future_providers.py — Unified future environmental condition providers.

╔══════════════════════════════════════════════════════════════════════════╗
║  SEMANTIC ALIGNMENT DESIGN  (UCGIS-2026 / TS-SatFire specification)    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  The future-condition branch feeds ONLY the 4 dynamic forecastable     ║
║  variables that are semantically shared across ALL data sources:       ║
║                                                                        ║
║    Channel 0 : Temperature       (K)                                   ║
║    Channel 1 : U-wind component  (m/s, eastward)                       ║
║    Channel 2 : V-wind component  (m/s, northward)                      ║
║    Channel 3 : Relative Humidity (%)                                   ║
║                                                                        ║
║  TRAINING:   4ch extracted from FirePred 19ch per future frame         ║
║              temp = band 5;  u,v = derived from speed(9)+dir(10);      ║
║              rh = band 7                                               ║
║                                                                        ║
║  INFERENCE Mode A:  HRRR 4ch  (TMP, UGRD, VGRD, RH)                   ║
║  INFERENCE Mode B:  WeatherNext2 4ch (temp_2m, u10, v10, rh_2m)       ║
║                                                                        ║
║  All remaining FirePred bands (DEM, slope, aspect, canopy, ERC,        ║
║  dewpoint, precip, fuel moisture, PDSI, fuel model, dead fuel, VPD,    ║
║  NDVI, soil temp, snow) stay in the HISTORICAL encoder (aux_hist 19ch) ║
║  and are never fed to the future decoder branch.                       ║
║                                                                        ║
║  This guarantees identical semantics (temp, u, v, rh) during training  ║
║  and inference — no 19ch→4ch mismatch.                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Providers
─────────
  1. TSSatFireFutureProvider   — FirePred → 4ch dynamic  (training)
  2. HRRRFutureProvider        — HRRR 4ch                (inference Mode A)
  3. WeatherNext2FutureProvider— WN2 4ch   ⚠ PLACEHOLDER (inference Mode B)
  4. PlaceholderFutureProvider — zeros fallback

Wind extraction:
    extract_wind(env_fut, source) → Tensor (B, H, 2, Hp, Wp)
    Channel 1 = u-wind, Channel 2 = v-wind — **uniform for ALL sources**.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  FirePred 19-band classification & dynamic extraction
# ═══════════════════════════════════════════════════════════════════════════

#: Complete FirePred 19-band index reference.
FIREPRED_BAND_NAMES = {
    0: "DEM",           1: "Slope",        2: "Aspect",
    3: "Canopy",        4: "ERC",          5: "Temperature",
    6: "Dewpoint",      7: "RH",           8: "Precipitation",
    9: "WindSpeed",     10: "WindDir",      11: "FuelMoisture",
    12: "PDSI",         13: "FuelModel",    14: "DeadFuel",
    15: "VPD",          16: "NDVI",         17: "SoilTemp",
    18: "Snow",
}

#: Classification of FirePred bands by temporal dynamics.
FIREPRED_CLASSIFICATION = {
    "static":                  [0, 1, 2],         # DEM, Slope, Aspect
    "slow_changing":           [3, 13, 16],        # Canopy, FuelModel, NDVI
    "dynamic_non_forecastable": [4, 11, 12, 14, 15],  # ERC, FuelMoist, PDSI, DeadFuel, VPD
    "dynamic_forecastable":    [5, 6, 7, 8, 9, 10, 17, 18],
    #                           Temp, Dew, RH, Precip, WSpd, WDir, SoilT, Snow
}

#: The 4 canonical future-dynamic channels extracted from FirePred.
#  These align semantically with HRRR (TMP, UGRD, VGRD, RH) and
#  WeatherNext2 (temp_2m, u10, v10, rh_2m).
DYNAMIC_FUTURE_CHANNELS = 4
DYNAMIC_FUTURE_MAP = {
    0: {"name": "temperature",  "firepred_band": 5,    "hrrr": "TMP",  "wn2": "temperature_2m"},
    1: {"name": "u_wind",       "firepred_derived": "speed(9)*sin(dir(10))", "hrrr": "UGRD", "wn2": "u_component_of_wind_10m"},
    2: {"name": "v_wind",       "firepred_derived": "speed(9)*cos(dir(10))", "hrrr": "VGRD", "wn2": "v_component_of_wind_10m"},
    3: {"name": "humidity",     "firepred_band": 7,    "hrrr": "RH",   "wn2": "relative_humidity_2m"},
}


def extract_dynamic_from_firepred(frame_19ch: np.ndarray) -> np.ndarray:
    """Extract 4 semantically aligned dynamic channels from FirePred 19ch.

    Parameters
    ----------
    frame_19ch : ndarray of shape (..., 19, H, W)
        FirePred frame(s).  Leading dims are preserved (e.g. (T, 19, H, W)
        or (19, H, W)).

    Returns
    -------
    ndarray of shape (..., 4, H, W)
        Channel 0: temperature  ← band 5
        Channel 1: u-wind       ← speed(band9) * sin(dir(band10))
        Channel 2: v-wind       ← speed(band9) * cos(dir(band10))
        Channel 3: humidity     ← band 7
    """
    temp = frame_19ch[..., 5:6, :, :]
    speed = frame_19ch[..., 9:10, :, :]
    dir_deg = frame_19ch[..., 10:11, :, :]
    rh = frame_19ch[..., 7:8, :, :]

    # Convert meteorological wind direction (degrees from north) to u,v
    dir_rad = dir_deg * (np.pi / 180.0)
    u = speed * np.sin(dir_rad)   # eastward component
    v = speed * np.cos(dir_rad)   # northward component

    return np.concatenate([temp, u, v, rh], axis=-3).astype(np.float32)


def extract_dynamic_from_firepred_torch(frame_19ch: torch.Tensor) -> torch.Tensor:
    """Torch version of extract_dynamic_from_firepred.

    Parameters
    ----------
    frame_19ch : Tensor (..., 19, H, W)

    Returns
    -------
    Tensor (..., 4, H, W)
    """
    temp = frame_19ch[..., 5:6, :, :]
    speed = frame_19ch[..., 9:10, :, :]
    dir_deg = frame_19ch[..., 10:11, :, :]
    rh = frame_19ch[..., 7:8, :, :]

    dir_rad = dir_deg * (math.pi / 180.0)
    u = speed * torch.sin(dir_rad)
    v = speed * torch.cos(dir_rad)

    return torch.cat([temp, u, v, rh], dim=-3)


# ═══════════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════════
class FutureConditionProvider:
    """Abstract base for future environmental condition providers.

    All providers MUST output exactly ``DYNAMIC_FUTURE_CHANNELS`` (4)
    channels in the canonical order: [temp, u_wind, v_wind, rh].
    """

    channels: int = DYNAMIC_FUTURE_CHANNELS
    source_name: str = "unknown"

    def __init__(
        self,
        channels: int = DYNAMIC_FUTURE_CHANNELS,
        source_name: str = "unknown",
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        self.channels = channels
        self.source_name = source_name
        self.spatial_size = spatial_size

    def get_future_env(
        self,
        indices: tuple,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Return future environmental conditions.

        Parameters
        ----------
        indices : tuple of int
        horizon : int, optional

        Returns
        -------
        ndarray of shape (H, 4, Hp, Wp)
            4 channels: [temperature, u_wind, v_wind, humidity]
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
#  1. TSSatFireFutureProvider  —  FirePred → 4ch dynamic (TRAINING)
# ═══════════════════════════════════════════════════════════════════════════
class TSSatFireFutureProvider(FutureConditionProvider):
    """TS-SatFire FirePred data → 4ch dynamic future conditioning.

    Loads the full 19-band FirePred npy but returns ONLY the 4 canonical
    dynamic channels (temp, u-wind, v-wind, humidity) via
    ``extract_dynamic_from_firepred``.

    This ensures the future decoder branch sees the **same 4 variables**
    during training (FirePred) and inference (HRRR / WeatherNext2),
    eliminating the semantic mismatch of 19ch-train → 4ch-infer.

    All remaining 15 FirePred bands (DEM, slope, aspect, canopy, ERC,
    dewpoint, precip, fuel moisture, PDSI, fuel model, dead fuel, VPD,
    NDVI, soil temp, snow) remain accessible through ``aux_hist`` in the
    historical encoder — they are NOT discarded, just not in the future
    decoder branch.

    FirePred 19 bands → 4ch extraction:
        band 5  (Temperature)  → channel 0
        band 9  (WindSpeed) }
        band 10 (WindDir)   }  → channel 1 (u), channel 2 (v)
        band 7  (RH)          → channel 3

    Parameters
    ----------
    firepred_npy_path : path to firepred_*.npy  (N, 19, T, H, W)
    spatial_size      : target spatial dims (for metadata only)
    """

    def __init__(
        self,
        firepred_npy_path: Union[str, Path],
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        firepred_npy_path = Path(firepred_npy_path)
        if not firepred_npy_path.exists():
            raise FileNotFoundError(f"FirePred npy not found: {firepred_npy_path}")
        self._data = np.load(str(firepred_npy_path), mmap_mode="r")
        N = self._data.shape[0]
        self._N = N
        super().__init__(
            channels=DYNAMIC_FUTURE_CHANNELS,
            source_name="firepred",
            spatial_size=spatial_size,
        )
        logger.info(
            f"TSSatFireFutureProvider: {N} samples, 19ch raw → "
            f"{DYNAMIC_FUTURE_CHANNELS}ch dynamic from {firepred_npy_path.name}"
        )

    def get_future_env(self, indices: tuple, horizon: Optional[int] = None) -> np.ndarray:
        """Return (H, 4, Hp, Wp) dynamic future from FirePred."""
        frames = []
        for idx in indices:
            i = idx % self._N
            frame_19 = self._data[i, :, 0, :, :]       # (19, H, W)
            frame_4 = extract_dynamic_from_firepred(frame_19)  # (4, H, W)
            frames.append(frame_4)
        return np.stack(frames, axis=0)                 # (H, 4, Hp, Wp)


# ═══════════════════════════════════════════════════════════════════════════
#  2. HRRRFutureProvider  —  HRRR 4ch (INFERENCE Mode A)
# ═══════════════════════════════════════════════════════════════════════════
class HRRRFutureProvider(FutureConditionProvider):
    """HRRR DOY climatology for single-step inference (Mode A).

    Loads pre-built doy_*.npy from ``hrrr_cache/climatology/``.
    4 channels already semantically aligned:
        0: TMP   (2m temperature, K)
        1: UGRD  (10m u-wind, m/s)
        2: VGRD  (10m v-wind, m/s)
        3: RH    (2m relative humidity, %)

    Parameters
    ----------
    clim_dir     : path to hrrr_cache/climatology/
    spatial_size : spatial dims for crop/pad
    """

    VARIABLES = ["TMP", "UGRD", "VGRD", "RH"]  # 4 channels

    def __init__(
        self,
        clim_dir: Union[str, Path],
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        clim_dir = Path(clim_dir)
        files = sorted(clim_dir.glob("doy_*.npy"))
        if not files:
            raise FileNotFoundError(
                f"No doy_*.npy in {clim_dir}. "
                "Run Step 1b in the notebook to download HRRR climatology."
            )
        self._data = np.stack(
            [self._load_patch(f, spatial_size) for f in files]
        )  # (D, 4, H, W)
        D, Cw = self._data.shape[:2]
        self._D = D
        super().__init__(channels=Cw, source_name="hrrr", spatial_size=spatial_size)
        logger.info(f"HRRRFutureProvider: {D} DOY frames, {Cw}ch")

    @staticmethod
    def _load_patch(path: Path, size: Tuple[int, int]) -> np.ndarray:
        arr = np.load(str(path)).astype(np.float32)
        th, tw = size
        # Handle scalar DOY files (4,) — broadcast to spatial
        if arr.ndim == 1:
            return arr[:, None, None] * np.ones((1, th, tw), dtype=np.float32)
        # Normal 3D array (4, Hp, Wp) — crop or pad
        _, H, W = arr.shape
        if H >= th and W >= tw:
            cy, cx = H // 2, W // 2
            arr = arr[:, cy - th // 2 : cy + th // 2, cx - tw // 2 : cx + tw // 2]
        else:
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)))
        return arr

    def get_future_env(self, indices: tuple, horizon: Optional[int] = None) -> np.ndarray:
        """Return (H, 4, Hp, Wp) HRRR weather for given indices."""
        frames = [self._data[idx % self._D] for idx in indices]
        return np.stack(frames, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
#  3. WeatherNext2FutureProvider  —  Google WN2 4ch (INFERENCE Mode B)
#
#  ⚠⚠⚠  PLACEHOLDER / MOCK  ⚠⚠⚠
#
#  This provider is a PLACEHOLDER for the Google WeatherNext2 integration.
#  WeatherNext2 is not yet publicly released as a general API.
#  This class gracefully falls back to zeros if the downloader is
#  unavailable (which is the expected state for now).
#
#  When WN2 becomes available, implement a real downloader module at
#  ``firest_graphnet.data.weathernext2_downloader``.
# ═══════════════════════════════════════════════════════════════════════════
class WeatherNext2FutureProvider(FutureConditionProvider):
    """Google WeatherNext2 multi-step forecast (inference Mode B).

    ⚠ **STATUS: PLACEHOLDER / MOCK** — NOT production-ready.
    Falls back to zeros when the downloader is unavailable.

    4 channels semantically aligned with HRRR / FirePred dynamic:
        0: temperature_2m             (K)
        1: u_component_of_wind_10m    (m/s)
        2: v_component_of_wind_10m    (m/s)
        3: relative_humidity_2m       (%)

    Parameters
    ----------
    cache_dir    : path to cache/weathernext2
    spatial_size : spatial dims
    """

    VARIABLES = [
        "temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "relative_humidity_2m",
    ]

    def __init__(
        self,
        cache_dir: Union[str, Path] = "cache/weathernext2",
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__(
            channels=DYNAMIC_FUTURE_CHANNELS,
            source_name="weathernext2",
            spatial_size=spatial_size,
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._downloader = None
        try:
            from firest_graphnet.data.weathernext2_downloader import WeatherNext2Downloader
            self._downloader = WeatherNext2Downloader(
                cache_dir=str(self.cache_dir),
                variables=self.VARIABLES,
                spatial_size=spatial_size,
            )
            n_doy = len(self._downloader._doy_data)
            if n_doy > 0:
                logger.info(f"WeatherNext2FutureProvider: {n_doy} DOY climatology frames ready")
            else:
                logger.warning(
                    "WeatherNext2FutureProvider: no DOY climatology built yet. "
                    "Run: python -m firest_graphnet.data.weathernext2_downloader build-climatology"
                )
        except Exception as e:
            logger.warning(f"WeatherNext2 downloader unavailable ({e}), falling back to zeros")

    def get_future_env(self, indices: tuple, horizon: Optional[int] = None) -> np.ndarray:
        """Return (H, 4, Hp, Wp) WN2 forecast."""
        H = len(indices) if horizon is None else horizon
        if self._downloader is not None:
            try:
                arr = self._downloader.fetch(t0=indices[0], horizon=H)
                if arr is not None:
                    return arr
            except Exception as e:
                logger.warning(f"WeatherNext2 fetch failed: {e}")
        # Fallback: zeros (expected for now)
        return np.zeros((H, self.channels, *self.spatial_size), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  4. PlaceholderFutureProvider  —  zeros fallback
# ═══════════════════════════════════════════════════════════════════════════
class PlaceholderFutureProvider(FutureConditionProvider):
    """Returns zero tensors — pipeline never crashes."""

    _warned = False

    def __init__(
        self,
        channels: int = DYNAMIC_FUTURE_CHANNELS,
        source_name: str = "firepred",
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__(
            channels=channels,
            source_name=source_name,
            spatial_size=spatial_size,
        )

    def get_future_env(self, indices: tuple, horizon: Optional[int] = None) -> np.ndarray:
        H = len(indices) if horizon is None else horizon
        if not PlaceholderFutureProvider._warned:
            logger.warning(
                f"PlaceholderFutureProvider({self.source_name}): returning zeros "
                f"({H}, {self.channels}, {self.spatial_size[0]}, {self.spatial_size[1]})."
            )
            PlaceholderFutureProvider._warned = True
        return np.zeros((H, self.channels, *self.spatial_size), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Wind extraction utility  (UNIFORM for all sources)
# ═══════════════════════════════════════════════════════════════════════════
def extract_wind(
    env_fut: torch.Tensor,
    source: str = "firepred",
) -> torch.Tensor:
    """Extract (u, v) wind components from future condition tensor.

    Because all sources now provide the SAME 4-channel canonical layout
    [temp, u_wind, v_wind, rh], wind extraction is trivial and uniform:
        channel 1 = u-wind  (m/s, eastward)
        channel 2 = v-wind  (m/s, northward)

    Parameters
    ----------
    env_fut : (B, H, 4, Hp, Wp)  future environmental conditions
    source  : str — ignored (kept for API compatibility)

    Returns
    -------
    wind : (B, H, 2, Hp, Wp)  where dim-2 = [u, v]  in m/s
    """
    return env_fut[:, :, 1:3].contiguous()  # (B, H, 2, Hp, Wp)


# ═══════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════
def build_future_provider(
    mode: str = "firepred",
    data_root: Optional[Union[str, Path]] = None,
    spatial_size: Tuple[int, int] = (256, 256),
    **kw,
) -> FutureConditionProvider:
    """Create a future condition provider by name.

    Parameters
    ----------
    mode : "firepred" | "hrrr" | "weathernext2" | "placeholder"
    data_root : workspace root for auto-locating npy files
    """
    if mode == "firepred":
        if data_root is not None:
            data_root = Path(data_root)
            for sub in ["dataset_train", "dataset_val", ""]:
                d = data_root / sub if sub else data_root
                if not d.exists():
                    continue
                hits = sorted(d.glob("firepred_*.npy"))
                if hits:
                    return TSSatFireFutureProvider(hits[0], spatial_size=spatial_size)
        logger.info("No firepred_*.npy found — placeholder for future env (4ch zeros).")
        return PlaceholderFutureProvider(
            channels=DYNAMIC_FUTURE_CHANNELS,
            source_name="firepred",
            spatial_size=spatial_size,
        )

    if mode == "hrrr":
        clim_dir = Path(data_root) / "hrrr_cache" / "climatology" if data_root else None
        if clim_dir and clim_dir.exists() and list(clim_dir.glob("doy_*.npy")):
            return HRRRFutureProvider(clim_dir, spatial_size=spatial_size)
        logger.warning("HRRR climatology not found — placeholder (4ch zeros).")
        return PlaceholderFutureProvider(
            channels=DYNAMIC_FUTURE_CHANNELS,
            source_name="hrrr",
            spatial_size=spatial_size,
        )

    if mode == "weathernext2":
        return WeatherNext2FutureProvider(
            cache_dir=kw.get("cache_dir", "cache/weathernext2"),
            spatial_size=spatial_size,
        )

    return PlaceholderFutureProvider(
        channels=kw.get("channels", DYNAMIC_FUTURE_CHANNELS),
        source_name=kw.get("source_name", "firepred"),
        spatial_size=spatial_size,
    )
