"""
weathernext2_downloader.py -- Google Weather API + Open-Meteo ERA5 fallback.

Primary source  : Google Weather API (maps.googleapis.com/weather/v1)
                  Real-time hourly forecasts up to 240h ahead.
                  Returns: temperature, wind speed+dir, relative humidity.
Fallback source : Open-Meteo ERA5 historical climatology (DOY averages).

Canonical 4-channel output (same as HRRR and FirePred):
    Channel 0 : temperature_2m   (K)
    Channel 1 : u_wind_10m       (m/s, eastward  = speed * sin(dir_rad))
    Channel 2 : v_wind_10m       (m/s, northward = speed * cos(dir_rad))
    Channel 3 : relative_humidity (%)

API key storage:
    - Colab : use Colab Secrets  (userdata.get('GOOGLE_WEATHER_API_KEY'))
    - Local : set env var        GOOGLE_WEATHER_API_KEY=<key>
    NEVER hardcode the key in source files.

Usage:
    dl = WeatherNext2Downloader(api_key=os.environ['GOOGLE_WEATHER_API_KEY'])
    arr = dl.fetch(t0=0, horizon=15)   # (15, 4, 256, 256)

Build DOY fallback climatology (one-time, ~3 min):
    python -m firest_graphnet.data.weathernext2_downloader build-climatology
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default location: Central California wildfire region
DEFAULT_LAT = 37.0
DEFAULT_LON = -119.0

GOOGLE_WEATHER_BASE = "https://weather.googleapis.com/v1"
OPEN_METEO_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"


def _kph_to_ms(kph: float) -> float:
    return kph / 3.6


def _celsius_to_kelvin(c: float) -> float:
    return c + 273.15


def _wind_to_uv(speed_ms: float, dir_deg: float):
    """Convert meteorological wind speed + direction to u, v components."""
    r = math.radians(dir_deg)
    return speed_ms * math.sin(r), speed_ms * math.cos(r)


class WeatherNext2Downloader:
    """
    Real-time Google Weather API + ERA5 fallback for multi-step conditioning.

    Parameters
    ----------
    cache_dir    : root cache; climatology stored in {cache_dir}/climatology/
    api_key      : Google Weather API key (AIzaSy...).
                   If None, reads from env GOOGLE_WEATHER_API_KEY.
                   Falls back to Open-Meteo ERA5 if key unavailable.
    variables    : ignored (kept for API compat)
    spatial_size : output spatial shape broadcast from point data
    lat, lon     : location for weather queries
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache/weathernext2",
        api_key: str | None = None,
        variables: List[str] | None = None,
        spatial_size: tuple[int, int] = (256, 256),
        lat: float = DEFAULT_LAT,
        lon: float = DEFAULT_LON,
    ):
        self.cache_dir = Path(cache_dir)
        self.clim_dir  = self.cache_dir / "climatology"
        self.clim_dir.mkdir(parents=True, exist_ok=True)
        self.Cw = 4
        self.spatial_size = spatial_size
        self.lat = lat
        self.lon = lon

        # Resolve API key: arg > env var > None (fallback to climatology)
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_WEATHER_API_KEY")
            or os.environ.get("WN2_API_KEY")
        )
        if self._api_key:
            logger.info("WeatherNext2: Google Weather API key found, live forecasts enabled")
        else:
            logger.info("WeatherNext2: no API key, using DOY climatology only")

        # Load pre-built DOY climatology as fallback
        self._doy_data: dict[int, np.ndarray] = {}
        self._load_climatology()

    # ── DOY climatology ───────────────────────────────────────────────────

    def _load_climatology(self) -> None:
        """Load all available DOY npy files into memory."""
        self._doy_data.clear()
        for f in sorted(self.clim_dir.glob("doy_*.npy")):
            try:
                doy = int(f.stem.split("_")[1])
                arr = np.load(str(f)).astype(np.float32)
                if arr.ndim == 1 and arr.shape[0] == 4:
                    # Scalar (4,) — broadcast to spatial
                    spatial = arr[:, None, None] * np.ones(
                        (1, *self.spatial_size), dtype=np.float32
                    )
                elif arr.ndim == 3 and arr.shape[0] == 4:
                    # Already (4, H, W) — crop/pad if needed
                    _, H, W = arr.shape
                    th, tw = self.spatial_size
                    if H != th or W != tw:
                        out = np.zeros((4, th, tw), dtype=np.float32)
                        ch, cw = min(H, th), min(W, tw)
                        out[:, :ch, :cw] = arr[:, :ch, :cw]
                        spatial = out
                    else:
                        spatial = arr
                else:
                    continue
                self._doy_data[doy] = spatial
            except Exception as e:
                logger.debug(f"Skip {f.name}: {e}")

        if self._doy_data:
            doys = sorted(self._doy_data)
            logger.info(
                f"WeatherNext2: {len(self._doy_data)} DOY climatology frames "
                f"(DOY {doys[0]}..{doys[-1]})"
            )

    def _clim_frame(self, idx: int) -> np.ndarray:
        """Return (4, H, W) climatology frame for index (cyclic over available DOYs)."""
        if not self._doy_data:
            return np.zeros((4, *self.spatial_size), dtype=np.float32)
        doys = sorted(self._doy_data)
        doy  = doys[idx % len(doys)]
        return self._doy_data[doy]

    # ── Google Weather API (live) ─────────────────────────────────────────

    def _fetch_live_forecast(self, horizon: int) -> Optional[np.ndarray]:
        """
        Fetch real-time hourly forecast from Google Weather API.

        Returns (horizon, 4, H, W) or None on failure.
        """
        if not self._api_key:
            return None

        try:
            import requests
        except ImportError:
            return None

        try:
            url = f"{GOOGLE_WEATHER_BASE}/forecast/hours:lookup"
            params = {
                "key":                    self._api_key,
                "location.latitude":      self.lat,
                "location.longitude":     self.lon,
                "hours":                  min(horizon, 240),  # API max 240h
                "languageCode":           "en",
                "unitsSystem":            "METRIC",
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Google Weather API fetch failed: {e}")
            return None

        forecast_hours = data.get("forecastHours", [])
        if not forecast_hours:
            logger.warning("Google Weather API: empty forecastHours")
            return None

        frames = []
        for h in forecast_hours[:horizon]:
            try:
                temp_c = float(h["temperature"]["degrees"])         # °C
                rh     = float(h["relativeHumidity"])               # %
                wind   = h.get("wind", {})
                speed_kph = float(wind.get("speed", {}).get("value", 0.0))
                dir_deg   = float(wind.get("direction", {}).get("degrees", 0.0))

                temp_k   = _celsius_to_kelvin(temp_c)
                speed_ms = _kph_to_ms(speed_kph)
                u, v     = _wind_to_uv(speed_ms, dir_deg)

                # Broadcast scalar to (4, H, W)
                ch = np.array([temp_k, u, v, rh], dtype=np.float32)
                frame = ch[:, None, None] * np.ones(
                    (1, *self.spatial_size), dtype=np.float32
                )
                frames.append(frame)
            except Exception as e:
                logger.debug(f"Parse error on hour {len(frames)}: {e}")
                frames.append(np.zeros((4, *self.spatial_size), dtype=np.float32))

        if not frames:
            return None

        # Pad with last frame if API returned fewer than horizon steps
        while len(frames) < horizon:
            frames.append(frames[-1].copy())

        arr = np.stack(frames[:horizon], axis=0)  # (horizon, 4, H, W)
        logger.info(
            f"Google Weather API: {len(frames)} hourly steps fetched "
            f"(temp[0]={arr[0,0,0,0]:.1f}K, rh[0]={arr[0,3,0,0]:.1f}%)"
        )
        return arr

    # ── Public API ────────────────────────────────────────────────────────

    def fetch(
        self,
        t0: int,
        horizon: int,
        bbox=None,
        **kw,
    ) -> Optional[np.ndarray]:
        """
        Fetch (horizon, 4, H, W) future weather conditions.

        Priority:
          1. Google Weather API (live hourly forecast) — if API key available
          2. DOY climatology fallback — always available if built

        Parameters
        ----------
        t0      : reference index (used for climatology cycling)
        horizon : number of future steps
        """
        # 1. Try live Google Weather API
        live = self._fetch_live_forecast(horizon)
        if live is not None:
            return live

        # 2. Fall back to DOY climatology
        if self._doy_data:
            frames = [self._clim_frame(t0 + t) for t in range(horizon)]
            return np.stack(frames, axis=0)

        logger.warning(
            "WeatherNext2: no live API and no DOY climatology. "
            "Run: python -m firest_graphnet.data.weathernext2_downloader build-climatology"
        )
        return None

    # ── Build DOY fallback climatology ───────────────────────────────────

    def build_doy_climatology(
        self,
        years: List[int] | None = None,
        lat: float | None = None,
        lon: float | None = None,
        force: bool = False,
    ) -> int:
        """Download Open-Meteo ERA5 and build DOY climatology as fallback."""
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")
        import datetime

        if years is None:
            years = [2017, 2018, 2019, 2020, 2021]
        lat = lat if lat is not None else self.lat
        lon = lon if lon is not None else self.lon

        accum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(4, np.float64))
        count: dict[int, int]        = defaultdict(int)

        print(f"Building DOY climatology from Open-Meteo ERA5 ({lat}N, {lon}E) ...")
        for year in years:
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
                "timezone": "UTC",
            }
            print(f"  {year}: fetching ...", end=" ", flush=True)
            try:
                resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"FAILED ({e})")
                continue

            hourly = data.get("hourly", {})
            times  = hourly.get("time", [])
            temp_c = np.array(hourly.get("temperature_2m", []), dtype=np.float64)
            wspd   = np.array(hourly.get("wind_speed_10m", []), dtype=np.float64)
            wdir   = np.array(hourly.get("wind_direction_10m", []), dtype=np.float64)
            rh     = np.array(hourly.get("relative_humidity_2m", []), dtype=np.float64)

            n_ok = 0
            for i, t_str in enumerate(times):
                vals = np.array([
                    temp_c[i] + 273.15,
                    wspd[i] * math.sin(math.radians(wdir[i])),
                    wspd[i] * math.cos(math.radians(wdir[i])),
                    rh[i],
                ])
                if np.any(np.isnan(vals)):
                    continue
                dt  = datetime.datetime.fromisoformat(t_str)
                doy = dt.timetuple().tm_yday
                accum[doy] += vals
                count[doy] += 1
                n_ok += 1
            print(f"OK ({n_ok} hours)")

        saved = 0
        for doy in sorted(accum.keys()):
            if count[doy] == 0:
                continue
            out = self.clim_dir / f"doy_{doy:03d}.npy"
            if out.exists() and not force:
                saved += 1
                continue
            avg = (accum[doy] / count[doy]).astype(np.float32)
            # Save as (4, H, W) spatial array
            arr = avg[:, None, None] * np.ones((1, *self.spatial_size), dtype=np.float32)
            np.save(str(out), arr)
            saved += 1

        print(f"Climatology: {saved} DOY files in {self.clim_dir}")
        self._load_climatology()
        return saved


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="WeatherNext2 downloader")
    sub = p.add_subparsers(dest="cmd")

    bc = sub.add_parser("build-climatology")
    bc.add_argument("--cache-dir", default="cache/weathernext2")
    bc.add_argument("--lat",   type=float, default=DEFAULT_LAT)
    bc.add_argument("--lon",   type=float, default=DEFAULT_LON)
    bc.add_argument("--years", default="2017,2018,2019,2020,2021")
    bc.add_argument("--force", action="store_true")

    test = sub.add_parser("test")
    test.add_argument("--lat",     type=float, default=DEFAULT_LAT)
    test.add_argument("--lon",     type=float, default=DEFAULT_LON)
    test.add_argument("--horizon", type=int,   default=5)

    args = p.parse_args()

    if args.cmd == "build-climatology":
        years = [int(y) for y in args.years.split(",")]
        dl = WeatherNext2Downloader(cache_dir=args.cache_dir, lat=args.lat, lon=args.lon)
        dl.build_doy_climatology(years=years, force=args.force)

    elif args.cmd == "test":
        api_key = os.environ.get("GOOGLE_WEATHER_API_KEY")
        if not api_key:
            print("Set GOOGLE_WEATHER_API_KEY env var to test live API")
            sys.exit(1)
        dl = WeatherNext2Downloader(api_key=api_key, lat=args.lat, lon=args.lon)
        arr = dl.fetch(t0=0, horizon=args.horizon)
        if arr is not None:
            print(f"shape: {arr.shape}")
            print(f"step0: temp={arr[0,0,0,0]:.1f}K  u={arr[0,1,0,0]:.2f}m/s  "
                  f"v={arr[0,2,0,0]:.2f}m/s  rh={arr[0,3,0,0]:.1f}%")
        else:
            print("fetch returned None")
    else:
        p.print_help()
