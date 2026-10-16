"""
Build HRRR DOY climatology using Open-Meteo ERA5 historical data.

Downloads 2017-2021 hourly weather for a Central California wildfire region,
averages by day-of-year, and saves (4, 256, 256) float32 .npy files to
hrrr_cache/climatology/doy_XXX.npy

4 channels (same as HRRR canonical format):
  0: temperature_2m  (K)
  1: u_wind_10m      (m/s, eastward)
  2: v_wind_10m      (m/s, northward)
  3: relative_humidity_2m (%)

Runtime: ~5-10 min (network dependent)
"""
import datetime
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Skip DOYs that already exist (correct shape)
CLIM_DIR = Path("hrrr_cache/climatology")
CLIM_DIR.mkdir(parents=True, exist_ok=True)

SPATIAL_SIZE = (256, 256)
YEARS = [2017, 2018, 2019, 2020, 2021]

# Central California wildfire region
LAT = 37.0
LON = -119.0

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

try:
    import requests
except ImportError:
    print("pip install requests"); sys.exit(1)


def build():
    # Check existing files
    existing = {int(f.stem.split("_")[1]) for f in CLIM_DIR.glob("doy_*.npy")}
    print(f"Existing DOY files: {len(existing)}/366")
    if len(existing) == 366:
        print("All 366 DOY files already exist. Done.")
        return

    # Accumulate hourly data by DOY
    accum = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    count = defaultdict(int)

    print(f"Downloading ERA5 from Open-Meteo ({LAT}N, {LON}E) for {YEARS} ...")
    for year in YEARS:
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
            "timezone": "UTC",
        }
        print(f"  {year}: fetching ...", end=" ", flush=True)
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        hourly = data.get("hourly", {})
        times   = hourly.get("time", [])
        temp_c  = np.array(hourly.get("temperature_2m", []), dtype=np.float64)
        wspd    = np.array(hourly.get("wind_speed_10m", []), dtype=np.float64)
        wdir    = np.array(hourly.get("wind_direction_10m", []), dtype=np.float64)
        rh      = np.array(hourly.get("relative_humidity_2m", []), dtype=np.float64)

        n_ok = 0
        for i, t_str in enumerate(times):
            if i >= len(temp_c):
                break
            r = math.radians(wdir[i]) if not math.isnan(wdir[i]) else 0.0
            vals = np.array([
                temp_c[i] + 273.15,          # °C → K
                wspd[i] * math.sin(r),        # u (eastward)
                wspd[i] * math.cos(r),        # v (northward)
                rh[i],
            ])
            if np.any(np.isnan(vals)):
                continue
            doy = datetime.datetime.fromisoformat(t_str).timetuple().tm_yday
            accum[doy] += vals
            count[doy] += 1
            n_ok += 1
        print(f"OK ({n_ok} valid hours)")

    # Save DOY files
    H, W = SPATIAL_SIZE
    saved = skipped = 0
    for doy in sorted(accum.keys()):
        if count[doy] == 0:
            continue
        out_path = CLIM_DIR / f"doy_{doy:03d}.npy"
        if out_path.exists():
            # Verify shape
            existing_arr = np.load(str(out_path))
            if existing_arr.shape == (4, H, W):
                skipped += 1
                continue
            # Wrong shape — overwrite
            print(f"  DOY {doy:03d}: fixing shape {existing_arr.shape} → (4,{H},{W})")

        avg = (accum[doy] / count[doy]).astype(np.float32)
        # Broadcast to (4, H, W)
        arr = avg[:, None, None] * np.ones((1, H, W), dtype=np.float32)
        np.save(str(out_path), arr)
        saved += 1

    total = len(list(CLIM_DIR.glob("doy_*.npy")))
    print(f"\nDone: {saved} new + {skipped} existing = {total} total DOY files")
    print(f"Location: {CLIM_DIR.resolve()}")

    # Quick validation
    files = sorted(CLIM_DIR.glob("doy_*.npy"))
    if files:
        sample = np.load(str(files[0]))
        print(f"Sample {files[0].name}: shape={sample.shape} dtype={sample.dtype}")
        print(f"  temp={sample[0,128,128]:.1f}K  u={sample[1,128,128]:.2f}m/s  "
              f"v={sample[2,128,128]:.2f}m/s  rh={sample[3,128,128]:.1f}%")


if __name__ == "__main__":
    os.makedirs(CLIM_DIR, exist_ok=True)
    build()
