"""Patch colab_weather_finetune.ipynb to use Google Weather API."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "colab_weather_finetune.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# New WN2 cell source (cell 11)
new_wn2_src = """\
# --- 4b: WeatherNext2 via Google Weather API + ERA5 fallback ---
# SECURITY: Store key in Colab Secrets (key icon in left sidebar).
#   Secret name: GOOGLE_WEATHER_API_KEY
import os
from google.colab import userdata

try:
    GOOGLE_API_KEY = userdata.get("GOOGLE_WEATHER_API_KEY")
    os.environ["GOOGLE_WEATHER_API_KEY"] = GOOGLE_API_KEY
    print(f"Google Weather API key loaded (len={len(GOOGLE_API_KEY)})")
except Exception:
    GOOGLE_API_KEY = None
    print("No GOOGLE_WEATHER_API_KEY in Colab Secrets.")
    print("Add it: left sidebar -> key icon -> Add new secret.")
    print("Falling back to ERA5 DOY climatology for WN2 source.")

WN2_CLIM  = "/content/cache/weathernext2/climatology"
WN2_CACHE = "/content/cache/weathernext2"
os.makedirs(WN2_CLIM, exist_ok=True)

DRIVE_WN2_CLIM = f"{PROJECT_PATH}/cache/weathernext2/climatology"
drive_wn2 = sorted(Path(DRIVE_WN2_CLIM).glob("doy_*.npy")) if Path(DRIVE_WN2_CLIM).exists() else []

if drive_wn2:
    print(f"Copying {len(drive_wn2)} WN2 DOY files from Drive ...")
    for f in drive_wn2:
        dest = Path(WN2_CLIM) / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
        arr = np.load(str(dest))
        if arr.ndim == 1:
            np.save(str(dest), arr[:, None, None] * np.ones((1, 256, 256), dtype=np.float32))
    print(f"WN2 climatology: {len(list(Path(WN2_CLIM).glob('doy_*.npy')))} files")
else:
    print("Building ERA5 DOY climatology from Open-Meteo (~3 min) ...")
    import requests, math, datetime
    from collections import defaultdict
    LAT_WN2, LON_WN2 = 37.0, -119.0
    accum2 = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    count2 = defaultdict(int)
    for year in [2017, 2018, 2019, 2020, 2021]:
        params = {
            "latitude": LAT_WN2, "longitude": LON_WN2,
            "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
            "timezone": "UTC",
        }
        print(f"  {year}: fetching ...", end=" ", flush=True)
        try:
            resp = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"FAILED ({e})"); continue
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        tc = np.array(hourly.get("temperature_2m", []), dtype=np.float64)
        ws = np.array(hourly.get("wind_speed_10m", []), dtype=np.float64)
        wd = np.array(hourly.get("wind_direction_10m", []), dtype=np.float64)
        rh = np.array(hourly.get("relative_humidity_2m", []), dtype=np.float64)
        n_ok = 0
        for i, t_str in enumerate(times):
            r = math.radians(wd[i])
            vals = np.array([tc[i]+273.15, ws[i]*math.sin(r), ws[i]*math.cos(r), rh[i]])
            if np.any(np.isnan(vals)):
                continue
            doy = datetime.datetime.fromisoformat(t_str).timetuple().tm_yday
            accum2[doy] += vals
            count2[doy] += 1
            n_ok += 1
        print(f"OK ({n_ok} hours)")
    saved2 = 0
    for doy in sorted(accum2):
        if count2[doy] == 0:
            continue
        avg = (accum2[doy] / count2[doy]).astype(np.float32)
        np.save(str(Path(WN2_CLIM) / f"doy_{doy:03d}.npy"),
                avg[:, None, None] * np.ones((1, 256, 256), dtype=np.float32))
        saved2 += 1
    print(f"WN2 ERA5 climatology: {saved2} DOY files -> {WN2_CLIM}")

# Test Google Weather API
if GOOGLE_API_KEY:
    import requests as _req
    try:
        r = _req.get(
            "https://weather.googleapis.com/v1/forecast/hours:lookup",
            params={"key": GOOGLE_API_KEY, "location.latitude": 37.0,
                    "location.longitude": -119.0, "hours": 3},
            timeout=15,
        )
        r.raise_for_status()
        hrs = r.json().get("forecastHours", [])
        h0  = hrs[0] if hrs else {}
        temp = h0.get("temperature", {}).get("degrees", "?")
        rh0  = h0.get("relativeHumidity", "?")
        print(f"Google Weather API OK: {len(hrs)} hours, temp={temp}C  rh={rh0}%")
        print("Live forecasts ENABLED for WeatherNext2 source.")
    except Exception as e:
        print(f"Google Weather API test FAILED: {e}")
        GOOGLE_API_KEY = None

# Verify shapes
print("\\n=== Weather Data Ready ===")
for label, clim_dir in [("HRRR", HRRR_CLIM), ("WN2", WN2_CLIM)]:
    files = sorted(Path(clim_dir).glob("doy_*.npy"))
    if files:
        s = np.load(str(files[0]))
        ok = s.ndim == 3 and s.shape == (4, 256, 256)
        print(f"  {label:5s}: {len(files)} files  shape={s.shape}  {'OK' if ok else 'FIXING...'}")
        if not ok:
            for ff in files:
                a = np.load(str(ff))
                if a.ndim == 1:
                    np.save(str(ff), a[:, None, None] * np.ones((1, 256, 256), dtype=np.float32))
            print(f"         Fixed all files to (4, 256, 256)")
    else:
        print(f"  {label:5s}: NO FILES!")
print(f"  Google Weather API: {'LIVE' if GOOGLE_API_KEY else 'ERA5 climatology fallback'}")
"""

# Find and replace WN2 cell
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "4b: Build WeatherNext2" in src or "4b: WeatherNext2" in src:
        nb["cells"][i]["source"] = [line + "\n" for line in new_wn2_src.splitlines()[:-1]] + [new_wn2_src.splitlines()[-1]]
        print(f"Replaced WN2 cell {i}")
        break

# Update dataset cell: pass api_key to WeatherNext2FutureProvider
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "===== BUILD DATASETS =====" in src:
        old = (
            '    wn2_prov = WeatherNext2FutureProvider(\n'
            '        cache_dir="/content/cache/weathernext2", spatial_size=(IMG_SIZE, IMG_SIZE)\n'
            '    )\n'
            '    n_wn2 = len(wn2_prov._downloader._doy_data) if wn2_prov._downloader else 0\n'
            '    print(f"WeatherNext2 provider: {n_wn2} DOY frames")'
        )
        new = (
            '    _wn2_key = os.environ.get("GOOGLE_WEATHER_API_KEY")\n'
            '    wn2_prov = WeatherNext2FutureProvider(\n'
            '        cache_dir="/content/cache/weathernext2",\n'
            '        spatial_size=(IMG_SIZE, IMG_SIZE),\n'
            '        api_key=_wn2_key,\n'
            '    )\n'
            '    n_wn2 = len(wn2_prov._downloader._doy_data) if wn2_prov._downloader else 0\n'
            '    has_api = bool(wn2_prov._downloader and wn2_prov._downloader._api_key)\n'
            '    print(f"WeatherNext2 provider: {n_wn2} DOY frames, live API={has_api}")'
        )
        if old in src:
            src = src.replace(old, new)
            nb["cells"][i]["source"] = [line + "\n" for line in src.splitlines()[:-1]] + [src.splitlines()[-1]]
            print(f"Updated dataset cell {i} with api_key")
        else:
            print(f"Dataset cell {i}: pattern not found (may already be updated)")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved")
