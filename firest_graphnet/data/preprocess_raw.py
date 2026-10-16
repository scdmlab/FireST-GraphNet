"""
preprocess_raw.py —— 从 TS-SatFire 原始 GeoTIFF 生成训练用 npy 文件。

TS-SatFire 原始数据结构 (Kaggle)
─────────────────────────────────
  ts-satfire/
    <event_id>/
      VIIRS_Day/      — 每天一个 8-band GeoTIFF
      VIIRS_Night/    — 每天一个 2-band GeoTIFF
      FirePred/       — 每天一个 19-band GeoTIFF (气象 + 地形 + 燃料)
      ESRI_LULC/      — 每年一个 1-band GeoTIFF (土地利用)

TS-SatFire VIIRS_Day 8 bands (per paper Table 1)
─────────────────────────────────────────────────
  Band 0: I1   (0.64 µm)   — visible reflectance
  Band 1: I2   (0.862 µm)  — NIR reflectance
  Band 2: I3   (1.61 µm)   — SWIR reflectance
  Band 3: M13  (4.05 µm)   — MWIR brightness temperature
  Band 4: M15  (10.76 µm)  — LWIR brightness temperature
  Band 5: M11  (2.25 µm)   — SWIR reflectance
  Band 6: (AF detection)    — used to derive AF label, NOT model input
  Band 7: (BA label)        — used to derive BA label, NOT model input

VIIRS_Night 2 bands
────────────────────
  Band 0: I4   (3.74 µm)   — MWIR brightness temperature
  Band 1: I5   (11.45 µm)  — LWIR brightness temperature

FirePred 19 bands (weather + topography + fuel)
────────────────────────────────────────────────
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

ESRI_LULC 1 band
─────────────────
  Class  1: No data
  Class  2: Trees
  Class  4: Flooded Vegetation
  Class  5: Crops
  Class  7: Built Area
  Class  8: Bare Ground
  Class  9: Snow/Ice
  Class 10: Clouds
  Class 11: Rangeland

输出格式
────────
  af_<split>_img_seqtoseq_alll_1i_1.npy    — (N, 8, T, H, W)  VIIRS image bands
  af_<split>_label_seqtoseq_alll_1i_1.npy  — (N, 3, T, H, W)  labels (BA, AF_acc, AF)
  firepred_<split>.npy                      — (N, 19, T, H, W) weather / topo / fuel
  lulc_<split>.npy                          — (N, 1, H, W)     land cover (static)

Usage
─────
  python -m firest_graphnet.data.preprocess_raw \\
      --raw_dir /path/to/ts-satfire_raw \\
      --out_dir /path/to/dataset_train \\
      --mode train \\
      --ts 1 --interval 1 \\
      --image_size 256
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  TIFF reader (rasterio or fallback)
# ═══════════════════════════════════════════════════════════════════════════
try:
    import rasterio

    def read_tiff(path: str) -> np.ndarray:
        with rasterio.open(path, "r") as src:
            return src.read().astype(np.float32)  # (C, H, W)

    def read_tiff_and_resample(path: str, target_h: int, target_w: int) -> np.ndarray:
        """Read a TIFF and resample to target resolution."""
        from rasterio.enums import Resampling
        with rasterio.open(path, "r") as src:
            arr = src.read(
                out_shape=(src.count, target_h, target_w),
                resampling=Resampling.nearest,
            ).astype(np.float32)
        return arr

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

    def read_tiff(path: str) -> np.ndarray:
        raise ImportError("rasterio is required. Install via: pip install rasterio")

    def read_tiff_and_resample(path: str, target_h: int, target_w: int) -> np.ndarray:
        raise ImportError("rasterio is required.")


# ═══════════════════════════════════════════════════════════════════════════
#  Core : process a single event folder
# ═══════════════════════════════════════════════════════════════════════════
def _center_crop(arr: np.ndarray, h: int, w: int, offset: int = 128) -> np.ndarray:
    """Center-crop with offset = 128 (matching TS-SatFire default)."""
    return arr[..., offset : offset + h, offset : offset + w]


def process_event(
    event_dir: str | Path,
    ts_length: int = 1,
    interval: int = 1,
    image_size: int = 256,
    usecase: str = "af",
) -> dict | None:
    """Process one event folder → dict of stacked arrays.

    Returns
    -------
    dict with keys:
        'viirs'    : (N_windows, ts_length, 8, H, W)
        'labels'   : (N_windows, ts_length, 3, H, W)
        'firepred' : (N_windows, ts_length, 19, H, W) or None
        'lulc'     : (1, H, W) or None
    or None if the event cannot be processed.
    """
    event_dir = Path(event_dir)
    day_dir = event_dir / "VIIRS_Day"
    night_dir = event_dir / "VIIRS_Night"
    pred_dir = event_dir / "FirePred"
    lulc_dir = event_dir / "ESRI_LULC"

    # List VIIRS_Day TIFFs (sorted by date)
    day_files = sorted(glob(str(day_dir / "*.tif")))
    if not day_files:
        logger.warning(f"Skipping {event_dir.name}: no VIIRS_Day TIFFs found")
        return None

    n_files = len(day_files)
    H, W = image_size, image_size
    offset = 128

    # Read first to get original dims
    first_day = read_tiff(day_files[0])
    orig_h, orig_w = first_day.shape[1], first_day.shape[2]

    # Whether FirePred exists
    has_firepred = pred_dir.exists() and len(glob(str(pred_dir / "*.tif"))) > 0

    # ESRI LULC (static, one per event — resample to VIIRS resolution)
    lulc_arr = None
    lulc_files = sorted(glob(str(lulc_dir / "*.tif")))
    if lulc_files:
        try:
            lulc_raw = read_tiff_and_resample(lulc_files[0], orig_h, orig_w)
            lulc_arr = _center_crop(lulc_raw, H, W, offset)
            # Normalize class IDs to [0, 1] range (max class = 11)
            lulc_arr = np.nan_to_num(lulc_arr / 11.0).astype(np.float32)
        except Exception as e:
            logger.warning(f"LULC read failed for {event_dir.name}: {e}")
            lulc_arr = None

    # BA / AF accumulated labels tracking
    ba_label = np.zeros((H, W), dtype=np.float32)
    af_acc_label = np.zeros((H, W), dtype=np.float32)
    new_base_acc = af_acc_label.copy()
    new_base_ba = ba_label.copy()

    viirs_windows = []
    label_windows = []
    firepred_windows = []

    for i in range(0, n_files, interval):
        if i + ts_length > n_files:
            i = n_files - ts_length
            if i < 0:
                break

        out_viirs = np.zeros((ts_length, 8, H, W), dtype=np.float32)
        out_labels = np.zeros((ts_length, 3, H, W), dtype=np.float32)
        out_pred = np.zeros((ts_length, 19, H, W), dtype=np.float32) if has_firepred else None

        valid = True
        for j in range(ts_length):
            file = day_files[i + j]

            # ── VIIRS Day (8 bands) ──────────────────────────────────
            arr_day = read_tiff(file)
            if arr_day.shape[0] < 7:
                logger.warning(f"Skip {file}: only {arr_day.shape[0]} bands")
                valid = False
                break

            # ── VIIRS Night (2 bands) ────────────────────────────────
            night_file = file.replace("VIIRS_Day", "VIIRS_Night")
            if os.path.exists(night_file):
                arr_night = read_tiff(night_file)
                if arr_night.shape[0] == 5:
                    arr_night = arr_night[3:, :, :]
                if arr_night.shape[0] < 2:
                    arr_night = np.zeros((2, orig_h, orig_w), dtype=np.float32)
                if arr_night.shape[1] != orig_h or arr_night.shape[2] != orig_w:
                    arr_night = np.zeros((2, orig_h, orig_w), dtype=np.float32)
            else:
                arr_night = np.zeros((2, orig_h, orig_w), dtype=np.float32)

            # Image: Day[:6] + Night[:2] = 8 bands (matching TS-SatFire AF usecase)
            img = np.concatenate(
                [
                    arr_day[:6, offset : offset + H, offset : offset + W],
                    arr_night[:2, offset : offset + H, offset : offset + W],
                ],
                axis=0,
            )
            img = np.nan_to_num(img[:, :H, :W])

            # For AF usecase: keep raw values (not max-accumulated)
            if usecase == "af":
                img = np.concatenate(
                    [img[:3], img[3:5], img[[5]], img[6:8]], axis=0
                )

            # ── Labels ───────────────────────────────────────────────
            if arr_day.shape[0] >= 8:
                ba_raw = np.nan_to_num(
                    arr_day[7, offset : offset + H, offset : offset + W], nan=-1
                )
            else:
                ba_raw = np.zeros((H, W), dtype=np.float32)

            af_raw = np.nan_to_num(
                arr_day[6, offset : offset + H, offset : offset + W]
            )

            ba_label = np.logical_or(ba_raw > 0, ba_label).astype(np.float32)
            af_acc_label = np.logical_or(af_raw > 0, af_acc_label).astype(
                np.float32
            )

            if j == interval - 1:
                new_base_acc = af_acc_label.copy()
                new_base_ba = ba_label.copy()

            out_viirs[j] = img
            out_labels[j, 0] = ba_raw  # BA raw (-1 for no data)
            out_labels[j, 1] = af_acc_label  # AF accumulated binary mask
            out_labels[j, 2] = af_raw  # AF raw detection

            # ── FirePred (19 bands) ──────────────────────────────────
            if has_firepred and out_pred is not None:
                pred_file = file.replace("VIIRS_Day", "FirePred")
                if os.path.exists(pred_file):
                    try:
                        arr_pred = read_tiff(pred_file)
                        # FirePred may have different spatial dims → resample
                        if arr_pred.shape[1] != orig_h or arr_pred.shape[2] != orig_w:
                            arr_pred = read_tiff_and_resample(
                                pred_file, orig_h, orig_w
                            )
                        arr_pred = arr_pred[:19, offset : offset + H, offset : offset + W]
                        arr_pred = np.nan_to_num(arr_pred[:, :H, :W])
                        out_pred[j] = arr_pred
                    except Exception as e:
                        logger.warning(f"FirePred read failed: {e}")

        if not valid:
            continue

        af_acc_label = new_base_acc
        ba_label = new_base_ba

        viirs_windows.append(out_viirs)
        label_windows.append(out_labels)
        if out_pred is not None:
            firepred_windows.append(out_pred)

    if not viirs_windows:
        logger.warning(f"No windows generated for {event_dir.name}")
        return None

    result = {
        "viirs": np.stack(viirs_windows, axis=0),  # (N, T, 8, H, W)
        "labels": np.stack(label_windows, axis=0),  # (N, T, 3, H, W)
    }

    if firepred_windows:
        result["firepred"] = np.stack(firepred_windows, axis=0)  # (N, T, 19, H, W)

    if lulc_arr is not None:
        result["lulc"] = lulc_arr  # (1, H, W)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main : process all events and save combined npy
# ═══════════════════════════════════════════════════════════════════════════
def process_all_events(
    raw_dir: str | Path,
    out_dir: str | Path,
    event_ids: List[str],
    mode: str = "train",
    ts_length: int = 1,
    interval: int = 1,
    image_size: int = 256,
    usecase: str = "af",
):
    """Process all events in raw_dir and save combined npy files.

    Saves to out_dir:
        af_{mode}_img_seqtoseq_alll_1i_1.npy     — (N, 8, T, H, W) VIIRS
        af_{mode}_label_seqtoseq_alll_1i_1.npy   — (N, 3, T, H, W) labels
        firepred_{mode}.npy                        — (N, 19, T, H, W) auxiliary
        lulc_{mode}.npy                            — (N, 1, H, W) static land cover
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_viirs = []
    all_labels = []
    all_firepred = []
    all_lulc = []

    for eid in event_ids:
        event_dir = raw_dir / str(eid)
        if not event_dir.exists():
            logger.warning(f"Event directory not found: {event_dir}")
            continue

        print(f"  Processing {eid} ...", end=" ")
        result = process_event(
            event_dir,
            ts_length=ts_length,
            interval=interval,
            image_size=image_size,
            usecase=usecase,
        )

        if result is None:
            print("SKIP")
            continue

        n = result["viirs"].shape[0]
        all_viirs.append(result["viirs"])
        all_labels.append(result["labels"])

        if "firepred" in result:
            all_firepred.append(result["firepred"])
        else:
            # Pad with zeros to keep alignment
            shape = list(result["viirs"].shape)
            shape[2] = 19  # (N, T, 19, H, W)
            all_firepred.append(np.zeros(shape, dtype=np.float32))

        if "lulc" in result:
            # Repeat LULC for each window (static → same for all windows of this event)
            lulc_repeat = np.tile(result["lulc"], (n, 1, 1, 1))  # (N, 1, H, W)
            all_lulc.append(lulc_repeat)
        else:
            shape_lulc = (n, 1, image_size, image_size)
            all_lulc.append(np.zeros(shape_lulc, dtype=np.float32))

        print(f"OK ({n} windows)")

    if not all_viirs:
        raise RuntimeError("No events processed successfully!")

    # Concatenate and transpose to match TS-SatFire format: (N, C, T, H, W)
    viirs_all = np.concatenate(all_viirs, axis=0).transpose(
        (0, 2, 1, 3, 4)
    )  # (N, 8, T, H, W)
    labels_all = np.concatenate(all_labels, axis=0).transpose(
        (0, 2, 1, 3, 4)
    )  # (N, 3, T, H, W)

    # Save VIIRS + labels (compatible with existing af_* format)
    suffix = f"seqtoseq_alll_{ts_length}i_{interval}"
    viirs_path = out_dir / f"af_{mode}_img_{suffix}.npy"
    labels_path = out_dir / f"af_{mode}_label_{suffix}.npy"
    np.save(str(viirs_path), viirs_all.astype(np.float32))
    np.save(str(labels_path), labels_all.astype(np.float32))
    print(f"\nSaved VIIRS     : {viirs_path}  shape={viirs_all.shape}")
    print(f"Saved Labels    : {labels_path}  shape={labels_all.shape}")

    # Save FirePred auxiliary
    if all_firepred:
        firepred_all = np.concatenate(all_firepred, axis=0).transpose(
            (0, 2, 1, 3, 4)
        )  # (N, 19, T, H, W)
        fp_path = out_dir / f"firepred_{mode}.npy"
        np.save(str(fp_path), firepred_all.astype(np.float32))
        print(f"Saved FirePred  : {fp_path}  shape={firepred_all.shape}")

    # Save LULC static features
    if all_lulc:
        lulc_all = np.concatenate(all_lulc, axis=0)  # (N, 1, H, W)
        lulc_path = out_dir / f"lulc_{mode}.npy"
        np.save(str(lulc_path), lulc_all.astype(np.float32))
        print(f"Saved LULC      : {lulc_path}  shape={lulc_all.shape}")

    return {
        "viirs": viirs_path,
        "labels": labels_path,
        "firepred": out_dir / f"firepred_{mode}.npy",
        "lulc": out_dir / f"lulc_{mode}.npy",
    }


def discover_event_ids(raw_dir: Path) -> List[str]:
    """Auto-discover event IDs from raw directory."""
    ids = []
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and (d / "VIIRS_Day").exists():
            ids.append(d.name)
    return ids


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TS-SatFire raw TIFFs → multi-source npy files"
    )
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Path to raw ts-satfire folder"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory for npy files"
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--ts", type=int, default=1, help="Time-series length")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval")
    parser.add_argument("--image_size", type=int, default=256, help="Output spatial size")

    args = parser.parse_args()

    if not HAS_RASTERIO:
        print("ERROR: rasterio is required. Install via: pip install rasterio")
        sys.exit(1)

    raw_dir = Path(args.raw_dir)
    event_ids = discover_event_ids(raw_dir)
    if not event_ids:
        print(f"No events found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(event_ids)} events in {raw_dir}")
    process_all_events(
        raw_dir=raw_dir,
        out_dir=args.out_dir,
        event_ids=event_ids,
        mode=args.mode,
        ts_length=args.ts,
        interval=args.interval,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
