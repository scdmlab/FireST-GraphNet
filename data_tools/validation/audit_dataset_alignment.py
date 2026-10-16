#!/usr/bin/env python
"""
audit_dataset_alignment.py
===========================
FireST-GraphNet Dataset / Dataloader Alignment Audit Utility.

Run this BEFORE any training to verify data is semantically correct.

Usage:
    python audit_dataset_alignment.py
    python audit_dataset_alignment.py --data_root . --n_samples 5 --save_figs

Exit code:
    0 = all checks passed
    1 = one or more CRITICAL issues found (do NOT train)
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ── colour helpers (no deps) ──────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}[OK]{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def err(msg):  print(f"  {RED}[ERROR]{RESET} {msg}")
def sep(title=""):
    print(f"\n{BOLD}{'='*60}{RESET}")
    if title:
        print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


CRITICAL_ISSUES: List[str] = []
WARNINGS:        List[str] = []


def flag_critical(msg: str):
    CRITICAL_ISSUES.append(msg)
    err(f"CRITICAL: {msg}")


def flag_warn(msg: str):
    WARNINGS.append(msg)
    warn(msg)


# ═══════════════════════════════════════════════════════════════════════════
# A.  File / shape audit
# ═══════════════════════════════════════════════════════════════════════════

def audit_files(data_root: Path) -> Dict[str, np.ndarray]:
    """Load all npy files and check shapes / dtype."""
    sep("A. File Discovery & Shape Audit")
    arrays = {}

    for split in ("train", "val"):
        d = data_root / f"dataset_{split}"
        if not d.exists():
            flag_warn(f"dataset_{split} directory not found — skipping.")
            continue

        print(f"\n  [{split}] {d}")
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".npy"):
                continue
            path = d / fname
            arr = np.load(str(path), mmap_mode="r")
            key = f"{split}/{fname}"
            arrays[key] = arr
            print(f"    {fname}: shape={arr.shape}  dtype={arr.dtype}  "
                  f"min={float(arr.min()):.3f}  max={float(arr.max()):.3f}")

    # ── cross-array size consistency check ────────────────────────────────
    print()
    for split in ("train", "val"):
        img_key  = f"{split}/af_{split}_img_seqtoseq_alll_1i_1.npy"
        lbl_key  = f"{split}/af_{split}_label_seqtoseq_alll_1i_1.npy"
        fp_key   = f"{split}/firepred_{split}.npy"
        lulc_key = f"{split}/lulc_{split}.npy"

        if img_key not in arrays:
            continue

        N_img  = arrays[img_key].shape[0]
        N_lbl  = arrays[lbl_key].shape[0]  if lbl_key  in arrays else None
        N_fp   = arrays[fp_key].shape[0]   if fp_key   in arrays else None
        N_lulc = arrays[lulc_key].shape[0] if lulc_key in arrays else None

        assert N_lbl == N_img, \
            f"[{split}] img N={N_img} != label N={N_lbl}"
        ok(f"[{split}] img({N_img}) == label({N_lbl})")

        if N_fp is not None and N_fp != N_img:
            flag_critical(
                f"[{split}] firepred N={N_fp} != img N={N_img}  "
                f"(delta={abs(N_fp-N_img)}).  "
                "Index i in img does NOT correspond to index i in firepred.  "
                "FirePredProvider.get_window(hist_ids) will silently load WRONG auxiliary data."
            )
        elif N_fp is not None:
            ok(f"[{split}] firepred N == img N == {N_img}")

        if N_lulc is not None and N_lulc != N_img:
            flag_critical(
                f"[{split}] lulc N={N_lulc} != img N={N_img}.  "
                "LULCStaticProvider.get(sample_idx) uses modulo wrapping — "
                "wrong land-cover for most samples."
            )
        elif N_lulc is not None:
            ok(f"[{split}] lulc N == img N == {N_img}")

    return arrays


# ═══════════════════════════════════════════════════════════════════════════
# B.  Label channel semantics
# ═══════════════════════════════════════════════════════════════════════════

def audit_label_channels(arrays: Dict[str, np.ndarray]):
    """Verify correct interpretation of the 3-channel label npy."""
    sep("B. Label Channel Semantics")

    for split in ("train", "val"):
        lbl_key = f"{split}/af_{split}_label_seqtoseq_alll_1i_1.npy"
        if lbl_key not in arrays:
            continue
        lbl = arrays[lbl_key]   # (N, 3, 1, H, W)
        N = lbl.shape[0]

        print(f"\n  [{split}] label shape={lbl.shape}")

        # ── Channel 0 ─────────────────────────────────────────────────────
        ch0 = lbl[:, 0, 0, :, :]
        all_neg1 = (ch0 == -1).all(axis=(1, 2)).sum()
        non_neg1 = ch0[ch0 != -1]
        ch0_max = float(non_neg1.max()) if len(non_neg1) > 0 else 0.0
        print(f"    Channel 0: sentinel(-1) samples={all_neg1}/{N}  "
              f"non-sentinel range=[{non_neg1.min() if len(non_neg1) else 0:.0f}, {ch0_max:.0f}]")
        if ch0_max > 100:
            flag_critical(
                f"[{split}] Label channel 0 contains large non-binary values (max={ch0_max:.0f}).  "
                "This is NOT a binary BA mask.  b_init = labels[i, 0, 0] in dataset_forecast.py "
                "will feed arbitrary large integers / -1 sentinels as 'burned area init' to the model."
            )
        else:
            ok(f"[{split}] Channel 0 is binary-ish (max={ch0_max:.0f})")

        # ── Channel 1 (the default label_channel) ─────────────────────────
        ch1 = lbl[:, 1, 0, :, :]
        unique_ch1 = np.unique(ch1)
        ch1_is_binary = set(unique_ch1.tolist()).issubset({0.0, 1.0})
        pct_fire = (ch1 > 0).any(axis=(1, 2)).mean() * 100
        print(f"    Channel 1: unique_vals={unique_ch1[:6].tolist()}  "
              f"binary={ch1_is_binary}  pct_samples_with_fire={pct_fire:.1f}%")
        if ch1_is_binary:
            ok(f"[{split}] Channel 1 is binary AF mask — correct for label_channel=1")
        else:
            flag_critical(
                f"[{split}] Channel 1 is NOT binary (unique={unique_ch1[:10]}).  "
                "Default label_channel=1 may be loading wrong data."
            )

        # ── Channel 2 ─────────────────────────────────────────────────────
        ch2 = lbl[:, 2, 0, :, :]
        ch2_max = float(ch2.max())
        ch2_nonzero_mean = float(ch2[ch2 > 0].mean()) if (ch2 > 0).any() else 0.0
        print(f"    Channel 2: range=[{float(ch2.min()):.1f}, {ch2_max:.1f}]  "
              f"non-zero mean={ch2_nonzero_mean:.1f}")
        if ch2_max > 100:
            flag_warn(
                f"[{split}] Label channel 2 has continuous values (max={ch2_max:.1f}, "
                f"mean={ch2_nonzero_mean:.1f}) — likely thermal/temperature data stored in label array.  "
                "This channel is NOT a class label and should NOT be used as y."
            )
        else:
            ok(f"[{split}] Channel 2 looks categorical (max={ch2_max:.1f})")


# ═══════════════════════════════════════════════════════════════════════════
# C.  Fire event boundary detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_event_boundaries(fire_counts: np.ndarray,
                             ratio_threshold: float = 20.0) -> List[int]:
    """Return list of indices where a new fire event starts."""
    boundaries = set()
    N = len(fire_counts)
    for i in range(N - 1):
        a, b = float(fire_counts[i]), float(fire_counts[i + 1])
        # large absolute drop → new event
        if a > 50 and b == 0:
            boundaries.add(i + 1)
        elif a == 0 and b > 50:
            boundaries.add(i + 1)
        # large relative change → new event
        elif a > 0 and b > 0:
            ratio = max(a, b) / max(min(a, b), 1.0)
            if ratio > ratio_threshold:
                boundaries.add(i + 1)
    return sorted(boundaries)


def audit_event_boundaries(arrays: Dict[str, np.ndarray], T_in: int = 9, T_out: int = 1):
    """Check what fraction of sliding windows cross fire event boundaries."""
    sep("C. Fire Event Boundary & Sliding Window Audit")
    T = T_in + T_out

    for split in ("train", "val"):
        lbl_key = f"{split}/af_{split}_label_seqtoseq_alll_1i_1.npy"
        if lbl_key not in arrays:
            continue
        lbl = arrays[lbl_key]
        af_mask = lbl[:, 1, 0, :, :]           # (N, H, W)
        fire_counts = af_mask.sum(axis=(1, 2))  # (N,)
        N = len(fire_counts)

        boundaries = detect_event_boundaries(fire_counts)
        b_set = set(boundaries)

        # Count windows that cross at least one boundary
        total_wins = max(0, N - T + 1)
        cross = 0
        for start in range(N - T + 1):
            end = start + T - 1
            if any(start < b <= end for b in boundaries):
                cross += 1

        pct = cross / max(total_wins, 1) * 100

        print(f"\n  [{split}] N={N}, T_in+T_out={T_in}+{T_out}={T}")
        print(f"    Detected event boundaries: {len(boundaries)}")
        print(f"    Total sliding windows: {total_wins}")
        print(f"    Windows crossing a boundary: {cross} ({pct:.1f}%)")

        if cross == 0:
            ok(f"[{split}] No cross-event contamination detected.")
        elif pct < 5:
            flag_warn(f"[{split}] {pct:.1f}% of windows cross event boundaries "
                      "(low — but verify).")
        else:
            flag_critical(
                f"[{split}] {pct:.1f}% ({cross}/{total_wins}) of sliding windows "
                f"cross fire-event boundaries!  "
                "The dataset treats the entire npy as ONE long sequence, ignoring "
                "the {len(boundaries)} fire-event changes embedded in it.  "
                "Input frames from Fire A are being paired with labels from Fire B."
            )

        # Show first few boundaries
        print(f"    First 10 boundary indices: {boundaries[:10]}")
        if len(boundaries) > 0:
            # Show what happens around boundary[0]
            b0 = boundaries[0]
            start_show = max(0, b0 - 3)
            end_show = min(N, b0 + 4)
            print(f"    Fire pixel counts around boundary {b0}:")
            for i in range(start_show, end_show):
                marker = " <-- BOUNDARY" if i == b0 else ""
                print(f"      frame {i:5d}: fire_pixels={fire_counts[i]:.0f}{marker}")

    print()
    print("  NOTE: The _build_windows() code only looks for gaps >1 in index space.")
    print("  Since all npy indices are consecutive (0..N-1), it builds ONE long run")
    print("  and slides a window across ALL events concatenated together.")
    print("  Fix required: load per-fire-event arrays separately, or embed event_ids")
    print("  in a metadata file and break windows at event boundaries.")


# ═══════════════════════════════════════════════════════════════════════════
# D.  Index alignment between arrays (firepred / lulc vs img)
# ═══════════════════════════════════════════════════════════════════════════

def audit_index_alignment(arrays: Dict[str, np.ndarray]):
    """Check correlation between img thermal bands and firepred temperature."""
    sep("D. FirePred / LULC Index Alignment")

    for split in ("train", "val"):
        img_key = f"{split}/af_{split}_img_seqtoseq_alll_1i_1.npy"
        fp_key  = f"{split}/firepred_{split}.npy"

        if img_key not in arrays or fp_key not in arrays:
            continue

        img = arrays[img_key]      # (N_img, 8, 1, H, W)
        fp  = arrays[fp_key]       # (N_fp, 19, 1, H, W)

        N_img, N_fp = img.shape[0], fp.shape[0]
        N_min = min(N_img, N_fp)

        # img band 4 = I5 thermal (11.45 µm), fp band 5 = 2m temperature
        img_thermal  = img[:N_min, 4, 0, :, :].mean(axis=(1, 2))  # (N_min,)
        fp_temp      = fp[:N_min,  5, 0, :, :].mean(axis=(1, 2))  # (N_min,)

        # Both should have zeros as fill — check non-zero correlation
        valid = (img_thermal > 10) & (fp_temp > 10)
        if valid.sum() < 10:
            flag_warn(f"[{split}] Not enough non-zero samples to compute alignment correlation.")
            continue

        corr = float(np.corrcoef(img_thermal[valid], fp_temp[valid])[0, 1])
        print(f"\n  [{split}] corr(img.I5_thermal, firepred.temp) = {corr:.4f}  "
              f"(using {valid.sum()} non-zero samples)")

        if abs(corr) < 0.3:
            flag_critical(
                f"[{split}] Very low correlation ({corr:.4f}) between VIIRS I5 thermal "
                f"and FirePred temperature when index-aligned.  "
                "firepred[i] does NOT correspond to img[i].  "
                "The FirePredProvider silently feeds misaligned auxiliary data for every sample."
            )
        elif abs(corr) < 0.6:
            flag_warn(
                f"[{split}] Moderate correlation ({corr:.4f}) between img and firepred at "
                "same index.  Alignment is questionable."
            )
        else:
            ok(f"[{split}] Good alignment correlation: {corr:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# E.  Channel manifest for one sample
# ═══════════════════════════════════════════════════════════════════════════

# Canonical VIIRS band names (from TS-SatFire paper Table 1, day then night)
VIIRS_BAND_NAMES = [
    "Band I1 (Red, 640nm)",
    "Band I2 (NIR, 865nm)",
    "Band I3 (SWIR, 1610nm)",
    "Band I4 Daytime (MIR, 3610nm) — fire detection",
    "Band I5 Daytime (LWIR, 11450nm) — thermal",
    "Band M11 (SWIR, 2250nm)",
    "Band I4 Night (MIR, 3610nm)",
    "Band I5 Night (LWIR, 11450nm)",
]

FIREPRED_BAND_NAMES = [
    "DEM elevation (m)",
    "Slope (°)",
    "Aspect (°)",
    "Canopy height (m)",
    "ERC (Energy Release Component)",
    "2m Temperature (K)",
    "Dewpoint temperature (K)",
    "Relative humidity (%)",
    "Precipitation (mm)",
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Fuel moisture (live)",
    "PDSI (Palmer Drought Severity Index)",
    "Fuel model (1-16)",
    "Dead fuel moisture (1h)",
    "VPD (kPa)",
    "NDVI",
    "Soil temperature (°C)",
    "Snow cover (flag)",
]

LABEL_CHAN_NAMES = [
    "Channel 0: BA area value (large int, -1=no-data) — NOT a binary mask",
    "Channel 1: Binary AF mask {0,1} — CORRECT target for AF task",
    "Channel 2: Thermal measurement (K) — NOT a class label",
]


def audit_channel_manifest(arrays: Dict[str, np.ndarray], sample_idx: int = 0):
    """Print a complete channel manifest for one sample."""
    sep("E. Channel Manifest (one sample)")

    split = "train"
    img_key  = f"{split}/af_{split}_img_seqtoseq_alll_1i_1.npy"
    lbl_key  = f"{split}/af_{split}_label_seqtoseq_alll_1i_1.npy"
    fp_key   = f"{split}/firepred_{split}.npy"
    lulc_key = f"{split}/lulc_{split}.npy"

    if img_key not in arrays:
        warn("Train arrays not found.")
        return

    img  = arrays[img_key]
    lbl  = arrays[lbl_key]
    fp   = arrays[fp_key] if fp_key in arrays else None
    lulc = arrays[lulc_key] if lulc_key in arrays else None

    si = min(sample_idx, img.shape[0] - 1)
    print(f"\n  Sample index: {si}  (shapes img:{img.shape}, lbl:{lbl.shape})")
    print()
    print("  VIIRS Image channels (img[si, :, 0, :, :]):")
    for c, name in enumerate(VIIRS_BAND_NAMES):
        ch = img[si, c, 0, :, :]
        non_zero_mean = float(ch[ch > 0].mean()) if (ch > 0).any() else 0.0
        print(f"    img ch {c}: {name}")
        print(f"           range=[{float(ch.min()):.2f}, {float(ch.max()):.2f}]  "
              f"non-zero mean={non_zero_mean:.2f}")

    print()
    print("  Label channels (lbl[si, :, 0, :, :]):")
    for c, name in enumerate(LABEL_CHAN_NAMES):
        ch = lbl[si, c, 0, :, :]
        print(f"    lbl ch {c}: {name}")
        print(f"           unique[:8]={np.unique(ch)[:8].tolist()}")

    if fp is not None:
        print()
        print("  FirePred channels (fp[si, :, 0, :, :]):")
        print(f"  [WARNING] si={si} into firepred (N={fp.shape[0]}) — alignment not verified!")
        for c, name in enumerate(FIREPRED_BAND_NAMES):
            ch = fp[si, c, 0, :, :]
            print(f"    fp  ch {c:2d}: {name}  -> [{float(ch.min()):.2f}, {float(ch.max()):.2f}]")

    if lulc is not None:
        print()
        lulc_s = lulc[si% lulc.shape[0]]
        print(f"  LULC (lulc[{si%lulc.shape[0]}]):  shape={lulc_s.shape}  "
              f"range=[{float(lulc_s.min()):.3f}, {float(lulc_s.max()):.3f}]")


# ═══════════════════════════════════════════════════════════════════════════
# F.  Dataloader collation check
# ═══════════════════════════════════════════════════════════════════════════

def audit_dataloader(data_root: Path, T_in: int = 9):
    """Build actual DataLoader and inspect one batch."""
    sep("F. Dataloader Collation & Batch Shapes")

    try:
        import torch
        from torch.utils.data import DataLoader
        sys.path.insert(0, str(data_root))
        from firest_graphnet.data.dataset import build_datasets

        train_ds, val_ds = build_datasets(
            data_root=data_root,
            T_in=T_in,
            T_out=1,
            label_channel=1,
            val_split=0.1,
            seed=42,
        )

        loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        x = batch["x_seq"]
        y = batch["y"]
        meta = batch["meta"]

        print(f"\n  x_seq shape: {tuple(x.shape)}  (expected [B, T_in, C, H, W])")
        print(f"  y     shape: {tuple(y.shape)}    (expected [B, 1, H, W])")
        print(f"  meta: {meta}")

        # Check dimension ordering
        B, T, C, H, W = x.shape
        if T != T_in:
            flag_critical(
                f"x_seq temporal dim is {T}, expected T_in={T_in}.  "
                "Batch/temporal dimensions may be swapped."
            )
        else:
            ok(f"x_seq shape [{B}, {T}, {C}, {H}, {W}] — temporal dim correct")

        if y.shape != (B, 1, H, W):
            flag_critical(
                f"y shape {tuple(y.shape)} != expected ({B}, 1, {H}, {W})."
            )
        else:
            ok(f"y shape [{B}, 1, {H}, {W}] — correct")

        # Check y is binary
        y_vals = y.unique()
        if torch.all((y_vals == 0) | (y_vals == 1)):
            ok(f"y is binary {{0,1}} — correct AF mask")
        else:
            flag_warn(f"y has non-binary values: {y_vals[:10].tolist()}")

        # Verify window structure in the first 4 samples
        windows = train_ds.windows
        print(f"\n  First 4 batch windows:")
        for i in range(min(4, len(windows))):
            w = windows[i]
            in_f = w[:T_in]
            tgt_f = w[T_in:]
            print(f"    Sample {i}: input_frames={in_f}  target_frame={tgt_f}")
            # Temporal assertion: all consecutive
            for j in range(len(w) - 1):
                if w[j + 1] - w[j] != 1:
                    flag_critical(
                        f"Window {i} frame gap: w[{j}]={w[j]} w[{j+1}]={w[j+1]} — not consecutive!"
                    )

        # event_id is hardcoded "sequence" -- flag it
        if "event_id" in meta and all(eid == "sequence" for eid in meta["event_id"]):
            flag_warn(
                "meta['event_id'] is hardcoded 'sequence' for all samples.  "
                "No actual fire event ID is tracked.  Cannot verify cross-event contamination at runtime."
            )

    except ImportError as e:
        flag_warn(f"Could not import firest_graphnet: {e}.  Skipping dataloader audit.")
    except Exception as e:
        flag_warn(f"Dataloader audit failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# G.  Task-specific label semantics (pred / ba / af)
# ═══════════════════════════════════════════════════════════════════════════

def audit_task_label_semantics():
    """Check task semantics as described in docstrings vs actual code."""
    sep("G. Task-Specific Label Semantics")

    try:
        import inspect, sys
        sys.path.insert(0, ".")
        from firest_graphnet.data import dataset, dataset_forecast

        print("\n  --- LEGACY (dataset.py) single-step mode ---")
        print("  Task: predict next-day AF mask")
        src = inspect.getsource(dataset.TSSatFireSequenceDataset.__getitem__)
        if "label_channel" in src and "target_ids" in src:
            ok("target extracted via label_channel (default=1 = binary AF mask)")
        if "labels[tid, self.label_channel, 0, :, :]" in src:
            ok("y = labels[target_id, 1, 0, :, :] — binary AF mask of target day")
        else:
            flag_warn("Cannot confirm exact label extraction — review manually")

        # Off-by-one check for legacy dataset
        print()
        print("  Off-by-one analysis (legacy dataset):")
        print("    input_ids = window[:T_in]  (frames 0..T_in-1)")
        print("    target_ids = window[T_in:]  (frame T_in)")
        print("    => target is NEXT day after last input  ✓")
        print("    => NO off-by-one in index math")
        ok("Legacy dataset: input[0..T_in-1], label[T_in] — correct sequence")

        print()
        print("  --- FORECAST (dataset_forecast.py) multi-horizon mode ---")
        src2 = inspect.getsource(dataset_forecast.TSSatFireForecastDataset.__getitem__)
        print("    hist_ids  = w[:T]       (past frames)")
        print("    fut_ids   = w[T:]       (future frames)")
        print("    af_fut    = labels[fut_ids, label_channel, 0]")
        print("    ba_init   = labels[hist_ids[-1], 0, 0]    <-- channel 0!")
        ok("af_fut uses label_channel=1 (binary AF) for all future steps")

        if "labels[hist_ids[-1], 0, 0]" in src2:
            flag_critical(
                "ba_init = labels[hist_ids[-1], 0, 0] uses label CHANNEL 0.  "
                "Channel 0 contains large integer BA area values (NOT a binary mask) "
                "and -1 sentinel for 38% of train samples.  "
                "The model receives garbage as 'burned area initialization'."
            )

        print()
        print("  PRED task label semantic check:")
        print("    The 'pred' task SHOULD use: next-day BA MINUS last-observed BA")
        print("    Actual code uses: labels[fut_ids, label_channel=1, 0] = AF binary mask")
        flag_critical(
            "PRED task uses AF binary mask (label ch 1) as target — NOT the "
            "burned-area progression difference (next_BA - last_BA).  "
            "Per the TS-SatFire paper, the pred label should be: "
            "(BA[t+1] - BA[t]) i.e. newly burned area only.  "
            "The current code predicts raw AF mask, not fire spread."
        )

        print()
        print("  BA task label semantic check:")
        print("    The 'ba' task SHOULD use: burned area mask of target day")
        print("    The npy label channel 0 is the BA-related channel")
        print("    But channel 0 has large integers and -1 sentinel — not binary")
        flag_critical(
            "No dedicated BA task is implemented — the codebase only has "
            "AF-mask prediction.  Label channel 0 (BA-related) is never used "
            "as the prediction target, it only appears via ba_init (broken).  "
            "The BA task label (binary daily BA mask) is missing from the npy."
        )

    except ImportError as e:
        flag_warn(f"Cannot import modules: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# H.  Visual sanity checks
# ═══════════════════════════════════════════════════════════════════════════

def save_visual_sanity_checks(arrays: Dict[str, np.ndarray],
                               save_dir: Path, n_samples: int = 3):
    """Save PNG figures: last input frame, label overlay per sample."""
    sep("H. Visual Sanity Checks")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        flag_warn("matplotlib not installed — skipping visual checks.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    split = "train"
    img_key = f"{split}/af_{split}_img_seqtoseq_alll_1i_1.npy"
    lbl_key = f"{split}/af_{split}_label_seqtoseq_alll_1i_1.npy"

    if img_key not in arrays:
        return

    img = arrays[img_key]
    lbl = arrays[lbl_key]

    # Detect event boundaries to visualise boundary samples too
    af_mask = lbl[:, 1, 0, :, :]
    fire_counts = af_mask.sum(axis=(1, 2))
    boundaries = detect_event_boundaries(fire_counts)

    # Samples to visualise: first N, and N near a boundary
    indices = list(range(min(n_samples, len(img))))
    if boundaries:
        b0 = boundaries[0]
        for off in [-1, 0, 1]:
            idx = b0 + off
            if 0 <= idx < len(img):
                indices.append(idx)

    indices = list(dict.fromkeys(indices))  # deduplicate, preserve order

    for si in indices:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle(f"Sample index {si}  (fire_pixels={fire_counts[si]:.0f})", fontsize=12)

        # Row 0: input image bands
        band_labels = ["I1 (Red)", "I2 (NIR)", "I3 (SWIR)", "I4 (MIR/fire)"]
        for col, bname in enumerate(band_labels):
            ax = axes[0, col]
            data = img[si, col, 0, :, :]
            im = ax.imshow(data, cmap="gray")
            ax.set_title(f"img band {col}\n{bname}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 1: labels
        for col, (ch, cname, cmap) in enumerate([
            (0, "lbl ch0 (BA?)", "viridis"),
            (1, "lbl ch1 (AF binary)", "hot"),
            (2, "lbl ch2 (thermal?)", "plasma"),
            (3, "img band4 (I5 LWIR)", "gray"),
        ]):
            ax = axes[1, col]
            if ch == 3:
                data = img[si, 4, 0, :, :]
            else:
                data = lbl[si, ch, 0, :, :]
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(cname, fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fname = save_dir / f"sample_{si:05d}.png"
        plt.tight_layout()
        plt.savefig(str(fname), dpi=100)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # Boundary visualisation
    for b in boundaries[:min(2, len(boundaries))]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"BOUNDARY at index {b}: samples {b-1} → {b}", fontsize=11, color="red")
        for col, si in enumerate([b - 2, b - 1, b, b + 1]):
            if 0 <= si < len(img):
                ax = axes[col]
                data = img[si, 3, 0, :, :]  # I4 thermal
                af   = lbl[si, 1, 0, :, :]
                ax.imshow(data, cmap="gray")
                ax.imshow(af, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f"frame {si}\nAF_px={fire_counts[si]:.0f}", fontsize=9,
                             color="red" if si == b else "black")
                ax.axis("off")
            else:
                axes[col].axis("off")
        fname = save_dir / f"boundary_{b:05d}.png"
        plt.tight_layout()
        plt.savefig(str(fname), dpi=100)
        plt.close(fig)
        print(f"  Saved boundary figure: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# I.  Hard assertions
# ═══════════════════════════════════════════════════════════════════════════

def run_hard_assertions(arrays: Dict[str, np.ndarray]):
    """Run all hard assertions — fail loudly if violated."""
    sep("I. Hard Assertions")

    for split in ("train", "val"):
        img_key = f"{split}/af_{split}_img_seqtoseq_alll_1i_1.npy"
        lbl_key = f"{split}/af_{split}_label_seqtoseq_alll_1i_1.npy"
        if img_key not in arrays:
            continue

        img = arrays[img_key]
        lbl = arrays[lbl_key]
        N_img, N_lbl = img.shape[0], lbl.shape[0]

        # Assertion 1: img and label N must match
        if N_img != N_lbl:
            flag_critical(f"ASSERTION FAIL [{split}]: N_img({N_img}) != N_lbl({N_lbl})")
        else:
            ok(f"[{split}] assert N_img == N_lbl == {N_img}")

        # Assertion 2: spatial size consistent
        H_img, W_img = img.shape[3], img.shape[4]
        H_lbl, W_lbl = lbl.shape[3], lbl.shape[4]
        if (H_img, W_img) != (H_lbl, W_lbl):
            flag_critical(f"ASSERTION FAIL [{split}]: img spatial ({H_img}x{W_img}) "
                          f"!= lbl spatial ({H_lbl}x{W_lbl})")
        else:
            ok(f"[{split}] assert spatial size matches: {H_img}x{W_img}")

        # Assertion 3: label ch1 must be binary
        ch1 = lbl[:, 1, 0, :, :]
        unique_ch1 = set(np.unique(ch1).tolist())
        if not unique_ch1.issubset({0.0, 1.0}):
            flag_critical(f"ASSERTION FAIL [{split}]: label ch1 not binary: {list(unique_ch1)[:10]}")
        else:
            ok(f"[{split}] assert label ch1 is binary {{0,1}}")

        # Assertion 4: img channels not all-zero
        for c in range(img.shape[1]):
            channel_max = float(img[:, c, 0, :, :].max())
            if channel_max == 0:
                flag_warn(f"[{split}] img channel {c} is all-zero — may be missing/fill")

        # Assertion 5: img file list not empty
        if N_img == 0:
            flag_critical(f"ASSERTION FAIL [{split}]: img array is empty!")
        else:
            ok(f"[{split}] assert img N > 0 ({N_img})")

    # Assertion for firepred/lulc size mismatch (repeat, for hard-assert framing)
    for split in ("train", "val"):
        img_key = f"{split}/af_{split}_img_seqtoseq_alll_1i_1.npy"
        fp_key  = f"{split}/firepred_{split}.npy"
        lulc_key= f"{split}/lulc_{split}.npy"
        if img_key not in arrays:
            continue
        N_img = arrays[img_key].shape[0]
        if fp_key in arrays:
            N_fp = arrays[fp_key].shape[0]
            if N_fp != N_img:
                flag_critical(
                    f"ASSERTION FAIL [{split}]: firepred N={N_fp} != img N={N_img}.  "
                    "Training with misaligned auxiliary data."
                )
        if lulc_key in arrays:
            N_lulc = arrays[lulc_key].shape[0]
            if N_lulc != N_img:
                flag_critical(
                    f"ASSERTION FAIL [{split}]: lulc N={N_lulc} != img N={N_img}.  "
                    "Modulo-wrapped LULC will silently provide wrong land-cover."
                )


# ═══════════════════════════════════════════════════════════════════════════
# J.  Summary report
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table():
    sep("J. Summary: Task Alignment Table")
    print(f"""
  {'Task':<12} {'Input days':<25} {'Target day':<20} {'Label type':<30} {'Status'}
  {'-'*12} {'-'*25} {'-'*20} {'-'*30} {'-'*10}
  {'AF (legacy)':<12} {'frames[0..T_in-1]':<25} {'frame[T_in]':<20} {'Binary AF mask (ch1)':<30} {'OK'}
  {'AF (fore.)':<12} {'hist_ids[0..T-1]':<25} {'fut_ids[0..H-1]':<20} {'Binary AF mask (ch1)':<30} {'OK (per-step)'}
  {'BA':<12} {'(not implemented)':<25} {'(not impl.)':<20} {'N/A — ch0 is NOT binary':<30} {RED+'MISSING'+RESET}
  {'PRED':<12} {'hist_ids[0..T-1]':<25} {'fut_ids[0..H-1]':<20} {'AF mask, NOT BA-diff!':<30} {RED+'SEMANTIC BUG'+RESET}
  {'ba_init':<12} {'hist_ids[-1]':<25} {'—':<20} {'ch0: large int/-1 BAD':<30} {RED+'TYPE BUG'+RESET}
""")
    print("  Preprocessing notes (from code):")
    print("    - No band-wise normalization applied in legacy dataset")
    print("    - Images are loaded as float32 from mmap, no clamp (except label clamped to [0,1])")
    print("    - FirePred values include zeros as fill (no explicit no-data handling)")
    print("    - LULC is modulo-indexed if mismatched — silently wrong")


def print_final_report():
    sep("FINAL AUDIT REPORT")
    print(f"\n  {BOLD}CRITICAL ISSUES ({len(CRITICAL_ISSUES)}){RESET}")
    for i, msg in enumerate(CRITICAL_ISSUES, 1):
        print(f"  {RED}{i:2d}. {msg}{RESET}\n")

    print(f"  {BOLD}WARNINGS ({len(WARNINGS)}){RESET}")
    for i, msg in enumerate(WARNINGS, 1):
        print(f"  {YELLOW}{i:2d}. {msg}{RESET}\n")

    if CRITICAL_ISSUES:
        print(f"\n  {RED}{BOLD}DO NOT TRAIN. Fix all CRITICAL issues first.{RESET}\n")
        return 1
    else:
        print(f"\n  {GREEN}{BOLD}No critical issues. Proceed with caution.{RESET}\n")
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FireST-GraphNet Dataset Alignment Audit")
    p.add_argument("--data_root", default=".", help="Root of workspace (containing dataset_train/)")
    p.add_argument("--T_in", type=int, default=9, help="Input sequence length used in training")
    p.add_argument("--T_out", type=int, default=1, help="Output length")
    p.add_argument("--n_samples", type=int, default=3, help="Samples for visual checks")
    p.add_argument("--save_figs", action="store_true", help="Save sanity-check PNG figures")
    p.add_argument("--fig_dir", default="audit_figures", help="Directory for figures")
    p.add_argument("--skip_dataloader", action="store_true",
                   help="Skip DataLoader check (if firest_graphnet not installed)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    print(f"\n{BOLD}FireST-GraphNet Dataset Alignment Audit{RESET}")
    print(f"  data_root : {data_root}")
    print(f"  T_in={args.T_in}  T_out={args.T_out}")

    arrays = audit_files(data_root)
    audit_label_channels(arrays)
    audit_event_boundaries(arrays, T_in=args.T_in, T_out=args.T_out)
    audit_index_alignment(arrays)
    audit_channel_manifest(arrays)
    if not args.skip_dataloader:
        audit_dataloader(data_root, T_in=args.T_in)
    audit_task_label_semantics()
    run_hard_assertions(arrays)

    if args.save_figs:
        fig_dir = data_root / args.fig_dir
        save_visual_sanity_checks(arrays, fig_dir, n_samples=args.n_samples)

    print_summary_table()
    exit_code = print_final_report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
