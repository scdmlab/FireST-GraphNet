#!/usr/bin/env python
r"""
inspect_dataset.py -- Mandatory first-step dataset inspection for TS-SatFire.

Scans the data root to locate dataset_train / dataset_val directories,
analyses .npy file shapes, reports statistics, and prints a directory tree.

Usage:
    python -m firest_graphnet.data.inspect_dataset --data_root "C:\Users\22125\Desktop\FireST-GraphNet"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── helpers ─────────────────────────────────────────────────────────────────
def _find_dataset_dirs(
    root: Path, max_depth: int = 4
) -> Tuple[Optional[Path], Optional[Path]]:
    """Recursively search for dataset_train and dataset_val under *root*."""
    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None

    # Priority candidates (explicit known paths)
    candidates = [
        root,
        root / "dataset",
        root / "TS-SatFire" / "dataset",
    ]

    for base in candidates:
        t = base / "dataset_train"
        v = base / "dataset_val"
        if t.is_dir():
            train_dir = t
        if v.is_dir():
            val_dir = v
        if train_dir and val_dir:
            return train_dir, val_dir

    # Fall back to recursive walk (depth ≤ max_depth)
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        for d in dirnames:
            full = Path(dirpath) / d
            if d == "dataset_train" and train_dir is None:
                train_dir = full
            elif d == "dataset_val" and val_dir is None:
                val_dir = full
        if train_dir and val_dir:
            break

    return train_dir, val_dir


def _print_tree(path: Path, prefix: str = "", depth: int = 2, max_files: int = 15):
    """Print a directory tree up to *depth* levels."""
    if depth < 0:
        return
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    shown = 0
    for entry in entries:
        connector = "├── " if entry != entries[-1] else "└── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "│   " if entry != entries[-1] else "    "
            _print_tree(entry, prefix + extension, depth - 1, max_files)
        shown += 1
        if shown >= max_files and entry != entries[-1]:
            remaining = len(entries) - shown
            print(f"{prefix}    ... ({remaining} more entries)")
            break


def _analyse_npy(path: Path) -> Dict:
    """Return shape / dtype / min / max for a single .npy file."""
    arr = np.load(str(path), mmap_mode="r")
    info: Dict = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    }
    try:
        info["min"] = float(np.nanmin(arr))
        info["max"] = float(np.nanmax(arr))
    except Exception:
        info["min"] = None
        info["max"] = None
    return info


# ── main inspection ────────────────────────────────────────────────────────
def inspect(data_root: str) -> Dict:
    """Run full dataset inspection and return summary dict."""
    root = Path(data_root).resolve()
    print(f"\n{'='*70}")
    print(f"  FireST-GraphNet Dataset Inspector")
    print(f"  Root: {root}")
    print(f"{'='*70}\n")

    train_dir, val_dir = _find_dataset_dirs(root)

    if train_dir is None:
        print("[ERROR] dataset_train not found — aborting.")
        sys.exit(1)
    print(f"[OK] Train dir : {train_dir}")
    print(f"[OK] Val dir   : {val_dir or 'NOT FOUND (will split from train)'}\n")

    summary: Dict = {"train_dir": str(train_dir), "val_dir": str(val_dir)}

    for label, ddir in [("TRAIN", train_dir), ("VAL", val_dir)]:
        if ddir is None:
            continue
        print(f"\n── {label} directory tree ──")
        _print_tree(ddir, depth=2, max_files=20)

        npy_files = sorted(ddir.glob("*.npy"))
        print(f"\n  Total .npy files: {len(npy_files)}")

        # Categorise files
        main_files = [f for f in npy_files if f.name.startswith("af_")]
        tmp_img = [f for f in npy_files if "tmp" in f.name and "_img" in f.name]
        tmp_lbl = [f for f in npy_files if "tmp" in f.name and "_label" in f.name]
        batch_files = [f for f in npy_files if f.name.startswith("temp_batch")]
        print(f"  Main (af_*)    : {len(main_files)}")
        print(f"  Tmp img        : {len(tmp_img)}")
        print(f"  Tmp label      : {len(tmp_lbl)}")
        print(f"  Batch files    : {len(batch_files)}")

        # Inspect main files in detail
        for f in main_files:
            info = _analyse_npy(f)
            print(
                f"\n  {f.name}\n"
                f"    shape : {info['shape']}\n"
                f"    dtype : {info['dtype']}\n"
                f"    range : [{info['min']}, {info['max']}]"
            )

        # Label channel breakdown
        main_labels = [f for f in main_files if "label" in f.name]
        for f in main_labels:
            arr = np.load(str(f), mmap_mode="r")
            print(f"\n  Label channel analysis ({f.name}):")
            for ch in range(arr.shape[1]):
                vals = arr[:min(50, arr.shape[0]), ch, 0].flatten()
                nz = np.count_nonzero(vals)
                uvals = np.unique(vals)
                desc = f"    ch{ch}: min={np.nanmin(vals):.2f}  max={np.nanmax(vals):.2f}  " \
                       f"nonzero={nz}/{len(vals)}"
                if len(uvals) <= 10:
                    desc += f"  unique={uvals}"
                print(desc)

        # Fire pixel stats from channel 1
        for f in main_labels:
            arr = np.load(str(f), mmap_mode="r")
            if arr.shape[1] >= 2:
                fire_ch = arr[:, 1, 0]  # (N, H, W)
                fire_per_sample = fire_ch.reshape(fire_ch.shape[0], -1).sum(axis=1)
                total_pix = fire_ch.shape[1] * fire_ch.shape[2]
                print(
                    f"\n  Fire pixel stats ({f.name}, channel 1):\n"
                    f"    Samples          : {fire_ch.shape[0]}\n"
                    f"    Pixels/sample    : {total_pix}\n"
                    f"    Mean fire pixels : {fire_per_sample.mean():.1f}\n"
                    f"    Max  fire pixels : {fire_per_sample.max():.0f}\n"
                    f"    Min  fire pixels : {fire_per_sample.min():.0f}\n"
                    f"    Samples w/ fire  : {np.sum(fire_per_sample > 0)}/{fire_ch.shape[0]}\n"
                    f"    Avg fire fraction: {fire_per_sample.mean()/total_pix:.5f}"
                )

        # Tmp file shape summary
        if tmp_img:
            shapes = set()
            for f in tmp_img[:20]:
                try:
                    info = _analyse_npy(f)
                    shapes.add(info["shape"])
                except Exception:
                    pass
            print(f"\n  Tmp img unique shapes ({len(tmp_img)} files sampled): {shapes}")
        if tmp_lbl:
            shapes = set()
            for f in tmp_lbl[:20]:
                try:
                    info = _analyse_npy(f)
                    shapes.add(info["shape"])
                except Exception:
                    pass
            print(f"  Tmp label unique shapes ({len(tmp_lbl)} files sampled): {shapes}")

        summary[label.lower()] = {
            "total_npy": len(npy_files),
            "main_files": [f.name for f in main_files],
            "tmp_img_count": len(tmp_img),
            "tmp_label_count": len(tmp_lbl),
        }

    print(f"\n{'='*70}")
    print("  Inspection complete.")
    print(f"{'='*70}\n")
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inspect TS-SatFire dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"C:\Users\22125\Desktop\FireST-GraphNet",
        help="Root directory containing dataset_train / dataset_val",
    )
    args = parser.parse_args()
    inspect(args.data_root)


if __name__ == "__main__":
    main()
