"""
dataset.py — TSSatFireSequenceDataset for FireST-GraphNet.

Loads the TS-SatFire pre-processed .npy arrays and exposes sliding-window
temporal sequences for training a sequence → next-frame predictor.

Dataset layout (after inspection):
    af_{split}_img_seqtoseq_alll_1i_1.npy   shape: (N, 8, 1, 256, 256)
    af_{split}_label_seqtoseq_alll_1i_1.npy  shape: (N, 3, 1, 256, 256)

    Image dim-1 holds 8 feature/spectral bands.
    Label channel 1 is the binary active-fire mask {0, 1}.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from firest_graphnet.utils.io import find_dataset_dirs


# ── fire event boundary detection ─────────────────────────────────────────
def _detect_event_boundaries(labels_mmap: np.ndarray,
                              ratio_threshold: float = 20.0) -> Tuple[set, np.ndarray]:
    """Detect fire-event boundaries from the binary AF mask (label channel 1).

    The TS-SatFire npy files concatenate multiple fire events sequentially.
    A boundary is detected when the AF fire-pixel count changes abruptly
    between consecutive frames (new spatial region = new fire event).

    Returns
    -------
    boundaries : set[int]
        Set of frame indices that START a new fire event.
    frame_event_id : ndarray shape (N,)
        Integer event-ID for every frame (0-based).
    """
    af_mask = labels_mmap[:, 1, 0, :, :]   # (N, H, W) binary {0, 1}
    fire_counts = af_mask.sum(axis=(1, 2))   # (N,)
    N = len(fire_counts)

    boundaries: set = set()
    for i in range(N - 1):
        a = float(fire_counts[i])
        b = float(fire_counts[i + 1])
        if a > 50 and b == 0:
            boundaries.add(i + 1)
        elif a == 0 and b > 50:
            boundaries.add(i + 1)
        elif a > 0 and b > 0:
            ratio = max(a, b) / max(min(a, b), 1.0)
            if ratio > ratio_threshold:
                boundaries.add(i + 1)

    # Assign event IDs per frame
    frame_event_id = np.zeros(N, dtype=np.int32)
    eid = 0
    for i in range(1, N):
        if i in boundaries:
            eid += 1
        frame_event_id[i] = eid

    return boundaries, frame_event_id


# ── helper: locate the canonical npy pair ──────────────────────────────────
def _find_npy_files(directory: Path) -> Tuple[Path, Path]:
    """Return (img_path, label_path) for the main af_*.npy files."""
    npy_files = sorted(directory.glob("af_*.npy"))
    img_files = [f for f in npy_files if "_img_" in f.name]
    lbl_files = [f for f in npy_files if "_label_" in f.name]
    if not img_files or not lbl_files:
        raise FileNotFoundError(
            f"Could not find af_*_img_*.npy / af_*_label_*.npy in {directory}"
        )
    return img_files[0], lbl_files[0]


# ── main dataset ───────────────────────────────────────────────────────────
class TSSatFireSequenceDataset(Dataset):
    """Sliding-window temporal-sequence dataset for wildfire progression.

    Each __getitem__ returns:
        x_seq : Tensor  [T_in, C, H, W]   — input sequence of feature maps
        y     : Tensor  [1, H, W]          — next-day active-fire mask
        meta  : dict                        — event_id, frame_idx
    """

    def __init__(
        self,
        data_dir: str | Path,
        T_in: int = 9,
        T_out: int = 1,
        label_channel: int = 1,
        transform: Optional[Callable] = None,
        indices: Optional[np.ndarray] = None,
        weather_dir: Optional[str | Path] = None,
        weather_channels: int = 4,
        default_doy: int = 200,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.T_in = T_in
        self.T_out = T_out
        self.label_channel = label_channel
        self.transform = transform
        self.default_doy = default_doy

        # Optional HRRR weather provider (DOY-based climatology)
        self.weather_provider = None
        self.weather_channels = 0
        if weather_dir is not None:
            wpath = Path(weather_dir)
            if wpath.is_dir() and any(wpath.glob("doy_*.npy")):
                from firest_graphnet.data.hrrr_alignment import HRRRClimatologyProvider
                self.weather_provider = HRRRClimatologyProvider(
                    wpath, channels=weather_channels, spatial_size=(256, 256)
                )
                self.weather_channels = weather_channels
                print(f"[Dataset] HRRR weather enabled: {wpath} ({weather_channels} ch)")
            else:
                print(f"[Dataset] weather_dir not found or empty: {wpath}, skipping.")

        img_path, lbl_path = _find_npy_files(self.data_dir)

        # Memory-map so we never load >1 sample into RAM at once
        self.images = np.load(str(img_path), mmap_mode="r")  # (N, 8, 1, H, W)
        self.labels = np.load(str(lbl_path), mmap_mode="r")  # (N, 3, 1, H, W)

        self.N = self.images.shape[0]
        self.num_bands = self.images.shape[1]  # 8
        self.H = self.images.shape[3]
        self.W = self.images.shape[4]

        # Detect fire-event boundaries from the AF label mask.
        # This prevents sliding windows from spanning across different fire events.
        self._event_boundaries, self._frame_event_id = _detect_event_boundaries(
            self.labels
        )
        n_events = int(self._frame_event_id.max()) + 1
        print(f"[Dataset] {self.data_dir.name}: N={self.N} frames, "
              f"{n_events} fire events detected, "
              f"{len(self._event_boundaries)} event boundaries")

        # Build valid sliding-window indices
        # Each sample at index i uses frames [i .. i+T_in-1] as input
        # and frame [i+T_in .. i+T_in+T_out-1] as label
        total_window = T_in + T_out
        if indices is not None:
            # Use only the supplied subset (for train/val splitting)
            self._base_indices = indices
        else:
            self._base_indices = np.arange(self.N)

        self.windows: List[Tuple[int, ...]] = []
        sorted_idx = np.sort(self._base_indices)
        self._build_windows(sorted_idx, total_window)

    # ── window construction ────────────────────────────────────────────────
    def _build_windows(self, sorted_idx: np.ndarray, window_size: int):
        """Build sliding windows, breaking at fire-event boundaries.

        A break occurs when:
        - There is a gap > 1 in the index sequence (non-contiguous subset), OR
        - The next index is a detected fire-event boundary (different spatial region).
        """
        if len(sorted_idx) == 0:
            return

        # Collect all break positions (as positions in sorted_idx, not frame values)
        break_positions: set = set()

        diffs = np.diff(sorted_idx)
        # Gap > 1 in index space
        for pos in np.where(diffs > 1)[0]:
            break_positions.add(int(pos))
        # Fire-event boundary: sorted_idx[pos+1] starts a new event
        for pos in range(len(sorted_idx) - 1):
            if int(sorted_idx[pos + 1]) in self._event_boundaries:
                break_positions.add(pos)

        runs: List[np.ndarray] = []
        start = 0
        for bp in sorted(break_positions):
            runs.append(sorted_idx[start : bp + 1])
            start = bp + 1
        runs.append(sorted_idx[start:])

        for run in runs:
            if len(run) < window_size:
                # Not enough frames for one full window — skip (no fallback padding)
                continue
            for i in range(len(run) - window_size + 1):
                self.windows.append(tuple(run[i : i + window_size]))

    # ── __len__ / __getitem__ ──────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window = self.windows[idx]

        if len(window) == self.T_in + self.T_out:
            input_ids = window[: self.T_in]
            target_ids = window[self.T_in :]
        else:
            # Fallback: single frame — replicate for T_in, use same as target
            fid = window[0]
            input_ids = tuple([fid] * self.T_in)
            target_ids = (fid,)

        # x_seq: (T_in, C, H, W) — squeeze the singleton dim-2 and stack bands
        frames = []
        for fid in input_ids:
            # self.images[fid] is (8, 1, H, W) → squeeze to (8, H, W)
            frame = np.array(self.images[fid, :, 0, :, :])  # (8, H, W)
            frames.append(frame)
        x_seq = np.stack(frames, axis=0)  # (T_in, 8, H, W)

        # y: take the last target frame's label channel → (1, H, W)
        tid = target_ids[-1]
        y = np.array(self.labels[tid, self.label_channel, 0, :, :])  # (H, W)
        y = y[np.newaxis, :, :]  # (1, H, W)

        # Convert to float tensors
        x_seq = torch.from_numpy(x_seq.copy()).float()
        y = torch.from_numpy(y.copy()).float()

        # Clamp label to [0, 1]
        y = y.clamp(0.0, 1.0)

        # ── optional HRRR weather (DOY-based climatology) ──────────────────
        if self.weather_provider is not None:
            w_arr = self.weather_provider.get(self.default_doy)  # (4, H, W)
            w_tensor = torch.from_numpy(w_arr).float()            # (4, H, W)
            # Normalize roughly: TMP ~280-310K, wind ~0-20m/s, RH 0-100%
            # Simple min-max scaling per channel
            w_tensor[0] = (w_tensor[0] - 260.0) / 60.0   # TMP K → ~[-1, 1]
            w_tensor[1] = w_tensor[1] / 20.0              # UGRD m/s → norm
            w_tensor[2] = w_tensor[2] / 20.0              # VGRD m/s → norm
            w_tensor[3] = w_tensor[3] / 100.0             # RH % → [0, 1]
            # Expand to (T_in, 4, H, W) and concatenate
            w_exp = w_tensor.unsqueeze(0).expand(self.T_in, -1, -1, -1)  # (T, 4, H, W)
            x_seq = torch.cat([x_seq, w_exp], dim=1)  # (T, 8+4=12, H, W)

        sample: Dict[str, Any] = {
            "x_seq": x_seq,
            "y": y,
            "meta": {
                "event_id": int(self._frame_event_id[window[0]]),
                "frame_idx": int(window[0]),
                "weather_channels": self.weather_channels,
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ── helpers ────────────────────────────────────────────────────────────────
def build_datasets(
    data_root: str | Path,
    T_in: int = 9,
    T_out: int = 1,
    label_channel: int = 1,
    val_split: float = 0.1,
    seed: int = 42,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    weather_dir: Optional[str | Path] = None,
    weather_channels: int = 4,
    default_doy: int = 200,
) -> Tuple["TSSatFireSequenceDataset", "TSSatFireSequenceDataset"]:
    """Build train and validation datasets.

    If dataset_val exists, use it directly.  Otherwise, split dataset_train.
    """
    train_dir, val_dir = find_dataset_dirs(data_root)

    if train_dir is None:
        raise FileNotFoundError(
            f"Cannot find dataset_train under {data_root}"
        )

    if val_dir is not None:
        # Both dirs exist — use as-is
        train_ds = TSSatFireSequenceDataset(
            train_dir, T_in=T_in, T_out=T_out,
            label_channel=label_channel, transform=train_transform,
            weather_dir=weather_dir, weather_channels=weather_channels,
            default_doy=default_doy,
        )
        val_ds = TSSatFireSequenceDataset(
            val_dir, T_in=T_in, T_out=T_out,
            label_channel=label_channel, transform=val_transform,
            weather_dir=weather_dir, weather_channels=weather_channels,
            default_doy=default_doy,
        )
    else:
        # Split train set
        img_path, _ = _find_npy_files(train_dir)
        N = np.load(str(img_path), mmap_mode="r").shape[0]

        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        n_val = max(1, int(N * val_split))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])

        train_ds = TSSatFireSequenceDataset(
            train_dir, T_in=T_in, T_out=T_out,
            label_channel=label_channel, transform=train_transform,
            indices=train_idx,
            weather_dir=weather_dir, weather_channels=weather_channels,
            default_doy=default_doy,
        )
        val_ds = TSSatFireSequenceDataset(
            train_dir, T_in=T_in, T_out=T_out,
            label_channel=label_channel, transform=val_transform,
            indices=val_idx,
            weather_dir=weather_dir, weather_channels=weather_channels,
            default_doy=default_doy,
        )

        # Save splits for reproducibility
        splits_path = Path(data_root) / "splits.json"
        if not splits_path.exists():
            from firest_graphnet.utils.io import save_json
            save_json(
                {"train": train_idx.tolist(), "val": val_idx.tolist()},
                splits_path,
            )

    return train_ds, val_ds
