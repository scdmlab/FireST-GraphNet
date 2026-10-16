"""dataset_forecast.py  --  Forecast-conditioned multi-horizon AF dataset.

╔══════════════════════════════════════════════════════════════════════════╗
║  SEMANTIC ALIGNMENT:  env_fut is ALWAYS 4ch (temp, u, v, rh)          ║
║                                                                        ║
║  TRAINING:   4ch extracted from FirePred 19ch future frames            ║
║              (temp=band5, u,v=derived from speed+dir, rh=band7)        ║
║  INFERENCE:  4ch from HRRR (TMP,UGRD,VGRD,RH) or WN2 (temp,u,v,rh)  ║
║                                                                        ║
║  All remaining FirePred bands stay in aux_hist (19ch historical).      ║
╚══════════════════════════════════════════════════════════════════════════╝

Sample layout  (TS-SatFire + FirePred + LULC)
─────────────
  af_hist  : (T_in, 8, H, W)           past VIIRS imagery
             • Day  bands 0-5:  I1, I2, I3, M13, M15, M11 (6 spectral)
             • Night bands 6-7: I4 (3.74µm), I5 (11.45µm) (2 thermal)
  aux_hist : (T_in, 19, H, W)          FirePred auxiliary per-date (historical)
             DEM / slope / aspect / canopy / ERC / temp / dewpoint / RH /
             precip / wind speed / wind dir / fuel moisture / PDSI /
             fuel model / dead fuel / VPD / NDVI / soil temp / snow
  env_fut  : (Horizon, 4, H, W)        future dynamic env conditions
             Channel 0: temperature  1: u_wind  2: v_wind  3: humidity
             TRAINING: extracted from FirePred (same semantic 4ch)
             INFERENCE: replaced externally by HRRR 4ch / WeatherNext2 4ch
  af_fut   : (Horizon, 1, H, W)        future AF ground truth
  ba_init  : (1, H, W)                 burned-area at last historical step
  static   : (C_s, H, W)              static features (ESRI LULC = 1ch)
  meta     : dict                      event_id, index, timestamps

Encoder input per timestep = cat(fire[8] + aux[19] + static[1]) = 28 channels
Decoder conditioning = future env 4ch via FiLM (adapter pattern)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from firest_graphnet.utils.io import find_dataset_dirs
from firest_graphnet.data.firepred_provider import FirePredProviderBase
from firest_graphnet.data.future_providers import (
    extract_dynamic_from_firepred,
    DYNAMIC_FUTURE_CHANNELS,
)
from firest_graphnet.data.dataset import _detect_event_boundaries


# ───────────────────────── helpers ─────────────────────────────────────────
def _find_npy(directory: Path, pattern: str = "af_*"):
    npy = sorted(directory.glob(f"{pattern}.npy"))
    img = [f for f in npy if "_img_" in f.name]
    lbl = [f for f in npy if "_label_" in f.name]
    if not img or not lbl:
        raise FileNotFoundError(f"Cannot find {pattern}_img/label in {directory}")
    return img[0], lbl[0]


# ───────────────────────── dataset ─────────────────────────────────────────
class TSSatFireForecastDataset(Dataset):
    """
    Forecast-conditioned multi-horizon AF dataset.

    Each sample spans (T_in + Horizon) consecutive frames from the npy arrays.

    Data sources (per TS-SatFire paper):
      - af_*_img.npy        : 8ch VIIRS — Day[I1-M11](6) + Night[I4,I5](2)
      - firepred_*.npy      : 19ch auxiliary — weather/topo/fuel from FirePred
      - lulc_*.npy          : 1ch static ESRI Land-Use / Land-Cover

    Future conditioning (semantically aligned, 4ch canonical):
      - TRAINING:  env_fut = 4ch dynamic extracted from FirePred future frames
                   (temp=band5, u,v=speed+dir→u,v, rh=band7)
      - INFERENCE: env_fut replaced externally by HRRR (4ch) / WN2 (4ch)
      - All sources provide [temp, u_wind, v_wind, rh] — NO semantic mismatch

    Parameters
    ----------
    data_dir          : path to dataset_train  or  dataset_val
    T_in              : number of historical frames (default 10)
    horizon           : future prediction horizon   (default 20)
    label_channel     : channel index for AF in the label npy (default 1)
    firepred_provider : FirePredProviderBase for 19ch auxiliary (hist + future)
    static_provider   : optional static feature provider with .get(sample_idx)
    transform         : optional augmentation callable
    indices           : optional subset of frame indices (for splits)
    img_as_input      : if True (default), af_hist uses all 8 VIIRS bands;
                        if False, uses the AF label mask only (1ch).
    future_provider   : optional FutureConditionProvider override
                        (for inference with HRRR/WN2 — replaces FirePred future)
    """

    def __init__(
        self,
        data_dir: str | Path,
        T_in: int = 10,
        horizon: int = 20,
        label_channel: int = 1,
        firepred_provider: FirePredProviderBase | None = None,
        static_provider: object | None = None,
        transform: Optional[Callable] = None,
        indices: Optional[np.ndarray] = None,
        img_as_input: bool = True,
        future_provider: object | None = None,
        # ── backward compat (ignored) ──
        weather_provider: object | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.T_in = T_in
        self.horizon = horizon
        self.label_channel = label_channel
        self.firepred_provider = firepred_provider
        self.static_provider = static_provider
        self.future_provider = future_provider  # optional HRRR/WN2 override
        self.transform = transform
        self.img_as_input = img_as_input

        img_path, lbl_path = _find_npy(self.data_dir)
        self.images = np.load(str(img_path), mmap_mode="r")  # (N, 8, 1, H, W)  VIIRS
        self.labels = np.load(str(lbl_path), mmap_mode="r")  # (N, 3, 1, H, W)  BA/AF_acc/AF

        self.N = self.images.shape[0]
        self.num_bands = self.images.shape[1]       # 8 = VIIRS Day(6) + Night(2)
        self.H = self.images.shape[3]
        self.W = self.images.shape[4]
        self.Cw = self.num_bands                    # back-compat

        # Detect fire-event boundaries to avoid cross-event window contamination.
        self._event_boundaries, self._frame_event_id = _detect_event_boundaries(
            self.labels
        )
        n_events = int(self._frame_event_id.max()) + 1
        print(f"[ForecastDataset] {Path(data_dir).name}: N={self.N} frames, "
              f"{n_events} fire events, "
              f"{len(self._event_boundaries)} boundaries")

        # Guard: if firepred/lulc N != img N, their indices are not aligned.
        # Fall back to zeros rather than silently feed wrong auxiliary data.
        if firepred_provider is not None and hasattr(firepred_provider, '_N'):
            if firepred_provider._N != self.N:
                import warnings
                warnings.warn(
                    f"[ForecastDataset] firepred N={firepred_provider._N} != "
                    f"img N={self.N} — index alignment cannot be guaranteed. "
                    "Disabling FirePred auxiliary (falling back to zeros). "
                    "Re-generate firepred npy from the same fire-event list as af_img npy.",
                    UserWarning, stacklevel=2,
                )
                self.firepred_provider = None
        if static_provider is not None and hasattr(static_provider, '_N'):
            if static_provider._N != self.N:
                import warnings
                warnings.warn(
                    f"[ForecastDataset] lulc N={static_provider._N} != "
                    f"img N={self.N} — disabling LULC static provider (falling back to zeros).",
                    UserWarning, stacklevel=2,
                )
                self.static_provider = None

        if indices is not None:
            self._base = indices
        else:
            self._base = np.arange(self.N)

        self.windows: List[Tuple[int, ...]] = []
        self._build_windows(np.sort(self._base), T_in + horizon)

    # ── sliding windows ──────────────────────────────────────────────────
    def _build_windows(self, idx: np.ndarray, win: int):
        """Build sliding windows, breaking at fire-event boundaries."""
        if len(idx) == 0:
            return

        break_positions: set = set()
        diffs = np.diff(idx)
        for pos in np.where(diffs > 1)[0]:
            break_positions.add(int(pos))
        for pos in range(len(idx) - 1):
            if int(idx[pos + 1]) in self._event_boundaries:
                break_positions.add(pos)

        runs: List[np.ndarray] = []
        s = 0
        for bp in sorted(break_positions):
            runs.append(idx[s : bp + 1])
            s = bp + 1
        runs.append(idx[s:])

        for run in runs:
            if len(run) < win:
                # Not enough frames for a full window in this event — skip
                continue
            for i in range(len(run) - win + 1):
                self.windows.append(tuple(run[i : i + win]))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        w = self.windows[idx]
        T, H_steps = self.T_in, self.horizon

        if len(w) == T + H_steps:
            hist_ids = w[:T]
            fut_ids = w[T:]
        else:
            fid = w[0]
            hist_ids = tuple([fid] * T)
            fut_ids = tuple([fid] * H_steps)

        # ── AF history: 8ch VIIRS (Day 6 spectral + Night 2 thermal) ─────
        if self.img_as_input:
            af_hist = np.stack(
                [self.images[i, :, 0] for i in hist_ids]
            )  # (T_in, 8, H, W)  — stack already copies from mmap
            af_hist = torch.from_numpy(np.ascontiguousarray(af_hist)).float()
        else:
            af_hist = np.stack(
                [self.labels[i, self.label_channel, 0][np.newaxis] for i in hist_ids]
            )  # (T_in, 1, H, W)
            af_hist = torch.from_numpy(np.ascontiguousarray(af_hist)).float().clamp(0, 1)

        # ── AF future GT: (Horizon, 1, H, W) ─────────────────────────────
        af_fut = np.stack(
            [self.labels[i, self.label_channel, 0][np.newaxis] for i in fut_ids]
        )
        af_fut = torch.from_numpy(np.ascontiguousarray(af_fut)).float().clamp(0, 1)

        # ── FirePred auxiliary: 19ch weather/topo/fuel per date ───────────
        if self.firepred_provider is not None:
            aux_hist_np = self.firepred_provider.get_window(hist_ids)
        else:
            aux_hist_np = np.zeros(
                (T, 19, self.H, self.W), dtype=np.float32
            )
        aux_hist = torch.from_numpy(np.ascontiguousarray(aux_hist_np)).float()

        # ── Future environmental conditions (ALWAYS 4ch canonical) ────────
        if self.future_provider is not None:
            env_fut_np = self.future_provider.get_future_env(fut_ids)
        elif self.firepred_provider is not None:
            env_fut_19 = self.firepred_provider.get_window(fut_ids)  # (H, 19, H, W)
            env_fut_np = extract_dynamic_from_firepred(env_fut_19)   # (H, 4, H, W)
        else:
            env_fut_np = np.zeros(
                (H_steps, DYNAMIC_FUTURE_CHANNELS, self.H, self.W), dtype=np.float32
            )
        env_fut = torch.from_numpy(np.ascontiguousarray(env_fut_np)).float()

        # Determine env source name for adapter selection
        if self.future_provider is not None and hasattr(self.future_provider, "source_name"):
            env_source = self.future_provider.source_name
        else:
            env_source = "firepred"

        # ── BA init: binary mask of burned area at last historical step ──────
        # Label channel 0 stores raw fire-area values (large integers) with
        # sentinel -1 for "no data".  Convert to binary: positive → 1, else → 0.
        ba_init_raw = np.array(self.labels[hist_ids[-1], 0, 0], dtype=np.float32)  # (H, W)
        ba_init_bin = np.where(ba_init_raw > 0, 1.0, 0.0).astype(np.float32)
        ba_init = torch.from_numpy(np.ascontiguousarray(ba_init_bin)).unsqueeze(0)  # (1, H, W)

        # ── Static features: ESRI LULC (1ch) ─────────────────────────────
        if self.static_provider is not None:
            static_np = self.static_provider.get(sample_idx=int(w[0]))
            static = torch.from_numpy(np.ascontiguousarray(static_np)).float()
        else:
            static = torch.zeros(1, self.H, self.W)

        sample: Dict[str, Any] = {
            "af_hist": af_hist,       # (T_in, 8, H, W)       VIIRS fire imagery
            "aux_hist": aux_hist,     # (T_in, 19, H, W)      FirePred hist auxiliary
            "env_fut": env_fut,       # (H, 4, H, W)          future dynamic env (4ch)
            "env_source": env_source, # str                    adapter key
            "af_fut": af_fut,         # (H, 1, H, W)          ground truth
            "ba_init": ba_init,       # (1, H, W)             burned area init (binary)
            "static": static,         # (C_s, H, W)           ESRI LULC
            "meta": {
                "event_id": int(self._frame_event_id[w[0]]),
                "index": int(w[0]),
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ───────────────────────── builder ─────────────────────────────────────────
def build_forecast_datasets(
    data_root: str | Path,
    T_in: int = 10,
    horizon: int = 20,
    label_channel: int = 1,
    val_split: float = 0.1,
    seed: int = 42,
    firepred_provider: FirePredProviderBase | None = None,
    static_provider: object | None = None,
    future_provider: object | None = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    img_as_input: bool = True,
    # backward compat (ignored)
    weather_provider: object | None = None,
) -> Tuple[TSSatFireForecastDataset, TSSatFireForecastDataset]:
    """Build train+val datasets for the forecast-conditioned pipeline.

    Data sources (all optional with graceful fallback to zeros):
      - firepred_provider : FirePred 19ch auxiliary (weather/topo/fuel) per date
      - static_provider   : ESRI LULC static features (1ch)
      - future_provider   : optional external future-condition override
                            (None → use FirePred for future env during training)
    """
    train_dir, val_dir = find_dataset_dirs(data_root)
    if train_dir is None:
        raise FileNotFoundError(f"No dataset_train under {data_root}")

    _kw = dict(
        T_in=T_in, horizon=horizon, label_channel=label_channel,
        firepred_provider=firepred_provider,
        static_provider=static_provider,
        future_provider=future_provider,
        img_as_input=img_as_input,
    )

    if val_dir is not None:
        return (
            TSSatFireForecastDataset(train_dir, transform=train_transform, **_kw),
            TSSatFireForecastDataset(val_dir, transform=val_transform, **_kw),
        )
    # split
    img_path, _ = _find_npy(train_dir)
    N = np.load(str(img_path), mmap_mode="r").shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * val_split))
    return (
        TSSatFireForecastDataset(
            train_dir, indices=np.sort(perm[n_val:]),
            transform=train_transform, **_kw,
        ),
        TSSatFireForecastDataset(
            train_dir, indices=np.sort(perm[:n_val]),
            transform=val_transform, **_kw,
        ),
    )
