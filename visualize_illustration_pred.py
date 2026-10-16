"""
visualize_illustration_pred.py — Illustration figure for the Prediction (Pred) task.

Pred = soft probability map output.
Unlike AF (hard binary mask) and BA (cumulative binary), the Pred task shows the
model's CONTINUOUS fire-spread probability [0, 1] — the raw confidence before
any thresholding.  High values = high predicted probability of fire.

3-row × 5-col figure:
  Row 1: Ground truth probability  (smoothed AF → soft GT)
  Row 2: Simulated prediction probability  (continuous heatmap)
  Row 3: Residual/error map  (signed difference: pred − gt, blue=under, red=over)

Metric: normalized MAE (nMAE) — both GT and prediction are rescaled to [0, 1]
with training-set global min/max statistics (see
firest_graphnet.training.normalization) before the absolute error is averaged.

Usage:
  python visualize_illustration_pred.py --event 5 --quality 0.75 --fragmentation 0.4
  python visualize_illustration_pred.py --all --top-k 5 --quality 0.75
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter, binary_dilation, binary_erosion, label as nd_label,
)

try:
    from firest_graphnet.training.normalization import (
        DEFAULT_PROB_STATS,
        GlobalProbStats,
        nmae as _nmae_fn,
    )
except ImportError:  # package not on sys.path — fall back to local import
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from firest_graphnet.training.normalization import (
        DEFAULT_PROB_STATS,
        GlobalProbStats,
        nmae as _nmae_fn,
    )

from visualize_roi_utils import (
    BOTTOM,
    HSPACE,
    LEFT,
    LEGEND_FONTSIZE,
    LEGEND_HANDLEHEIGHT,
    LEGEND_HANDLELENGTH,
    LEGEND_Y,
    METRIC_FONTSIZE,
    RIGHT,
    ROW_LABEL_FONTSIZE,
    ROW_LABEL_X,
    TITLE_FONTSIZE,
    TITLE_PAD,
    TOP,
    WSPACE,
    apply_roi,
    compute_auto_roi,
    format_event_id,
    load_roi,
    save_roi,
    select_roi_interactive,
)


# ─────────────────────────────────────────────────────────────────────────────
# GPU acceleration (PyTorch) — transparent fallback to scipy on CPU
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as _TF
    _DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _DEVICE    = None


def _gauss_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur — PyTorch conv2d (GPU) or scipy (CPU)."""
    if not _HAS_TORCH or sigma <= 0:
        return gaussian_filter(x.astype(np.float32), sigma=sigma)
    k   = max(3, int(4 * sigma + 1) * 2 + 1) | 1
    t   = torch.from_numpy(x.astype(np.float32)).to(_DEVICE)[None, None]
    c   = torch.arange(k, dtype=torch.float32, device=_DEVICE) - k // 2
    g   = torch.exp(-0.5 * (c / sigma) ** 2); g /= g.sum()
    kh  = g.view(1, 1, k, 1);  kw = g.view(1, 1, 1, k)
    out = _TF.conv2d(_TF.conv2d(t, kh, padding=(k // 2, 0)),
                     kw, padding=(0, k // 2))
    return out[0, 0].cpu().numpy()


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation via max-pool (GPU) or scipy (CPU)."""
    if not _HAS_TORCH or radius <= 0:
        return binary_dilation(binary, structure=_disk_struct(radius)).astype(np.float32)
    k   = 2 * radius + 1
    t   = torch.from_numpy(binary.astype(np.float32)).to(_DEVICE)[None, None]
    out = _TF.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    return out[0, 0].cpu().numpy()


def _erode(binary: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion via negative max-pool (GPU) or scipy (CPU)."""
    if not _HAS_TORCH or radius <= 0:
        return binary_erosion(binary, structure=_disk_struct(radius)).astype(np.float32)
    k   = 2 * radius + 1
    t   = torch.from_numpy(binary.astype(np.float32)).to(_DEVICE)[None, None]
    out = -_TF.max_pool2d(-t, kernel_size=k, stride=1, padding=radius)
    return out[0, 0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Low-frequency noise
# ─────────────────────────────────────────────────────────────────────────────

def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


def low_freq_noise(shape: Tuple[int, int], scale: float = 32.0,
                   rng: np.random.Generator = None) -> np.ndarray:
    """Smooth low-frequency noise in [0, 1] via bilinear upsampling."""
    H, W = shape
    if rng is None:
        rng = np.random.default_rng(0)
    gh = max(2, int(math.ceil(H / scale)) + 2)
    gw = max(2, int(math.ceil(W / scale)) + 2)
    grid = rng.random((gh, gw), dtype=np.float32)
    ys = np.linspace(0, gh - 1, H, dtype=np.float32)
    xs = np.linspace(0, gw - 1, W, dtype=np.float32)
    y0 = np.floor(ys).astype(int);  y1 = np.minimum(y0 + 1, gh - 1)
    x0 = np.floor(xs).astype(int);  x1 = np.minimum(x0 + 1, gw - 1)
    fy = _fade((ys - y0).astype(np.float32))
    fx = _fade((xs - x0).astype(np.float32))
    top    = _lerp(grid[y0][:, x0], grid[y0][:, x1], fx[None, :])
    bottom = _lerp(grid[y1][:, x0], grid[y1][:, x1], fx[None, :])
    return _lerp(top, bottom, fy[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# Build soft GT probability from AF binary mask
# ─────────────────────────────────────────────────────────────────────────────

def af_to_prob(af_frame: np.ndarray, sigma_interior: float = 3.0,
               sigma_exterior: float = 8.0) -> np.ndarray:
    """
    Convert a binary AF frame to a smooth probability map.

    Strategy:
      - Interior (fire pixels) → smooth Gaussian blob (high confidence, ~0.85–0.98)
      - Exterior → distance-decaying probability (uncertainty halo around boundary)

    This mimics the soft sigmoid output of the trained model.
    """
    binary = (af_frame >= 0.5).astype(np.float32)

    # Interior: smooth the fire region to get confidence ~0.85–0.95
    interior = _gauss_blur(binary * 0.95, sigma_interior)

    # Exterior halo: probability decays from boundary outward
    dilated  = _gauss_blur(binary, sigma_exterior)
    halo     = np.clip(dilated * 0.55, 0, 0.45)

    # Combine: fire pixels take interior value; outside takes halo
    prob = np.where(binary >= 0.5, interior, halo)
    prob = np.clip(prob, 0, 1)
    return prob.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _disk_struct(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)

def _extract_boundary_band(binary: np.ndarray, width: int = 4) -> np.ndarray:
    """Return a band of pixels near the mask boundary (dilate XOR erode)."""
    dilated = _dilate(binary.astype(np.float32), width)
    eroded  = _erode (binary.astype(np.float32), width)
    return np.clip(dilated - eroded, 0, 1)

def _apply_fragmentation(
    soft: np.ndarray,         # (H, W) soft float mask [0, 1]
    binary: np.ndarray,       # (H, W) current binary mask
    fragmentation: float,
    rng: np.random.Generator,
    base_noise: np.ndarray,   # (H, W) temporally-shared low-freq field
    frame_noise: np.ndarray,  # (H, W) per-frame low-freq field
) -> np.ndarray:
    """
    Add structured boundary roughness to a soft mask.

    Strategy
    --------
    1. Extract boundary band of the current binary mask.
    2. Generate a multi-scale noise field (medium + fine) — spatially coherent.
    3. Apply signed displacement ONLY inside the boundary band:
         soft += noise * fragmentation * band
    4. Re-threshold to get a rough binary, then clean tiny artefacts.

    This preserves the interior (no holes) while creating jagged, irregular
    perimeter details that scale with `fragmentation`.
    """
    if fragmentation <= 0.0:
        return soft
    H, W = soft.shape
    band_width = max(2, int(round(3 + 7 * fragmentation)))
    band   = _extract_boundary_band(binary, width=band_width)
    medium = low_freq_noise((H, W), scale=10.0, rng=rng)
    fine   = low_freq_noise((H, W), scale=4.5,  rng=rng)
    alpha  = 0.3 + 0.5 * fragmentation
    combined  = (1 - alpha) * medium + alpha * fine
    noise     = (combined - 0.5) * 2.0
    noise     = 0.55 * noise + 0.45 * (base_noise - 0.5) * 2.0
    amplitude = 0.22 * (fragmentation ** 0.7)   # slightly gentler for soft maps
    return np.clip(soft + noise * amplitude * band, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation — Pred (soft probability) version
# ─────────────────────────────────────────────────────────────────────────────

def simulate_prediction(
    gt_prob: np.ndarray,
    seed: int = 42,
    quality: float = 0.8,
    smoothness: float = 2.0,
    noise_level: float = 0.1,
    fragmentation: float = 0.0,
    mode: Literal["presentation", "realistic"] = "presentation",
) -> np.ndarray:
    """
    Simulate soft probability prediction maps from GT probability sequence.

    Key difference from AF/BA:
      - Output is CONTINUOUS [0, 1] — not binarised
      - Errors appear as miscalibrated confidence (slight spatial shift, halo distortion)
      - Temporal consistency: probability fields vary smoothly across frames
      - Fragmentation: distorts the probability boundary iso-contour, not a hard edge

    Parameters
    ----------
    gt_prob       : (T, H, W) soft probability ground truth [0, 1]
    quality       : 0.0 → large probability miscalibration; 1.0 → near-perfect
    smoothness    : controls spatial smoothness of output
    noise_level   : volumetric noise amplitude
    fragmentation : distorts the boundary probability contour
    mode          : "presentation" | "realistic"
    """
    rng = np.random.default_rng(seed)
    T, H, W = gt_prob.shape
    gt_prob = np.clip(gt_prob, 0, 1)

    if mode == "presentation":
        smoothness  *= 1.10
        noise_level *= 0.50
    else:
        smoothness  *= 0.80
        noise_level *= 1.25

    # Temporally correlated noise fields
    base_noise = low_freq_noise((H, W), scale=30.0, rng=rng)
    noise_fields = [
        0.55 * base_noise + 0.45 * low_freq_noise((H, W), scale=22.0, rng=rng)
        for _ in range(T)
    ]
    frag_base = low_freq_noise((H, W), scale=18.0, rng=rng)

    preds     = []
    prev_pred = None

    for t in range(T):
        frame = gt_prob[t].copy()

        # ── (A) Spatial perturbation of the probability field ──────────────
        # Slight shift of the spatial peak (simulates position error)
        dy = rng.integers(-max(1, int(5 * (1 - quality))),
                           max(2, int(5 * (1 - quality)) + 1))
        dx = rng.integers(-max(1, int(5 * (1 - quality))),
                           max(2, int(5 * (1 - quality)) + 1))
        frame = np.roll(np.roll(frame, dy, axis=0), dx, axis=1)

        # ── (B) Amplitude miscalibration (under/over-confidence) ───────────
        # Model may be slightly under- or over-confident at boundaries
        calibration_shift = rng.uniform(-0.08, 0.08) * (1.0 - quality)
        binary  = (gt_prob[t] >= 0.5).astype(np.float32)
        bnd     = _extract_boundary_band(binary, width=4)
        frame   = frame + calibration_shift * bnd  # only shifts boundary

        # ── (C) Structured volumetric noise ───────────────────────────────
        vol_noise = (noise_fields[t] - 0.5) * 2.0 * noise_level * (1.0 - quality * 0.65)
        frame = frame + vol_noise

        # ── (D) Temporal consistency ───────────────────────────────────────
        if prev_pred is not None:
            tw    = 0.18 if mode == "presentation" else 0.10
            frame = (1.0 - tw) * frame + tw * prev_pred

        frame = np.clip(frame, 0, 1)

        # ── (E) Smoothing (presentation mode: cleaner halo) ───────────────
        sigma = max(0.5, smoothness * 0.45 * (1.0 + 0.20 * (1.0 - quality)))
        frame = _gauss_blur(frame, sigma)

        # ── (F) Fragmentation: distort the probability iso-contour ────────
        # Operate on the mid-probability boundary band (~0.4–0.6 iso-contour)
        mid_binary = (frame > 0.40).astype(np.float32)
        if fragmentation > 0.0:
            frame = _apply_fragmentation(
                frame, mid_binary, fragmentation, rng, frag_base, noise_fields[t],
            )

        frame = np.clip(frame, 0, 1)
        preds.append(frame.astype(np.float32))
        prev_pred = frame.copy()

    return np.stack(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Row 3: Residual / error map  (pred − gt, signed)
# ─────────────────────────────────────────────────────────────────────────────

def residual_rgb(gt_prob: np.ndarray, pred_prob: np.ndarray) -> np.ndarray:
    """
    Signed residual visualisation for probability maps.

      pred > gt  (over-prediction) → red   shades
      pred < gt  (under-prediction) → blue  shades
      near zero  (correct)          → near-black

    Returns (H, W, 3) float32
    """
    diff = pred_prob - gt_prob        # typically very small, e.g. ~0.005-0.02
    H, W = gt_prob.shape
    rgb  = np.full((H, W, 3), 0.01, dtype=np.float32)
    rgb[..., 2] = 0.02

    # Over-prediction: red tones
    over_raw  = np.clip( diff, 0, None)
    under_raw = np.clip(-diff, 0, None)

    # Residual magnitudes are often tiny in this task, so raw values make the
    # red/blue row look too dark. Normalize by a robust frame-wise scale so
    # small but meaningful differences still render as bright colors.
    scale = float(np.percentile(np.abs(diff), 97))
    scale = max(scale, 0.02)
    over = np.clip(over_raw / scale, 0, 1)
    under = np.clip(under_raw / scale, 0, 1)

    rgb[..., 0] += over  * 1.20    # push pure R channel harder
    rgb[..., 1] += over  * 0.06    # keep only a tiny warm lift
    rgb[..., 2] += over  * 0.02    # avoid dull dark red
    rgb[..., 0] += under * 0.02    # tiny violet lift only
    rgb[..., 1] += under * 0.06    # keep blue mostly in B channel
    rgb[..., 2] += under * 1.20    # push pure B channel harder
    rgb          = np.clip(rgb, 0, 1)
    return rgb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ROI crop
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(
    af_seq: np.ndarray,
    seed: int = 42,
    quality: float = 0.75,
    smoothness: float = 2.0,
    noise_level: float = 0.1,
    fragmentation: float = 0.0,
    mode: Literal["presentation", "realistic"] = "presentation",
    save_path: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
    roi: Optional[Tuple[int, int, int, int]] = None,
    auto_crop: bool = True,
) -> Tuple:
    """
    Build 3x5 Pred illustration figure using binary GT/pred masks.

    Parameters
    ----------
    af_seq : (T, H, W) AF binary frames
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    T_show = min(5, af_seq.shape[0])
    af5 = np.clip(af_seq[:T_show].copy(), 0, 1)
    gt_bin = (af5 >= 0.5).astype(np.float32)

    # Use the existing soft simulator as an intermediate step to create
    # plausible forecast errors, then threshold back to binary for display.
    gt_prob = np.stack([af_to_prob(gt_bin[t]) for t in range(T_show)], axis=0)
    pred_prob = simulate_prediction(
        gt_prob, seed=seed, quality=quality,
        smoothness=smoothness, noise_level=noise_level,
        fragmentation=fragmentation, mode=mode,
    )
    pred_bin = (pred_prob >= 0.5).astype(np.float32)

    if roi is not None:
        gt_c = apply_roi(gt_bin, roi)
        pred_c = apply_roi(pred_bin, roi)
    elif auto_crop:
        auto_roi = compute_auto_roi(gt_bin, pred_bin, blur_fn=_gauss_blur)
        gt_c = apply_roi(gt_bin, auto_roi)
        pred_c = apply_roi(pred_bin, auto_roi)
    else:
        gt_c = gt_bin
        pred_c = pred_bin

    crop_h, crop_w = gt_c.shape[1:]
    aspect = crop_w / max(crop_h, 1)
    panel_h = 2.8
    fig_h = panel_h * 3 + 0.55
    fig_w = panel_h * T_show * aspect + 1.25
    fig, axes = plt.subplots(
        3, T_show, figsize=(fig_w, fig_h),
        gridspec_kw={
            "hspace": HSPACE, "wspace": WSPACE,
            "left": LEFT, "right": RIGHT,
            "top": TOP, "bottom": BOTTOM,
        },
    )

    binary_cmap = LinearSegmentedColormap.from_list(
        "pred_binary",
        ["#08080c", "#fff6b0"],
        N=2,
    )
    bg_color = "#08080c"

    time_labels = [f"T+{t}" for t in range(T_show)]
    row_labels = ["Ground Truth", "Prediction", "Residual"]

    for t in range(T_show):
        gt_f = gt_c[t]
        pred_f = pred_c[t]

        ax = axes[0, t]
        ax.set_facecolor(bg_color)
        ax.imshow(gt_f, cmap=binary_cmap, vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.set_title(time_labels[t], fontsize=TITLE_FONTSIZE, fontweight="bold",
                     pad=TITLE_PAD, color="#2a2a2a")
        ax.axis("off")

        ax = axes[1, t]
        ax.set_facecolor(bg_color)
        ax.imshow(pred_f, cmap=binary_cmap, vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.axis("off")

        ax = axes[2, t]
        ax.set_facecolor(bg_color)
        ax.imshow(residual_rgb(gt_f, pred_f), interpolation="nearest", aspect="equal")
        ax.axis("off")

    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            ROW_LABEL_X, 0.5, label, transform=axes[i, 0].transAxes,
            fontsize=ROW_LABEL_FONTSIZE, fontweight="bold", va="center", ha="right",
            rotation=90, color="#1a1a1a",
        )

    from matplotlib.patches import Patch as _Patch
    fig.legend(
        handles=[
            _Patch(facecolor="#ff2a1a", lw=0.8, label="Over-prediction"),
            _Patch(facecolor="#1f5fff", lw=0.8, label="Under-prediction"),
        ],
        loc="lower center", ncol=2, fontsize=LEGEND_FONTSIZE, frameon=True, fancybox=True,
        edgecolor="#cccccc", facecolor="white", bbox_to_anchor=(0.54, LEGEND_Y),
        handlelength=LEGEND_HANDLELENGTH, handleheight=LEGEND_HANDLEHEIGHT,
        borderpad=0.5, columnspacing=1.2,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, pred_bin, []


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (identical to AF version)
# ─────────────────────────────────────────────────────────────────────────────

def make_temporal_sequence(fire_mask: np.ndarray, T: int = 5, seed: int = 0) -> np.ndarray:
    """
    Synthesize a T-step growing fire sequence from a single observed mask.

    Frame 0  ≈ 20 % of final area  (early ignition)
    Frame T-1 = full observed mask  (current state)

    Each frame is a spatial subset of the next — valid as AF input and
    BA (cumulative burned area) ground truth.
    """
    fire = (fire_mask >= 0.5).astype(np.float32)
    area = int(fire.sum())
    if area < 50:
        return np.zeros((T,) + fire.shape, dtype=np.float32)

    frames: list = []
    for t in range(T):
        progress = t / max(T - 1, 1)           # 0.0 → 1.0
        target_f = 0.20 + 0.80 * progress      # 20 % → 100 %

        if target_f >= 0.98 or t == T - 1:
            frames.append(fire.copy())
            continue

        # Binary-search for the erosion radius hitting ~target_f area fraction
        lo, hi = 0, 50
        while lo < hi - 1:
            mid  = (lo + hi) // 2
            cand = (binary_erosion(fire, structure=_disk_struct(mid)).astype(np.float32)
                    if mid > 0 else fire)
            lo, hi = ((mid, hi) if float(cand.sum()) / (area + 1e-6) > target_f
                      else (lo, mid))

        frame = (binary_erosion(fire, structure=_disk_struct(lo)).astype(np.float32)
                 if lo > 0 else fire.copy())
        # Fallback: erosion over-killed the mask
        if frame.sum() < max(20, area * 0.10) and lo > 1:
            frame = binary_erosion(
                fire, structure=_disk_struct(max(1, lo - 2))
            ).astype(np.float32)
        frames.append((frame >= 0.5).astype(np.float32))

    return np.stack(frames, axis=0)  # (T, H, W)


def load_gt_sequences(data_dir: str = "dataset_val",
                      label_channel: int = 1, T: int = 5) -> list:
    data_dir = Path(data_dir)
    for name in ["af_val_label_seqtoseq_alll_1i_1.npy",
                 "af_train_label_seqtoseq_alll_1i_1.npy"]:
        lf = data_dir / name
        if lf.exists():
            break
    else:
        raise FileNotFoundError(f"No label file in {data_dir}")

    labels = np.load(str(lf), mmap_mode="r")
    dev = str(_DEVICE) if _HAS_TORCH else "cpu"
    print(f"Labels shape: {labels.shape}  |  device: {dev}")
    sequences = []
    for i in range(labels.shape[0]):
        if labels.ndim == 5:                                  # (N, C, T, H, W)
            obs = np.array(labels[i, label_channel, 0], dtype=np.float32)
        elif labels.ndim == 4:                                # (N, C, H, W)
            obs = np.array(labels[i, label_channel],    dtype=np.float32)
        else:
            continue
        obs = np.clip(obs, 0, 1)
        if (obs >= 0.5).sum() < 200:                          # require substantial fire
            continue
        gt_seq = make_temporal_sequence(obs, T=T, seed=i)    # synthesize T-step sequence
        sequences.append((i, gt_seq))
    print(f"  -> {len(sequences)} events with sufficient fire area")
    return sequences


def find_best_events(sequences: list, top_k: int = 5) -> list:
    """
    Select top_k diverse fire events.

    Scoring  : final-frame fire area (larger = more illustrative).
    Diversity: greedy IoU-threshold selection — each new event must have
               < 15 % mask overlap with all already-selected events.
    """
    def _iou(m1: np.ndarray, m2: np.ndarray) -> float:
        b1, b2 = m1 >= 0.5, m2 >= 0.5
        return float((b1 & b2).sum()) / float((b1 | b2).sum() + 1e-6)

    scored: list = []
    for idx, gt in sequences:
        final_area = int((gt[-1] >= 0.5).sum())
        scored.append((final_area, idx, gt))
    scored.sort(reverse=True)

    selected:  list = []
    sel_masks: list = []
    iou_thr = 0.15

    for area, idx, gt in scored:
        mask = (gt[-1] >= 0.5).astype(np.float32)
        if not sel_masks or all(_iou(mask, m) < iou_thr for m in sel_masks):
            selected.append((idx, gt))
            sel_masks.append(mask)
        if len(selected) >= top_k:
            break

    # Relax diversity if we couldn't fill top_k
    if len(selected) < top_k:
        sel_indices = {s[0] for s in selected}
        for area, idx, gt in scored:
            if idx not in sel_indices:
                selected.append((idx, gt))
                sel_indices.add(idx)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def load_multihorizon_gt(data_dir: str = "dataset_val",
                         event_idx: int = 0, horizon: int = 5) -> np.ndarray:
    try:
        import yaml, torch, sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
        from firest_graphnet.data.firepred_provider import build_firepred_provider
        from firest_graphnet.data.static_provider   import build_static_provider

        cfg_path = Path("outputs/output-clean/run20/config.yaml")
        if not cfg_path.exists():
            cfg_path = Path("config_run20.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        fc = cfg.get("forecast", {}); dc = cfg.get("data", {})
        img_size = int(dc.get("img_size", 256))

        fp = build_firepred_provider(data_dir, spatial_size=(img_size, img_size))
        st = build_static_provider(data_dir,   spatial_size=(img_size, img_size))
        ds = TSSatFireForecastDataset(
            data_dir=data_dir,
            T_in=int(fc.get("T_in", 10)), horizon=int(fc.get("horizon", 20)),
            label_channel=int(dc.get("label_channel", 1)),
            firepred_provider=fp, static_provider=st,
        )
        if ds.firepred_provider is None and getattr(fp, "_N", 0) > 0:
            ds.firepred_provider = fp
        if ds.static_provider  is None and getattr(st, "_N", 0) > 0:
            ds.static_provider  = st

        event_idx = event_idx % len(ds)
        sample = ds[event_idx]
        af_fut = sample["af_fut"]
        if hasattr(af_fut, "numpy"):
            af_fut = af_fut.numpy()
        af_fut = np.clip(af_fut.squeeze(1), 0, 1)
        idxs = np.linspace(0, af_fut.shape[0] - 1, horizon, dtype=int)
        gt   = af_fut[idxs] if af_fut.shape[0] >= horizon else af_fut[:horizon]
        print(f"Loaded event {event_idx}: "
              f"fire pixels={[(gt[t]>=0.5).sum() for t in range(gt.shape[0])]}")
        return gt
    except Exception as e:
        print(f"Forecast dataset failed ({e}), falling back to raw labels")
        for name in ["af_val_label_seqtoseq_alll_1i_1.npy",
                     "af_train_label_seqtoseq_alll_1i_1.npy"]:
            lf = Path(data_dir) / name
            if lf.exists():
                labels = np.load(str(lf), mmap_mode="r")
                seqs = []
                for i in range(labels.shape[0]):
                    gt = np.clip(np.array(labels[i, 1], dtype=np.float32), 0, 1)
                    if gt.ndim == 3 and (gt >= 0.5).sum() >= 50:
                        seqs.append((i, gt[:horizon]))
                if seqs:
                    return seqs[event_idx % len(seqs)][1]
        raise FileNotFoundError("No usable label file found")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Pred probability illustration figures")
    p.add_argument("--event",         type=int,   default=5)
    p.add_argument("--quality",       type=float, default=0.75)
    p.add_argument("--smoothness",    type=float, default=2.0)
    p.add_argument("--noise",         type=float, default=0.1)
    p.add_argument("--fragmentation", type=float, default=0.0)
    p.add_argument("--mode", choices=["presentation", "realistic"],
                   default="presentation")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--data-dir", type=str, default="dataset_val")
    p.add_argument("--horizon",  type=int, default=5)
    p.add_argument("--out-dir",  type=str, default="figures")
    p.add_argument("--all",      action="store_true")
    p.add_argument("--top-k",    type=int, default=5)
    p.add_argument("--dpi",      type=int, default=200)
    p.add_argument("--show",     action="store_true")
    p.add_argument("--select-roi", action="store_true")
    p.add_argument("--use-roi",    action="store_true")
    p.add_argument("--roi-cache",  type=str, default="roi_cache.json")
    p.add_argument("--pred-stats", type=str, default=None,
                   help="Path to a GlobalProbStats JSON "
                        "(training-set min/max for nMAE normalization). "
                        "If omitted, canonical [0,1] probability stats are used.")
    args = p.parse_args()

    if args.pred_stats:
        globals()["_PRED_STATS"] = GlobalProbStats.load(args.pred_stats)
        print(f"[nMAE] Loaded global stats: "
              f"min={globals()['_PRED_STATS'].x_min:.4f} "
              f"max={globals()['_PRED_STATS'].x_max:.4f} "
              f"(from {args.pred_stats})")
    else:
        globals()["_PRED_STATS"] = DEFAULT_PROB_STATS
        print(f"[nMAE] Using default stats: "
              f"min={DEFAULT_PROB_STATS.x_min:.4f} "
              f"max={DEFAULT_PROB_STATS.x_max:.4f} "
              f"({DEFAULT_PROB_STATS.source})")

    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frag_tag = f"_f{int(args.fragmentation * 10):02d}"

    if args.all:
        print(f"Scanning {args.data_dir} for best events ...")
        seqs = load_gt_sequences(args.data_dir, T=args.horizon)
        best = find_best_events(seqs, top_k=args.top_k)
        print(f"Selected {len(best)} events\n")
        for rank, (ev_idx, af_gt) in enumerate(best):
            event_id = format_event_id(ev_idx)
            roi = None
            if args.select_roi:
                roi = select_roi_interactive(af_gt[min(4, af_gt.shape[0] - 1)])
                save_roi(event_id, roi, args.roi_cache)
                print(f"Saved ROI for {event_id}: {roi}")
            elif args.use_roi:
                roi = load_roi(event_id, args.roi_cache)
                if roi is None:
                    raise FileNotFoundError(
                        f"ROI for {event_id} not found in {args.roi_cache}. "
                        "Run with --select-roi first."
                    )
            sp = out_dir / f"pred_event{ev_idx:04d}_{args.mode}{frag_tag}.png"
            build_figure(
                af_gt, seed=args.seed, quality=args.quality,
                smoothness=args.smoothness, noise_level=args.noise,
                fragmentation=args.fragmentation,
                mode=args.mode, save_path=str(sp),
                dpi=args.dpi, show=args.show,
                roi=roi,
                auto_crop=not (args.select_roi or args.use_roi),
            )
            print(f"  [{rank+1}] event {ev_idx}: saved {sp.name}")
    else:
        af_gt = load_multihorizon_gt(args.data_dir, args.event, args.horizon)
        event_id = format_event_id(args.event)
        roi = None
        if args.select_roi:
            roi = select_roi_interactive(af_gt[min(4, af_gt.shape[0] - 1)])
            save_roi(event_id, roi, args.roi_cache)
            print(f"Saved ROI for {event_id}: {roi}")
        elif args.use_roi:
            roi = load_roi(event_id, args.roi_cache)
            if roi is None:
                raise FileNotFoundError(
                    f"ROI for {event_id} not found in {args.roi_cache}. "
                    "Run with --select-roi first."
                )
        sp    = out_dir / f"pred_event{args.event:04d}_{args.mode}{frag_tag}.png"
        build_figure(
            af_gt, seed=args.seed, quality=args.quality,
            smoothness=args.smoothness, noise_level=args.noise,
            fragmentation=args.fragmentation,
            mode=args.mode, save_path=str(sp),
            dpi=args.dpi, show=args.show,
            roi=roi,
            auto_crop=not (args.select_roi or args.use_roi),
        )
        print(f"\nEvent {args.event}: saved {sp.name}")

    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
