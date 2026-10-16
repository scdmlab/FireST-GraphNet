"""
visualize_illustration_ba.py — Illustration figure for the Burned Area (BA) task.

BA = cumulative burn extent: each frame shows the TOTAL area burned up to that time.
Unlike AF (active fire at a single instant), BA masks are monotonically growing —
they only expand, never shrink.

3-row × 5-col figure:
  Row 1: Ground truth BA  (T → T+4, cumulative)
  Row 2: Simulated BA prediction
  Row 3: Overlap  (TP=green / FP=red / FN=blue)

Usage:
  python visualize_illustration_ba.py --event 5 --quality 0.75 --fragmentation 0.4
  python visualize_illustration_ba.py --all --top-k 5 --quality 0.75 --fragmentation 0.35
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _disk_struct(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)

def _remove_small_components(mask: np.ndarray, min_area: int = 25) -> np.ndarray:
    labeled, n = nd_label(mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_area:
            mask[labeled == i] = 0.0
    return mask

def _extract_boundary_band(binary: np.ndarray, width: int = 4) -> np.ndarray:
    """Return a band of pixels near the mask boundary (dilate XOR erode)."""
    dilated = _dilate(binary.astype(np.float32), width)
    eroded  = _erode (binary.astype(np.float32), width)
    return np.clip(dilated - eroded, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal sequence synthesis
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


# ─────────────────────────────────────────────────────────────────────────────
# Build GT burned-area sequence from AF frames
# ─────────────────────────────────────────────────────────────────────────────

def af_to_ba(af_seq: np.ndarray) -> np.ndarray:
    """
    Convert per-timestep AF masks to cumulative burned area.

    BA[t] = union of AF[0..t]  → monotonically growing binary mask.
    """
    T = af_seq.shape[0]
    ba = np.zeros_like(af_seq[0], dtype=np.float32)
    ba_seq = []
    for t in range(T):
        ba = np.maximum(ba, (af_seq[t] >= 0.5).astype(np.float32))
        ba_seq.append(ba.copy())
    return np.stack(ba_seq, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Fragmentation: boundary-only roughness (same logic as AF version)
# ─────────────────────────────────────────────────────────────────────────────

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

    # Band width scales with fragmentation
    band_width = max(2, int(round(3 + 7 * fragmentation)))
    band = _extract_boundary_band(binary, width=band_width)

    # Medium-scale component (shapes the major boundary irregularities)
    medium = low_freq_noise((H, W), scale=10.0, rng=rng)
    # Fine-scale component (jagged micro-details)
    fine   = low_freq_noise((H, W), scale=4.5,  rng=rng)

    # Blend: higher fragmentation → more fine relative to medium
    alpha = 0.3 + 0.5 * fragmentation          # fine weight
    combined = (1 - alpha) * medium + alpha * fine
    # Map [0,1] → [-1, 1], blend with shared base for temporal coherence
    noise = (combined - 0.5) * 2.0
    noise = 0.55 * noise + 0.45 * (base_noise - 0.5) * 2.0

    # Amplitude: grows non-linearly with fragmentation so low values stay clean
    amplitude = 0.28 * (fragmentation ** 0.7)

    # Apply only in the boundary band
    perturbed = soft + noise * amplitude * band

    return np.clip(perturbed, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation — BA version
# ─────────────────────────────────────────────────────────────────────────────

def simulate_prediction(
    gt: np.ndarray,
    seed: int = 42,
    quality: float = 0.8,
    smoothness: float = 2.0,
    noise_level: float = 0.1,
    fragmentation: float = 0.0,
    mode: Literal["presentation", "realistic"] = "presentation",
) -> np.ndarray:
    """
    Simulate a plausible BA prediction from ground truth BA sequence.

    Key difference from AF:
      - Predictions are MONOTONICALLY ENFORCED: pred[t] ⊇ pred[t-1]
        (burned area never shrinks)
      - Errors are mostly at the growing frontier (leading edge), not interior
      - Interior (already-burned) pixels are very stable

    Parameters
    ----------
    gt            : (T, H, W) cumulative burned-area ground truth (binary)
    quality       : 0.0 (poor) → 1.0 (near-perfect)
    smoothness    : controls boundary shaping
    noise_level   : amplitude of structured noise at frontier
    fragmentation : 0.0 (smooth) → 1.0 (jagged frontier boundary)
    mode          : "presentation" | "realistic"
    """
    rng = np.random.default_rng(seed)
    T, H, W = gt.shape
    gt = (gt >= 0.5).astype(np.float32)

    if mode == "presentation":
        smoothness  *= 1.1
        noise_level *= 0.50
    else:
        smoothness  *= 0.80
        noise_level *= 1.20

    # Shared temporally-correlated base noise
    base_noise = low_freq_noise((H, W), scale=28.0, rng=rng)
    noise_fields = [
        0.55 * base_noise + 0.45 * low_freq_noise((H, W), scale=20.0, rng=rng)
        for _ in range(T)
    ]
    frag_base = low_freq_noise((H, W), scale=18.0, rng=rng)

    preds = []
    # Running binary mask — enforces monotonicity across frames
    cumulative_pred = np.zeros((H, W), dtype=np.float32)

    for t in range(T):
        # New-growth region in this frame
        if t == 0:
            new_growth = gt[t].copy()
        else:
            new_growth = np.clip(gt[t] - gt[t - 1], 0, 1)

        # ── Perturb the new-growth frontier ─────────────────────────────────
        sigma = max(0.4, smoothness * 0.55 * (1.0 + 0.20 * (1.0 - quality)))
        frontier_soft = _gauss_blur(new_growth, sigma)

        # Morphological perturbation on frontier only
        perturb_r    = max(1, int(round(3 * (1.0 - quality))))
        frontier_bin = (frontier_soft > 0.35).astype(np.float32)
        morph_choice = rng.random()
        if morph_choice < 0.28:
            frontier_bin = _dilate(frontier_bin, perturb_r)
        elif morph_choice < 0.55:
            frontier_bin = _erode(frontier_bin, perturb_r)

        blend_w = 0.48 + 0.18 * (1.0 - quality)
        frontier_soft = (1 - blend_w) * frontier_soft + blend_w * frontier_bin

        # Volumetric noise on frontier
        vol_noise = (noise_fields[t] - 0.5) * 2.0 * noise_level * (1.0 - quality * 0.65)
        frontier_soft = np.clip(frontier_soft + vol_noise, 0, 1)

        # Fragmentation on frontier boundary
        rough_bin = (frontier_soft > (0.45 + 0.10 * (1.0 - quality))).astype(np.float32)
        if fragmentation > 0.0:
            frontier_soft = _apply_fragmentation(
                frontier_soft, rough_bin, fragmentation, rng, frag_base, noise_fields[t],
            )

        thr = 0.48 + 0.10 * (1.0 - quality)
        frontier_pred = (frontier_soft > thr).astype(np.float32)

        # ── Enforce monotonicity: accumulate with previous prediction ────────
        # Interior (already burned at t-1) stays burned with very high prob
        # Only the frontier has prediction error
        if t == 0:
            frame_pred = frontier_pred.copy()
        else:
            # Interior pixels: keep with probability proportional to quality
            interior_keep_prob = 0.90 + 0.09 * quality
            interior = (cumulative_pred >= 0.5).astype(np.float32)
            # Rare interior pixel dropout (simulates slight over-erosion)
            dropout  = (rng.random((H, W)) < (1.0 - interior_keep_prob)).astype(np.float32)
            interior = np.clip(interior - dropout, 0, 1)
            frame_pred = np.maximum(interior, frontier_pred)

        # Clean up tiny fragments
        min_clean = 8 if fragmentation > 0.4 else 20
        frame_pred = _remove_small_components(frame_pred, min_area=min_clean)

        cumulative_pred = np.maximum(cumulative_pred, frame_pred)
        preds.append(cumulative_pred.copy())

    return np.stack(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap visualization
# ─────────────────────────────────────────────────────────────────────────────

def overlap_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_b  = (gt   >= 0.5)
    pr_b  = (pred >= 0.5)
    H, W  = gt.shape
    rgb   = np.full((H, W, 3), 0.08, dtype=np.float32)
    rgb[..., 2] = 0.10
    rgb[gt_b & pr_b]    = [0.20, 0.82, 0.25]   # TP — green
    rgb[(~gt_b) & pr_b] = [1.00, 0.24, 0.16]   # FP — brighter red
    rgb[gt_b & (~pr_b)] = [0.18, 0.52, 1.00]   # FN — brighter blue
    return rgb


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
    Build 3×5 BA illustration figure.  No title or caption.

    Parameters
    ----------
    af_seq : (T, H, W) per-timestep AF masks — converted internally to BA
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap

    T_show = min(5, af_seq.shape[0])
    af5    = np.clip(af_seq[:T_show].copy(), 0, 1)

    # Convert AF → cumulative BA ground truth
    gt5 = af_to_ba(af5)

    # Simulate BA prediction
    pred5 = simulate_prediction(
        gt5, seed=seed, quality=quality,
        smoothness=smoothness, noise_level=noise_level,
        fragmentation=fragmentation, mode=mode,
    )

    # Crop
    if roi is not None:
        gt5_c = apply_roi(gt5, roi)
        pred5_c = apply_roi(pred5, roi)
    elif auto_crop:
        auto_roi = compute_auto_roi(gt5, pred5, blur_fn=_gauss_blur)
        gt5_c = apply_roi(gt5, auto_roi)
        pred5_c = apply_roi(pred5, auto_roi)
    else:
        gt5_c = gt5;  pred5_c = pred5

    # IoU per frame
    ious = []
    for t in range(T_show):
        g = gt5[t] >= 0.5;  p = pred5[t] >= 0.5
        tp = (g & p).sum();  fp = (~g & p).sum();  fn = (g & ~p).sum()
        ious.append(tp / (tp + fp + fn + 1e-9))

    # ── Figure ───────────────────────────────────────────────────────────────
    crop_h, crop_w = gt5_c.shape[1:]
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

    # BA colormap: black → charcoal → ash brown → warm gray → light ash
    # Burnt earth tones — distinct from the fire-orange AF palette
    ba_cmap = LinearSegmentedColormap.from_list(
        "burned",
        ["#080808", "#1c0e05", "#3d1f0a", "#7a4010",
         "#b87040", "#d4a070", "#e8c8a0", "#f5e8d0"],
        N=256,
    )
    bg_color = "#0d0d10"

    time_labels = [f"T+{t}" for t in range(T_show)]
    row_labels  = ["Burned Area (GT)", "BA Prediction", "Overlap"]

    for t in range(T_show):
        gt_f  = gt5_c[t]
        pr_f  = pred5_c[t]

        # Row 0 — GT burned area
        ax = axes[0, t]
        ax.set_facecolor(bg_color)
        ax.imshow(gt_f, cmap=ba_cmap, vmin=0, vmax=1,
                  interpolation="bilinear", aspect="equal")
        ax.set_title(time_labels[t], fontsize=TITLE_FONTSIZE, fontweight="bold",
                     pad=TITLE_PAD, color="#2a2a2a")
        ax.axis("off")

        # Row 1 — BA prediction
        ax = axes[1, t]
        ax.set_facecolor(bg_color)
        ax.imshow(pr_f, cmap=ba_cmap, vmin=0, vmax=1,
                  interpolation="bilinear", aspect="equal")
        ax.axis("off")

        # Row 2 — Overlap
        ax = axes[2, t]
        ax.set_facecolor(bg_color)
        ax.imshow(overlap_rgb(gt_f, pr_f), interpolation="bilinear", aspect="equal")
        ax.axis("off")

    # Row labels
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            ROW_LABEL_X, 0.5, label, transform=axes[i, 0].transAxes,
            fontsize=ROW_LABEL_FONTSIZE, fontweight="bold", va="center", ha="right",
            rotation=90, color="#1a1a1a",
        )

    # Legend
    fig.legend(
        handles=[
            Patch(facecolor="#34d139", edgecolor="#1a7a1f", lw=0.8, label="True Positive"),
            Patch(facecolor="#ff3d29", edgecolor="#a61b10", lw=0.8, label="False Positive"),
            Patch(facecolor="#2e85ff", edgecolor="#1b4f99", lw=0.8, label="False Negative"),
        ],
        loc="lower center", ncol=3, fontsize=LEGEND_FONTSIZE, frameon=True, fancybox=True,
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

    return fig, pred5, ious


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_multihorizon_gt(data_dir="dataset_val", event_idx=0, horizon=5):
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
                    if gt.ndim == 3:
                        gt = gt
                    elif gt.ndim == 2:
                        gt = gt[None]
                    if (gt >= 0.5).sum() >= 50:
                        seqs.append((i, gt[:horizon]))
                if seqs:
                    idx = event_idx % len(seqs)
                    return seqs[idx][1]
        raise FileNotFoundError("No usable label file found")


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

    Scoring  : final-frame burned area (larger = more illustrative).
    Diversity: greedy IoU-threshold selection — each new event must have
               < 15 % mask overlap with all already-selected events.
    """
    def _iou(m1: np.ndarray, m2: np.ndarray) -> float:
        b1, b2 = m1 >= 0.5, m2 >= 0.5
        return float((b1 & b2).sum()) / float((b1 | b2).sum() + 1e-6)

    scored: list = []
    for idx, gt in sequences:
        ba         = af_to_ba(gt)
        final_area = int((ba[-1] >= 0.5).sum())
        scored.append((final_area, idx, gt))
    scored.sort(reverse=True)

    selected:  list = []
    sel_masks: list = []
    iou_thr = 0.15

    for area, idx, gt in scored:
        mask = (af_to_ba(gt)[-1] >= 0.5).astype(np.float32)
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="BA illustration figures")
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
    args = p.parse_args()

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
            sp = out_dir / f"ba_event{ev_idx:04d}_{args.mode}{frag_tag}.png"
            _, _, ious = build_figure(
                af_gt, seed=args.seed, quality=args.quality,
                smoothness=args.smoothness, noise_level=args.noise,
                fragmentation=args.fragmentation,
                mode=args.mode, save_path=str(sp),
                dpi=args.dpi, show=args.show,
                roi=roi,
                auto_crop=not (args.select_roi or args.use_roi),
            )
            print(f"  [{rank+1}] event {ev_idx}: mean IoU={np.mean(ious):.3f}  "
                  f"{[f'{x:.2f}' for x in ious]}")
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
        sp    = out_dir / f"ba_event{args.event:04d}_{args.mode}{frag_tag}.png"
        _, _, ious = build_figure(
            af_gt, seed=args.seed, quality=args.quality,
            smoothness=args.smoothness, noise_level=args.noise,
            fragmentation=args.fragmentation,
            mode=args.mode, save_path=str(sp),
            dpi=args.dpi, show=args.show,
            roi=roi,
            auto_crop=not (args.select_roi or args.use_roi),
        )
        print(f"\nEvent {args.event}: mean IoU={np.mean(ious):.3f}")
        print(f"Per-frame: {[f'{x:.3f}' for x in ious]}")

    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
