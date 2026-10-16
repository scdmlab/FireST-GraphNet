"""
visualize_illustration.py — Illustration-quality wildfire prediction figures.

3-row × 5-col figure:
  Row 1: Ground truth  (T → T+4)
  Row 2: Simulated predictions
  Row 3: Overlap  (TP=green / FP=red / FN=blue)

Usage:
  python visualize_illustration.py --event 5 --quality 0.75 --fragmentation 0.4
  python visualize_illustration.py --event 5 --quality 0.65 --fragmentation 0.7 --mode realistic
  python visualize_illustration.py --all --top-k 5 --quality 0.75 --fragmentation 0.4
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
# Low-frequency noise  (no external dependency)
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
# Helper morphological utilities
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
# Boundary-aware fragmentation
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
    amplitude = 0.30 * (fragmentation ** 0.7)

    # Apply only in the boundary band
    perturbed = soft + noise * amplitude * band

    return np.clip(perturbed, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
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
    Generate a plausible simulated prediction from ground truth.

    Parameters
    ----------
    gt            : (T, H, W) binary or soft ground truth
    seed          : random seed for reproducibility
    quality       : 0.0 (poor) → 1.0 (near-perfect match)
    smoothness    : Gaussian blur sigma for large-scale shaping
    noise_level   : amplitude of structured volumetric noise
    fragmentation : 0.0 (smooth) → 1.0 (jagged, irregular perimeter)
    mode          : "presentation" (cleaner) | "realistic" (noisier)

    Returns
    -------
    pred : (T, H, W) binary simulated prediction
    """
    rng = np.random.default_rng(seed)
    T, H, W = gt.shape
    gt = (gt >= 0.5).astype(np.float32)

    # ── Mode adjustments ────────────────────────────────────────────────────
    if mode == "presentation":
        smoothness  = smoothness * 1.1          # slightly smoother interior
        noise_level = noise_level * 0.55
    else:
        smoothness  = smoothness * 0.75
        noise_level = noise_level * 1.25

    # ── Pre-generate temporally-correlated noise ─────────────────────────────
    # One shared base gives temporal coherence; per-frame adds variation
    base_noise = low_freq_noise((H, W), scale=28.0, rng=rng)
    noise_fields = []
    for _ in range(T):
        frame_n = low_freq_noise((H, W), scale=20.0, rng=rng)
        noise_fields.append(0.55 * base_noise + 0.45 * frame_n)

    # Fragmentation base field (same for all frames → temporal coherence)
    frag_base = low_freq_noise((H, W), scale=18.0, rng=rng)

    preds   = []
    prev_soft = None

    for t in range(T):
        frame = gt[t].copy()

        # ── (A) Light smoothing of GT to soften input ─────────────────────
        # Use LESS smoothing than before so GT roughness is partly preserved
        sigma = max(0.5, smoothness * 0.6 * (1.0 + 0.25 * (1.0 - quality)))
        frame = _gauss_blur(frame, sigma)

        # ── (B) Morphological perturbation (global shape, quality-controlled)
        perturb_r = max(1, int(round(4 * (1.0 - quality))))
        frame_bin = (frame > 0.35).astype(np.float32)

        morph_choice = rng.random()
        if morph_choice < 0.30:
            frame_bin = _dilate(frame_bin, perturb_r)
        elif morph_choice < 0.58:
            frame_bin = _erode(frame_bin, perturb_r)

        # Blend: keep more of the rough binary at low smoothness
        blend_w = 0.50 + 0.20 * (1.0 - quality)   # binary weight 0.50–0.70
        frame = (1.0 - blend_w) * frame + blend_w * frame_bin

        # ── (C) Volumetric structured noise ──────────────────────────────
        vol_noise = (noise_fields[t] - 0.5) * 2.0 * noise_level * (1.0 - quality * 0.65)
        frame = frame + vol_noise

        # ── (D) Temporal consistency ──────────────────────────────────────
        if prev_soft is not None:
            temporal_w = 0.20 if mode == "presentation" else 0.12
            frame = (1.0 - temporal_w) * frame + temporal_w * prev_soft
            # Fire tends to grow: keep pixels that both GT and prev agree on
            grow_mask = (gt[t] > 0.5) & (prev_soft > 0.28)
            frame[grow_mask] = np.maximum(frame[grow_mask], 0.72)

        frame = np.clip(frame, 0, 1)

        # ── (E) Fragmentation: boundary-only roughness ───────────────────
        # Applied BEFORE final threshold so it shapes the boundary
        rough_bin = (frame > (0.48 + 0.12 * (1.0 - quality))).astype(np.float32)
        if fragmentation > 0.0:
            frame = _apply_fragmentation(
                frame, rough_bin, fragmentation, rng, frag_base, noise_fields[t]
            )

        # ── (F) Threshold ─────────────────────────────────────────────────
        thr = 0.50 + 0.12 * (1.0 - quality)
        pred_bin = (frame > thr).astype(np.float32)

        # Light cleanup: remove very tiny isolated fragments
        # but keep larger edge fragments that fragmentation creates
        min_clean = 8 if fragmentation > 0.4 else 20
        pred_bin = _remove_small_components(pred_bin, min_area=min_clean)

        preds.append(pred_bin)
        prev_soft = frame.copy()   # keep soft for temporal blending

    return np.stack(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap visualization
# ─────────────────────────────────────────────────────────────────────────────

def overlap_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    TP → green | FP → red | FN → blue | TN → near-black
    Returns (H, W, 3) float32
    """
    gt_b = (gt   >= 0.5)
    pr_b = (pred >= 0.5)
    H, W = gt.shape
    rgb = np.full((H, W, 3), 0.08, dtype=np.float32)
    rgb[..., 2] = 0.10
    rgb[gt_b & pr_b]   = [0.20, 0.82, 0.25]   # green  — TP
    rgb[(~gt_b) & pr_b] = [1.00, 0.24, 0.16]  # brighter red  — FP
    rgb[gt_b & (~pr_b)] = [0.18, 0.52, 1.00]  # brighter blue — FN
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# ROI crop
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(
    gt_seq: np.ndarray,
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
    Build the 3×5 illustration figure.  No title or caption text.

    Parameters
    ----------
    gt_seq        : (T, H, W), T ≥ 5 — uses first 5 frames
    fragmentation : 0.0 (smooth perimeter) → 1.0 (jagged, irregular edges)
    auto_crop     : zoom into the fire region for better visibility
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap

    T_show = min(5, gt_seq.shape[0])
    gt5 = np.clip(gt_seq[:T_show].copy(), 0, 1)

    # ── Simulate ─────────────────────────────────────────────────────────────
    pred5 = simulate_prediction(
        gt5, seed=seed, quality=quality,
        smoothness=smoothness, noise_level=noise_level,
        fragmentation=fragmentation, mode=mode,
    )

    # ── Crop to fire region ───────────────────────────────────────────────────
    if roi is not None:
        gt5_c = apply_roi(gt5, roi)
        pred5_c = apply_roi(pred5, roi)
    elif auto_crop:
        auto_roi = compute_auto_roi(gt5, pred5, blur_fn=_gauss_blur)
        gt5_c = apply_roi(gt5, auto_roi)
        pred5_c = apply_roi(pred5, auto_roi)
    else:
        gt5_c = gt5;  pred5_c = pred5

    # ── IoU per frame ─────────────────────────────────────────────────────────
    ious = []
    for t in range(T_show):
        g  = gt5[t]   >= 0.5
        p  = pred5[t] >= 0.5
        tp = (g & p).sum();  fp = (~g & p).sum();  fn = (g & ~p).sum()
        ious.append(tp / (tp + fp + fn + 1e-9))

    # ── Layout ────────────────────────────────────────────────────────────────
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

    # ── Colormaps ─────────────────────────────────────────────────────────────
    fire_cmap = LinearSegmentedColormap.from_list(
        "fire",
        ["#080808", "#1a0500", "#6b1a00", "#c44000",
         "#ff7b00", "#ffb833", "#ffd966", "#fff8e0"],
        N=256,
    )
    bg_color = "#0d0d10"

    time_labels = [f"T+{t}" for t in range(T_show)]
    row_labels  = ["Ground Truth", "Prediction", "Overlap"]

    for t in range(T_show):
        gt_f  = gt5_c[t]
        pr_f  = pred5_c[t]

        # Row 0 — Ground truth
        ax = axes[0, t]
        ax.set_facecolor(bg_color)
        ax.imshow(gt_f, cmap=fire_cmap, vmin=0, vmax=1,
                  interpolation="bilinear", aspect="equal")
        ax.set_title(time_labels[t], fontsize=TITLE_FONTSIZE, fontweight="bold",
                     pad=TITLE_PAD, color="#2a2a2a")
        ax.axis("off")

        # Row 1 — Prediction
        ax = axes[1, t]
        ax.set_facecolor(bg_color)
        ax.imshow(pr_f, cmap=fire_cmap, vmin=0, vmax=1,
                  interpolation="bilinear", aspect="equal")
        ax.axis("off")

        # Row 2 — Overlap
        ax = axes[2, t]
        ax.set_facecolor(bg_color)
        ax.imshow(overlap_rgb(gt_f, pr_f), interpolation="bilinear", aspect="equal")
        ax.axis("off")

    # ── Row labels (left side) ────────────────────────────────────────────────
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            ROW_LABEL_X, 0.5, label, transform=axes[i, 0].transAxes,
            fontsize=ROW_LABEL_FONTSIZE, fontweight="bold", va="center", ha="right",
            rotation=90, color="#1a1a1a",
        )

    # ── Legend (bottom, no title) ─────────────────────────────────────────────
    legend_patches = [
        Patch(facecolor="#34d139", edgecolor="#1a7a1f", lw=0.8, label="True Positive"),
        Patch(facecolor="#ff3d29", edgecolor="#a61b10", lw=0.8, label="False Positive"),
        Patch(facecolor="#2e85ff", edgecolor="#1b4f99", lw=0.8, label="False Negative"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=3,
        fontsize=LEGEND_FONTSIZE, frameon=True, fancybox=True,
        edgecolor="#cccccc", facecolor="white",
        bbox_to_anchor=(0.54, LEGEND_Y),
        handlelength=LEGEND_HANDLELENGTH,
        handleheight=LEGEND_HANDLEHEIGHT,
        borderpad=0.5,
        columnspacing=1.2,
    )

    # ── Save / show ───────────────────────────────────────────────────────────
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
# Data loading helpers
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
        af_fut = np.clip(af_fut.squeeze(1), 0, 1)          # (H_total, H, W)

        idxs = np.linspace(0, af_fut.shape[0] - 1, horizon, dtype=int)
        gt   = af_fut[idxs] if af_fut.shape[0] >= horizon else af_fut[:horizon]
        print(f"Loaded event {event_idx}: "
              f"fire pixels={[(gt[t]>=0.5).sum() for t in range(gt.shape[0])]}")
        return gt

    except Exception as e:
        print(f"Forecast dataset failed ({e}), falling back to raw labels")
        seqs = load_gt_sequences(data_dir)
        return seqs[event_idx % len(seqs)][1]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate illustration wildfire prediction figures")
    p.add_argument("--event",         type=int,   default=42)
    p.add_argument("--quality",       type=float, default=0.75,
                   help="Prediction accuracy 0–1 (default 0.75)")
    p.add_argument("--smoothness",    type=float, default=2.0)
    p.add_argument("--noise",         type=float, default=0.1)
    p.add_argument("--fragmentation", type=float, default=0.0,
                   help="Boundary roughness 0–1 (0=smooth, 1=jagged)")
    p.add_argument("--mode",   choices=["presentation", "realistic"],
                   default="presentation")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--data-dir",      type=str,   default="dataset_val")
    p.add_argument("--horizon",       type=int,   default=5)
    p.add_argument("--out-dir",       type=str,   default="figures")
    p.add_argument("--all",           action="store_true")
    p.add_argument("--top-k",         type=int,   default=5)
    p.add_argument("--dpi",           type=int,   default=200)
    p.add_argument("--show",          action="store_true")
    p.add_argument("--select-roi",    action="store_true")
    p.add_argument("--use-roi",       action="store_true")
    p.add_argument("--roi-cache",     type=str,   default="roi_cache.json")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frag_tag = f"_f{int(args.fragmentation*10):02d}"

    if args.all:
        print(f"Scanning {args.data_dir} for best events ...")
        seqs = load_gt_sequences(args.data_dir, T=args.horizon)
        best = find_best_events(seqs, top_k=args.top_k)
        print(f"Selected {len(best)} events\n")
        for rank, (ev_idx, gt) in enumerate(best):
            event_id = format_event_id(ev_idx)
            roi = None
            if args.select_roi:
                roi = select_roi_interactive(gt[min(4, gt.shape[0] - 1)])
                save_roi(event_id, roi, args.roi_cache)
                print(f"Saved ROI for {event_id}: {roi}")
            elif args.use_roi:
                roi = load_roi(event_id, args.roi_cache)
                if roi is None:
                    raise FileNotFoundError(
                        f"ROI for {event_id} not found in {args.roi_cache}. "
                        "Run with --select-roi first."
                    )
            sp = out_dir / f"illustration_event{ev_idx:04d}_{args.mode}{frag_tag}.png"
            _, _, ious = build_figure(
                gt, seed=args.seed, quality=args.quality,
                smoothness=args.smoothness, noise_level=args.noise,
                fragmentation=args.fragmentation,
                mode=args.mode, save_path=str(sp),
                dpi=args.dpi, show=args.show,
                roi=roi,
                auto_crop=not (args.select_roi or args.use_roi),
            )
            print(f"  [{rank+1}/{len(best)}] event {ev_idx}: "
                  f"mean IoU={np.mean(ious):.3f}  "
                  f"per-frame={[f'{x:.2f}' for x in ious]}")
    else:
        print(f"Loading event {args.event} ...")
        gt = load_multihorizon_gt(args.data_dir, args.event, args.horizon)
        event_id = format_event_id(args.event)
        roi = None
        if args.select_roi:
            roi = select_roi_interactive(gt[min(4, gt.shape[0] - 1)])
            save_roi(event_id, roi, args.roi_cache)
            print(f"Saved ROI for {event_id}: {roi}")
        elif args.use_roi:
            roi = load_roi(event_id, args.roi_cache)
            if roi is None:
                raise FileNotFoundError(
                    f"ROI for {event_id} not found in {args.roi_cache}. "
                    "Run with --select-roi first."
                )
        sp = out_dir / f"illustration_event{args.event:04d}_{args.mode}{frag_tag}.png"
        _, _, ious = build_figure(
            gt, seed=args.seed, quality=args.quality,
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
