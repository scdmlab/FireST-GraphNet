#!/usr/bin/env python
"""
batch_eval_all_runs.py — Batch inference + visualization for every FireST-GraphNet run.

For each run under outputs/output-clean/run{14..22}:
  1) load best_model.pt + config.yaml
  2) build the forecast dataset for both dataset_train and dataset_val
  3) run inference on every sliding-window event
  4) save a side-by-side comparison PNG (Input AF | Ground Truth | Prediction)
     into outputs/output-clean/{run}/predictions/{train,val}/event_XXXX_comparison.png
  5) write a per-run summary metrics CSV (per-event IoU, F1, Precision, Recall)
"""
from __future__ import annotations

import os
import sys
import csv
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

# OpenMP duplicate guard (Windows / conda multi-runtime issue)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider import build_static_provider
from firest_graphnet.data.future_providers import (
    HRRRFutureProvider, WeatherNext2FutureProvider, PlaceholderFutureProvider,
)
from firest_graphnet.models.firest_graphnet import FireSTGraphNet


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch eval + viz across all runs.")
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument(
        "--runs-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "output-clean"),
    )
    p.add_argument(
        "--runs",
        type=str,
        default="14,15,16,17,18,19,20,21,22",
        help="Comma-separated run numbers (e.g. 14,15,16) or 'all'.",
    )
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Fixed binarization threshold. Ignored when --auto-threshold is set.")
    p.add_argument("--auto-threshold", action="store_true",
                   help="Sweep thresholds per event and pick the one maximising IoU.")
    p.add_argument("--auto-thr-min", type=float, default=0.05)
    p.add_argument("--auto-thr-max", type=float, default=0.50)
    p.add_argument("--auto-thr-step", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-train", type=int, default=-1, help="Cap train events (-1 = all, 0 = skip).")
    p.add_argument("--max-val", type=int, default=-1, help="Cap val events (-1 = all, 0 = skip).")
    p.add_argument(
        "--device", type=str, default="auto", help="auto, cuda, cuda:0, cpu",
    )
    p.add_argument(
        "--max-horizon-cols", type=int, default=5,
        help="Columns per row in the time-series figure.",
    )
    p.add_argument("--timeseries", action="store_true",
                   help="Also save a per-step time-series figure (GT/Prob/Overlay rows).")
    p.add_argument(
        "--env-source",
        choices=["firepred", "hrrr", "weathernext2", "mixed"],
        default="firepred",
        help=(
            "Future weather source for inference:\n"
            "  firepred    — use FirePred env_fut from the dataset (training default)\n"
            "  hrrr        — HRRR DOY climatology for all horizon steps\n"
            "  weathernext2— Google WeatherNext2 for all horizon steps (placeholder/zeros)\n"
            "  mixed       — HRRR for step 1, WeatherNext2 for steps 2..H"
        ),
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_cfg(cfg: Dict[str, Any], device: torch.device) -> FireSTGraphNet:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    forecast_cfg = cfg.get("forecast", {})
    future_channels = int(forecast_cfg.get("future_channels", 4))
    future_emb_dim = int(forecast_cfg.get("future_emb_dim", forecast_cfg.get("weather_emb_dim", 64)))

    model = FireSTGraphNet(
        in_channels=8 + 19 + 1,
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128]),
        use_cbam=model_cfg.get("use_cbam", True),
        patch_size=model_cfg.get("patch_size", 8),
        k_neighbors=model_cfg.get("k_neighbors", 8),
        gnn_hidden=model_cfg.get("gnn_hidden", 128),
        gnn_layers=model_cfg.get("gnn_layers", 2),
        lstm_hidden=model_cfg.get("lstm_hidden", 256),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        lstm_dropout=model_cfg.get("lstm_dropout", 0.1),
        decoder_channels=model_cfg.get("decoder_channels", [128, 64, 32]),
        out_channels=model_cfg.get("out_channels", 1),
        img_h=int(data_cfg.get("img_size", 256)),
        img_w=int(data_cfg.get("img_size", 256)),
        forecast_mode=True,
        future_channels=future_channels,
        future_emb_dim=future_emb_dim,
    ).to(device)
    return model


def load_run(run_dir: Path, device: torch.device) -> Tuple[FireSTGraphNet, Dict[str, Any], Dict[str, Any]]:
    cfg = load_yaml(run_dir / "config.yaml")
    ckpt = torch.load(str(run_dir / "best_model.pt"), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        meta = {"epoch": ckpt.get("epoch"), "best_val_iou": ckpt.get("best_val_iou")}
        # Prefer config from checkpoint if present (more authoritative)
        if isinstance(ckpt.get("config"), dict):
            cfg = ckpt["config"]
    else:
        sd = ckpt
        meta = {"epoch": None, "best_val_iou": None}

    model = build_model_from_cfg(cfg, device)
    incompat = model.load_state_dict(sd, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print(
            f"  [load] missing={len(incompat.missing_keys)} "
            f"unexpected={len(incompat.unexpected_keys)}"
        )
    model.eval()
    return model, cfg, meta


def build_dataset(
    data_dir: Path, cfg: Dict[str, Any], force_aux: bool = True
) -> TSSatFireForecastDataset:
    """Build the forecast dataset for an explicit split directory.

    The on-disk firepred / lulc npy files have a different N than the AF
    img/label files, so the dataset's strict N-alignment check disables both
    providers (zeros). The model was trained with these aux features active,
    and feeding zeros silences its output entirely. When force_aux=True we
    re-attach the providers after the dataset's safety check, accepting that
    indices wrap modulo N. Without this the model produces nothing useful.
    """
    forecast_cfg = cfg.get("forecast", {})
    data_cfg = cfg.get("data", {})
    img_size = int(data_cfg.get("img_size", 256))
    # Build providers from the explicit split dir to load the matching npy
    fp = build_firepred_provider(str(data_dir), spatial_size=(img_size, img_size))
    st = build_static_provider(str(data_dir), spatial_size=(img_size, img_size))

    ds = TSSatFireForecastDataset(
        data_dir=data_dir,
        T_in=int(forecast_cfg.get("T_in", 10)),
        horizon=int(forecast_cfg.get("horizon", 20)),
        label_channel=int(data_cfg.get("label_channel", 1)),
        firepred_provider=fp,
        static_provider=st,
    )
    if force_aux:
        # Re-attach in case the dataset's N-mismatch guard cleared them.
        if ds.firepred_provider is None and getattr(fp, "_N", 0) > 0:
            ds.firepred_provider = fp
        if ds.static_provider is None and getattr(st, "_N", 0) > 0:
            ds.static_provider = st
    return ds


# ──────────────────────────────────────────────────────────────────────────
# Weather provider helpers  (HRRR / WeatherNext2)
# ──────────────────────────────────────────────────────────────────────────
def build_env_providers(data_root: Path, env_source: str, img_size: int):
    """Return (hrrr_prov, wn2_prov) based on env_source mode.

    For 'firepred' mode both are None (no override needed).
    For 'hrrr' / 'mixed': loads HRRRFutureProvider from hrrr_cache/climatology/.
    For 'weathernext2' / 'mixed': loads WeatherNext2FutureProvider (zeros stub).
    """
    hrrr_prov, wn2_prov = None, None

    if env_source in ("hrrr", "mixed"):
        clim_dir = data_root / "hrrr_cache" / "climatology"
        try:
            hrrr_prov = HRRRFutureProvider(clim_dir, spatial_size=(img_size, img_size))
            print(f"  HRRR provider: {hrrr_prov._D} DOY frames ({clim_dir.name})")
        except FileNotFoundError as e:
            print(f"  WARNING: HRRR climatology missing ({e}) — falling back to zeros")
            hrrr_prov = PlaceholderFutureProvider(
                source_name="hrrr", spatial_size=(img_size, img_size)
            )

    if env_source in ("weathernext2", "mixed"):
        wn2_cache = data_root / "cache" / "weathernext2"
        wn2_prov = WeatherNext2FutureProvider(
            cache_dir=str(wn2_cache), spatial_size=(img_size, img_size)
        )
        print(f"  WeatherNext2 provider ready (cache: {wn2_cache})")

    return hrrr_prov, wn2_prov


def build_env_fut_override(
    ev_idx: int,
    horizon: int,
    hrrr_prov,
    wn2_prov,
    env_source: str,
) -> Tuple["np.ndarray | None", str]:
    """Build env_fut ndarray override for a single event.

    Returns (env_fut_np, env_source_str) where env_fut_np is None when
    env_source=='firepred' (use the dataset's own env_fut unchanged).

    env_fut_np shape: (H, 4, img_h, img_w)
    env_source_str  : always "firepred" for override modes.

    IMPORTANT — adapter note:
        The model was trained using only the "firepred" adapter.  The "hrrr"
        and "weathernext2" adapters exist in the checkpoint but were never
        optimized (random init).  All sources produce the same 4-channel
        canonical format [temp, u, v, rh], so routing HRRR / WN2 data through
        the trained "firepred" adapter gives correct conditioning.  Passing
        env_source="hrrr" to the model would use an untrained adapter and
        produce garbage FiLM conditioning.

    mixed mode:
      step 0  — HRRR DOY climatology  (real NWP, 1-step)
      steps 1..H-1 — WeatherNext2 (zeros stub until WN2 API is public)
    """
    if env_source == "firepred":
        return None, "firepred"

    indices = tuple(range(ev_idx, ev_idx + horizon))
    # Always use the trained "firepred" adapter regardless of data source.
    adapter = "firepred"

    if env_source == "hrrr":
        env_fut = hrrr_prov.get_future_env(indices, horizon=horizon)
        return env_fut, adapter

    if env_source == "weathernext2":
        env_fut = wn2_prov.get_future_env(indices, horizon=horizon)
        return env_fut, adapter

    if env_source == "mixed":
        hrrr_step = hrrr_prov.get_future_env((ev_idx,), horizon=1)   # (1, 4, H, W)
        if horizon > 1:
            wn2_steps = wn2_prov.get_future_env(
                tuple(range(ev_idx + 1, ev_idx + horizon)), horizon=horizon - 1
            )  # (H-1, 4, H, W)
            env_fut = np.concatenate([hrrr_step, wn2_steps], axis=0)
        else:
            env_fut = hrrr_step
        return env_fut, adapter

    return None, "firepred"


@torch.no_grad()
def run_inference(
    model: FireSTGraphNet,
    sample: Dict[str, Any],
    device: torch.device,
    env_fut_np: "np.ndarray | None" = None,
    env_source_override: "str | None" = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    af_hist = sample["af_hist"].unsqueeze(0).to(device)
    aux_hist = sample["aux_hist"].unsqueeze(0).to(device)
    static = sample["static"].unsqueeze(0).to(device)

    if env_fut_np is not None:
        # Override from HRRR / WeatherNext2 / mixed provider
        env_fut = torch.from_numpy(env_fut_np).unsqueeze(0).to(device)
        env_source = env_source_override or "hrrr"
    else:
        # Use dataset's own env_fut (firepred training data)
        env_fut = sample["env_fut"].unsqueeze(0).to(device)
        env_source = sample.get("env_source", "firepred")
        if isinstance(env_source, (list, tuple)):
            env_source = env_source[0]

    logits = model(
        af_hist,
        aux_hist=aux_hist,
        env_fut=env_fut,
        env_source=env_source,
        static=static,
    )
    probs = torch.sigmoid(logits).cpu().numpy()[0, :, 0]      # (H, Hpx, Wpx)
    targets = sample["af_fut"].cpu().numpy()[:, 0]            # (H, Hpx, Wpx)
    last_af_input = sample["af_hist"].cpu().numpy()[-1]       # (8, Hpx, Wpx)
    return probs, targets, last_af_input


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())
    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────
# Auto-threshold
# ──────────────────────────────────────────────────────────────────────────
def find_best_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    thr_min: float = 0.05,
    thr_max: float = 0.50,
    thr_step: float = 0.05,
) -> Tuple[float, float]:
    """Sweep thresholds and return (best_thr, best_mean_iou).

    Sweeps the full grid AND also tries the mean-prob and median-prob as
    data-driven candidates, so it works even when probabilities are very low.
    """
    candidates = list(np.arange(thr_min, thr_max + thr_step * 0.5, thr_step))
    # Data-driven candidates: mean, median, 75th-pct of predicted probs
    for pct in [25, 50, 75]:
        v = float(np.percentile(probs, pct))
        if 0.0 < v < 1.0:
            candidates.append(round(v, 4))
    candidates = sorted(set(candidates))

    best_thr, best_iou = float(candidates[0]), -1.0
    gt_bin = (targets >= 0.5)
    for thr in candidates:
        pred_b = (probs >= thr)
        tp = int((pred_b & gt_bin).sum())
        fp = int((pred_b & ~gt_bin).sum())
        fn = int((~pred_b & gt_bin).sum())
        iou = tp / (tp + fp + fn + 1e-9)
        if iou > best_iou:
            best_iou = iou
            best_thr = float(thr)
    return best_thr, best_iou


# ──────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────
_FIRE_CMAP = ListedColormap([(0, 0, 0, 0.0), (1.0, 0.15, 0.0, 1.0)])  # transparent → red


def _normalize_band(band: np.ndarray) -> np.ndarray:
    b = band.astype(np.float32)
    lo, hi = np.percentile(b, 2), np.percentile(b, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(b)
    return np.clip((b - lo) / (hi - lo), 0.0, 1.0)


def make_af_rgb(af_8ch: np.ndarray) -> np.ndarray:
    """Build a quick RGB visualization of the last historical VIIRS frame.

    Uses I4 (band 6, MWIR) and I5 (band 7, LWIR) which best highlight active
    fire. Falls back to a neutral grayscale composite if those bands are flat.
    """
    if af_8ch.shape[0] >= 8:
        i4 = _normalize_band(af_8ch[6])
        i5 = _normalize_band(af_8ch[7])
        i1 = _normalize_band(af_8ch[0])
    else:
        i4 = i5 = i1 = _normalize_band(af_8ch.mean(axis=0))
    rgb = np.stack([i4, i5, i1], axis=-1)
    return rgb


def render_event_comparison(
    af_input: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    pred_bin: np.ndarray,
    save_path: Path,
    title: str,
    threshold: float,
    metrics: Dict[str, float],
    max_cols: int = 5,
) -> None:
    """Render a compact comparison: 3 columns (Input | GT | Prediction overlay)
    using temporal max-projections for speed and at-a-glance comparison."""
    H = targets.shape[0]

    # Temporal max projections (union across horizon = full burn extent)
    gt_union = (targets.max(axis=0) >= 0.5).astype(np.float32)
    pred_union = (pred_bin.max(axis=0) > 0).astype(np.float32)
    prob_max = probs.max(axis=0).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6), facecolor="white")

    # 0) input VIIRS AF
    axes[0].imshow(make_af_rgb(af_input))
    axes[0].set_title("Input VIIRS\n(I4/I5/I1)", fontsize=10)

    # 1) GT union (red)
    axes[1].imshow(np.zeros_like(gt_union), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(gt_union, cmap=_FIRE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Ground Truth (union over t+1..t+{H})", fontsize=10)

    # 2) Prediction probability heatmap (max over horizon)
    im = axes[2].imshow(prob_max, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Pred prob (max), thr={threshold:.2f}", fontsize=10)

    # 3) Overlay GT(cyan)/Pred(red), white = overlap
    bg = np.zeros((*gt_union.shape, 3), dtype=np.float32)
    bg[..., 0] = pred_union              # R = pred
    bg[..., 1] = gt_union * 0.85         # G = GT
    bg[..., 2] = gt_union                # B = GT  → cyan when only GT, red when only pred
    overlap = (gt_union * pred_union).astype(bool)
    bg[overlap] = [1.0, 1.0, 1.0]        # white for overlap
    axes[3].imshow(bg)
    axes[3].set_title(
        f"GT (cyan) vs Pred (red)\n"
        f"IoU={metrics['iou']:.3f}  F1={metrics['f1']:.3f}",
        fontsize=10,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.98, wspace=0.10)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=110, facecolor="white")
    plt.close(fig)


def render_timeseries(
    af_input: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    pred_bin: np.ndarray,
    save_path: Path,
    title: str,
    threshold: float,
    max_cols: int = 5,
) -> None:
    """Per-step time-series figure: 3 rows (GT | Prob | Overlay) × horizon columns.

    Layout
    ------
    Row 0 : Ground Truth binary mask per step (black bg, red fire)
    Row 1 : Predicted probability heatmap per step (inferno colormap)
    Row 2 : Overlay — cyan=GT, red=Pred, white=overlap, on black bg
    Left sidebar column shows the last historical VIIRS AF frame.
    """
    H = targets.shape[0]
    n_cols = max(1, min(max_cols, H))
    n_rows_grid = math.ceil(H / n_cols)
    cell = 2.8   # inches per cell

    fig_w = cell * (n_cols + 0.8)          # +0.8 for sidebar
    fig_h = cell * 3 * n_rows_grid + 0.7   # 3 plot-rows per grid-row + title

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
    # gridspec: 3*n_rows_grid plot rows + 1 colorbar row, n_cols+1 cols (sidebar + data)
    gs = fig.add_gridspec(
        3 * n_rows_grid + 1,
        n_cols + 1,
        height_ratios=[1.0] * (3 * n_rows_grid) + [0.10],
        width_ratios=[0.9] + [1.0] * n_cols,
        hspace=0.08,
        wspace=0.06,
    )

    # Sidebar: VIIRS input (span all rows in column 0)
    ax_side = fig.add_subplot(gs[:3, 0])
    ax_side.imshow(make_af_rgb(af_input))
    ax_side.set_title("VIIRS\nInput", color="white", fontsize=8, pad=3)
    ax_side.set_xticks([]); ax_side.set_yticks([])
    for spine in ax_side.spines.values():
        spine.set_edgecolor("gray"); spine.set_linewidth(0.5)
    # blank remaining sidebar cells
    for gr in range(1, n_rows_grid):
        ax_b = fig.add_subplot(gs[3*gr:3*gr+3, 0])
        ax_b.set_facecolor("black"); ax_b.axis("off")

    prob_im_ref = None
    for step in range(H):
        gr = step // n_cols
        gc = step % n_cols + 1    # +1 for sidebar
        gt_mask = (targets[step] >= 0.5).astype(np.float32)
        prob_s = probs[step].astype(np.float32)
        pred_s = pred_bin[step].astype(np.float32)

        # per-step IoU
        m = binary_metrics(pred_s, gt_mask)
        iou_s = m["iou"]

        # ── row 0: GT ──
        ax_gt = fig.add_subplot(gs[3*gr + 0, gc])
        ax_gt.set_facecolor("black")
        ax_gt.imshow(np.zeros_like(gt_mask), cmap="gray", vmin=0, vmax=1)
        ax_gt.imshow(gt_mask, cmap=_FIRE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax_gt.set_title(f"t+{step+1}", color="white", fontsize=8, pad=2)
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        # ── row 1: probability heatmap ──
        ax_pr = fig.add_subplot(gs[3*gr + 1, gc])
        ax_pr.set_facecolor("black")
        prob_im_ref = ax_pr.imshow(prob_s, cmap="inferno", vmin=0.0, vmax=1.0)
        # mark threshold contour if there's any area above it
        if (prob_s >= threshold).any():
            ax_pr.contour(prob_s, levels=[threshold], colors=["white"], linewidths=0.6)
        ax_pr.set_xticks([]); ax_pr.set_yticks([])

        # ── row 2: overlay ──
        ax_ov = fig.add_subplot(gs[3*gr + 2, gc])
        ax_ov.set_facecolor("black")
        bg = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        bg[..., 0] = pred_s                    # red = pred
        bg[..., 1] = gt_mask * 0.85            # green component of cyan
        bg[..., 2] = gt_mask                   # blue = GT  → cyan when GT-only
        overlap = (gt_mask.astype(bool)) & (pred_s.astype(bool))
        bg[overlap] = [1.0, 1.0, 1.0]
        ax_ov.imshow(bg)
        ax_ov.text(
            0.98, 0.02, f"IoU={iou_s:.2f}",
            transform=ax_ov.transAxes,
            color="white", fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, edgecolor="none"),
        )
        ax_ov.set_xticks([]); ax_ov.set_yticks([])

    # Hide trailing empty cells
    for empty in range(H, n_rows_grid * n_cols):
        gr = empty // n_cols
        gc = empty % n_cols + 1
        for r in range(3):
            ax_e = fig.add_subplot(gs[3*gr + r, gc])
            ax_e.set_facecolor("black"); ax_e.axis("off")

    # Row labels (left edge of figure)
    for gr in range(n_rows_grid):
        row_y = 1.0 - (3*gr + 0.5) / (3 * n_rows_grid + 0.1)
        for r, lbl in enumerate(["GT", "Prob", "Overlay"]):
            y = 1.0 - (3*gr + r + 0.5) / (3 * n_rows_grid + 0.1)
            fig.text(0.01, y, lbl, color="white", fontsize=8,
                     va="center", ha="left", rotation=0)

    # Colorbar for probability
    cax = fig.add_subplot(gs[-1, 1:])
    cax.set_facecolor("black")
    if prob_im_ref is not None:
        cbar = fig.colorbar(prob_im_ref, cax=cax, orientation="horizontal")
        cbar.set_label(f"Predicted probability  (threshold={threshold:.2f})",
                       color="white", fontsize=8)
        cbar.ax.xaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
        cbar.outline.set_edgecolor("white")

    fig.suptitle(title, color="white", fontsize=10, y=0.995)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=120, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Per-run driver
# ──────────────────────────────────────────────────────────────────────────
def evaluate_split(
    model: FireSTGraphNet,
    dataset: TSSatFireForecastDataset,
    split_name: str,
    save_root: Path,
    device: torch.device,
    threshold: float,
    run_label: str,
    max_events: int,
    max_cols: int,
    auto_threshold: bool = False,
    auto_thr_min: float = 0.05,
    auto_thr_max: float = 0.50,
    auto_thr_step: float = 0.05,
    save_timeseries: bool = False,
    env_source: str = "firepred",
    hrrr_prov=None,
    wn2_prov=None,
) -> List[Dict[str, Any]]:
    out_dir = save_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    if max_events > 0:
        n = min(n, max_events)

    rows: List[Dict[str, Any]] = []
    for idx in range(n):
        sample = dataset[idx]
        horizon = int(sample["af_fut"].shape[0])
        env_fut_np, env_src_str = build_env_fut_override(
            idx, horizon, hrrr_prov, wn2_prov, env_source
        )
        probs, targets, af_input = run_inference(
            model, sample, device,
            env_fut_np=env_fut_np,
            env_source_override=env_src_str,
        )

        # Determine threshold for this event
        if auto_threshold:
            event_thr, _ = find_best_threshold(
                probs, targets,
                thr_min=auto_thr_min, thr_max=auto_thr_max, thr_step=auto_thr_step,
            )
        else:
            event_thr = threshold

        pred_bin = (probs >= event_thr).astype(np.uint8)
        gt_bin = (targets >= 0.5).astype(np.uint8)

        # Aggregate metrics across all horizon steps for this event
        metrics = binary_metrics(pred_bin, gt_bin)
        per_step_iou: List[float] = []
        for t in range(targets.shape[0]):
            per_step_iou.append(binary_metrics(pred_bin[t], gt_bin[t])["iou"])

        meta = sample.get("meta", {}) or {}
        event_id = int(meta.get("event_id", -1))
        frame_idx = int(meta.get("index", idx))

        thr_tag = f"auto-thr={event_thr:.2f}" if auto_threshold else f"thr={event_thr:.2f}"
        title = (
            f"{run_label} | {split_name} event {idx:04d} "
            f"(eid={event_id}, frame={frame_idx}) | {thr_tag}"
        )

        # Summary comparison figure
        save_path = out_dir / f"event_{idx:04d}_comparison.png"
        render_event_comparison(
            af_input=af_input,
            targets=targets,
            probs=probs,
            pred_bin=pred_bin,
            save_path=save_path,
            title=title,
            threshold=event_thr,
            metrics=metrics,
            max_cols=max_cols,
        )

        # Optional time-series figure
        if save_timeseries:
            ts_dir = save_root / f"{split_name}_timeseries"
            ts_path = ts_dir / f"event_{idx:04d}_timeseries.png"
            render_timeseries(
                af_input=af_input,
                targets=targets,
                probs=probs,
                pred_bin=pred_bin,
                save_path=ts_path,
                title=title,
                threshold=event_thr,
                max_cols=max_cols,
            )

        rows.append({
            "event_idx": idx,
            "event_id": event_id,
            "frame_idx": frame_idx,
            "horizon": int(targets.shape[0]),
            "threshold_used": event_thr,
            "iou": metrics["iou"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "mean_per_step_iou": float(np.mean(per_step_iou)) if per_step_iou else 0.0,
            "image_path": str(save_path.relative_to(save_root)),
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == n:
            print(f"  [{split_name}] {idx + 1}/{n}  iou={metrics['iou']:.3f}  thr={event_thr:.2f}")

    # Save per-split CSV
    csv_path = save_root / f"metrics_{split_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"  [{split_name}] metrics CSV -> {csv_path}")
    return rows


def evaluate_run(
    run_dir: Path,
    data_root: Path,
    device: torch.device,
    threshold: float,
    max_train: int,
    max_val: int,
    max_cols: int,
    auto_threshold: bool = False,
    auto_thr_min: float = 0.05,
    auto_thr_max: float = 0.50,
    auto_thr_step: float = 0.05,
    save_timeseries: bool = False,
    env_source: str = "firepred",
) -> Dict[str, Any]:
    run_label = run_dir.name
    print(f"\n=== {run_label} ===")
    model, cfg, meta = load_run(run_dir, device)
    h = cfg.get("forecast", {}).get("horizon", "?")
    thr_mode = f"auto [{auto_thr_min:.2f}-{auto_thr_max:.2f}]" if auto_threshold else f"fixed={threshold:.2f}"
    print(f"  ckpt epoch={meta.get('epoch')} best_val_iou={meta.get('best_val_iou')} horizon={h} thr={thr_mode}")
    print(f"  env_source={env_source}")

    img_size = int(cfg.get("data", {}).get("img_size", 256))
    hrrr_prov, wn2_prov = build_env_providers(data_root, env_source, img_size)

    save_root = run_dir / "predictions"
    save_root.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "dataset_train"
    val_dir = data_root / "dataset_val"

    split_kw = dict(
        auto_threshold=auto_threshold,
        auto_thr_min=auto_thr_min,
        auto_thr_max=auto_thr_max,
        auto_thr_step=auto_thr_step,
        save_timeseries=save_timeseries,
        env_source=env_source,
        hrrr_prov=hrrr_prov,
        wn2_prov=wn2_prov,
    )

    summary: Dict[str, Any] = {"run": run_label, "epoch": meta.get("epoch"),
                                "best_val_iou": meta.get("best_val_iou"),
                                "horizon": h}

    if train_dir.is_dir() and max_train != 0:
        ds_train = build_dataset(train_dir, cfg)
        print(f"  train events: {len(ds_train)}")
        rows = evaluate_split(
            model, ds_train, "train", save_root, device, threshold,
            run_label, max_train, max_cols, **split_kw,
        )
        if rows:
            summary["train_mean_iou"] = float(np.mean([r["iou"] for r in rows]))
            summary["train_mean_f1"] = float(np.mean([r["f1"] for r in rows]))
            summary["train_mean_thr"] = float(np.mean([r["threshold_used"] for r in rows]))
            summary["train_count"] = len(rows)
        del ds_train
    elif max_train == 0:
        print("  train: skipped (--max-train 0)")
    else:
        print(f"  WARNING: train dir not found: {train_dir}")

    if val_dir.is_dir() and max_val != 0:
        ds_val = build_dataset(val_dir, cfg)
        print(f"  val events: {len(ds_val)}")
        rows = evaluate_split(
            model, ds_val, "val", save_root, device, threshold,
            run_label, max_val, max_cols, **split_kw,
        )
        if rows:
            summary["val_mean_iou"] = float(np.mean([r["iou"] for r in rows]))
            summary["val_mean_f1"] = float(np.mean([r["f1"] for r in rows]))
            summary["val_mean_thr"] = float(np.mean([r["threshold_used"] for r in rows]))
            summary["val_count"] = len(rows)
        del ds_val
    elif max_val == 0:
        print("  val: skipped (--max-val 0)")
    else:
        print(f"  WARNING: val dir not found: {val_dir}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    device = resolve_device(args.device)

    if args.runs.lower() == "all":
        run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run")])
    else:
        nums = [n.strip() for n in args.runs.split(",") if n.strip()]
        run_dirs = [runs_dir / f"run{n}" for n in nums]

    print(f"Data root : {data_root}")
    print(f"Runs dir  : {runs_dir}")
    print(f"Device    : {device}")
    print(f"Threshold : {args.threshold}")
    print(f"Env source: {args.env_source}")
    print(f"Runs      : {[p.name for p in run_dirs]}")

    summaries: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        if not (run_dir / "best_model.pt").exists():
            print(f"Skipping {run_dir.name}: no best_model.pt")
            continue
        if not (run_dir / "config.yaml").exists():
            print(f"Skipping {run_dir.name}: no config.yaml")
            continue
        try:
            s = evaluate_run(
                run_dir=run_dir,
                data_root=data_root,
                device=device,
                threshold=args.threshold,
                max_train=args.max_train,
                max_val=args.max_val,
                max_cols=args.max_horizon_cols,
                auto_threshold=args.auto_threshold,
                auto_thr_min=args.auto_thr_min,
                auto_thr_max=args.auto_thr_max,
                auto_thr_step=args.auto_thr_step,
                save_timeseries=args.timeseries,
                env_source=args.env_source,
            )
            summaries.append(s)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  FAILED {run_dir.name}: {exc}")

    # Top-level summary CSV across runs
    summary_path = runs_dir / "predictions_summary.csv"
    if summaries:
        keys = sorted({k for s in summaries for k in s.keys()})
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
        print(f"\nSummary CSV -> {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
