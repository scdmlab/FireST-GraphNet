#!/usr/bin/env python
"""
render_paper_figure.py
Generates a publication-quality figure showing the best FireST-GraphNet predictions.

Layout:
  Two "event blocks" stacked vertically (train best + val best).
  Each block has columns for every horizon step:
    Row A: Ground Truth (fire = red on black)
    Row B: Predicted probability (inferno heatmap + threshold contour)
    Row C: Overlay (GT cyan / Pred red / Overlap white)
  A shared VIIRS input thumbnail is shown on the left of each block.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"C:\Users\22125\Desktop\FireST-GraphNet")
os.chdir(r"C:\Users\22125\Desktop\FireST-GraphNet")

import math
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
from firest_graphnet.data.firepred_provider import build_firepred_provider, FirePredProvider
from firest_graphnet.data.static_provider import build_static_provider, LULCStaticProvider
from firest_graphnet.models.firest_graphnet import FireSTGraphNet

# ── config ────────────────────────────────────────────────────────────────
RUNS_DIR  = Path(r"C:\Users\22125\Desktop\FireST-GraphNet\outputs\output-clean")
DATA_ROOT = Path(r"C:\Users\22125\Desktop\FireST-GraphNet")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = RUNS_DIR / "paper_figure_best_predictions.pdf"
SAVE_PNG  = RUNS_DIR / "paper_figure_best_predictions.png"

# Best events discovered from metrics sweep
EVENTS = [
    # (run, split, event_idx, label)
    ("run20", "train", 335, "Best Training Event  (IoU = 0.52)"),
    ("run16", "val",   14,  "Best Validation Event  (IoU = 0.28)"),
]

# Display at most this many horizon steps (crop to keep figure readable)
MAX_STEPS = 8   # keep manageable for paper width

# Threshold
THRESHOLD = 0.50

_FIRE_CMAP = ListedColormap([(0,0,0,0), (1.0, 0.12, 0.0, 1.0)])


# ── helpers ───────────────────────────────────────────────────────────────
def _norm(band):
    b = band.astype(np.float32)
    lo, hi = np.percentile(b, 2), np.percentile(b, 98)
    return np.clip((b - lo) / (hi - lo + 1e-9), 0, 1)


def viirs_rgb(af_8ch):
    r = _norm(af_8ch[6])  # I4 MWIR
    g = _norm(af_8ch[7])  # I5 LWIR
    b = _norm(af_8ch[0])  # I1 VIS
    return np.stack([r, g, b], -1)


def load_model(run_dir, device):
    cfg  = yaml.safe_load(open(run_dir / "config.yaml"))
    ckpt = torch.load(str(run_dir / "best_model.pt"), map_location=device, weights_only=False)
    sd   = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt.get("config"), dict):
        cfg = ckpt["config"]
    mc, dc, fc = cfg["model"], cfg["data"], cfg.get("forecast", {})
    model = FireSTGraphNet(
        in_channels=8+19+1,
        encoder_channels=mc["encoder_channels"],
        use_cbam=mc["use_cbam"],
        patch_size=mc["patch_size"],
        k_neighbors=mc["k_neighbors"],
        gnn_hidden=mc["gnn_hidden"],
        gnn_layers=mc["gnn_layers"],
        lstm_hidden=mc["lstm_hidden"],
        lstm_layers=mc["lstm_layers"],
        lstm_dropout=mc["lstm_dropout"],
        decoder_channels=mc["decoder_channels"],
        out_channels=1,
        img_h=dc["img_size"], img_w=dc["img_size"],
        forecast_mode=True,
        future_channels=fc.get("future_channels", 4),
        future_emb_dim=fc.get("future_emb_dim", 64),
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)
    return model, cfg


def load_sample(data_dir, cfg, event_idx):
    fc, dc = cfg.get("forecast", {}), cfg["data"]
    fp = build_firepred_provider(str(data_dir), spatial_size=(dc["img_size"], dc["img_size"]))
    st = build_static_provider(str(data_dir), spatial_size=(dc["img_size"], dc["img_size"]))
    ds = TSSatFireForecastDataset(
        data_dir, T_in=fc.get("T_in", 10), horizon=fc.get("horizon", 20),
        label_channel=dc.get("label_channel", 1),
        firepred_provider=fp, static_provider=st,
    )
    # Force re-attach providers (N-mismatch guard)
    if ds.firepred_provider is None and getattr(fp, "_N", 0) > 0:
        ds.firepred_provider = fp
    if ds.static_provider is None and getattr(st, "_N", 0) > 0:
        ds.static_provider = st
    return ds[event_idx]


@torch.no_grad()
def infer(model, sample, device):
    af   = sample["af_hist"].unsqueeze(0).to(device)
    aux  = sample["aux_hist"].unsqueeze(0).to(device)
    env  = sample["env_fut"].unsqueeze(0).to(device)
    stat = sample["static"].unsqueeze(0).to(device)
    src  = sample.get("env_source", "firepred")
    if isinstance(src, (list, tuple)):
        src = src[0]
    logits = model(af, aux_hist=aux, env_fut=env, env_source=src, static=stat)
    probs   = torch.sigmoid(logits).cpu().numpy()[0, :, 0]   # (H, Hpx, Wpx)
    targets = sample["af_fut"].numpy()[:, 0]                  # (H, Hpx, Wpx)
    af_last = sample["af_hist"].numpy()[-1]                   # (8, Hpx, Wpx)
    return probs, targets, af_last


def iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    return tp / (tp + fp + fn + 1e-9)


# ── gather data ───────────────────────────────────────────────────────────
print("Loading models and running inference...")
event_data = []
for run_name, split, ev_idx, label in EVENTS:
    run_dir  = RUNS_DIR / run_name
    data_dir = DATA_ROOT / f"dataset_{split}"
    model, cfg = load_model(run_dir, device=DEVICE)
    sample     = load_sample(data_dir, cfg, ev_idx)
    probs, targets, af_last = infer(model, sample, DEVICE)
    del model
    torch.cuda.empty_cache()

    H = min(MAX_STEPS, probs.shape[0])
    probs   = probs[:H]
    targets = targets[:H]
    pred_bin = (probs >= THRESHOLD).astype(np.float32)

    per_step_iou = [iou(pred_bin[t], targets[t]) for t in range(H)]
    agg_iou = iou(pred_bin, (targets >= 0.5).astype(np.float32))

    event_data.append({
        "label":   label,
        "af":      af_last,
        "probs":   probs,
        "targets": targets,
        "pred":    pred_bin,
        "iou":     agg_iou,
        "step_iou": per_step_iou,
        "H":       H,
    })
    print(f"  {run_name}/{split} event {ev_idx}: H={H} agg_iou={agg_iou:.4f}")

# ── figure layout ─────────────────────────────────────────────────────────
# For each event block: 3 rows (GT / Prob / Overlay), n_cols = min(H, MAX_STEPS)
# Two blocks stacked vertically, separated by a divider row
n_cols = max(e["H"] for e in event_data)
cell   = 2.2   # inches per cell
side   = 2.5   # sidebar (VIIRS input)
row_lbl_w = 0.55  # inches for row label column on right

fig_w  = side + n_cols * cell + row_lbl_w
# rows: 3 per event × n_blocks + dividers + colorbar
n_blocks    = len(event_data)
total_rows  = n_blocks * 3 + (n_blocks - 1) + 1
height_ratios = []
for bi in range(n_blocks):
    height_ratios += [1.0, 1.0, 1.0]
    if bi < n_blocks - 1:
        height_ratios.append(0.18)   # divider
height_ratios.append(0.18)           # colorbar

fig_h = cell * (n_blocks * 3) * 0.78 + 0.9
fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
# Extra rightmost column for row labels
gs  = gridspec.GridSpec(
    total_rows, n_cols + 2,   # col 0=sidebar, 1..n_cols=data, n_cols+1=row labels
    height_ratios=height_ratios,
    width_ratios=[side / cell] + [1.0] * n_cols + [row_lbl_w / cell],
    hspace=0.07,
    wspace=0.05,
    figure=fig,
)

row_cursor = 0
prob_im    = None   # keep for colorbar

for bi, ev in enumerate(event_data):
    H      = ev["H"]
    probs  = ev["probs"]
    tgts   = ev["targets"]
    pred   = ev["pred"]
    agg_iou = ev["iou"]
    siou   = ev["step_iou"]

    # ── Sidebar: VIIRS input (spans 3 rows) ─────────────────────────────
    ax_side = fig.add_subplot(gs[row_cursor:row_cursor+3, 0])
    ax_side.imshow(viirs_rgb(ev["af"]))
    ax_side.set_title(
        f"VIIRS Input\n(I4·I5·I1)",
        color="white", fontsize=8, pad=3,
    )
    ax_side.set_xticks([]); ax_side.set_yticks([])
    for sp in ax_side.spines.values():
        sp.set_edgecolor("#555"); sp.set_linewidth(0.5)

    # Event block title above sidebar
    ax_side.set_xlabel(
        ev["label"] + f"\n(Aggregate IoU = {agg_iou:.3f})",
        color="white", fontsize=8, labelpad=4,
    )

    for step in range(n_cols):
        gc = step + 1   # column (0 = sidebar)

        if step < H:
            gt_mask  = (tgts[step] >= 0.5).astype(np.float32)
            prob_s   = probs[step].astype(np.float32)
            pred_s   = pred[step].astype(np.float32)
            step_iou = siou[step]
        else:
            # Empty pad columns
            for r in range(3):
                ax = fig.add_subplot(gs[row_cursor + r, gc])
                ax.set_facecolor("black"); ax.axis("off")
            continue

        # Row 0: Ground Truth
        ax_gt = fig.add_subplot(gs[row_cursor + 0, gc])
        ax_gt.set_facecolor("black")
        ax_gt.imshow(np.zeros_like(gt_mask), cmap="gray", vmin=0, vmax=1)
        ax_gt.imshow(gt_mask, cmap=_FIRE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax_gt.set_title(f"t+{step+1}", color="white", fontsize=9, pad=3)
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        # Row 1: Predicted probability
        ax_pr = fig.add_subplot(gs[row_cursor + 1, gc])
        ax_pr.set_facecolor("black")
        prob_im = ax_pr.imshow(prob_s, cmap="inferno", vmin=0.0, vmax=1.0)
        if (prob_s >= THRESHOLD).any():
            ax_pr.contour(prob_s, levels=[THRESHOLD], colors=["white"], linewidths=0.8, alpha=0.9)
        ax_pr.set_xticks([]); ax_pr.set_yticks([])

        # Row 2: Overlay
        ax_ov = fig.add_subplot(gs[row_cursor + 2, gc])
        ax_ov.set_facecolor("black")
        bg = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        bg[..., 0] = pred_s
        bg[..., 1] = gt_mask * 0.85
        bg[..., 2] = gt_mask
        bg[(gt_mask.astype(bool)) & (pred_s.astype(bool))] = [1.0, 1.0, 1.0]
        ax_ov.imshow(bg)
        ax_ov.text(
            0.96, 0.04, f"IoU={step_iou:.2f}",
            transform=ax_ov.transAxes,
            color="white", fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65, edgecolor="none"),
        )
        ax_ov.set_xticks([]); ax_ov.set_yticks([])

        # Row labels in rightmost column (only first column of each row)
        if step == 0:
            for ri, lbl in enumerate(["Ground\nTruth", "Predicted\nProbability", "GT vs\nPrediction"]):
                ax_rl = fig.add_subplot(gs[row_cursor + ri, n_cols + 1])
                ax_rl.set_facecolor("black"); ax_rl.axis("off")
                ax_rl.text(0.05, 0.5, lbl, color="white", fontsize=8.5,
                           va="center", ha="left", transform=ax_rl.transAxes)

    row_cursor += 3

    # Thin divider between event blocks
    if bi < n_blocks - 1:
        for gc in range(n_cols + 2):
            ax_div = fig.add_subplot(gs[row_cursor, gc])
            ax_div.set_facecolor("#333333"); ax_div.axis("off")
        row_cursor += 1

# ── Colorbar + legend ─────────────────────────────────────────────────────
cax = fig.add_subplot(gs[-1, 1:n_cols+1])
cax.set_facecolor("black")
if prob_im is not None:
    cbar = fig.colorbar(prob_im, cax=cax, orientation="horizontal")
    cbar.set_label(f"Predicted fire probability  (threshold = {THRESHOLD:.2f})",
                   color="white", fontsize=8)
    cbar.ax.xaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    cbar.outline.set_edgecolor("white")

# Legend patches
legend_elements = [
    Patch(facecolor=(1.0, 0.12, 0.0), edgecolor="none", label="Ground Truth only"),
    Patch(facecolor=(1.0, 0.0,  0.0), edgecolor="none", label="Prediction only"),
    Patch(facecolor=(1.0, 1.0,  1.0), edgecolor="none", label="Overlap (TP)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower left", bbox_to_anchor=(0.01, 0.0),
    ncol=3, framealpha=0.0, labelcolor="white", fontsize=8,
)

fig.suptitle(
    "FireST-GraphNet  —  Multi-Horizon Active Fire Prediction",
    color="white", fontsize=13, y=1.002,
)

# ── Save ──────────────────────────────────────────────────────────────────
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(SAVE_PNG), dpi=200, facecolor="black", bbox_inches="tight")
fig.savefig(str(SAVE_PATH), dpi=200, facecolor="black", bbox_inches="tight")
print(f"\nSaved:\n  {SAVE_PNG}\n  {SAVE_PATH}")
plt.close(fig)
