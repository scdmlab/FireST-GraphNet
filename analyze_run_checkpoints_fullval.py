#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, gaussian_filter
from skimage.morphology import remove_small_objects

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firest_graphnet.data.dataset_forecast import build_forecast_datasets
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider import build_static_provider
from firest_graphnet.models.firest_graphnet import FireSTGraphNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze run checkpoints on full validation set.")
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--outputs-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--runs", type=str, default="4,5,6,7")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--n-cols", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-dir", type=str, default="outputs/analysis")
    p.add_argument("--front-threshold", type=float, default=0.5,
        help="Probability threshold for binarizing prediction into a burn mask.")
    p.add_argument("--front-sigma", type=float, default=1.0,
        help="Gaussian blur sigma applied before thresholding the probability map.")
    p.add_argument("--front-min-size", type=int, default=20,
        help="Minimum connected-component size (pixels) kept in the binarized mask.")
    p.add_argument("--front-erosion-iters", type=int, default=1,
        help="Binary erosion iterations used to extract the fire-front boundary.")
    p.add_argument("--front-dilation-iters", type=int, default=1,
        help="Binary dilation iterations to thicken the extracted front for visibility.")
    p.add_argument("--no-overlay", action="store_true", default=False,
        help="Show predicted front only (no GT overlay) in the bottom row.")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_base_config(project_root: Path, outputs_dir: Path, explicit_config: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if explicit_config:
        candidates.append(Path(explicit_config))
    candidates.append(outputs_dir / "config.yaml")
    candidates.append(project_root / "firest_graphnet" / "configs" / "base.yaml")

    for c in candidates:
        if c.exists():
            print(f"Using config: {c}")
            return load_yaml(c)
    raise FileNotFoundError("No usable config found.")


def build_model_from_cfg(cfg: dict[str, Any], device: torch.device) -> FireSTGraphNet:
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
        img_h=data_cfg.get("img_size", 256),
        img_w=data_cfg.get("img_size", 256),
        forecast_mode=True,
        future_channels=future_channels,
        future_emb_dim=future_emb_dim,
    ).to(device)
    return model


def load_checkpoint(checkpoint_path: Path, fallback_cfg: dict[str, Any], device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
        meta = {
            "epoch": checkpoint.get("epoch", "?"),
            "best_val_iou": checkpoint.get("best_val_iou"),
        }
    else:
        state_dict = checkpoint
        checkpoint_cfg = {}
        meta = {
            "epoch": "?",
            "best_val_iou": None,
        }

    cfg = deep_update(fallback_cfg, checkpoint_cfg)
    cfg.setdefault("forecast", {})
    cfg["forecast"]["enabled"] = True

    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, cfg, meta


def build_validation_dataset(data_root: Path, cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    forecast_cfg = cfg.get("forecast", {})
    img_size = int(data_cfg.get("img_size", 256))

    firepred_provider = build_firepred_provider(data_root=str(data_root), spatial_size=(img_size, img_size))
    static_provider = build_static_provider(data_root=str(data_root), spatial_size=(img_size, img_size))

    _, val_ds = build_forecast_datasets(
        data_root=str(data_root),
        T_in=int(forecast_cfg.get("T_in", 10)),
        horizon=int(forecast_cfg.get("horizon", 20)),
        label_channel=int(data_cfg.get("label_channel", 1)),
        val_split=float(data_cfg.get("val_split", 0.1)),
        seed=int(cfg.get("seed", 42)),
        firepred_provider=firepred_provider,
        static_provider=static_provider,
    )
    return val_ds


def compute_sample_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.float32)
    gt = (targets >= 0.5).astype(np.float32)

    tp = float((preds * gt).sum())
    fp = float((preds * (1.0 - gt)).sum())
    fn = float(((1.0 - preds) * gt).sum())
    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_mean": float(probs.mean()),
    }


def run_inference(model: FireSTGraphNet, sample: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    af_hist = sample["af_hist"].unsqueeze(0).to(device)
    aux_hist = sample["aux_hist"].unsqueeze(0).to(device)
    env_fut = sample["env_fut"].unsqueeze(0).to(device)
    static = sample["static"].unsqueeze(0).to(device)
    env_source = sample.get("env_source", "firepred")
    if isinstance(env_source, (list, tuple)):
        env_source = env_source[0]

    with torch.no_grad():
        logits = model(
            af_hist,
            aux_hist=aux_hist,
            env_fut=env_fut,
            env_source=str(env_source),
            static=static,
        )
    probs = torch.sigmoid(logits).cpu().numpy()[0, :, 0]   # (T, H, W)
    targets = sample["af_fut"].cpu().numpy()[:, 0]         # (T, H, W)
    return probs, targets


# ── Firefront extraction helpers ──────────────────────────────────────────────

def extract_boundary(mask: np.ndarray, erosion_iters: int = 1) -> np.ndarray:
    """Return the outer boundary ring of a binary mask (bool array)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=erosion_iters)
    return mask & ~eroded


def render_qualitative_grid(
    probs: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    title: str,
    max_steps: int,
    n_cols: int,
    threshold: float = 0.5,
    sigma: float = 1.0,
    min_size: int = 20,
    front_erosion_iters: int = 1,
    front_dilation_iters: int = 1,
    overlay: bool = True,
) -> None:
    """
    Visualise per-step firefront boundaries.

    Each time step is processed independently.  GT and prediction at
    consecutive steps are allowed to overlap — no temporal subtraction
    is applied.  The front at step t is simply the boundary of the burn
    mask at step t.
    """
    steps = list(range(min(max_steps, probs.shape[0], targets.shape[0])))
    n_cols = max(1, min(n_cols, len(steps)))
    n_rows = math.ceil(len(steps) / n_cols)
    total_cells = n_rows * n_cols

    fig = plt.figure(figsize=(3.0 * n_cols, 3.0 * (2 * n_rows) + 0.95), facecolor="black")
    gs = fig.add_gridspec(
        2 * n_rows + 1,
        n_cols,
        height_ratios=[1.0] * (2 * n_rows) + [0.15],
        hspace=0.12,
        wspace=0.06,
    )

    pred_mappable = None
    for plot_idx, step in enumerate(steps):
        row_offset = plot_idx // n_cols
        col = plot_idx % n_cols

        # ── GT firefront boundary (independent per step) ──────────────────
        gt_mask = targets[step] >= 0.5
        gt_front = extract_boundary(gt_mask, erosion_iters=front_erosion_iters)
        if front_dilation_iters > 0:
            gt_front = binary_dilation(gt_front, iterations=front_dilation_iters)

        gt_ax = fig.add_subplot(gs[row_offset, col])
        gt_ax.imshow(gt_front.astype(np.float32), cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
        gt_ax.set_title(f"t+{step + 1}", color="white", fontsize=10, pad=6)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        gt_ax.set_facecolor("black")
        for spine in gt_ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.2)

        # ── Predicted firefront boundary (independent per step) ───────────
        pred_prob = probs[step].astype(np.float32)
        pred_mask = gaussian_filter(pred_prob, sigma=sigma) > threshold
        if pred_mask.any():
            pred_mask = remove_small_objects(pred_mask, min_size=min_size)
        if pred_mask.any():
            pred_mask = binary_closing(pred_mask, iterations=2)
        pred_front = extract_boundary(pred_mask, erosion_iters=front_erosion_iters)
        if front_dilation_iters > 0:
            pred_front = binary_dilation(pred_front, iterations=front_dilation_iters)

        pred_ax = fig.add_subplot(gs[n_rows + row_offset, col])
        if overlay:
            H, W = gt_front.shape
            rgb = np.zeros((H, W, 3), dtype=np.float32)
            rgb[gt_front]   = [0.0, 1.0, 1.0]             # cyan   — GT only
            rgb[pred_front] = [1.0, 0.8, 0.0]             # yellow — Pred only
            rgb[gt_front & pred_front] = [1.0, 1.0, 1.0]  # white  — overlap
            pred_mappable = pred_ax.imshow(rgb, interpolation="nearest")
        else:
            pred_mappable = pred_ax.imshow(
                pred_front.astype(np.float32),
                cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest",
            )
        pred_ax.set_xticks([])
        pred_ax.set_yticks([])
        pred_ax.set_facecolor("black")
        for spine in pred_ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.2)

    for empty_idx in range(len(steps), total_cells):
        row_offset = empty_idx // n_cols
        col = empty_idx % n_cols
        for row in [row_offset, n_rows + row_offset]:
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            ax.set_facecolor("black")

    fig.add_artist(
        Line2D(
            [0.06, 0.98],
            [0.49, 0.49],
            transform=fig.transFigure,
            color="white",
            linewidth=0.8,
            alpha=0.6,
        )
    )
    fig.text(0.025, 0.74, "GT Front",  color="white", rotation=90, va="center", ha="center", fontsize=12)
    fig.text(
        0.025, 0.31,
        "GT vs Pred" if overlay else "Pred Front",
        color="white", rotation=90, va="center", ha="center", fontsize=12,
    )
    fig.suptitle(title, color="white", fontsize=14, y=0.99)

    cax = fig.add_subplot(gs[-1, :])
    cax.set_facecolor("black")
    if overlay:
        cax.axis("off")
        legend_elements = [
            Patch(facecolor=(0.0, 1.0, 1.0), edgecolor="none", label="GT only"),
            Patch(facecolor=(1.0, 0.8, 0.0), edgecolor="none", label="Pred only"),
            Patch(facecolor=(1.0, 1.0, 1.0), edgecolor="none", label="Overlap"),
        ]
        cax.legend(
            handles=legend_elements,
            loc="center",
            ncol=3,
            framealpha=0.0,
            labelcolor="white",
            fontsize=11,
            handlelength=2.0,
        )
    else:
        cbar = fig.colorbar(pred_mappable, cax=cax, orientation="horizontal")
        cbar.set_label(
            f"Fire-front boundary  (\u03c3={sigma}, thresh={threshold}, min_size={min_size})",
            color="white",
        )
        cbar.ax.xaxis.set_tick_params(color="white", labelcolor="white")
        cbar.outline.set_edgecolor("white")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    outputs_dir = Path(args.outputs_dir).resolve() if args.outputs_dir else data_root / "outputs"
    save_dir = (Path(args.save_dir) if Path(args.save_dir).is_absolute() else data_root / args.save_dir).resolve()
    device = resolve_device(args.device)

    base_cfg = resolve_base_config(PROJECT_ROOT, outputs_dir, args.config)
    base_cfg.setdefault("data", {})
    base_cfg["data"]["root"] = str(data_root)

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    summary_rows: list[dict[str, Any]] = []

    for run_id in run_ids:
        ckpt = outputs_dir / f"run{run_id}" / "best_model.pt"
        if not ckpt.exists():
            print(f"[WARN] missing checkpoint: {ckpt}")
            continue

        print(f"\n=== run{run_id} ===")
        model, cfg, meta = load_checkpoint(ckpt, base_cfg, device)
        val_ds = build_validation_dataset(data_root, cfg)
        print(f"Validation samples: {len(val_ds)}")

        score_rows: list[dict[str, Any]] = []
        for sample_idx in range(len(val_ds)):
            sample = val_ds[sample_idx]
            probs, targets = run_inference(model, sample, device)
            m = compute_sample_metrics(probs, targets, threshold=0.5)
            row = {
                "run": f"run{run_id}",
                "sample_idx": sample_idx,
                "iou": m["iou"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "pred_mean": m["pred_mean"],
                "epoch": meta.get("epoch", "?"),
                "best_val_iou": meta.get("best_val_iou"),
            }
            score_rows.append(row)

        score_rows_sorted = sorted(score_rows, key=lambda x: x["iou"], reverse=True)
        topk_rows = score_rows_sorted[: args.topk]
        worst_rows = list(reversed(score_rows_sorted[-args.topk :]))

        all_csv = save_dir / f"run{run_id}_sample_scores.csv"
        top_csv = save_dir / f"run{run_id}_top{args.topk}.csv"
        worst_csv = save_dir / f"run{run_id}_worst{args.topk}.csv"

        fieldnames = list(score_rows[0].keys())
        write_csv(all_csv, score_rows, fieldnames)
        write_csv(top_csv, topk_rows, fieldnames)
        write_csv(worst_csv, worst_rows, fieldnames)

        ious = np.array([r["iou"] for r in score_rows], dtype=np.float32)
        summary_rows.append(
            {
                "run": f"run{run_id}",
                "n_samples": len(score_rows),
                "mean_iou": float(np.mean(ious)),
                "median_iou": float(np.median(ious)),
                "p10_iou": float(np.percentile(ious, 10)),
                "p90_iou": float(np.percentile(ious, 90)),
                "best_sample_idx": topk_rows[0]["sample_idx"],
                "best_sample_iou": topk_rows[0]["iou"],
                "worst_sample_idx": worst_rows[0]["sample_idx"],
                "worst_sample_iou": worst_rows[0]["iou"],
            }
        )

        for tag, pick in [("best", topk_rows[0]), ("worst", worst_rows[0])]:
            idx = int(pick["sample_idx"])
            sample = val_ds[idx]
            probs, targets = run_inference(model, sample, device)
            out_png = save_dir / "extremes_visuals" / f"run{run_id}_{tag}_sample{idx}.png"
            title = f"run{run_id} {tag} sample={idx} IoU={pick['iou']:.4f}"
            render_qualitative_grid(
                probs=probs,
                targets=targets,
                save_path=out_png,
                title=title,
                max_steps=args.max_steps,
                n_cols=args.n_cols,
                threshold=args.front_threshold,
                sigma=args.front_sigma,
                min_size=args.front_min_size,
                front_erosion_iters=args.front_erosion_iters,
                front_dilation_iters=args.front_dilation_iters,
                overlay=not args.no_overlay,
            )

        print(
            f"run{run_id}: mean_iou={float(np.mean(ious)):.4f}, "
            f"best={topk_rows[0]['sample_idx']}({topk_rows[0]['iou']:.4f}), "
            f"worst={worst_rows[0]['sample_idx']}({worst_rows[0]['iou']:.4f})"
        )

    if summary_rows:
        summary_csv = save_dir / "runs_summary.csv"
        write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))
        print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
