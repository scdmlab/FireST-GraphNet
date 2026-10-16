#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, gaussian_filter
from skimage.morphology import remove_small_objects

# Windows Conda environments can load multiple OpenMP runtimes through PyTorch
# and matplotlib/numpy. Allowing duplicate initialization avoids a hard crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize saved FireST-GraphNet checkpoints as multi-horizon qualitative figures."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project/data root containing dataset_train, dataset_val, and outputs/.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Directory containing best_model_runX.pt files. Defaults to <data_root>/outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path. If omitted, tries outputs/config.yaml then firest_graphnet/configs/base.yaml.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="4",
        help="Run number to visualize (for example: 1, 2, 3, 4) or 'all'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. Overrides --run.",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Validation sample index to visualize.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum future steps to display.")
    parser.add_argument("--n-cols", type=int, default=5, help="Number of columns in the figure grid.")
    parser.add_argument(
        "--env-source",
        type=str,
        default="auto",
        choices=["auto", "firepred", "hrrr", "weathernext2"],
        help="Decoder adapter source. 'auto' uses sample['env_source'].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save generated figures. Defaults to <outputs_dir>/qualitative_run_visuals.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window in addition to saving the image.",
    )
    parser.add_argument("--front-threshold", type=float, default=0.5,
        help="Probability threshold for binarizing prediction into a burn mask.")
    parser.add_argument("--front-sigma", type=float, default=1.0,
        help="Gaussian blur sigma applied before thresholding the probability map.")
    parser.add_argument("--front-min-size", type=int, default=20,
        help="Minimum connected-component size (pixels) kept in the binarized mask.")
    parser.add_argument("--front-erosion-iters", type=int, default=1,
        help="Binary erosion iterations used to extract the fire-front boundary.")
    parser.add_argument("--front-dilation-iters", type=int, default=1,
        help="Binary dilation iterations to thicken the extracted front for visibility.")
    parser.add_argument("--no-overlay", action="store_true", default=False,
        help="Show predicted front only (no GT overlay) in the bottom row.")
    parser.add_argument("--auto-threshold", action="store_true", default=False,
        help="Sweep thresholds and pick the one maximising mean IoU for this sample.")
    parser.add_argument("--roi-pad", type=int, default=24,
        help="Pixels of padding around the fire ROI crop (0 = full frame).")
    return parser.parse_args()


ARGS = parse_args()

if not ARGS.show:
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


def load_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_base_config(project_root: Path, outputs_dir: Path, explicit_config: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if explicit_config:
        candidates.append(Path(explicit_config))
    candidates.append(outputs_dir / "config.yaml")
    candidates.append(project_root / "firest_graphnet" / "configs" / "base.yaml")

    for candidate in candidates:
        if candidate.exists():
            cfg = load_yaml_config(candidate)
            print(f"Using config: {candidate}")
            return cfg

    raise FileNotFoundError("Could not find a usable YAML config. Pass --config explicitly.")


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


def load_checkpoint(checkpoint_path: Path, fallback_cfg: dict[str, Any], device: torch.device) -> tuple[FireSTGraphNet, dict[str, Any], dict[str, Any]]:
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
    cfg.setdefault("data", {})

    model = build_model_from_cfg(cfg, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"Strict checkpoint load failed for {checkpoint_path.name}: {exc}")
        print("Retrying with strict=False.")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

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


def select_env_source(sample: dict[str, Any], requested_source: str) -> str:
    if requested_source != "auto":
        return requested_source
    source = sample.get("env_source", "firepred")
    if isinstance(source, (list, tuple)):
        source = source[0]
    return str(source)


def run_inference(model: FireSTGraphNet, sample: dict[str, Any], device: torch.device, env_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    af_hist = sample["af_hist"].unsqueeze(0).to(device)
    aux_hist = sample["aux_hist"].unsqueeze(0).to(device)
    env_fut = sample["env_fut"].unsqueeze(0).to(device)
    static = sample["static"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            af_hist,
            aux_hist=aux_hist,
            env_fut=env_fut,
            env_source=env_source,
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


def find_best_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    sigma: float = 1.0,
    candidates: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep probability thresholds and return (best_threshold, best_mean_iou)."""
    if candidates is None:
        candidates = np.arange(0.10, 0.91, 0.05)
    best_thr, best_iou = 0.5, -1.0
    for thr in candidates:
        ious = []
        for step in range(probs.shape[0]):
            pred = gaussian_filter(probs[step].astype(np.float32), sigma=sigma) > thr
            gt   = targets[step] >= 0.5
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            ious.append(tp / (tp + fp + fn + 1e-7))
        mean_iou = float(np.mean(ious))
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_thr = float(thr)
    return best_thr, best_iou


def compute_sample_roi(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    sigma: float,
    pad: int = 24,
) -> tuple[int, int, int, int] | None:
    """Return (r0, r1, c0, c1) tight bounding box over all fire activity.

    Unions the GT burn mask and the predicted burn mask across all time
    steps, then pads by `pad` pixels and clamps to image bounds.
    Returns None when neither GT nor any prediction contains fire pixels.
    """
    H, W = targets.shape[1], targets.shape[2]
    rows, cols = [], []
    for step in range(targets.shape[0]):
        gt = targets[step] >= 0.5
        # strip isolated specks before bbox so they don't inflate the crop
        if gt.any():
            gt_clean = remove_small_objects(gt.copy(), min_size=50)
            src = gt_clean if gt_clean.any() else gt
            r, c = np.where(src)
            rows.extend(r.tolist()); cols.extend(c.tolist())
        pred = gaussian_filter(probs[step].astype(np.float32), sigma=sigma) > threshold
        pred_clean = remove_small_objects(pred.copy(), min_size=50) if pred.any() else pred
        if pred_clean.any():
            r, c = np.where(pred_clean)
            rows.extend(r.tolist()); cols.extend(c.tolist())
    if not rows:
        return None
    rows_arr = np.array(rows); cols_arr = np.array(cols)
    r0 = max(0, int(np.percentile(rows_arr, 1)) - pad)
    r1 = min(H, int(np.percentile(rows_arr, 99)) + pad + 1)
    c0 = max(0, int(np.percentile(cols_arr, 1)) - pad)
    c1 = min(W, int(np.percentile(cols_arr, 99)) + pad + 1)
    return r0, r1, c0, c1


def render_qualitative_grid(
    probs: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    title: str,
    max_steps: int,
    n_cols: int,
    show: bool,
    threshold: float = 0.5,
    sigma: float = 1.0,
    min_size: int = 20,
    front_erosion_iters: int = 1,
    front_dilation_iters: int = 1,
    overlay: bool = True,
    roi_pad: int = 24,
) -> None:
    """
    Visualise per-step firefront boundaries.

    Panels are cropped to a tight ROI around the union of all GT + predicted
    fire pixels (padded by `roi_pad`). Each overlay cell shows the per-step
    mask IoU as a text annotation.
    """
    steps = list(range(min(max_steps, probs.shape[0], targets.shape[0])))
    if not steps:
        raise ValueError("No future steps available to render.")

    # ── ROI crop ──────────────────────────────────────────────────────────
    roi = compute_sample_roi(probs, targets, threshold, sigma, pad=roi_pad) if roi_pad > 0 else None

    def _crop(arr2d: np.ndarray) -> np.ndarray:
        if roi is None:
            return arr2d
        r0, r1, c0, c1 = roi
        return arr2d[r0:r1, c0:c1]

    n_cols = max(1, min(n_cols, len(steps)))
    n_rows = math.ceil(len(steps) / n_cols)
    total_cells = n_rows * n_cols

    fig = plt.figure(
        figsize=(3.0 * n_cols, 3.0 * (2 * n_rows) + 0.95),
        facecolor="black",
    )
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

        # ── GT panel: filled cumulative mask (easier to compare than boundary alone) ──
        gt_mask_raw = targets[step] >= 0.5
        # filter isolated noise pixels for clean display (keep large components only)
        gt_mask = remove_small_objects(gt_mask_raw.copy(), min_size=30) if gt_mask_raw.any() else gt_mask_raw
        if not gt_mask.any():
            gt_mask = gt_mask_raw
        gt_front = extract_boundary(gt_mask, erosion_iters=front_erosion_iters)
        if front_dilation_iters > 0:
            gt_front = binary_dilation(gt_front, iterations=front_dilation_iters)

        # show filled interior as dim grey + bright boundary ring
        gt_vis = gt_mask.astype(np.float32) * 0.35
        gt_vis[gt_front] = 1.0

        gt_ax = fig.add_subplot(gs[row_offset, col])
        gt_ax.imshow(_crop(gt_vis), cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
        gt_ax.set_title(f"t+{step + 1}", color="white", fontsize=10, pad=6)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        gt_ax.set_facecolor("black")
        for spine in gt_ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.2)

        # ── Predicted firefront boundary (independent per step) ────────────
        pred_prob = probs[step].astype(np.float32)
        pred_mask = gaussian_filter(pred_prob, sigma=sigma) > threshold
        if pred_mask.any():
            pred_mask = remove_small_objects(pred_mask, min_size=min_size)
        if pred_mask.any():
            pred_mask = binary_closing(pred_mask, iterations=2)
        pred_front = extract_boundary(pred_mask, erosion_iters=front_erosion_iters)
        if front_dilation_iters > 0:
            pred_front = binary_dilation(pred_front, iterations=front_dilation_iters)

        # per-step mask IoU (on full-frame masks, not boundaries)
        tp = int((pred_mask & gt_mask).sum())
        fp = int((pred_mask & ~gt_mask).sum())
        fn = int((~pred_mask & gt_mask).sum())
        step_iou = tp / (tp + fp + fn + 1e-7)

        pred_ax = fig.add_subplot(gs[n_rows + row_offset, col])
        if overlay:
            Hf, Wf = gt_front.shape
            # ─ background: normalised probability heatmap (inferno, dimmed) ─
            p_crop = _crop(gaussian_filter(pred_prob, sigma=sigma))
            pmin, pmax = p_crop.min(), p_crop.max()
            p_norm = (p_crop - pmin) / (pmax - pmin + 1e-7)
            import matplotlib.cm as _cm
            prob_rgb = _cm.get_cmap("inferno")(p_norm)[:, :, :3].astype(np.float32) * 0.45
            # ─ overlay front boundaries ─────────────────────────────────
            gt_fc  = _crop(gt_front)
            prd_fc = _crop(pred_front)
            prob_rgb[gt_fc]            = [0.0, 1.0, 1.0]   # cyan  — GT
            prob_rgb[prd_fc]           = [1.0, 0.8, 0.0]   # yellow — Pred
            prob_rgb[gt_fc & prd_fc]   = [1.0, 1.0, 1.0]   # white  — overlap
            pred_mappable = pred_ax.imshow(prob_rgb, interpolation="nearest")
        else:
            pred_mappable = pred_ax.imshow(
                _crop(pred_front.astype(np.float32)),
                cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest",
            )
        pred_ax.set_xticks([])
        pred_ax.set_yticks([])
        pred_ax.set_facecolor("black")
        for spine in pred_ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.2)
        # IoU annotation bottom-right
        pred_ax.text(
            0.97, 0.03, f"IoU {step_iou:.2f}",
            transform=pred_ax.transAxes,
            color="white", fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
        )

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
    print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def resolve_checkpoints(outputs_dir: Path, run_arg: str, explicit_checkpoint: str | None) -> list[Path]:
    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint)
        return [checkpoint_path]

    if run_arg.lower() == "all":
        checkpoints = sorted(outputs_dir.glob("best_model_run*.pt"))
        return checkpoints

    checkpoint_path = outputs_dir / f"best_model_run{run_arg}.pt"
    return [checkpoint_path]


def main() -> None:
    data_root = Path(ARGS.data_root).resolve()
    outputs_dir = Path(ARGS.outputs_dir).resolve() if ARGS.outputs_dir else data_root / "outputs"
    save_dir = Path(ARGS.save_dir).resolve() if ARGS.save_dir else outputs_dir / "qualitative_run_visuals"
    device = resolve_device(ARGS.device)

    base_cfg = resolve_base_config(PROJECT_ROOT, outputs_dir, ARGS.config)
    base_cfg.setdefault("data", {})
    base_cfg["data"]["root"] = str(data_root)
    base_cfg.setdefault("forecast", {})
    base_cfg["forecast"]["enabled"] = True

    checkpoints = resolve_checkpoints(outputs_dir, ARGS.run, ARGS.checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {outputs_dir}")

    built_dataset = None

    print(f"Data root: {data_root}")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Save dir: {save_dir}")
    print(f"Device: {device}")

    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue

        model, cfg, meta = load_checkpoint(checkpoint_path, base_cfg, device)
        if built_dataset is None:
            built_dataset = build_validation_dataset(data_root, cfg)
            print(f"Validation samples: {len(built_dataset)}")

        if ARGS.sample_idx < 0 or ARGS.sample_idx >= len(built_dataset):
            raise IndexError(f"sample-idx {ARGS.sample_idx} is out of range for val set size {len(built_dataset)}")

        sample = built_dataset[ARGS.sample_idx]
        env_source = select_env_source(sample, ARGS.env_source)
        probs, targets = run_inference(model, sample, device, env_source)

        checkpoint_label = checkpoint_path.stem
        best_iou = meta.get("best_val_iou")
        if best_iou is None:
            title = f"{checkpoint_label} | sample {ARGS.sample_idx} | source={env_source}"
            print(f"Loaded {checkpoint_label} | epoch={meta.get('epoch', '?')}")
        else:
            title = (
                f"{checkpoint_label} | sample {ARGS.sample_idx} | source={env_source} | "
                f"best IoU={float(best_iou):.4f}"
            )
            print(f"Loaded {checkpoint_label} | epoch={meta.get('epoch', '?')} | best_val_iou={float(best_iou):.4f}")

        save_path = save_dir / f"{checkpoint_label}_sample{ARGS.sample_idx}_{env_source}.png"
        thr = ARGS.front_threshold
        if ARGS.auto_threshold:
            thr, found_iou = find_best_threshold(probs, targets, sigma=ARGS.front_sigma)
            print(f"Auto-threshold: {thr:.2f}  (mean IoU = {found_iou:.4f})")
            title += f"  [auto-thr={thr:.2f}]"

        render_qualitative_grid(
            probs=probs,
            targets=targets,
            save_path=save_path,
            title=title,
            max_steps=ARGS.max_steps,
            n_cols=ARGS.n_cols,
            show=ARGS.show,
            threshold=thr,
            sigma=ARGS.front_sigma,
            min_size=ARGS.front_min_size,
            front_erosion_iters=ARGS.front_erosion_iters,
            front_dilation_iters=ARGS.front_dilation_iters,
            overlay=not ARGS.no_overlay,
            roi_pad=ARGS.roi_pad,
        )


if __name__ == "__main__":
    main()