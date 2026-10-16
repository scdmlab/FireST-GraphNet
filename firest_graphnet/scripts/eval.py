#!/usr/bin/env python
"""
eval.py — Evaluation script for FireSTGraphNet.

Supports legacy (single-step) and forecast (multi-horizon) modes.

Usage:
    python -m firest_graphnet.scripts.eval --checkpoint outputs/best_model.pt
    python -m firest_graphnet.scripts.eval --checkpoint outputs_forecast/best_model.pt --mode forecast
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from firest_graphnet.models.firest_graphnet import FireSTGraphNet
from firest_graphnet.training.metrics import (
    compute_metrics,
    compute_metrics_per_horizon,
    metrics_summary_str,
)
from firest_graphnet.utils.io import load_yaml, ensure_dir
from firest_graphnet.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FireSTGraphNet")
    p.add_argument("--data_root", type=str,
                    default=r"C:\Users\22125\Desktop\FireST-GraphNet")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, default="legacy",
                    choices=["legacy", "forecast"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save_vis", action="store_true")
    p.add_argument("--output_dir", type=str, default="eval_outputs")
    return p.parse_args()


def _find_config(ckpt_path: str) -> str | None:
    ckpt_dir = Path(ckpt_path).parent
    for c in [ckpt_dir / "config.yaml",
              _project_root / "firest_graphnet" / "configs" / "base.yaml"]:
        if c.exists():
            return str(c)
    return None


def main():
    args = parse_args()

    cfg_path = args.config or _find_config(args.checkpoint)
    if not cfg_path or not Path(cfg_path).exists():
        print("[ERROR] Config not found. Use --config.")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    seed_everything(cfg.get("seed", 42))

    dev_str = args.device
    device = torch.device(
        "cuda" if (dev_str == "auto" and torch.cuda.is_available()) else
        dev_str if dev_str != "auto" else "cpu"
    )

    if args.mode == "forecast":
        _eval_forecast(cfg, device, args)
    else:
        _eval_legacy(cfg, device, args)


def _eval_legacy(cfg, device, args):
    from firest_graphnet.data.dataset import build_datasets

    dcfg = cfg["data"]
    _, val_ds = build_datasets(
        data_root=args.data_root,
        T_in=dcfg["T_in"],
        T_out=dcfg.get("T_out", 1),
        label_channel=dcfg.get("label_channel", 1),
        seed=cfg.get("seed", 42),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    mcfg = cfg["model"]
    model = FireSTGraphNet(
        in_channels=mcfg.get("in_channels", dcfg.get("num_bands", 8)),
        encoder_channels=mcfg.get("encoder_channels", [32, 64, 128]),
        use_cbam=mcfg.get("use_cbam", True),
        patch_size=mcfg.get("patch_size", 8),
        k_neighbors=mcfg.get("k_neighbors", 8),
        gnn_hidden=mcfg.get("gnn_hidden", 128),
        gnn_layers=mcfg.get("gnn_layers", 2),
        lstm_hidden=mcfg.get("lstm_hidden", 256),
        lstm_layers=mcfg.get("lstm_layers", 2),
        lstm_dropout=mcfg.get("lstm_dropout", 0.1),
        decoder_channels=mcfg.get("decoder_channels", [128, 64, 32]),
        out_channels=mcfg.get("out_channels", 1),
        img_h=dcfg.get("img_size", 256),
        img_w=dcfg.get("img_size", 256),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    print(f"Loaded: {args.checkpoint} | Val: {len(val_ds)} samples")

    all_metrics = {"iou": 0, "f1": 0, "precision": 0, "recall": 0}
    n = 0
    vis_dir = ensure_dir(args.output_dir) / "vis" if args.save_vis else None

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            x_seq = batch["x_seq"].to(device)
            y = batch["y"].to(device)
            logits = model(x_seq)
            m = compute_metrics(logits, y)
            for k in all_metrics:
                all_metrics[k] += m[k]
            n += 1

            if vis_dir is not None:
                probs = torch.sigmoid(logits).cpu().numpy()
                gt = y.cpu().numpy()
                for i in range(probs.shape[0]):
                    idx = (n - 1) * args.batch_size + i
                    combined = np.concatenate([probs[i, 0], gt[i, 0]], axis=1)
                    np.save(str(vis_dir / f"sample_{idx:04d}.npy"), combined)

    avg = {k: v / max(n, 1) for k, v in all_metrics.items()}
    print(f"\n{'='*40}\n  Evaluation Results\n{'='*40}")
    for k, v in avg.items():
        print(f"  {k:>12s} : {v:.4f}")
    print(f"{'='*40}\n")


def _eval_forecast(cfg, device, args):
    """Forecast-mode evaluation with per-horizon metrics."""
    from firest_graphnet.data.dataset_forecast import build_forecast_datasets
    from firest_graphnet.data.weather_provider import build_weather_provider

    dcfg = cfg["data"]
    fcfg = cfg.get("forecast", {})
    T_in = fcfg.get("T_in", 10)
    horizon = fcfg.get("horizon", 20)
    Cw = fcfg.get("weather_channels", 8)

    wp = build_weather_provider(
        mode=fcfg.get("weather_provider", "dataset"),
        data_root=args.data_root,
    )

    _, val_ds = build_forecast_datasets(
        data_root=args.data_root,
        T_in=T_in, horizon=horizon,
        label_channel=dcfg.get("label_channel", 1),
        seed=cfg.get("seed", 42),
        weather_provider=wp,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    mcfg = cfg["model"]
    model = FireSTGraphNet(
        in_channels=fcfg.get("in_channels", 1),
        encoder_channels=mcfg.get("encoder_channels", [32, 64, 128]),
        use_cbam=mcfg.get("use_cbam", True),
        patch_size=mcfg.get("patch_size", 8),
        k_neighbors=mcfg.get("k_neighbors", 8),
        gnn_hidden=mcfg.get("gnn_hidden", 128),
        gnn_layers=mcfg.get("gnn_layers", 2),
        lstm_hidden=mcfg.get("lstm_hidden", 256),
        lstm_layers=mcfg.get("lstm_layers", 2),
        lstm_dropout=mcfg.get("lstm_dropout", 0.1),
        decoder_channels=mcfg.get("decoder_channels", [128, 64, 32]),
        out_channels=1,
        img_h=dcfg.get("img_size", 256),
        img_w=dcfg.get("img_size", 256),
        forecast_mode=True,
        weather_channels=Cw,
        weather_emb_dim=fcfg.get("weather_emb_dim", 64),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    print(f"Loaded: {args.checkpoint} | Val: {len(val_ds)} | H={horizon}")

    # Aggregate per-horizon metrics
    per_h_sums = {h: {"iou": 0, "f1": 0, "precision": 0, "recall": 0} for h in range(horizon)}
    n = 0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Forecast eval"):
            af_hist = batch["af_hist"].to(device)
            w_fut = batch["w_fut"].to(device)
            af_fut = batch["af_fut"].to(device)

            logits = model(af_hist, w_fut=w_fut)
            per_h = compute_metrics_per_horizon(logits, af_fut)
            for h in per_h:
                for k in per_h_sums[h]:
                    per_h_sums[h][k] += per_h[h][k]
            n += 1

    # Average
    for h in per_h_sums:
        for k in per_h_sums[h]:
            per_h_sums[h][k] /= max(n, 1)

    # Print
    print(f"\n{'='*50}\n  Per-Horizon Metrics (H={horizon})\n{'='*50}")
    print(metrics_summary_str(per_h_sums))

    # Aggregate
    agg = {k: sum(per_h_sums[h][k] for h in per_h_sums) / horizon for k in ["iou", "f1", "precision", "recall"]}
    print(f"\n  Aggregate:  IoU={agg['iou']:.4f}  F1={agg['f1']:.4f}")

    # Save CSV
    csv_path = out / "metrics_by_horizon.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "iou", "f1", "precision", "recall"])
        writer.writeheader()
        for h in sorted(per_h_sums.keys()):
            row = {"horizon": h}
            row.update(per_h_sums[h])
            writer.writerow(row)
    print(f"  Saved: {csv_path}\n")


if __name__ == "__main__":
    main()
