#!/usr/bin/env python
"""
train.py — Training entry-point for FireSTGraphNet.

Usage (legacy single-step):
    python -m firest_graphnet.scripts.train --data_root .

Usage (forecast multi-horizon + PINN):
    python -m firest_graphnet.scripts.train --data_root . --mode forecast --horizon 20
"""

from __future__ import annotations

import argparse
import sys
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── ensure project root on sys.path ───────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from firest_graphnet.models.firest_graphnet import FireSTGraphNet
from firest_graphnet.utils.io import load_yaml, save_yaml, ensure_dir
from firest_graphnet.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train FireSTGraphNet")
    p.add_argument("--data_root", type=str,
                    default=r"C:\Users\22125\Desktop\FireST-GraphNet")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, default="legacy",
                    choices=["legacy", "forecast"],
                    help="Training mode: legacy (single-step) or forecast (multi-horizon+PINN)")
    p.add_argument("--horizon", type=int, default=None,
                    help="Forecast horizon (overrides config)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--T_in", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--weather_dir", type=str, default=None,
                    help="Path to HRRR DOY climatology dir")
    p.add_argument("--weather_channels", type=int, default=4)
    p.add_argument("--default_doy", type=int, default=200)
    return p.parse_args()


def _find_config():
    for candidate in [
        Path(__file__).resolve().parent.parent / "configs" / "base.yaml",
        _project_root / "firest_graphnet" / "configs" / "base.yaml",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def main():
    args = parse_args()

    # ── config ─────────────────────────────────────────────────────────────
    cfg_path = args.config or _find_config()
    if cfg_path is None or not Path(cfg_path).exists():
        print("[ERROR] Config not found. Use --config.")
        sys.exit(1)
    cfg = load_yaml(cfg_path)

    # ── CLI overrides ──────────────────────────────────────────────────────
    cfg["data"]["root"] = args.data_root
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.patch_size is not None:
        cfg["model"]["patch_size"] = args.patch_size
    if args.k is not None:
        cfg["model"]["k_neighbors"] = args.k
    if args.T_in is not None:
        cfg["data"]["T_in"] = args.T_in
    if args.device is not None:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    # Weather overrides (legacy mode)
    weather_dir = args.weather_dir
    weather_channels = args.weather_channels
    default_doy = args.default_doy
    if weather_dir:
        cfg.setdefault("weather", {})
        cfg["weather"]["climatology_dir"] = weather_dir
        cfg["weather"]["channels"] = weather_channels
        cfg["weather"]["default_doy"] = default_doy
    else:
        w = cfg.get("weather", {})
        weather_dir = w.get("climatology_dir")
        weather_channels = w.get("channels", 4)
        default_doy = w.get("default_doy", 200)

    # Forecast overrides
    if args.horizon is not None:
        cfg.setdefault("forecast", {})
        cfg["forecast"]["horizon"] = args.horizon

    # ── seed + device ──────────────────────────────────────────────────────
    seed_everything(cfg["seed"])
    if platform.system() == "Windows" and cfg["data"].get("num_workers", 0) > 0:
        print("[Windows] Auto-setting num_workers=0")
        cfg["data"]["num_workers"] = 0

    dev_str = cfg.get("device", "auto")
    device = torch.device(
        "cuda" if (dev_str == "auto" and torch.cuda.is_available()) else
        dev_str if dev_str != "auto" else "cpu"
    )
    print(f"Device: {device}")

    # ══════════════════════════════════════════════════════════════════════
    #  FORECAST MODE
    # ══════════════════════════════════════════════════════════════════════
    if args.mode == "forecast":
        _train_forecast(cfg, device, weather_dir)
    else:
        _train_legacy(cfg, device, weather_dir, weather_channels, default_doy)


def _train_forecast(cfg, device, weather_dir):
    """Multi-horizon forecast training with PINN."""
    from firest_graphnet.data.dataset_forecast import build_forecast_datasets
    from firest_graphnet.data.firepred_provider import build_firepred_provider
    from firest_graphnet.data.static_provider import build_static_provider
    from firest_graphnet.training.trainer import ForecastTrainer

    dcfg = cfg["data"]
    fcfg = cfg.get("forecast", {})
    T_in = fcfg.get("T_in", 10)
    horizon = fcfg.get("horizon", 20)
    Cw = fcfg.get("future_channels", fcfg.get("weather_channels", 4))
    in_ch = fcfg.get("in_channels", 1)
    img_size = int(dcfg.get("img_size", 256))

    # Build historical auxiliary and static providers from preprocessed npy.
    firepred_provider = build_firepred_provider(
        data_root=dcfg["root"],
        spatial_size=(img_size, img_size),
    )
    static_provider = build_static_provider(
        data_root=dcfg["root"],
        spatial_size=(img_size, img_size),
    )
    print(
        f"Aux provider: {type(firepred_provider).__name__} "
        f"(C_aux={firepred_provider.channels}) | "
        f"Static provider: {type(static_provider).__name__} "
        f"(C_static={static_provider.channels})"
    )

    train_ds, val_ds = build_forecast_datasets(
        data_root=dcfg["root"],
        T_in=T_in,
        horizon=horizon,
        label_channel=dcfg.get("label_channel", 1),
        val_split=dcfg.get("val_split", 0.1),
        seed=cfg["seed"],
        firepred_provider=firepred_provider,
        static_provider=static_provider,
    )
    print(f"Forecast: T_in={T_in} H={horizon} | Train={len(train_ds)} Val={len(val_ds)}")

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", False), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", False),
    )

    # Model in forecast mode
    mcfg = cfg["model"]
    model = FireSTGraphNet(
        in_channels=in_ch,
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
        future_channels=Cw,
        future_emb_dim=fcfg.get("future_emb_dim", fcfg.get("weather_emb_dim", 64)),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    out_dir = ensure_dir(cfg.get("output_dir", "outputs_forecast"))
    save_yaml(cfg, out_dir / "config.yaml")

    trainer = ForecastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=out_dir,
    )
    trainer.fit()


def _train_legacy(cfg, device, weather_dir, weather_channels, default_doy):
    """Legacy single-step training."""
    from firest_graphnet.data.dataset import build_datasets
    from firest_graphnet.data.transforms import get_train_transforms, get_val_transforms
    from firest_graphnet.training.trainer import Trainer

    dcfg = cfg["data"]
    train_ds, val_ds = build_datasets(
        data_root=dcfg["root"],
        T_in=dcfg["T_in"],
        T_out=dcfg.get("T_out", 1),
        label_channel=dcfg.get("label_channel", 1),
        val_split=dcfg.get("val_split", 0.1),
        seed=cfg["seed"],
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        weather_dir=weather_dir,
        weather_channels=weather_channels,
        default_doy=default_doy,
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", False), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", False),
    )

    mcfg = cfg["model"]
    base_bands = dcfg.get("num_bands", 8)
    actual_in_channels = mcfg.get("in_channels", base_bands)
    if weather_dir and Path(weather_dir).is_dir() and any(Path(weather_dir).glob("doy_*.npy")):
        actual_in_channels = base_bands + weather_channels
        print(f"in_channels: {actual_in_channels} (bands + weather)")

    model = FireSTGraphNet(
        in_channels=actual_in_channels,
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    out_dir = ensure_dir(cfg.get("output_dir", "outputs"))
    save_yaml(cfg, out_dir / "config.yaml")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, device=device, output_dir=out_dir,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
