#!/usr/bin/env python
"""
finetune_weather.py — Fine-tune FireST-GraphNet with real weather data.

Strategy
--------
  The original model (run20) was trained using only FirePred-derived env_fut.
  It has three source adapters in its decoder:
    "firepred"     — trained (good weights)
    "hrrr"         — untrained (random weights)
    "weathernext2" — untrained (random weights)

  To enable real-weather inference we:
    1. Load run20 best checkpoint
    2. Copy firepred adapter → hrrr and weathernext2 adapters (warm start)
    3. Freeze the encoder (CNN + GNN + LSTM) — preserve its strong features
    4. Fine-tune the decoder with a multi-source collator that randomly
       replaces env_fut with HRRR or Open-Meteo/WN2 DOY climatology data
    5. Save best val-IoU model to outputs/output-clean/run23/

Data sources
------------
  firepred  : original training data, extracted from firepred_*.npy
  hrrr      : HRRR DOY climatology  (hrrr_cache/climatology/doy_NNN.npy)
  weathernext2 : Open-Meteo ERA5 DOY climatology (cache/weathernext2/climatology/)

Run
---
  python finetune_weather.py [options]
  python finetune_weather.py --epochs 30 --lr 5e-5 --source-mix 0.34,0.33,0.33
"""
from __future__ import annotations

import os, sys, csv, math, argparse, random, logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
import torch.amp
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider import build_static_provider
from firest_graphnet.data.future_providers import (
    HRRRFutureProvider, WeatherNext2FutureProvider, PlaceholderFutureProvider,
    DYNAMIC_FUTURE_CHANNELS,
)
from firest_graphnet.data.future_providers import extract_wind
from firest_graphnet.models.firest_graphnet import FireSTGraphNet
from firest_graphnet.training.losses import ForecastPINNLoss

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune with HRRR + WeatherNext2")
    p.add_argument("--base-run",    default="run20",  help="Checkpoint to start from")
    p.add_argument("--out-run",     default="run23",  help="Output run name")
    p.add_argument("--runs-dir",    default=str(PROJECT_ROOT / "outputs" / "output-clean"))
    p.add_argument("--data-root",   default=str(PROJECT_ROOT))
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=3e-5,
                   help="Decoder learning rate (adapters use 3× this)")
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--device",      default="auto")
    p.add_argument("--num-workers", type=int,   default=0)
    p.add_argument(
        "--source-mix",
        default="0.40,0.30,0.30",
        help="Probabilities for firepred,hrrr,weathernext2 (must sum to 1.0)",
    )
    p.add_argument("--freeze-encoder", action="store_true", default=True,
                   help="Freeze encoder (CNN+GNN+LSTM) — only train decoder")
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--pos-weight",      type=float, default=4.0)
    p.add_argument("--focal-gamma",     type=float, default=2.5)
    p.add_argument("--grad-clip",       type=float, default=0.5)
    p.add_argument("--mixed-precision", action="store_true", default=True)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-source dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────
class MultiSourceDataset(Dataset):
    """Wraps TSSatFireForecastDataset; randomly overrides env_fut per sample.

    For each __getitem__ call, picks a source with probability:
        firepred     : source_probs[0]  — use dataset's own env_fut unchanged
        hrrr         : source_probs[1]  — replace with HRRR DOY climatology
        weathernext2 : source_probs[2]  — replace with WN2/ERA5 DOY climatology

    All sources produce 4-channel canonical [temp, u_wind, v_wind, rh], so the
    label supervision is identical — only the weather conditioning changes.
    """

    def __init__(
        self,
        base_ds: TSSatFireForecastDataset,
        hrrr_prov: Optional[HRRRFutureProvider],
        wn2_prov: Optional[WeatherNext2FutureProvider],
        source_probs: Tuple[float, float, float] = (0.40, 0.30, 0.30),
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        self.ds = base_ds
        self.hrrr = hrrr_prov
        self.wn2  = wn2_prov
        self.probs = list(source_probs)
        self.sources = ["firepred", "hrrr", "weathernext2"]
        self.spatial_size = spatial_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        horizon = int(sample["env_fut"].shape[0])
        indices = tuple(range(idx, idx + horizon))

        # Pick source for this sample
        src = random.choices(self.sources, weights=self.probs, k=1)[0]

        if src == "firepred":
            # Use dataset env_fut as-is
            sample["env_source"] = "firepred"
            return sample

        if src == "hrrr" and self.hrrr is not None:
            env_np = self.hrrr.get_future_env(indices, horizon=horizon)
            sample["env_fut"] = torch.from_numpy(env_np)
            sample["env_source"] = "hrrr"
            return sample

        if src == "weathernext2" and self.wn2 is not None:
            env_np = self.wn2.get_future_env(indices, horizon=horizon)
            sample["env_fut"] = torch.from_numpy(env_np)
            sample["env_source"] = "weathernext2"
            return sample

        # Fallback: use firepred
        sample["env_source"] = "firepred"
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# Collate function (handles env_source as list of strings)
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict]) -> Dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            out[k] = vals   # keep as list of strings
        elif isinstance(vals[0], dict):
            out[k] = vals   # meta dicts
        else:
            out[k] = vals
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build model from cfg
# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg: Dict, device: torch.device) -> FireSTGraphNet:
    mc = cfg["model"]
    dc = cfg["data"]
    fc = cfg.get("forecast", {})
    model = FireSTGraphNet(
        in_channels=8 + 19 + 1,
        encoder_channels=mc.get("encoder_channels", [32, 64, 128]),
        use_cbam=mc.get("use_cbam", True),
        patch_size=mc.get("patch_size", 8),
        k_neighbors=mc.get("k_neighbors", 8),
        gnn_hidden=mc.get("gnn_hidden", 128),
        gnn_layers=mc.get("gnn_layers", 2),
        lstm_hidden=mc.get("lstm_hidden", 256),
        lstm_layers=mc.get("lstm_layers", 2),
        lstm_dropout=mc.get("lstm_dropout", 0.1),
        decoder_channels=mc.get("decoder_channels", [128, 64, 32]),
        out_channels=1,
        img_h=int(dc.get("img_size", 256)),
        img_w=int(dc.get("img_size", 256)),
        forecast_mode=True,
        future_channels=int(fc.get("future_channels", 4)),
        future_emb_dim=int(fc.get("future_emb_dim", fc.get("weather_emb_dim", 64))),
    ).to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset with force-reattach for provider N-mismatch
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(data_dir: Path, cfg: Dict) -> TSSatFireForecastDataset:
    fc = cfg.get("forecast", {})
    dc = cfg.get("data", {})
    img_size = int(dc.get("img_size", 256))
    fp = build_firepred_provider(str(data_dir), spatial_size=(img_size, img_size))
    st = build_static_provider(str(data_dir), spatial_size=(img_size, img_size))
    ds = TSSatFireForecastDataset(
        data_dir=data_dir,
        T_in=int(fc.get("T_in", 10)),
        horizon=int(fc.get("horizon", 20)),
        label_channel=int(dc.get("label_channel", 1)),
        firepred_provider=fp,
        static_provider=st,
    )
    # Force-reattach past N-mismatch guard
    if ds.firepred_provider is None and getattr(fp, "_N", 0) > 0:
        ds.firepred_provider = fp
    if ds.static_provider is None and getattr(st, "_N", 0) > 0:
        ds.static_provider = st
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iou_f1(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Dict:
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    gt   = (targets >= 0.5).float()
    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    iou  = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning trainer
# ─────────────────────────────────────────────────────────────────────────────
class WeatherFineTuner:
    def __init__(
        self,
        model: FireSTGraphNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        device: torch.device,
        epochs: int = 30,
        lr: float = 3e-5,
        pos_weight: float = 4.0,
        focal_gamma: float = 2.5,
        grad_clip: float = 0.5,
        mixed_precision: bool = True,
        patience: int = 10,
        freeze_encoder: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience

        # ── Enable fast matmul ──
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # ── AMP ──
        self.mixed_precision = mixed_precision
        self.amp_dtype = torch.float32
        if mixed_precision and device.type == "cuda":
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(mixed_precision and self.amp_dtype == torch.float16)
        )

        # ── Loss ──
        self.criterion = ForecastPINNLoss(
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
            boundary_weight=0.10,
            temporal_weight=0.08,
            horizon_decay=0.08,
            dice_weight=0.15,
            gamma=0.25,
            pde_D=0.01,
            pde_mu=0.12,
            beta_max=0.005,   # light PINN during fine-tune
            eta_max=0.08,
            xi_max=0.04,
            warmup_epochs=5,
            start_epoch=3,
        ).to(device)

        # ── Optimizer with parameter groups ──
        # Adapters (hrrr + weathernext2) get 3× LR — they start from scratch
        # Decoder backbone gets base LR
        # Encoder stays frozen
        if freeze_encoder:
            self._freeze_encoder()

        adapter_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "adapters.hrrr" in name or "adapters.weathernext2" in name:
                adapter_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": adapter_params, "lr": lr * 3.0,  "name": "adapters"},
            {"params": decoder_params, "lr": lr,         "name": "decoder"},
        ], weight_decay=5e-3)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        n_adapt = sum(p.numel() for p in adapter_params)
        n_dec   = sum(p.numel() for p in decoder_params)
        n_total = sum(p.numel() for p in model.parameters())
        n_frozen = n_total - n_adapt - n_dec
        print(f"\n  Trainable params: adapters={n_adapt:,}  decoder={n_dec:,}")
        print(f"  Frozen params:    {n_frozen:,} / {n_total:,}")
        print(f"  Adapter LR: {lr*3:.2e}  |  Decoder LR: {lr:.2e}")

        self.best_val_iou = -1.0
        self.patience_counter = 0
        self.history: List[Dict] = []

    def _freeze_encoder(self):
        """Freeze CNN encoder, GNN, and LSTM layers."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["encoder", "gnn", "lstm", "patch", "cbam"]):
                param.requires_grad_(False)
                frozen += param.numel()
        print(f"  Froze encoder: {frozen:,} params")

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> Dict:
        self.model.train(train)
        if train:
            self.criterion.set_epoch(epoch)

        totals = defaultdict(float)
        met_acc = defaultdict(float)
        n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                af_hist    = batch["af_hist"].to(self.device, non_blocking=True)
                aux_hist   = batch["aux_hist"].to(self.device, non_blocking=True)
                env_fut    = batch["env_fut"].to(self.device, non_blocking=True)
                af_fut     = batch["af_fut"].to(self.device, non_blocking=True)
                ba_init    = batch["ba_init"].to(self.device, non_blocking=True)
                static     = batch["static"].to(self.device, non_blocking=True)
                env_source = batch.get("env_source", ["firepred"])[0]

                with torch.amp.autocast("cuda", enabled=self.mixed_precision, dtype=self.amp_dtype):
                    logits = self.model(
                        af_hist,
                        aux_hist=aux_hist,
                        env_fut=env_fut,
                        env_source=env_source,
                        static=static,
                    )
                    wind = extract_wind(env_fut, source=env_source)
                    losses = self.criterion(logits, af_fut, ba_init=ba_init, wind=wind)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(losses["total"]).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.grad_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                for k, v in losses.items():
                    totals[k] += v.item()
                m = compute_iou_f1(logits.detach(), af_fut.detach())
                for k, v in m.items():
                    met_acc[k] += v
                n += 1

        N = max(n, 1)
        out = {k: v / N for k, v in totals.items()}
        out.update({k: v / N for k, v in met_acc.items()})
        out["loss"] = out.get("total", 0.0)
        return out

    def fit(self) -> Path:
        print(f"\n{'='*60}")
        print(f"  Weather Fine-Tuning — {self.epochs} epochs")
        print(f"  Device: {self.device}  |  AMP: {self.amp_dtype}")
        print(f"  Train: {len(self.train_loader)} batches  |  Val: {len(self.val_loader)} batches")
        print(f"{'='*60}\n")

        csv_path = self.output_dir / "finetune_metrics.csv"
        header_written = False

        for epoch in range(1, self.epochs + 1):
            train_m = self._run_epoch(self.train_loader, train=True,  epoch=epoch)
            val_m   = self._run_epoch(self.val_loader,   train=False, epoch=epoch)
            self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "lr_adapter": lr_now * 3, "lr_decoder": lr_now}
            for k, v in train_m.items():
                row[f"train_{k}"] = round(v, 6)
            for k, v in val_m.items():
                row[f"val_{k}"] = round(v, 6)

            # Save CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not header_written:
                    w.writeheader()
                    header_written = True
                w.writerow(row)

            # Print
            print(
                f"  Ep {epoch:3d}/{self.epochs}"
                f"  loss={train_m['loss']:.4f}"
                f"  train_iou={train_m['iou']:.4f}"
                f"  val_iou={val_m['iou']:.4f}"
                f"  val_f1={val_m['f1']:.4f}"
                f"  lr={lr_now:.2e}"
            )

            # Save best
            if val_m["iou"] > self.best_val_iou:
                self.best_val_iou = val_m["iou"]
                self.patience_counter = 0
                best_path = self.output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_iou": self.best_val_iou,
                    "config": self._cfg,
                    "finetune_note": "multi-source: firepred+hrrr+weathernext2",
                }, best_path)
                print(f"    *** New best val IoU: {self.best_val_iou:.4f}  -> {best_path}")
            else:
                self.patience_counter += 1
                if self.patience > 0 and self.patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        self.history = []
        print(f"\n  Best val IoU: {self.best_val_iou:.4f}")
        return self.output_dir / "best_model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helper
# ─────────────────────────────────────────────────────────────────────────────
_FIRE_CMAP = ListedColormap([(0,0,0,0), (1.0,0.12,0.0,1.0)])

def plot_finetune_curves(csv_path: Path):
    import csv as csvmod
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csvmod.DictReader(f))
    if not rows:
        return
    epochs  = [int(r["epoch"])           for r in rows]
    t_loss  = [float(r.get("train_loss", r.get("train_total", 0))) for r in rows]
    t_iou   = [float(r.get("train_iou",  0)) for r in rows]
    v_iou   = [float(r.get("val_iou",    0)) for r in rows]
    v_f1    = [float(r.get("val_f1",     0)) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, t_loss, label="train loss")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, t_iou, label="train IoU")
    axes[1].plot(epochs, v_iou, label="val IoU")
    axes[1].plot(epochs, v_f1,  label="val F1", linestyle="--")
    axes[1].set_title("IoU / F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    out = csv_path.parent / "finetune_curves.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Training curves -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    runs_dir  = Path(args.runs_dir)
    data_root = Path(args.data_root)
    base_dir  = runs_dir / args.base_run
    out_dir   = runs_dir / args.out_run
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse source mix ──
    probs = [float(x) for x in args.source_mix.split(",")]
    assert len(probs) == 3, "--source-mix must have 3 values: firepred,hrrr,wn2"
    total = sum(probs)
    probs = [p / total for p in probs]  # normalize
    print(f"\nSource mix: firepred={probs[0]:.2f}  hrrr={probs[1]:.2f}  wn2={probs[2]:.2f}")

    # ── Load base checkpoint + config ──
    print(f"\nLoading base checkpoint from {base_dir}")
    cfg_path = base_dir / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(str(base_dir / "best_model.pt"), map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    if isinstance(ckpt.get("config"), dict):
        cfg = ckpt["config"]
    print(f"  Base config: horizon={cfg.get('forecast',{}).get('horizon','?')}  "
          f"epoch={ckpt.get('epoch','?')}  best_val_iou={ckpt.get('best_val_iou','?')}")

    # ── Build and load model ──
    model = build_model(cfg, device)
    incompat = model.load_state_dict(sd, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print(f"  State dict: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} total params")

    # ── Warm-start: copy firepred adapter → hrrr and weathernext2 ──
    print("\nWarm-starting HRRR and WN2 adapters from firepred adapter weights ...")
    with torch.no_grad():
        fp_w = model.decoder.future_enc.adapters["firepred"].weight.data.clone()
        fp_b = model.decoder.future_enc.adapters["firepred"].bias.data.clone()
        model.decoder.future_enc.adapters["hrrr"].weight.data.copy_(fp_w)
        model.decoder.future_enc.adapters["hrrr"].bias.data.copy_(fp_b)
        model.decoder.future_enc.adapters["weathernext2"].weight.data.copy_(fp_w)
        model.decoder.future_enc.adapters["weathernext2"].bias.data.copy_(fp_b)
    print("  Done (all three adapters now share firepred weights as starting point)")

    # ── Save updated config for run23 ──
    out_cfg = dict(cfg)
    out_cfg["output_dir"] = str(out_dir)
    out_cfg["finetune"] = {
        "base_run": args.base_run,
        "source_mix": {"firepred": probs[0], "hrrr": probs[1], "weathernext2": probs[2]},
        "freeze_encoder": args.freeze_encoder,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(out_cfg, f, default_flow_style=False)

    img_size = int(cfg.get("data", {}).get("img_size", 256))

    # ── Build weather providers ──
    print("\nBuilding weather providers ...")
    hrrr_prov = None
    if probs[1] > 0:
        clim_dir = data_root / "hrrr_cache" / "climatology"
        try:
            hrrr_prov = HRRRFutureProvider(clim_dir, spatial_size=(img_size, img_size))
            print(f"  HRRR: {hrrr_prov._D} DOY frames")
        except FileNotFoundError:
            print("  WARNING: HRRR climatology not found — treating as zeros")
            hrrr_prov = PlaceholderFutureProvider(source_name="hrrr", spatial_size=(img_size, img_size))

    wn2_prov = None
    if probs[2] > 0:
        wn2_prov = WeatherNext2FutureProvider(
            cache_dir=str(data_root / "cache" / "weathernext2"),
            spatial_size=(img_size, img_size),
        )
        n_wn2 = len(wn2_prov._downloader._doy_data) if wn2_prov._downloader else 0
        if n_wn2 == 0:
            print("  WARNING: WeatherNext2 DOY climatology not built — run build-climatology first")
            print("           Falling back to zeros for WN2 samples")
        else:
            print(f"  WeatherNext2 (Open-Meteo ERA5): {n_wn2} DOY frames")

    # ── Build datasets ──
    print("\nBuilding datasets ...")
    train_dir = data_root / "dataset_train"
    val_dir   = data_root / "dataset_val"

    base_train = build_dataset(train_dir, cfg)
    base_val   = build_dataset(val_dir,   cfg)

    ds_train = MultiSourceDataset(base_train, hrrr_prov, wn2_prov, tuple(probs), (img_size, img_size))
    ds_val   = MultiSourceDataset(base_val,   hrrr_prov, wn2_prov,
                                  (1.0, 0.0, 0.0),  # val always uses firepred for fair comparison
                                  (img_size, img_size))

    print(f"  Train: {len(ds_train)} events  |  Val: {len(ds_val)} events")

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    # ── Fine-tune ──
    trainer = WeatherFineTuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        patience=args.patience,
        freeze_encoder=args.freeze_encoder,
    )
    trainer._cfg = out_cfg   # store for checkpoint saving

    best_path = trainer.fit()

    # ── Plot training curves ──
    csv_path = out_dir / "finetune_metrics.csv"
    if csv_path.exists():
        try:
            plot_finetune_curves(csv_path)
        except Exception as e:
            print(f"  (curve plot skipped: {e})")

    print(f"\nFine-tuning complete.")
    print(f"  Best model: {best_path}")
    print(f"  Best val IoU: {trainer.best_val_iou:.4f}")
    print(f"\nTo evaluate run23:")
    print(f"  python batch_eval_all_runs.py --runs 23 --env-source firepred --max-train 0")
    print(f"  python batch_eval_all_runs.py --runs 23 --env-source hrrr --max-train 0")
    print(f"  python batch_eval_all_runs.py --runs 23 --env-source mixed --max-train 0")


if __name__ == "__main__":
    main()
