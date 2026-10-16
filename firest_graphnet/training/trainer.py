"""
trainer.py — Training loops for FireSTGraphNet.

Trainers:
    - Trainer            : single-step (legacy)
    - ForecastTrainer    : multi-horizon with PINN physics constraints

Both handle AMP, cosine LR, early stopping, checkpoint saving, metric logging.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from firest_graphnet.training.losses import FireProgressionLoss, ForecastPINNLoss
from firest_graphnet.training.metrics import (
    compute_metrics,
    compute_metrics_per_horizon,
    compute_nmae,
    compute_nmae_per_horizon,
    metrics_summary_str,
)
from firest_graphnet.training.normalization import DEFAULT_PROB_STATS, GlobalProbStats
from firest_graphnet.utils.logging import MetricLogger, print_metrics
from firest_graphnet.data.future_providers import extract_wind


def _save_vis(
    logits: torch.Tensor,
    targets: torch.Tensor,
    save_dir: Path,
    epoch: int,
    max_samples: int = 4,
):
    """Save side-by-side pred vs gt images as .npy (and optionally .png)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    gt = targets.detach().cpu().numpy()
    n = min(max_samples, probs.shape[0])
    for i in range(n):
        pred_i = probs[i, 0]
        gt_i = gt[i, 0]
        combined = np.concatenate([pred_i, gt_i], axis=1)
        np.save(str(save_dir / f"epoch{epoch:03d}_sample{i}.npy"), combined)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(pred_i, cmap="hot", vmin=0, vmax=1)
            axes[0].set_title("Prediction")
            axes[0].axis("off")
            axes[1].imshow(gt_i, cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(str(save_dir / f"epoch{epoch:03d}_sample{i}.png"), dpi=100)
            plt.close(fig)
        except ImportError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy Trainer  (single-step)
# ═══════════════════════════════════════════════════════════════════════════
class Trainer:
    """Single-step training manager (legacy)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: torch.device,
        output_dir: str | Path = "outputs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg["training"]
        self.epochs = tcfg["epochs"]
        self.grad_clip = tcfg.get("grad_clip", 0.0)
        self.mixed_precision = tcfg.get("mixed_precision", False)
        self.save_every = tcfg.get("save_every", 5)
        self.patience = tcfg.get("patience", 0)

        self.criterion = FireProgressionLoss(pos_weight=tcfg.get("pos_weight", 10.0))
        self.criterion.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg.get("weight_decay", 1e-4),
        )

        sched_type = tcfg.get("scheduler", "cosine")
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif sched_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=tcfg.get("step_size", 15),
                gamma=tcfg.get("gamma", 0.5),
            )
        else:
            self.scheduler = None

        self.scaler = GradScaler("cuda", enabled=self.mixed_precision)
        self.logger = MetricLogger(self.output_dir / "metrics.csv")
        self.best_val_iou = -1.0
        self.patience_counter = 0

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_metrics: Dict[str, float] = {"iou": 0, "f1": 0, "precision": 0, "recall": 0}
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc="train" if train else "val  ", leave=False):
                x_seq = batch["x_seq"].to(self.device)
                y = batch["y"].to(self.device)

                with autocast("cuda", enabled=self.mixed_precision):
                    logits = self.model(x_seq)
                    loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                m = compute_metrics(logits.detach(), y.detach())
                for k in total_metrics:
                    total_metrics[k] += m[k]
                n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
        avg["loss"] = total_loss / max(n_batches, 1)
        return avg

    def fit(self):
        print(f"\n{'='*60}")
        print(f"  FireSTGraphNet Training — {self.epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}  |  Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            train_m = self._run_epoch(self.train_loader, train=True)
            if self.scheduler is not None:
                self.scheduler.step()
            val_m = self._run_epoch(self.val_loader, train=False)

            row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            for k, v in train_m.items():
                row[f"train_{k}"] = v
            for k, v in val_m.items():
                row[f"val_{k}"] = v
            self.logger.log(row)
            print_metrics(epoch, row)

            if val_m["iou"] > self.best_val_iou:
                self.best_val_iou = val_m["iou"]
                self.patience_counter = 0
                ckpt_path = self.output_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_iou": self.best_val_iou,
                    },
                    str(ckpt_path),
                )
                print(f"  ★ New best val IoU: {self.best_val_iou:.4f}  →  saved {ckpt_path.name}")
            else:
                self.patience_counter += 1

            if epoch % self.save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    str(self.output_dir / f"checkpoint_epoch{epoch:03d}.pt"),
                )

            if self.patience > 0 and self.patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={self.patience})")
                break

        torch.save(self.model.state_dict(), str(self.output_dir / "final_model.pt"))
        print(f"\n  Training complete.  Best val IoU: {self.best_val_iou:.4f}")
        print(f"  Outputs saved to: {self.output_dir}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Forecast Trainer  (multi-horizon + PINN)
# ═══════════════════════════════════════════════════════════════════════════
class ForecastTrainer:
    """Multi-horizon training with PINN physics constraints.

    Parameters
    ----------
    model        : FireSTGraphNet (forecast_mode=True)
    train_loader : DataLoader yielding forecast-format batches
    val_loader   : DataLoader yielding forecast-format batches
    cfg          : full config dict
    device       : torch.device
    output_dir   : save directory
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: torch.device,
        output_dir: str | Path = "outputs_forecast",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg["training"]
        pcfg = cfg.get("pinn", {})

        self.epochs = tcfg["epochs"]
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.mixed_precision = tcfg.get("mixed_precision", False)
        self.save_every = tcfg.get("save_every", 5)
        self.patience = tcfg.get("patience", 0)
        self.val_every = tcfg.get("val_every", 1)          # validate every N epochs
        self.metric_every = tcfg.get("metric_every", 10)   # compute metrics every N batches

        # ── Enable TF32 for A100/H100 matmul & convolution ────────────────
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Auto-select AMP dtype: bfloat16 on Ampere+, float16 otherwise
        self.amp_dtype = torch.float32
        if self.mixed_precision and device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16

        # Forecast PINN loss
        self.criterion = ForecastPINNLoss(
            pos_weight=tcfg.get("pos_weight", 10.0),
            focal_gamma=tcfg.get("focal_gamma", 0.0),
            boundary_weight=tcfg.get("boundary_weight", 0.0),
            temporal_weight=tcfg.get("temporal_weight", 0.0),
            horizon_decay=tcfg.get("horizon_decay", 0.0),
            dice_weight=tcfg.get("dice_weight", 0.0),
            gamma=pcfg.get("gamma", 0.3),
            pde_D=pcfg.get("pde_D", 0.01),
            pde_mu=pcfg.get("pde_mu", 0.1),
            beta_max=pcfg.get("beta_max", 0.01),
            eta_max=pcfg.get("eta_max", 0.1),
            xi_max=pcfg.get("xi_max", 0.05),
            warmup_epochs=pcfg.get("warmup_epochs", 10),
            start_epoch=pcfg.get("start_epoch", 5),
        )
        self.criterion.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg.get("weight_decay", 1e-4),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        self.scaler = GradScaler(
            "cuda",
            enabled=self.mixed_precision and self.amp_dtype == torch.float16,
        )
        self.logger = MetricLogger(self.output_dir / "metrics.csv")
        self.best_val_iou = -1.0
        self.patience_counter = 0

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int = 0) -> Dict[str, float]:
        if train:
            self.model.train()
            self.criterion.set_epoch(epoch)
        else:
            self.model.eval()

        totals = {"total": 0, "sup": 0, "dice": 0, "boundary": 0, "temporal": 0, "pde": 0, "mono": 0, "mask": 0}
        met_totals = {"iou": 0, "f1": 0, "precision": 0, "recall": 0, "nmae": 0.0}
        pos_ratio_pred = 0.0
        pos_ratio_gt = 0.0
        horizon_iou_acc: Dict[int, float] = defaultdict(float)
        horizon_nmae_acc: Dict[int, float] = defaultdict(float)
        # Pred-task normalization: training-set global stats (may be overridden
        # via cfg['pred']['stats_path']); defaults to canonical [0,1] range.
        stats_path = self.cfg.get("pred", {}).get("stats_path")
        if stats_path and Path(stats_path).exists():
            self._pred_stats = GlobalProbStats.load(stats_path)
        else:
            self._pred_stats = DEFAULT_PROB_STATS
        n = 0
        n_met = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(tqdm(loader, desc="train" if train else "val  ", leave=False)):
                af_hist    = batch["af_hist"].to(self.device, non_blocking=True)
                aux_hist   = batch["aux_hist"].to(self.device, non_blocking=True)
                env_fut    = batch["env_fut"].to(self.device, non_blocking=True)
                af_fut     = batch["af_fut"].to(self.device, non_blocking=True)
                ba_init    = batch["ba_init"].to(self.device, non_blocking=True)
                static     = batch["static"].to(self.device, non_blocking=True)
                env_source = batch.get("env_source", ["firepred"])[0]  # str

                with autocast("cuda", enabled=self.mixed_precision, dtype=self.amp_dtype):
                    logits = self.model(
                        af_hist,
                        aux_hist=aux_hist,
                        env_fut=env_fut,
                        env_source=env_source,
                        static=static,
                    )  # (B, H, 1, H, W)

                    # Extract wind (u, v) for PDE convection term
                    # env_fut is 4ch canonical: [temp, u, v, rh]
                    # extract_wind is source-agnostic (always channels 1,2)
                    wind = extract_wind(env_fut, source=env_source)  # (B, H, 2, Hp, Wp)
                    losses = self.criterion(
                        logits, af_fut, ba_init=ba_init, wind=wind,
                    )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(losses["total"]).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                for k in totals:
                    totals[k] += losses[k].item()
                # Compute metrics periodically to reduce overhead
                if (batch_idx + 1) % self.metric_every == 0 or (batch_idx + 1) == len(loader):
                    m = compute_metrics(logits.detach(), af_fut.detach())
                    # Normalized MAE on the Pred continuous probability field
                    m["nmae"] = compute_nmae(
                        logits.detach(), af_fut.detach(), self._pred_stats
                    )
                    for k in met_totals:
                        met_totals[k] += m[k]
                    # Per-horizon IoU / nMAE (val only)
                    if not train:
                        phm = compute_metrics_per_horizon(logits.detach(), af_fut.detach())
                        for h, hm in phm.items():
                            horizon_iou_acc[h] += hm["iou"]
                        phn = compute_nmae_per_horizon(
                            logits.detach(), af_fut.detach(), self._pred_stats
                        )
                        for h, v in phn.items():
                            horizon_nmae_acc[h] += v
                    n_met += 1
                # Track predicted / GT positive-pixel ratio (diagnostic)
                with torch.no_grad():
                    pos_ratio_pred += torch.sigmoid(logits.detach()).gt(0.5).float().mean().item()
                    pos_ratio_gt   += af_fut.mean().item()
                n += 1

        avg = {}
        for k in totals:
            avg[k] = totals[k] / max(n, 1)
        avg["loss"] = avg["total"]
        for k in met_totals:
            avg[k] = met_totals[k] / max(n_met, 1)
        avg["pred_pos_ratio"] = pos_ratio_pred / max(n, 1)
        avg["gt_pos_ratio"]   = pos_ratio_gt   / max(n, 1)
        # Per-horizon IoU / nMAE (val only — empty dict for train)
        avg["_horizon_iou"] = (
            {h: v / max(n_met, 1) for h, v in horizon_iou_acc.items()}
            if not train else {}
        )
        avg["_horizon_nmae"] = (
            {h: v / max(n_met, 1) for h, v in horizon_nmae_acc.items()}
            if not train else {}
        )
        return avg

    def fit(self):
        print(f"\n{'='*60}")
        print(f"  FireSTGraphNet FORECAST Training — {self.epochs} epochs")
        print(f"  Device: {self.device} | PINN: enabled")
        amp_info = f"AMP: {self.amp_dtype}" if self.mixed_precision else "AMP: off"
        print(f"  {amp_info} | TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  Train: {len(self.train_loader)} batches | Val: {len(self.val_loader)} batches")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            train_m = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            self.scheduler.step()

            # Validate every val_every epochs (always validate on last epoch)
            if epoch % self.val_every == 0 or epoch == self.epochs:
                val_m = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            else:
                val_m = None

            row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            for k, v in train_m.items():
                if k.startswith("_"):
                    continue   # skip internal keys like _horizon_iou
                row[f"train_{k}"] = v
            if val_m is not None:
                for k, v in val_m.items():
                    if k.startswith("_"):
                        continue
                    row[f"val_{k}"] = v
            self.logger.log(row)
            print_metrics(epoch, row)
            # ── Diagnostic: positive-pixel ratio ──────────────────────────
            print(
                f"  Pos-ratio  train: pred={train_m['pred_pos_ratio']:.4f}  "
                f"gt={train_m['gt_pos_ratio']:.4f}"
                + (f"  |  val: pred={val_m['pred_pos_ratio']:.4f}  gt={val_m['gt_pos_ratio']:.4f}" if val_m else "")
            )
            # ── Per-horizon val IoU / nMAE ─────────────────────────────────
            if val_m and val_m.get("_horizon_iou"):
                hiou = val_m["_horizon_iou"]
                idxs = sorted(hiou.keys())
                summary = "  ".join(f"h{i+1}={hiou[i]:.3f}" for i in idxs[:8])
                if len(idxs) > 8:
                    summary += f"  ...h{idxs[-1]+1}={hiou[idxs[-1]]:.3f}"
                print(f"  Horizon IoU : {summary}")
            if val_m and val_m.get("_horizon_nmae"):
                hnm = val_m["_horizon_nmae"]
                idxs = sorted(hnm.keys())
                summary = "  ".join(f"h{i+1}={hnm[i]:.4f}" for i in idxs[:8])
                if len(idxs) > 8:
                    summary += f"  ...h{idxs[-1]+1}={hnm[idxs[-1]]:.4f}"
                print(f"  Horizon nMAE: {summary}  (normalization: {self._pred_stats.source})")

            # Physics loss info
            w = self.criterion.schedule.get_weights(epoch)
            print(f"  PINN weights: β={w['beta']:.4f}  η={w['eta']:.4f}  ξ={w['xi']:.4f}")

            if val_m is not None and val_m["iou"] > self.best_val_iou:
                self.best_val_iou = val_m["iou"]
                self.patience_counter = 0
                ckpt = self.output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_iou": self.best_val_iou,
                    "config": self.cfg,
                }, str(ckpt))
                print(f"  ★ Best val IoU: {self.best_val_iou:.4f}")
            elif val_m is not None:
                self.patience_counter += 1

            if epoch % self.save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    str(self.output_dir / f"checkpoint_epoch{epoch:03d}.pt"),
                )

            if self.patience > 0 and self.patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        torch.save(self.model.state_dict(), str(self.output_dir / "final_model.pt"))
        print(f"\n  Forecast training complete.  Best val IoU: {self.best_val_iou:.4f}")
        print(f"  Outputs: {self.output_dir}\n")
