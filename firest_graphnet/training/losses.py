"""
losses.py — Loss functions for wildfire progression prediction.

Includes:
    - FireProgressionLoss        : single-step BCEWithLogits
    - FocalBCELoss               : Focal loss + pos_weight (Run12+)
    - BoundaryAwareLoss          : boundary-weighted BCE on GT firefront (Run12+)
    - TemporalConsistencyLoss    : L1 penalty on consecutive prediction deltas (Run12+)
    - ForecastPINNLoss           : multi-horizon supervision + PINN constraints
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from firest_graphnet.physics.constraints import PhysicsConstraints
from firest_graphnet.physics.schedules import PINNSchedule


class FireProgressionLoss(nn.Module):
    """BCEWithLogitsLoss with configurable pos_weight for class imbalance."""

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", pw)
        self.loss_fn = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is None or self.pos_weight.device != logits.device:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
            )
        return self.loss_fn(logits, targets)


class FocalBCELoss(nn.Module):
    """Focal BCE: -(1-p_t)^gamma * w_t * log(p_t).

    Down-weights easy negative pixels (background), forces focus on
    hard fire-boundary pixels.  Replaces plain BCE when focal_gamma > 0.

    Parameters
    ----------
    pos_weight : float   per-pixel class-imbalance weight for fire pixels
    gamma      : float   focusing exponent (2.0 recommended)
    """

    def __init__(self, pos_weight: float = 10.0, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight_buf", pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # p_t = probability of the true class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        # Per-pixel class weight: fire=pos_weight, background=1
        pw = self.pos_weight_buf.to(logits.device)
        pixel_weight = targets * pw + (1.0 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_weight * pixel_weight * bce).mean()


class BoundaryAwareLoss(nn.Module):
    """BCE with elevated weight near GT firefront boundaries.

    GT boundary pixels are extracted via morphological dilation − erosion.
    Boundary pixels receive boundary_factor × the base BCE loss, penalising
    the model more heavily for firefront misalignment than interior errors.

    Parameters
    ----------
    boundary_factor : float   loss multiplier at boundary pixels (default 4×)
    kernel_size     : int     morphological kernel size (default 3)
    """

    def __init__(self, boundary_factor: float = 4.0, kernel_size: int = 3):
        super().__init__()
        self.boundary_factor = boundary_factor
        self.ks = kernel_size
        self.pad = kernel_size // 2

    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Morphological boundary: dilation − erosion on binary mask (N,1,H,W)."""
        dilated = F.max_pool2d(mask, self.ks, stride=1, padding=self.pad)
        eroded = -F.max_pool2d(-mask, self.ks, stride=1, padding=self.pad)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Accept (B,H,1,Hp,Wp) and flatten horizon into batch dim
        if logits.dim() == 5:
            B, H, C, Hp, Wp = logits.shape
            logits = logits.reshape(B * H, C, Hp, Wp)
            targets = targets.reshape(B * H, C, Hp, Wp)
        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        boundary = self._extract_boundary(targets)
        weights = 1.0 + boundary * (self.boundary_factor - 1.0)
        return (bce_map * weights).mean()


class TemporalConsistencyLoss(nn.Module):
    """Penalise mismatch between predicted and GT frame-to-frame deltas.

    Encourages the model to capture fire spread direction and rate across
    consecutive steps rather than producing static or incoherent blobs.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits / targets: (B, H, 1, Hp, Wp)
        if logits.shape[1] < 2:
            return torch.tensor(0.0, device=logits.device)
        prob = torch.sigmoid(logits)
        pred_delta = prob[:, 1:] - prob[:, :-1]      # (B, H-1, 1, Hp, Wp)
        gt_delta = targets[:, 1:] - targets[:, :-1]
        return F.l1_loss(pred_delta, gt_delta)


class ForecastPINNLoss(nn.Module):
    """Multi-horizon supervised loss + PINN physics constraints.

    Total = horizon_weighted(sup_loss)
          + boundary_weight  * L_boundary
          + temporal_weight  * L_temporal
          + dice_weight      * L_dice
          + beta * L_pde + eta * L_mono + xi * L_mask

    Parameters
    ----------
    pos_weight      : float  class imbalance weight for fire pixels
    focal_gamma     : float  Focal loss exponent (>0 enables Focal; 0 = plain BCE)
    boundary_weight : float  weight for boundary-aware loss (0 = disabled)
    temporal_weight : float  weight for temporal consistency loss (0 = disabled)
    horizon_decay   : float  exp decay across horizon steps: w_h=exp(-d*h) (0=uniform)
    dice_weight     : float  weight for soft Dice loss (0 = disabled)
    gamma           : float  BA update rate
    pde_D           : float  diffusion coeff
    pde_mu          : float  decay rate
    beta_max        : float  max PDE weight
    eta_max         : float  max monotonicity weight
    xi_max          : float  max AF-BA mask weight
    warmup_epochs   : int    PINN warmup duration
    start_epoch     : int    PINN start epoch
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        focal_gamma: float = 0.0,
        boundary_weight: float = 0.0,
        temporal_weight: float = 0.0,
        horizon_decay: float = 0.0,
        dice_weight: float = 0.0,
        gamma: float = 0.3,
        pde_D: float = 0.01,
        pde_mu: float = 0.1,
        beta_max: float = 0.01,
        eta_max: float = 0.1,
        xi_max: float = 0.05,
        warmup_epochs: int = 10,
        start_epoch: int = 5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.temporal_weight = temporal_weight
        self.horizon_decay = horizon_decay

        # Supervised loss: Focal BCE if focal_gamma > 0, else plain BCE
        if focal_gamma > 0.0:
            self.sup_loss_fn: nn.Module = FocalBCELoss(pos_weight=pos_weight, gamma=focal_gamma)
        else:
            self.sup_loss_fn = FireProgressionLoss(pos_weight=pos_weight)

        self.boundary_loss_fn = BoundaryAwareLoss() if boundary_weight > 0 else None
        self.temporal_loss_fn = TemporalConsistencyLoss() if temporal_weight > 0 else None

        self.physics = PhysicsConstraints(gamma=gamma, pde_D=pde_D, pde_mu=pde_mu)
        self.schedule = PINNSchedule(
            beta_max=beta_max, eta_max=eta_max, xi_max=xi_max,
            warmup_epochs=warmup_epochs, start_epoch=start_epoch,
        )
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        """Called by trainer at start of each epoch."""
        self._current_epoch = epoch

    def forward(
        self,
        af_logits: torch.Tensor,
        af_targets: torch.Tensor,
        ba_init: Optional[torch.Tensor] = None,
        wind: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        af_logits  : (B, H, 1, Hp, Wp)  predicted AF logits
        af_targets : (B, H, 1, Hp, Wp)  ground truth AF
        ba_init    : (B, 1, Hp, Wp)     initial BA state  (optional)
        wind       : (B, H, 2, Hp, Wp)  wind field        (optional)

        Returns
        -------
        dict with 'total', 'sup', 'boundary', 'temporal', 'dice', 'pde', 'mono', 'mask'
        """
        H = af_logits.shape[1]

        # ── Supervised loss (optionally horizon-weighted) ───────────────────
        if self.horizon_decay > 0.0 and H > 1:
            # w_h = exp(-d*h), normalised so mean weight = 1
            hw = torch.exp(
                -self.horizon_decay
                * torch.arange(H, dtype=torch.float32, device=af_logits.device)
            )
            hw = hw * H / hw.sum()
            loss_sup = torch.tensor(0.0, device=af_logits.device)
            for h in range(H):
                loss_sup = loss_sup + hw[h] * self.sup_loss_fn(
                    af_logits[:, h], af_targets[:, h]
                )
            loss_sup = loss_sup / H
        else:
            loss_sup = self.sup_loss_fn(af_logits, af_targets)

        # ── Soft Dice loss (optional) ───────────────────────────────────────
        if self.dice_weight > 0:
            prob = torch.sigmoid(af_logits)
            p = prob.reshape(-1)
            t = af_targets.reshape(-1)
            intersection = (p * t).sum()
            loss_dice = 1.0 - (2.0 * intersection + 1.0) / (p.sum() + t.sum() + 1.0)
        else:
            loss_dice = torch.tensor(0.0, device=af_logits.device)

        # ── Boundary-aware loss ─────────────────────────────────────────────
        if self.boundary_weight > 0.0 and self.boundary_loss_fn is not None:
            loss_boundary = self.boundary_loss_fn(af_logits, af_targets)
        else:
            loss_boundary = torch.tensor(0.0, device=af_logits.device)

        # ── Temporal consistency loss ───────────────────────────────────────
        if self.temporal_weight > 0.0 and self.temporal_loss_fn is not None:
            loss_temporal = self.temporal_loss_fn(af_logits, af_targets)
        else:
            loss_temporal = torch.tensor(0.0, device=af_logits.device)

        # ── PINN physics losses ─────────────────────────────────────────────
        w = self.schedule.get_weights(self._current_epoch)
        has_physics = (ba_init is not None) and any(v > 0 for v in w.values())

        if has_physics:
            phys = self.physics(af_logits, ba_init, wind)
            loss_pde = phys["pde"]
            loss_mono = phys["mono"]
            loss_mask = phys["mask"]
        else:
            z = torch.tensor(0.0, device=af_logits.device)
            loss_pde = loss_mono = loss_mask = z

        total = (
            loss_sup
            + self.dice_weight     * loss_dice
            + self.boundary_weight * loss_boundary
            + self.temporal_weight * loss_temporal
            + w["beta"] * loss_pde
            + w["eta"]  * loss_mono
            + w["xi"]   * loss_mask
        )

        return {
            "total":    total,
            "sup":      loss_sup.detach(),
            "dice":     loss_dice.detach(),
            "boundary": loss_boundary.detach(),
            "temporal": loss_temporal.detach(),
            "pde":      loss_pde.detach(),
            "mono":     loss_mono.detach(),
            "mask":     loss_mask.detach(),
        }
