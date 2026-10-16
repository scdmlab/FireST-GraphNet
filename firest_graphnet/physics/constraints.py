"""
constraints.py  --  Physics-informed constraint losses.

Provides three additional loss terms:
    1. BA Monotonicity   — burned area must never decrease
    2. AF-BA Consistency — active fire should not exist in already burned area
    3. PDE Residual      — convection-diffusion-reaction equation residual
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from firest_graphnet.physics.ba_update import BAStateUpdate
from firest_graphnet.physics.pde_residual import PDEResidual


class PhysicsConstraints(nn.Module):
    """Combines all PINN constraints into a single module.

    Parameters
    ----------
    gamma      : float   BA update rate
    pde_D      : float   diffusion coefficient
    pde_mu     : float   decay rate
    """

    def __init__(
        self,
        gamma: float = 0.3,
        pde_D: float = 0.01,
        pde_mu: float = 0.1,
    ):
        super().__init__()
        self.ba_updater = BAStateUpdate(gamma=gamma)
        self.pde_module = PDEResidual(D=pde_D, mu=pde_mu)

    def forward(
        self,
        af_logits: torch.Tensor,
        ba_init: torch.Tensor,
        wind: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        af_logits : (B, H, 1, Hp, Wp)   predicted AF logits for horizon
        ba_init   : (B, 1, Hp, Wp)      BA state at end of history
        wind      : (B, H, 2, Hp, Wp)   wind  (optional)

        Returns
        -------
        dict  keys: 'mono', 'mask', 'pde'  — each is a scalar loss tensor
        """
        B, H_steps, C, Hp, Wp = af_logits.shape

        # ── BA state sequence ──────────────────────────────────────────────
        ba_seq = self.ba_updater(af_logits, ba_init)  # (B, H, 1, Hp, Wp)
        af_prob = torch.sigmoid(af_logits)

        # ── 1. Monotonicity: B_{i+1} >= B_i ───────────────────────────────
        if H_steps > 1:
            ba_diff = ba_seq[:, :-1] - ba_seq[:, 1:]           # should be <= 0
            loss_mono = F.relu(ba_diff).mean()
        else:
            loss_mono = torch.tensor(0.0, device=af_logits.device)

        # ── 2. AF-BA Consistency: fire ∉ burned ────────────────────────────
        loss_mask = (af_prob * ba_seq).mean()

        # ── 3. PDE Residual ────────────────────────────────────────────────
        pde_losses = []
        for i in range(H_steps - 1):
            A_curr = af_prob[:, i]     # (B, 1, H, W)
            A_next = af_prob[:, i + 1]
            w = wind[:, i] if wind is not None else None
            R = self.pde_module(A_curr, A_next, wind=w)
            pde_losses.append((R ** 2).mean())

        if pde_losses:
            loss_pde = torch.stack(pde_losses).mean()
        else:
            loss_pde = torch.tensor(0.0, device=af_logits.device)

        return {
            "mono": loss_mono,
            "mask": loss_mask,
            "pde": loss_pde,
        }
