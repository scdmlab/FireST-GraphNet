"""
ba_update.py  --  Burned-Area (BA) state update from predicted Active Fire.

Rule:   B_{i+1} = 1 - (1 - B_i)(1 - gamma * sigmoid(AF_i))

This ensures B is monotonically non-decreasing in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BAStateUpdate(nn.Module):
    """Compute burned-area trajectory from AF logits.

    Parameters
    ----------
    gamma : float   fire-to-burn conversion rate  (default 0.3)
    """

    def __init__(self, gamma: float = 0.3):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        af_logits: torch.Tensor,
        ba_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        af_logits : (B, H, 1, Hp, Wp)   raw AF logits for each horizon step
        ba_init   : (B, 1, Hp, Wp)      last historical BA (0-1)

        Returns
        -------
        ba_seq : (B, H, 1, Hp, Wp)  burned-area state at each future step
        """
        B, H_steps, C, Hp, Wp = af_logits.shape
        device = af_logits.device
        ba_seq = []
        b_prev = ba_init.clone()  # (B, 1, Hp, Wp)

        for i in range(H_steps):
            af_prob = torch.sigmoid(af_logits[:, i])       # (B, 1, Hp, Wp)
            b_next = 1.0 - (1.0 - b_prev) * (1.0 - self.gamma * af_prob)
            ba_seq.append(b_next)
            b_prev = b_next

        return torch.stack(ba_seq, dim=1)   # (B, H, 1, Hp, Wp)
