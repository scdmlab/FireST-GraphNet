"""
schedules.py  --  Warm-up schedules for PINN loss weights.

During early epochs the network hasn't learned the basic mapping yet,
so physics constraints would only add noise.  We linearly ramp them in.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PINNSchedule:
    """Linear warm-up schedule for physics loss weights.

    Parameters
    ----------
    beta_max  : float   max PDE residual weight
    eta_max   : float   max BA monotonicity weight
    xi_max    : float   max AF-BA mask weight
    warmup_epochs : int   how many epochs before full weight
    start_epoch   : int   first epoch where weights > 0
    """
    beta_max: float = 0.01
    eta_max: float = 0.1
    xi_max: float = 0.05
    warmup_epochs: int = 10
    start_epoch: int = 5

    def get_weights(self, epoch: int) -> dict[str, float]:
        """Return dict {'beta': ..., 'eta': ..., 'xi': ...} for current epoch."""
        if epoch < self.start_epoch:
            return {"beta": 0.0, "eta": 0.0, "xi": 0.0}

        progress = min(
            (epoch - self.start_epoch) / max(self.warmup_epochs, 1), 1.0
        )
        return {
            "beta": self.beta_max * progress,
            "eta": self.eta_max * progress,
            "xi": self.xi_max * progress,
        }
