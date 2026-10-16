"""
pde_residual.py  --  Convection-Diffusion-Reaction PDE residual for PINN loss.

PDE:
    dA/dt + div(v*A) - div(D*grad(A)) + mu*A - S = 0

  A    : active fire field       (B, 1, H, W)
  v    : 2D wind field           (B, 2, H, W)  from weather
  D    : diffusion coefficient   scalar or (B, 1, H, W)
  mu   : decay rate              scalar
  S    : source (ignition)       (B, 1, H, W) or scalar

Spatial gradients are computed using Sobel-like convolution kernels
(no autograd needed — keeps the computation graph lightweight).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_kernels(device: torch.device):
    """3x3 Sobel kernels for dx and dy."""
    kx = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32, device=device) / 8.0
    ky = kx.t()
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def _laplacian_kernel(device: torch.device):
    """3x3 Laplacian kernel."""
    k = torch.tensor([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=torch.float32, device=device)
    return k.view(1, 1, 3, 3)


class PDEResidual(nn.Module):
    """Compute PDE residual for the fire convection-diffusion-reaction equation.

    Parameters
    ----------
    D       : float   diffusion coefficient
    mu      : float   decay rate
    dt      : float   time step (normalised, default 1.0)
    """

    def __init__(self, D: float = 0.01, mu: float = 0.1, dt: float = 1.0):
        super().__init__()
        self.D = D
        self.mu = mu
        self.dt = dt

        # Pre-register spatial gradient kernels as persistent buffers
        kx, ky = _sobel_kernels(torch.device("cpu"))
        k_lap = _laplacian_kernel(torch.device("cpu"))
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("k_lap", k_lap)

    def forward(
        self,
        A_curr: torch.Tensor,
        A_next: torch.Tensor,
        wind: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        A_curr : (B, 1, H, W)  AF probability at time t
        A_next : (B, 1, H, W)  AF probability at time t+1
        wind   : (B, 2, H, W)  [u, v] wind components (optional)
        source : (B, 1, H, W)  ignition source term  (optional)

        Returns
        -------
        residual : (B, 1, H, W)   PDE residual R  (should be → 0)
        """
        # dA/dt
        dAdt = (A_next - A_curr) / self.dt

        # Laplacian (diffusion)
        lap_A = F.conv2d(A_curr, self.k_lap, padding=1)
        diffusion = self.D * lap_A

        # Convection div(v A) ≈ u dA/dx + v dA/dy  (divergence form)
        if wind is not None:
            dAdx = F.conv2d(A_curr, self.kx, padding=1)
            dAdy = F.conv2d(A_curr, self.ky, padding=1)
            u = wind[:, 0:1]   # (B, 1, H, W)
            v = wind[:, 1:2]
            convection = u * dAdx + v * dAdy
        else:
            convection = torch.zeros_like(A_curr)

        # Decay
        decay = self.mu * A_curr

        # Source
        if source is None:
            source = torch.zeros_like(A_curr)

        # Residual:  dA/dt + conv - diff + decay - source = 0
        R = dAdt + convection - diffusion + decay - source
        return R
