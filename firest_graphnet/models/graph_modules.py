"""
graph_modules.py — Graph construction, GNN layers, and spatial reshape.

Supports two backends:
    1. torch_geometric (preferred)  — uses knn_graph + MessagePassing
    2. Pure PyTorch fallback         — manual adjacency + scatter_add

Workflow:
    feature_map (B, C, H, W)
    → patchify  → node features  (B, N_nodes, D)
    → build kNN graph
    → GNN message passing (2 layers)
    → unpatchify / upsample → (B, C, H, W)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── backend detection ──────────────────────────────────────────────────────
try:
    import torch_geometric  # noqa: F401
    from torch_geometric.nn import knn_graph, MessagePassing

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ── pure-PyTorch fallback utilities ────────────────────────────────────────
def _knn_graph_pure(
    x: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a kNN graph from node features using brute-force pairwise distance.

    Parameters
    ----------
    x : (N, D) node feature matrix
    k : number of neighbours

    Returns
    -------
    edge_index : (2, N*k) long tensor   [source, target]
    """
    # Pairwise L2 distance
    dist = torch.cdist(x, x)  # (N, N)
    # Exclude self-loops by setting diagonal to inf
    dist.fill_diagonal_(float("inf"))
    # Top-k nearest
    _, idx = dist.topk(k, dim=1, largest=False)  # (N, k)

    N = x.size(0)
    source = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
    target = idx.reshape(-1)
    edge_index = torch.stack([source, target], dim=0)  # (2, N*k)
    return edge_index


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Simple scatter-add: src (E, D), index (E,) → out (dim_size, D)."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


# ── Graph layer — pure PyTorch ─────────────────────────────────────────────
class GraphConvFallback(nn.Module):
    """Simple message-passing GNN layer (mean aggregation)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """x: (N, D_in)  edge_index: (2, E) → (N, D_out)."""
        src, tgt = edge_index  # (E,)
        msgs = x[src]  # (E, D_in)
        agg = _scatter_add(msgs, tgt, x.size(0))  # (N, D_in)

        # Degree for mean normalisation
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.scatter_add_(0, tgt.unsqueeze(1), torch.ones(src.size(0), 1, device=x.device))
        deg = deg.clamp(min=1.0)
        agg = agg / deg

        out = self.linear(agg + x)  # residual
        return F.gelu(self.norm(out))


# ── Graph layer — PyG backend ─────────────────────────────────────────────
if _HAS_PYG:

    class GraphConvPyG(MessagePassing):
        """GCN-like layer using PyG MessagePassing."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__(aggr="mean")
            self.linear = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            out = self.propagate(edge_index, x=x)  # aggregated messages
            out = self.linear(out + x)  # residual
            return F.gelu(self.norm(out))

        def message(self, x_j: torch.Tensor) -> torch.Tensor:
            return x_j


# ── high-level graph module ────────────────────────────────────────────────
class GraphFilter(nn.Module):
    """Patchify → kNN graph → GNN layers → spatial upsample.

    Parameters
    ----------
    feat_dim : int          feature channels from encoder
    patch_size : int        spatial patch size (each patch → 1 node)
    k : int                 kNN neighbours
    gnn_hidden : int        hidden dim of GNN layers
    num_layers : int        number of GNN layers
    img_h, img_w : int      original spatial dimensions
    """

    def __init__(
        self,
        feat_dim: int,
        patch_size: int = 8,
        k: int = 8,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        img_h: int = 256,
        img_w: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.k = k
        self.n_h = img_h // patch_size
        self.n_w = img_w // patch_size
        self.num_nodes = self.n_h * self.n_w

        # Projection into GNN space
        self.node_proj = nn.Linear(feat_dim, gnn_hidden)

        # GNN layers
        LayerCls = GraphConvPyG if _HAS_PYG else GraphConvFallback
        layers = []
        for i in range(num_layers):
            d_in = gnn_hidden
            d_out = gnn_hidden
            layers.append(LayerCls(d_in, d_out))
        self.gnn_layers = nn.ModuleList(layers)

        # Back-projection
        self.out_proj = nn.Linear(gnn_hidden, feat_dim)

    # ── static graph caching ──────────────────────────────────────────────
    def _get_static_edge_index(self, device: torch.device) -> torch.Tensor:
        """Build kNN graph from a regular grid of patch centres (static topology).

        The graph depends only on the spatial grid (n_h × n_w) and k, which
        are fixed at construction time, so we compute it once and cache it.
        """
        if hasattr(self, "_cached_edge_index") and self._cached_edge_index.device == device:
            return self._cached_edge_index

        # Regular grid coordinates for patch centres
        rows = torch.arange(self.n_h, dtype=torch.float32, device=device)
        cols = torch.arange(self.n_w, dtype=torch.float32, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_r.reshape(-1), grid_c.reshape(-1)], dim=1)  # (N, 2)

        if _HAS_PYG:
            edge_index = knn_graph(coords, self.k, loop=False)
        else:
            edge_index = _knn_graph_pure(coords, self.k)

        self._cached_edge_index = edge_index
        return edge_index

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) → (B, C, H, W) graph-refined features."""
        B, C, H, W = feat.shape
        ps = self.patch_size

        # ── patchify: mean-pool each patch ─────────────────────────────────
        # (B, C, n_h, ps, n_w, ps) → (B, C, n_h, n_w) via mean
        x = feat.unfold(2, ps, ps).unfold(3, ps, ps)  # (B, C, n_h, n_w, ps, ps)
        x = x.mean(dim=(-2, -1))  # (B, C, n_h, n_w)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, N_nodes, C)

        x = self.node_proj(x)  # (B, N_nodes, gnn_hidden)

        # Static kNN topology — computed once and cached
        edge_index = self._get_static_edge_index(feat.device)

        # Batch all samples into a single disjoint graph for GNN layers
        N = self.num_nodes
        # Offset edge indices for each sample in the batch
        offsets = torch.arange(B, device=feat.device).unsqueeze(1) * N  # (B, 1)
        # Expand edge_index for B samples: (2, E) → (2, B*E)
        ei = edge_index.unsqueeze(0).expand(B, -1, -1)  # (B, 2, E)
        ei = (ei + offsets.unsqueeze(1)).reshape(2, -1)  # (2, B*E)

        x_flat = x.reshape(B * N, -1)  # (B*N, gnn_hidden)
        for layer in self.gnn_layers:
            x_flat = layer(x_flat, ei)
        x = x_flat.reshape(B, N, -1)   # (B, N_nodes, gnn_hidden)

        x = self.out_proj(x)  # (B, N_nodes, C)

        # ── unpatchify: reshape to grid and upsample ───────────────────────
        x = x.permute(0, 2, 1).reshape(B, C, self.n_h, self.n_w)  # (B, C, n_h, n_w)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


class GraphUpdate(nn.Module):
    """Fuse graph-refined features with the original feature map (residual)."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, original: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        """Both inputs: (B, C, H, W)."""
        gate = self.gate(torch.cat([original, graph_feat], dim=1))
        return original + gate * graph_feat
