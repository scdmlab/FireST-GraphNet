"""
decoder.py — Forecast decoder with semantically aligned future conditioning.

╔══════════════════════════════════════════════════════════════════════════╗
║  SEMANTIC ALIGNMENT:  ALL sources → 4ch (temp, u, v, rh)              ║
║                                                                        ║
║  Per-source adapters (4 → 32 each) handle distribution calibration:    ║
║    "firepred"     : extracted 4ch from FirePred 19-band data           ║
║    "hrrr"         : HRRR NWP forecast (TMP, UGRD, VGRD, RH)           ║
║    "weathernext2" : Google WN2 forecast (temp, u10, v10, rh)           ║
║                                                                        ║
║  Shared backbone (32 → 64 → emb_dim) is frozen across all sources.    ║
║  For HRRR/WN2 domain adaptation, only per-source adapter weights are   ║
║  fine-tuned while backbone+downstream remain frozen.                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Components:
    FiLMLayer              — Feature-wise Linear Modulation
    FutureConditionEncoder — per-source 4→32 adapters + shared backbone → emb
    SpatialHead            — conv blocks + FiLM → single-step logits
    ForecastDecoder        — LSTM auto-regressive multi-horizon decoder
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
#  FiLM — Feature-wise Linear Modulation
# ═══════════════════════════════════════════════════════════════════════════
class FiLMLayer(nn.Module):
    """Affine modulation:  y = γ(cond) ⊙ x + β(cond)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, channels)
        self.beta_fc = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ═══════════════════════════════════════════════════════════════════════════
#  FutureConditionEncoder — source-specific adapters + shared backbone
# ═══════════════════════════════════════════════════════════════════════════
UNIFIED_DIM = 32   # adapter output dim (before shared backbone)


class FutureConditionEncoder(nn.Module):
    """Encode future environmental conditions into a conditioning vector.

    All sources provide the same 4 semantically aligned channels:
        [temperature, u_wind, v_wind, relative_humidity]

    Per-source 1×1 conv adapters (4 → 32) handle distribution differences
    (e.g. FirePred temp might have different normalisation than HRRR temp).
    The shared backbone then processes the unified 32-channel features.

    Parameters
    ----------
    future_channels : int   input channels per source (default 4)
    emb_dim         : int   output embedding dimension
    """

    def __init__(
        self,
        future_channels: int = 4,
        emb_dim: int = 64,
        # backward compat — accepted but ignored if future_channels is set
        train_channels: Optional[int] = None,
        inference_channels: Optional[int] = None,
    ):
        super().__init__()
        # Backward compat: if old-style params given, use future_channels anyway
        ch = future_channels

        # Per-source adapter: 4ch → UNIFIED_DIM (lightweight 1×1 conv)
        self.adapters = nn.ModuleDict({
            "firepred":     nn.Conv2d(ch, UNIFIED_DIM, 1),
            "hrrr":         nn.Conv2d(ch, UNIFIED_DIM, 1),
            "weathernext2": nn.Conv2d(ch, UNIFIED_DIM, 1),
        })

        # Shared backbone: UNIFIED_DIM → emb_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(UNIFIED_DIM, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, w: torch.Tensor, source: str = "firepred") -> torch.Tensor:
        """
        Parameters
        ----------
        w      : (B, 4, H, W)   future-condition frame (4ch canonical)
        source : adapter key — "firepred" | "hrrr" | "weathernext2"

        Returns
        -------
        emb : (B, emb_dim)
        """
        x = self.adapters[source](w)   # → (B, 32, H, W)
        return self.backbone(x)         # → (B, emb_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  SpatialHead — conv blocks + FiLM → single-step logits
# ═══════════════════════════════════════════════════════════════════════════
class SpatialHead(nn.Module):
    """Conv decoder blocks with FiLM → (B, 1, H, W)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        cond_dim: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()

        ch = in_channels
        for ch_out in hidden_channels:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.GELU(),
            ))
            self.films.append(FiLMLayer(cond_dim, ch_out))
            ch = ch_out
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W)  cond: (B, D) → (B, 1, H, W)."""
        x = feat
        for block, film in zip(self.blocks, self.films):
            x = block(x)
            x = film(x, cond)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════
#  ForecastDecoder  — the full multi-horizon decoder
# ═══════════════════════════════════════════════════════════════════════════
class ForecastDecoder(nn.Module):
    """Auto-regressive forecast decoder with semantically aligned future
    conditioning.

    All sources (FirePred training / HRRR inference / WN2 inference) feed
    the SAME 4-channel canonical input [temp, u, v, rh] through per-source
    adapters (4 → 32) to a shared backbone.  This eliminates the
    19ch-train → 4ch-infer semantic mismatch.

    Parameters
    ----------
    enc_feat_dim      : int   spatial feature channels from encoder  (e.g. 128)
    temporal_dim      : int   LSTM hidden from temporal encoder  (e.g. 256)
    future_channels   : int   future env channels (default 4)
    future_emb_dim    : int   future-condition encoder output dim
    decoder_hidden    : list  conv channel progression in spatial head
    lstm_hidden       : int   decoder LSTM hidden size
    """

    def __init__(
        self,
        enc_feat_dim: int = 128,
        temporal_dim: int = 256,
        future_channels: int = 4,
        future_emb_dim: int = 64,
        decoder_hidden: Optional[List[int]] = None,
        lstm_hidden: int = 256,
        # backward compat — accepted but not used
        train_future_channels: Optional[int] = None,
        inference_future_channels: Optional[int] = None,
    ):
        super().__init__()
        if decoder_hidden is None:
            decoder_hidden = [128, 64, 32]

        # Resolve backward compat
        ch = future_channels
        if train_future_channels is not None:
            ch = train_future_channels

        self.future_enc = FutureConditionEncoder(
            future_channels=ch,
            emb_dim=future_emb_dim,
        )

        # Decoder LSTM input = temporal_dim + future_emb
        self.lstm_input_dim = temporal_dim + future_emb_dim
        self.lstm = nn.LSTMCell(self.lstm_input_dim, lstm_hidden)

        # Spatial head gets encoder features + FiLM from decoder LSTM
        self.spatial_head = SpatialHead(
            in_channels=enc_feat_dim,
            hidden_channels=decoder_hidden,
            cond_dim=lstm_hidden,
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_cond: torch.Tensor,
        env_fut: torch.Tensor,
        source: str = "firepred",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_feat  : (B, C_enc, H, W)        last spatial feature from encoder
        temporal_cond : (B, D_temporal)          temporal summary (last LSTM hidden)
        env_fut       : (B, Horizon, 4, H, W)   future environmental conditions
                          4ch = [temp, u_wind, v_wind, rh]  for ALL sources
        source        : adapter key — "firepred" | "hrrr" | "weathernext2"

        Returns
        -------
        logits : (B, Horizon, 1, H, W)
        """
        B, Horizon = env_fut.shape[0], env_fut.shape[1]
        device = spatial_feat.device

        # Init LSTM state
        h = torch.zeros(B, self.lstm.hidden_size, device=device)
        c = torch.zeros(B, self.lstm.hidden_size, device=device)

        logits_list = []
        for t in range(Horizon):
            # Future-condition encoding for step t
            env_t = env_fut[:, t]                            # (B, 4, H, W)
            env_emb = self.future_enc(env_t, source=source)  # (B, future_emb_dim)

            # LSTM input = concat(temporal_summary, future_embedding)
            lstm_in = torch.cat([temporal_cond, env_emb], dim=1)  # (B, D)
            h, c = self.lstm(lstm_in, (h, c))

            # Decode spatial features with FiLM conditioning from h
            logit_t = self.spatial_head(spatial_feat, h)  # (B, 1, H, W)
            logits_list.append(logit_t)

        return torch.stack(logits_list, dim=1)  # (B, Horizon, 1, H, W)
