"""
firest_graphnet.py — FireSTGraphNet: the full model.

Supports two forward modes:

  Legacy mode  (single-step):
      Input:  x_seq (B, T_in, C, H, W)
      Output: logits (B, 1, H, W)

  Forecast mode  (multi-horizon, semantically aligned multi-source):
      Input:  af_hist    (B, T_in, 8, H, W)        VIIRS imagery
                • Day bands 0-5:  I1, I2, I3, M13, M15, M11 (6 spectral)
                • Night bands 6-7: I4 (MWIR), I5 (LWIR) (2 thermal)
              aux_hist   (B, T_in, 19, H, W)       FirePred auxiliary per-date
                DEM / slope / aspect / canopy / ERC / temp / dewpoint / RH /
                precip / wind speed / wind dir / fuel moisture / PDSI /
                fuel model / dead fuel / VPD / NDVI / soil temp / snow
              env_fut    (B, Horizon, 4, H, W)      future dynamic env conditions
                4ch canonical: [temperature, u_wind, v_wind, humidity]
                TRAINING :   extracted from FirePred (band5, speed→u,v, band7)
                INFERENCE A: HRRR 4ch     (TMP, UGRD, VGRD, RH)
                INFERENCE B: WeatherNext2 4ch (temp_2m, u10, v10, rh_2m)
              env_source (str)  adapter key — "firepred" | "hrrr" | "weathernext2"
              static     (B, C_s, H, W)              static features (ESRI LULC 1ch)
      Output: logits     (B, Horizon, 1, H, W)

      Encoder input per timestep = cat(fire[8] + aux[19] + static[1]) = 28 channels

Architecture:
    (A) Multi-Source CNN Encoder + Dual-CBAM  [per timestep]
        – Input: concat(fire, firepred_aux, lulc_static) = C_total channels
    (B) Graph Construction + GNN              [per timestep]
    (C) LSTM Temporal Encoder                 [across timesteps]
    (D) ForecastDecoder (FiLM + semantically aligned future env)
         ALL SOURCES → 4ch (temp, u, v, rh) → per-source adapter → backbone → FiLM
         TRAINING:   FirePred dynamic 4ch → "firepred"     adapter (4→32) → backbone
         INFERENCE A: HRRR 4ch            → "hrrr"          adapter (4→32) → backbone
         INFERENCE B: WN2 4ch             → "weathernext2"  adapter (4→32) → backbone
         or legacy Decoder for single-step
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from firest_graphnet.models.encoder import MultiSourceEncoder
from firest_graphnet.models.graph_modules import GraphFilter, GraphUpdate


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy single-step support (still fully functional)
# ═══════════════════════════════════════════════════════════════════════════
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, channels)
        self.beta_fc = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class Decoder(nn.Module):
    """Legacy conv decoder with FiLM (single-step output)."""

    def __init__(self, in_channels, hidden_channels, out_channels=1, cond_dim=256):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        ch = in_channels
        for ch_out in hidden_channels:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out), nn.GELU(),
            ))
            self.films.append(FiLMLayer(cond_dim, ch_out))
            ch = ch_out
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x, cond):
        for blk, film in zip(self.blocks, self.films):
            x = film(blk(x), cond)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════
#  FireSTGraphNet  — unified model
# ═══════════════════════════════════════════════════════════════════════════
class FireSTGraphNet(nn.Module):
    """Full FireST-GraphNet model (legacy + forecast mode).

    Parameters
    ----------
    in_channels       : int   input channels per frame (AF bands)
    encoder_channels  : list  CNN channel progression
    use_cbam          : bool
    patch_size        : int   graph patch size
    k_neighbors       : int   kNN
    gnn_hidden        : int
    gnn_layers        : int
    lstm_hidden       : int
    lstm_layers       : int
    lstm_dropout      : float
    decoder_channels  : list  decoder conv channels
    out_channels      : int
    img_h, img_w      : int
    forecast_mode     : bool  enable multi-horizon forecast decoder
    future_channels   : int   future env channels (4 = temp/u/v/rh)
    future_emb_dim    : int   future-condition encoder output dim
    """

    def __init__(
        self,
        in_channels: int = 8,
        encoder_channels: Optional[List[int]] = None,
        use_cbam: bool = True,
        patch_size: int = 8,
        k_neighbors: int = 8,
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        decoder_channels: Optional[List[int]] = None,
        out_channels: int = 1,
        img_h: int = 256,
        img_w: int = 256,
        # ── forecast-mode extras ──
        forecast_mode: bool = False,
        future_channels: int = 4,
        future_emb_dim: int = 64,
        # ── backward compat aliases ──
        train_future_channels: Optional[int] = None,
        inference_future_channels: Optional[int] = None,
        weather_channels: Optional[int] = None,
        weather_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        if decoder_channels is None:
            decoder_channels = [128, 64, 32]

        # Backward compat: old parameter names → future_channels
        if weather_channels is not None:
            future_channels = weather_channels
        if train_future_channels is not None:
            future_channels = train_future_channels
        # inference_future_channels ignored — all sources now use same ch
        if weather_emb_dim is not None:
            future_emb_dim = weather_emb_dim

        self.forecast_mode = forecast_mode
        enc_out = encoder_channels[-1]

        # (A) CNN Encoder + CBAM
        self.encoder = MultiSourceEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_channels,
            use_cbam=use_cbam,
        )

        # (B) Graph module
        self.graph_filter = GraphFilter(
            feat_dim=enc_out,
            patch_size=patch_size,
            k=k_neighbors,
            gnn_hidden=gnn_hidden,
            num_layers=gnn_layers,
            img_h=img_h,
            img_w=img_w,
        )
        self.graph_update = GraphUpdate(enc_out)

        # (C) Temporal LSTM encoder
        self.lstm = nn.LSTM(
            input_size=enc_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # (D) Decoder
        if forecast_mode:
            from firest_graphnet.models.decoder import ForecastDecoder
            self.decoder = ForecastDecoder(
                enc_feat_dim=enc_out,
                temporal_dim=lstm_hidden,
                future_channels=future_channels,
                future_emb_dim=future_emb_dim,
                decoder_hidden=decoder_channels,
                lstm_hidden=lstm_hidden,
            )
        else:
            self.decoder = Decoder(
                in_channels=enc_out,
                hidden_channels=decoder_channels,
                out_channels=out_channels,
                cond_dim=lstm_hidden,
            )

    # ── shared encoder backbone ────────────────────────────────────────────
    def _encode_sequence(
        self,
        x_seq: torch.Tensor,
        aux_hist: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a multi-source temporal sequence through CNN+GNN+LSTM.

        Batches all timesteps through encoder and GNN in a single pass
        by reshaping (B, T, ...) → (B*T, ...) to maximise GPU utilisation.

        Parameters
        ----------
        x_seq    : (B, T, 8, H, W)   VIIRS fire imagery
        aux_hist : (B, T, 19, H, W)  FirePred auxiliary data
        static   : (B, C_s, H, W)    static environmental features (ESRI LULC)

        Returns
        -------
        spatial_feat  : (B, enc_out, H, W)   last spatial feature map
        temporal_cond : (B, lstm_hidden)      last LSTM hidden state
        """
        B, T, C, H, W = x_seq.shape

        # ── Fuse all timesteps into one big batch ──────────────────────────
        parts = [x_seq.reshape(B * T, C, H, W)]          # (B*T, 8, H, W)
        if aux_hist is not None:
            parts.append(aux_hist.reshape(B * T, aux_hist.shape[2], H, W))
        if static is not None:
            # static is (B, C_s, H, W) — expand to (B*T, C_s, H, W)
            parts.append(static.unsqueeze(1).expand(-1, T, -1, -1, -1)
                         .reshape(B * T, static.shape[1], H, W))

        frame = torch.cat(parts, dim=1)                  # (B*T, C_total, H, W)

        # Encoder + GNN in one fused pass
        feat = self.encoder(frame)                        # (B*T, enc_out, H, W)
        gf = self.graph_filter(feat)                      # (B*T, enc_out, H, W)
        feat = self.graph_update(feat, gf)                # (B*T, enc_out, H, W)

        # Spatial pool → temporal sequence for LSTM
        pooled = feat.mean(dim=(2, 3))                    # (B*T, enc_out)
        seq = pooled.reshape(B, T, -1)                    # (B, T, enc_out)

        # Keep last spatial feature map for decoder
        last_spatial = feat.reshape(B, T, feat.shape[1], H, W)[:, -1]  # (B, enc_out, H, W)

        lstm_out, _ = self.lstm(seq)
        cond = lstm_out[:, -1, :]
        return last_spatial, cond

    # ── legacy forward ─────────────────────────────────────────────────────
    def forward(self, x_seq: torch.Tensor, **kw) -> torch.Tensor:
        """
        Legacy mode:
            x_seq : (B, T, C, H, W)  →  logits (B, 1, H, W)

        Forecast mode (pass keyword args):
            x_seq      : (B, T, 8, H, W)          VIIRS fire imagery
            aux_hist   : (B, T, 19, H, W)         FirePred hist auxiliary
            env_fut    : (B, H, 4, H, W)          future env conditions (4ch canonical)
                          4ch = [temp, u_wind, v_wind, rh] — ALL sources
            env_source : str   adapter key — "firepred" | "hrrr" | "weathernext2"
            static     : (B, C_s, H, W)           ESRI LULC static features
            →  logits (B, H, 1, H, W)

        Backward compat: w_fut accepted as alias for env_fut.
        """
        if not self.forecast_mode:
            spatial, cond = self._encode_sequence(x_seq)
            return self.decoder(spatial, cond)

        # ── Forecast mode ──────────────────────────────────────────────────
        aux_hist   = kw.get("aux_hist")
        env_fut    = kw.get("env_fut")
        if env_fut is None:
            env_fut = kw.get("w_fut")   # backward compat
        env_source = kw.get("env_source", "firepred")
        static     = kw.get("static")
        if env_fut is None:
            raise ValueError(
                "forecast_mode=True requires env_fut (or w_fut) keyword arg"
            )

        spatial, cond = self._encode_sequence(x_seq, aux_hist=aux_hist, static=static)
        return self.decoder(spatial, cond, env_fut, source=env_source)
