"""
create_notebook_v5.py
─────────────────────
Generator for `colab_run26_version2.ipynb` — FireST-GraphNet **V2**.

Why v2?  4 prior runs (23/24/25/26) plateau'd at ≤0.60 val-IoU:
    23: 0.598  |  24: 0.579  |  25: 0.580  |  26: 0.525
Run 26's threshold grid [0.30..0.50] was too narrow and pegged at 0.50 for
ALL 90 epochs — the true operating point lives >0.5.  Standard loss /
regularisation tuning has clearly hit diminishing returns.

Run 27 = **architectural redesign** while keeping GNN+LSTM core:
    GATv2 attention graph, multi-scale patch pyramid (8+16), U-Net skips,
    cross-attention weather fusion, attention-LSTM, auxiliary monotonic BA
    head, curriculum horizon (5->15 ep 1-30), mixup, WIDER threshold scan
    [0.35..0.65], pos_weight 4.5.

Notebook layout (11 top-level cells):
    0 (md) Report : current status -> improvements -> expected outcome
    1 Section 0  Install dependencies
    2 Section 1  Mount Google Drive + project setup
    3 Section 2  Imports + env check
    4 Section 3  Dataloader (existing TSSatFireForecastDataset + v2 augmenter)
    5 Section 4  Inline FireSTGraphNetV2 model  (~500 lines, GNN+LSTM+attn)
    6 Section 5  Config (write config_run27.yaml)
    7 Section 6  Train loop (patched trainer w/ curriculum horizon + mixup)
    8 Section 7  Validation metrics (IoU/F1/Prec/Rec/AUROC + threshold curve)
    9 Section 8  Inference visualisation (15 horizon frames side-by-side)
    10 Section 9 Export best checkpoint to Drive
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DST = (
    'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/'
    'data_tools/colab/colab_run26_version2.ipynb'
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _nl(lines: list[str]) -> list[str]:
    """Ensure every line (except last) ends in \\n, as required by .ipynb."""
    out: list[str] = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith('\n'):
            line = line + '\n'
        out.append(line)
    return out


def code_cell(src: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': _nl(src.splitlines(keepends=False)),
    }


def md_cell(src: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': _nl(src.splitlines(keepends=False)),
    }


cells: list[dict] = []

# ═══════════════════════════════════════════════════════════════════════════
# Cell 0 — Top-of-notebook report
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""# FireST-GraphNet **V2** — run26_version2 (run27)
### Multi-step wildfire progression forecasting (T_in=10, horizon=15)

> **Task**: binary **future Active Fire mask** prediction — direct multi-horizon forecast.
> **Inputs**: 8-band VIIRS active-fire history + 19-band FirePred auxiliary + 1-band LULC static + 4-channel future weather (temp, u-wind, v-wind, RH) for 15 horizons.
> **Output**: per-pixel fire probability for each of the next 15 time steps.
> **Architecture**: CNN encoder → **GNN (GATv2 + multi-scale pyramid)** → **Attention-LSTM** → decoder with **cross-attention weather fusion** + **U-Net skips** + auxiliary monotonic BA head.

---

## 1. Current Run26 Diagnosis

| Run | best val_iou (best-thr) | thr_best | Precision | Recall | Overfit gap | Status |
|-----|---|---|---|---|---|---|
| 23 | **0.598** @ep70 | — | — | — | 0.137 | overfit |
| 24 | 0.579 @ep52 | — | — | — | 0.077 | over-regularised |
| 25 | 0.580 @ep86 | 0.38 | 0.68 | 0.80 | 0.135 | EMA ✓ but plateau |
| **26** | **0.525 @ep90** | **0.50 (90/90 ep!)** | 0.65 | 0.73 | **0.20** | **worse** |

### Key observations from run26
- **`thr_best = 0.50` on every single epoch** ← the scan grid `[0.30..0.50]` was too narrow; the true optimum lives `> 0.50`.
- **Under-predicts** (`pos_weight=3.0` dropped too hard from run25's 5.0) — precision fell, recall fell, IoU fell.
- **Still climbing at ep90** — run was not converged.
- **Overfit gap 0.20** — train IoU 0.73 vs val 0.52.
- Four runs in a row plateau at ≤0.60 val-IoU ceiling → **standard HP tuning exhausted**; we need an architectural overhaul.

---

## 2. Why new V2 should break the 0.58 ceiling

### A. Model changes (inline `FireSTGraphNetV2` in Section 4)
| # | Change | Rationale |
|---|---|---|
| 1 | **GATv2 graph attention** | Per-edge attention >> mean aggregation for sparse fire structure |
| 2 | **Multi-scale graph pyramid** (patch=8 + patch=16 fused) | Captures both fine flame-front and plume-scale spread context |
| 3 | **Encoder→decoder U-Net skips** | Restores spatial detail the patch-level GNN discards |
| 4 | **Cross-attention weather fusion** | Pixel can "look up" weather by semantic similarity, not just scalar FiLM |
| 5 | **Attention-LSTM** (MHA over encoder T steps before LSTM) | Relevant past steps weighted, not squashed to last state |
| 6 | **Auxiliary monotonic BA head** | BA must grow over time — strong physical regulariser |

### B. Training changes (Section 6, patched trainer)
| # | Change | Rationale |
|---|---|---|
| 7 | **Curriculum horizon** H=5 (ep 1-10) → 15 (ep 30+) | Easy targets first — stabilises early training |
| 8 | **Wider threshold scan** `[0.35..0.65]` (step 0.05) | Fixes run26's 0.50-ceiling bug |
| 9 | **`pos_weight = 4.5`** (run25=5.0, run26=3.0 both bad) | Midpoint — avoids under-predict and over-predict |
| 10 | **Tversky α=0.4 β=0.6** | Mild recall bias (run26's 0.5/0.5 under-predicted) |
| 11 | **Mixup (p=0.2, α=0.4)** on img+aux | Cheap strong regulariser — cuts overfit gap |
| 12 | **AdamW WD=0.015** | More aggressive than run26's 0.01 |

### C. Keep (proven winners)
- **EMA (decay=0.999, start=ep10)** — run25 cut val_iou std from 0.030 to 0.003
- Label smoothing 0.05, decoder_dropout 0.15, cosine LR + 5-ep warmup, bf16 AMP, grad_clip=0.5
- Lovász hinge (0.25), BCE+focal (γ=2), Dice (0.15), boundary (0.12), PINN warm-up after ep10

---

## 3. Expected outcome

| Metric | Run26 | V2 target |
|--------|-------|-----------|
| val_iou (best-thr) | 0.525 | **0.64 – 0.68** |
| val_precision | 0.65 | **0.75+** |
| val_recall | 0.73 | 0.72 – 0.77 (balanced) |
| thr_best distribution | pegged at 0.50 | **spread 0.45-0.60** (grid not pegging at ends) |
| Overfit gap | 0.20 | **< 0.10** (mixup + aux BA) |
| val_iou std (last 20 ep) | — | **≤ 0.005** (EMA keeps it smooth) |

---

## 4. Notebook roadmap

| Section | Content |
|---|---|
| 0 | Install deps (torch, torch-geometric, timm, matplotlib, tqdm, pyyaml) |
| 1 | Mount Google Drive + set `PROJECT_PATH` |
| 2 | Python imports + env sanity check (GPU, memory) |
| 3 | Improved dataloader (`TSSatFireForecastDataset` + v2 augmenter: mixup + jitter) |
| 4 | **Inline `FireSTGraphNetV2`** — full model class in one cell (~500 LOC) |
| 5 | Write `config_run27.yaml` to disk |
| 6 | Patched training loop (curriculum horizon + mixup + aux BA) |
| 7 | Post-training validation curves (IoU/F1/P/R/AUROC + threshold grid) |
| 8 | Inference visualisation — 15 horizon frames side-by-side for 3 val events |
| 9 | Export `best_model.pt` + `metrics.csv` to Drive |

---

### Important notes
- **GNN+LSTM core is preserved** — no change to the overall paradigm, only the *inside* of each block is improved.
- The notebook is **Colab-A100 ready** (~2.5–3 hours for 100 epochs at `batch_size=8`).
- `config_run27.yaml` is written from the notebook itself — you do **not** need to upload a config.
- V2 improvements are opt-in via `model.v2_enabled = true` so the same train script can still run v1 configs.
"""))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 1 — Section 0: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## Section 0 · Install dependencies"))

cells.append(code_cell(r"""# ─── Section 0 · Install dependencies ──────────────────────────────────────
# Colab already ships torch ≥ 2.1 + CUDA 12.  We only need the extras.
# torch-geometric is optional (FireSTGraphNetV2 falls back to a hand-rolled
# GATv2 if PyG is absent), but we install it for speed.

import subprocess, sys, importlib

_PKGS = [
    ('torch_geometric', 'torch-geometric'),
    ('timm',            'timm'),
    ('einops',          'einops'),
    ('yaml',            'pyyaml'),
    ('tqdm',            'tqdm'),
]

def _ensure(import_name, pip_name):
    try:
        importlib.import_module(import_name)
        print(f'  ✓ {pip_name} already installed')
    except ImportError:
        print(f'  ⬇ installing {pip_name} ...')
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', pip_name]
        )

for imp, pip in _PKGS:
    _ensure(imp, pip)

print('\nAll dependencies ready.')
"""))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 2 — Section 1: Mount Drive + project setup
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## Section 1 · Mount Google Drive + project setup"))

cells.append(code_cell(r"""# ─── Section 1 · Mount Drive + project setup ───────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, sys, shutil
from pathlib import Path

# ---- CUSTOMISE THESE TWO PATHS IF YOUR DRIVE LAYOUT DIFFERS ----
PROJECT_PATH = '/content/drive/MyDrive/FireST-GraphNet-Clean'
DATASET_ZIP_TRAIN = '/content/drive/MyDrive/FireST-GraphNet/dataset_train.zip'
DATASET_ZIP_VAL   = '/content/drive/MyDrive/FireST-GraphNet/dataset_val.zip'
# -----------------------------------------------------------------

assert Path(PROJECT_PATH).is_dir(), f'PROJECT_PATH missing: {PROJECT_PATH}'

# Unpack datasets into Colab local SSD (fast I/O)
for zip_path, out_dir in [
    (DATASET_ZIP_TRAIN, '/content/dataset_train'),
    (DATASET_ZIP_VAL,   '/content/dataset_val'),
]:
    if Path(out_dir).is_dir() and any(Path(out_dir).iterdir()):
        print(f'  ✓ {out_dir} already present')
        continue
    if not Path(zip_path).exists():
        print(f'  ⚠ zip not found: {zip_path} (continuing — maybe unzipped elsewhere)')
        continue
    os.makedirs(out_dir, exist_ok=True)
    print(f'  ⬇ unzipping {Path(zip_path).name} -> {out_dir}')
    shutil.unpack_archive(zip_path, out_dir)

os.chdir('/content')
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

print('\nProject:', PROJECT_PATH)
print('CWD    :', os.getcwd())
"""))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 3 — Section 2: Imports + env check
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell("## Section 2 · Imports + environment check"))

cells.append(code_cell(r"""# ─── Section 2 · Imports + env check ───────────────────────────────────────
import math, json, time, random, warnings, csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Torch        : {torch.__version__}')
print(f'CUDA         : {torch.version.cuda}')
print(f'Device       : {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "-"})')
if DEVICE.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f'GPU memory   : {props.total_memory / 1024**3:.1f} GB')
    # A100 → bf16 is native & fast
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Existing project backend (data loader, loss, baseline trainer)
from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider import build_static_provider
from firest_graphnet.data.future_providers import extract_wind
from firest_graphnet.data.transforms import get_forecast_train_transforms
from firest_graphnet.training.losses import ForecastPINNLoss
from firest_graphnet.utils.io import find_dataset_dirs

print('\nBackend modules imported ✓')
"""))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 4 — Section 3: Dataloader + v2 augmenter
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 3 · Improved dataloader

We reuse the existing `TSSatFireForecastDataset` (event-boundary-aware sliding
windows, 8-band VIIRS + 19-band FirePred + 4-band future weather) but wrap it
in a `V2Augmenter` that adds:

- **Random horizontal/vertical flip** (p=0.5 each) — applied consistently to
  `af_hist`, `aux_hist`, `env_fut`, `af_fut`, `ba_init`, `static`.
- **Gaussian input jitter** on VIIRS + FirePred (std=0.01).
- **Channel dropout** (p=0.05 per aux channel) — forces the model not to
  depend on any single weather input.

Mixup is applied **at the batch level** inside the training loop, not here."""))

cells.append(code_cell(r"""# ─── Section 3 · Dataloader ────────────────────────────────────────────────
#
# Real project API (from firest_graphnet/scripts/train.py):
#   train_ds = TSSatFireForecastDataset(
#       train_dir,
#       firepred_provider=build_firepred_provider(train_dir, spatial_size=(H,W)),
#       static_provider =build_static_provider (train_dir, spatial_size=(H,W)),
#       transform=get_forecast_train_transforms(noise_std=...),
#       T_in=T_in, horizon=horizon, label_channel=1,
#   )
#
# We wrap the base dataset with V2Augmenter which ADDS channel dropout on top
# of the project's built-in spatial flip+jitter transform.  Each sample is a
# dict: af_hist, aux_hist, env_fut, env_source (str), af_fut, ba_init, static.

class V2Augmenter(torch.utils.data.Dataset):
    '''Wrap TSSatFireForecastDataset with extra channel dropout at train time.

    Flips + noise jitter are already applied by the project transform
    (`get_forecast_train_transforms`).  We only add channel dropout so the
    model does not depend on any single aux (weather/fuel) channel.
    '''
    def __init__(self, base, train: bool, channel_drop_p: float = 0.05):
        self.base = base
        self.train = train
        self.channel_drop_p = channel_drop_p

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if not self.train or self.channel_drop_p <= 0:
            return sample

        aux_hist = sample['aux_hist']              # (T_in, 19, H, W)
        C = aux_hist.shape[1]
        drop = (torch.rand(C) < self.channel_drop_p).float()
        keep = (1 - drop).view(1, C, 1, 1)
        sample = dict(sample)                       # shallow copy
        sample['aux_hist'] = aux_hist * keep
        return sample


def build_loaders(cfg):
    dcfg = cfg['data']
    fcfg = cfg['forecast']
    tcfg = cfg['training']
    img_size = int(dcfg.get('img_size', 256))
    T_in    = fcfg['T_in']
    horizon = fcfg['horizon']

    train_dir, val_dir = find_dataset_dirs(dcfg['root'])
    if train_dir is None:
        raise FileNotFoundError(f"No dataset_train under {dcfg['root']}")

    # Per-split providers (train has firepred_train.npy, val has firepred_val.npy)
    fp_train = build_firepred_provider(data_root=train_dir, spatial_size=(img_size, img_size))
    st_train = build_static_provider (data_root=train_dir, spatial_size=(img_size, img_size))
    if val_dir is not None:
        fp_val = build_firepred_provider(data_root=val_dir, spatial_size=(img_size, img_size))
        st_val = build_static_provider (data_root=val_dir, spatial_size=(img_size, img_size))
    else:
        fp_val, st_val = fp_train, st_train
    print(f'  firepred_provider (train C_aux={fp_train.channels}), '
          f'static_provider (C_static={st_train.channels})')

    noise_std = tcfg.get('augmentation_noise_std', 0.0)
    train_tf = get_forecast_train_transforms(noise_std=noise_std) if tcfg.get('augmentation', True) else None

    ds_kw = dict(T_in=T_in, horizon=horizon, label_channel=dcfg.get('label_channel', 1))
    train_base = TSSatFireForecastDataset(
        train_dir, firepred_provider=fp_train, static_provider=st_train,
        transform=train_tf, **ds_kw,
    )
    val_base = TSSatFireForecastDataset(
        val_dir if val_dir is not None else train_dir,
        firepred_provider=fp_val, static_provider=st_val,
        transform=None, **ds_kw,
    )
    train_ds = V2Augmenter(train_base, train=True)
    val_ds   = V2Augmenter(val_base,   train=False)
    print(f'  datasets: train={len(train_ds)}  val={len(val_ds)}')

    bs = tcfg['batch_size']
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=dcfg.get('num_workers', 0),
        pin_memory=dcfg.get('pin_memory', True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=dcfg.get('num_workers', 0),
        pin_memory=dcfg.get('pin_memory', True),
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds

print('V2Augmenter + build_loaders defined ✓')
"""))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 5 — Section 4: Inline FireSTGraphNetV2
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 4 · Inline `FireSTGraphNetV2` model

**Architecture diagram**

```
        x_seq (B,T,8,H,W)  aux_hist (B,T,19,H,W)  static (B,1,H,W)
                 │                  │                   │
                 └──────────┬───────┘                   │
                            ▼  concat→28ch              │
                   ┌──────────────────┐                 │
                   │ MultiSourceEnc   │ (CBAM + IN)     │
                   └──────────────────┘  features ×3 scales (32,64,128)
                            │
                   ┌──────────────────┐   ▲ skip-to-decoder (U-Net)
                   │ Multi-scale      │
                   │ Graph Pyramid    │   patch=8  GATv2 → 128ch spatial
                   │   • patch=8      │   patch=16 GATv2 → 128ch spatial
                   │   • patch=16     │   fuse via conv
                   │   (GATv2 attn)   │
                   └──────────────────┘
                            │
            spatial (B,T,C,H',W')  →  mean-pool → (B,T,C)
                            │
                   ┌──────────────────┐
                   │ Attention-LSTM   │ MHA over T steps → LSTM → h,c
                   └──────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
  env_fut (B,H,4,H,W)   spatial (B,C,H',W')   static
  cross-attn ↓               ↓ FiLM               │
       └──────────────→ ForecastDecoder ←─────────┘
                       (LSTMCell + SpatialHead
                       + U-Net skip concat)
                            │
                       ┌─────┴──────┐
                       ▼            ▼
                  af_pred (B,H,H,W)  ba_pred (B,H,H,W) monotonic
```
"""))

# Big inline model cell
MODEL_CELL = r'''# ─── Section 4 · FireSTGraphNetV2 (inline) ─────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------- CBAM ----
class _ChannelAttn(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, max(c // r, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(c // r, 4), c),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        a = F.adaptive_avg_pool2d(x, 1).view(B, C)
        m = F.adaptive_max_pool2d(x, 1).view(B, C)
        w = torch.sigmoid(self.mlp(a) + self.mlp(m)).view(B, C, 1, 1)
        return x * w


class _SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)
        m = x.max(dim=1, keepdim=True).values
        w = torch.sigmoid(self.conv(torch.cat([a, m], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = _ChannelAttn(c)
        self.sa = _SpatialAttn()

    def forward(self, x):
        return self.sa(self.ca(x))


# ---------------------------------------------------------- Encoder ----
class _ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1, bias=False)
        self.n1 = nn.InstanceNorm2d(co, affine=True)
        self.c2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm2d(co, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.n1(self.c1(x)))
        x = self.act(self.n2(self.c2(x)))
        return x


class MultiSourceEncoderV2(nn.Module):
    """Produces features at 3 scales (for U-Net skips)."""
    def __init__(self, in_ch=28, channels=(32, 64, 128), use_cbam=True):
        super().__init__()
        self.input_norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.b1 = _ConvBlock(in_ch,       channels[0])
        self.b2 = _ConvBlock(channels[0], channels[1])
        self.b3 = _ConvBlock(channels[1], channels[2])
        self.cbam = nn.ModuleList([CBAM(c) for c in channels]) if use_cbam else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):          # x (B,28,H,W)
        x = self.input_norm(x)
        s1 = self.b1(x)            # H
        if self.cbam: s1 = self.cbam[0](s1)
        s2 = self.b2(self.pool(s1))  # H/2
        if self.cbam: s2 = self.cbam[1](s2)
        s3 = self.b3(self.pool(s2))  # H/4
        if self.cbam: s3 = self.cbam[2](s3)
        return s1, s2, s3


# -------------------------------------------------------- GATv2 layer ----
class GATv2Layer(nn.Module):
    """Lightweight GATv2 (Brody et al. 2022).  Works on dense kNN graph."""
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        assert dim % heads == 0, 'dim must divide heads'
        self.W_src = nn.Linear(dim, dim, bias=False)
        self.W_dst = nn.Linear(dim, dim, bias=False)
        self.att = nn.Linear(self.dk, 1, bias=False)
        self.out = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, idx):
        """x (B,N,D), idx (B,N,K) kNN indices including self."""
        B, N, D = x.shape
        K = idx.shape[-1]
        src = self.W_src(x)        # (B,N,D)
        dst = self.W_dst(x)        # (B,N,D)
        # gather neighbours
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)   # (B,N,K,D)
        src_nei = torch.gather(src.unsqueeze(2).expand(-1, -1, K, -1), 1,
                               idx_exp)                      # (B,N,K,D)
        # GATv2: tanh of sum BEFORE attention projection
        e = torch.tanh(dst.unsqueeze(2) + src_nei)           # (B,N,K,D)
        # split heads
        e = e.view(B, N, K, self.h, self.dk)                 # (B,N,K,h,dk)
        alpha = self.att(e).squeeze(-1)                      # (B,N,K,h)
        alpha = F.softmax(alpha, dim=2)                      # over K
        alpha = self.drop(alpha)
        # values: the same src_nei split into heads
        v = src_nei.view(B, N, K, self.h, self.dk)
        agg = (alpha.unsqueeze(-1) * v).sum(dim=2)           # (B,N,h,dk)
        agg = agg.reshape(B, N, D)
        out = self.out(agg)
        # residual + LN
        return self.ln(x + self.drop(F.gelu(out)))


# ------------------------------------------- Multi-scale Graph Pyramid ----
def _build_knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """x (B,N,D) -> idx (B,N,K) incl. self at position 0."""
    B, N, _ = x.shape
    with torch.no_grad():
        d2 = torch.cdist(x, x, p=2)
        _, idx = torch.topk(-d2, k=k, dim=-1)      # nearest = smallest distance
    return idx


class MultiScaleGraphPyramid(nn.Module):
    """Patch-ify features at two scales, run GATv2, fuse, back-project."""
    def __init__(self, ch=128, patches=(8, 16), heads=4, n_layers=2):
        super().__init__()
        self.patches = patches
        self.layers = nn.ModuleDict()
        for p in patches:
            self.layers[f'p{p}'] = nn.ModuleList([
                GATv2Layer(ch, heads=heads) for _ in range(n_layers)
            ])
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * len(patches), ch, 1),
            nn.GELU(),
        )
        self.k = 8

    def forward(self, feat):          # feat (B,C,H,W)
        B, C, H, W = feat.shape
        outs = []
        for p in self.patches:
            # mean-pool to patch grid
            Hp, Wp = H // p, W // p
            if Hp * p != H or Wp * p != W:
                # fallback: adaptive
                patched = F.adaptive_avg_pool2d(feat, (Hp, Wp))
            else:
                patched = F.avg_pool2d(feat, p)
            B_, C_, hp, wp = patched.shape
            tok = patched.flatten(2).transpose(1, 2)         # (B,N,C)
            idx = _build_knn(tok, min(self.k, tok.shape[1]))
            for layer in self.layers[f'p{p}']:
                tok = layer(tok, idx)
            tok = tok.transpose(1, 2).reshape(B_, C_, hp, wp)
            # upsample back to HxW
            up = F.interpolate(tok, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(up)
        return self.fuse(torch.cat(outs, dim=1))


# -------------------------------------------------- Attention-LSTM ----
class AttentionLSTM(nn.Module):
    """Multi-head attention over the T encoder steps before feeding LSTM."""
    def __init__(self, dim, hidden, num_layers=2, dropout=0.4, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(dim, hidden, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

    def forward(self, x):                  # (B,T,D)
        res = x
        x, _ = self.attn(x, x, x)
        x = self.ln(res + x)
        out, (h, c) = self.lstm(x)
        return out, (h, c)                 # out (B,T,hidden)


# -------------------------------------- Cross-attention weather fusion ----
class CrossAttnWeatherFusion(nn.Module):
    """Spatial tokens query weather tokens."""
    def __init__(self, dim_spatial, dim_weather, heads=4):
        super().__init__()
        self.q = nn.Linear(dim_spatial, dim_spatial, bias=False)
        self.k = nn.Linear(dim_weather, dim_spatial, bias=False)
        self.v = nn.Linear(dim_weather, dim_spatial, bias=False)
        self.h = heads
        self.dk = dim_spatial // heads
        self.out = nn.Linear(dim_spatial, dim_spatial)
        self.ln = nn.LayerNorm(dim_spatial)

    def forward(self, spatial, weather):
        """
        spatial  (B, Ns, Ds) — flattened feature map tokens
        weather  (B, Nw, Dw) — weather tokens (e.g. global pooled per horizon)
        """
        B, Ns, Ds = spatial.shape
        _, Nw, _ = weather.shape
        q = self.q(spatial).view(B, Ns, self.h, self.dk).transpose(1, 2)
        k = self.k(weather).view(B, Nw, self.h, self.dk).transpose(1, 2)
        v = self.v(weather).view(B, Nw, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, Ns, Ds)
        return self.ln(spatial + self.out(out))


# ---------------------------------------------- Forecast Decoder V2 ----
class ForecastDecoderV2(nn.Module):
    """LSTMCell + cross-attn weather + U-Net skip + monotonic BA head."""
    def __init__(self, spatial_ch, lstm_hidden, horizon,
                 skip_channels=(32, 64),        # from encoder scales s1, s2
                 weather_ch=4, weather_emb=64,
                 dropout=0.15, heads=4, aux_ba=True):
        super().__init__()
        self.H = horizon
        self.spatial_ch = spatial_ch

        # Weather encoder — per horizon, map 4ch → weather_emb
        self.weather_embed = nn.Sequential(
            nn.Linear(weather_ch, 32), nn.GELU(),
            nn.Linear(32, weather_emb),
        )
        self.cross_attn = CrossAttnWeatherFusion(spatial_ch, weather_emb, heads=heads)

        # LSTMCell for horizon rollout
        self.lstm_cell = nn.LSTMCell(spatial_ch, lstm_hidden)
        self.proj_cond = nn.Linear(lstm_hidden, spatial_ch)  # condition the spatial map

        # U-Net decoder: spatial_ch → 64 (+skip64) → 32 (+skip32) → head
        self.up1 = nn.ConvTranspose2d(spatial_ch, 64, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + skip_channels[1], 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True), nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(32 + skip_channels[0], 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True), nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.head_af = nn.Conv2d(32, 1, 1)
        self.aux_ba = aux_ba
        if aux_ba:
            self.head_ba = nn.Conv2d(32, 1, 1)

    def forward(self, spatial, lstm_state, env_fut, skip1, skip2):
        """
        spatial   (B, C, Hs, Ws)          — graph-pyramid output
        lstm_state ((B,hid),(B,hid))      — (h, c) from Attention-LSTM last step
        env_fut   (B, H, 4, Hf, Wf)       — future weather
        skip1     (B, 32, H,   W)         — encoder scale 0 for U-Net
        skip2     (B, 64, H/2, W/2)       — encoder scale 1 for U-Net
        """
        B, C, Hs, Ws = spatial.shape
        H_out = env_fut.shape[1]
        h, c = lstm_state

        af_seq = []
        ba_seq = []
        ba_cum = torch.zeros(B, 1, skip1.shape[-2], skip1.shape[-1],
                             device=spatial.device, dtype=spatial.dtype)

        for t in range(H_out):
            # 1) Pool spatial → vector for LSTMCell
            sp_vec = F.adaptive_avg_pool2d(spatial, 1).flatten(1)  # (B,C)
            h, c = self.lstm_cell(sp_vec, (h, c))
            cond = self.proj_cond(h)                                # (B,C)
            spatial_t = spatial + cond.view(B, C, 1, 1)

            # 2) Cross-attention with weather at this horizon
            #    weather for this step: pool over space → one weather token
            w_now = env_fut[:, t]                                   # (B,4,Hf,Wf)
            w_vec = F.adaptive_avg_pool2d(w_now, 1).flatten(1)      # (B,4)
            w_tok = self.weather_embed(w_vec).unsqueeze(1)          # (B,1,emb)
            sp_tok = spatial_t.flatten(2).transpose(1, 2)           # (B,Ns,C)
            sp_tok = self.cross_attn(sp_tok, w_tok)
            spatial_t = sp_tok.transpose(1, 2).reshape(B, C, Hs, Ws)

            # 3) U-Net upsample with encoder skips
            x = self.up1(spatial_t)                                  # (B,64,H/2,W/2)
            if x.shape[-2:] != skip2.shape[-2:]:
                x = F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse1(torch.cat([x, skip2], dim=1))
            x = self.up2(x)                                          # (B,32,H,W)
            if x.shape[-2:] != skip1.shape[-2:]:
                x = F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse2(torch.cat([x, skip1], dim=1))

            af_logit = self.head_af(x)                              # (B,1,H,W)
            af_seq.append(af_logit)

            if self.aux_ba:
                ba_delta = torch.sigmoid(self.head_ba(x))            # positive increment
                ba_cum = torch.clamp(ba_cum + ba_delta, 0.0, 1.0)    # monotonic non-decreasing
                ba_seq.append(ba_cum)

        af_pred = torch.stack(af_seq, dim=1)                         # (B,H,1,H,W)
        out = {'af_pred': af_pred}
        if self.aux_ba:
            out['ba_pred'] = torch.stack(ba_seq, dim=1)              # (B,H,1,H,W), monotonic
        return out


# -------------------------------------------- FireSTGraphNetV2 (top) ----
class FireSTGraphNetV2(nn.Module):
    """
    GNN + LSTM core preserved.  Improvements:
      * GATv2 attention graph at 2 patch scales
      * Attention-LSTM (MHA over T)
      * Cross-attention weather fusion
      * U-Net encoder→decoder skips
      * Monotonic auxiliary BA head
    """
    def __init__(self,
                 in_ch_img=8, in_ch_aux=19, in_ch_static=1,
                 encoder_channels=(32, 64, 128),
                 gnn_patches=(8, 16), gat_heads=4,
                 lstm_hidden=256, lstm_layers=2, lstm_dropout=0.4,
                 lstm_attn_heads=4,
                 weather_ch=4, weather_emb=64,
                 horizon=15,
                 dropout=0.15, aux_ba=True):
        super().__init__()
        in_total = in_ch_img + in_ch_aux + in_ch_static
        self.encoder = MultiSourceEncoderV2(in_ch=in_total,
                                            channels=encoder_channels, use_cbam=True)
        self.pyramid = MultiScaleGraphPyramid(
            ch=encoder_channels[2], patches=gnn_patches, heads=gat_heads)
        self.attn_lstm = AttentionLSTM(
            dim=encoder_channels[2], hidden=lstm_hidden,
            num_layers=lstm_layers, dropout=lstm_dropout, heads=lstm_attn_heads)
        # project LSTM hidden back to spatial channels
        self.proj = nn.Linear(lstm_hidden, encoder_channels[2])
        self.decoder = ForecastDecoderV2(
            spatial_ch=encoder_channels[2],
            lstm_hidden=lstm_hidden,
            horizon=horizon,
            skip_channels=(encoder_channels[0], encoder_channels[1]),
            weather_ch=weather_ch, weather_emb=weather_emb,
            dropout=dropout, heads=gat_heads, aux_ba=aux_ba,
        )
        self.horizon = horizon

    def _encode_step(self, img, aux, static):
        """One time step: (B,8,H,W)+(B,19,H,W)+(B,1,H,W) -> 3 scale features."""
        x = torch.cat([img, aux, static], dim=1)
        s1, s2, s3 = self.encoder(x)
        return s1, s2, s3

    def forward(self, batch, horizon_override: int | None = None):
        x_seq   = batch['af_hist']          # (B,T,8,H,W)
        aux_hist= batch['aux_hist']         # (B,T,19,H,W)
        env_fut = batch['env_fut']          # (B,H,4,H,W)
        static  = batch['static']           # (B,1,H,W)

        B, T = x_seq.shape[:2]
        H_out = env_fut.shape[1] if horizon_override is None else horizon_override

        # per-step encode (U-Net skips from LAST step)
        s1_list, s2_list, s3_list = [], [], []
        for t in range(T):
            s1, s2, s3 = self._encode_step(x_seq[:, t], aux_hist[:, t], static)
            s1_list.append(s1); s2_list.append(s2); s3_list.append(s3)
        skip1 = s1_list[-1]
        skip2 = s2_list[-1]
        feats_deep = torch.stack(s3_list, dim=1)          # (B,T,C,Hs,Ws)
        Bf, Tf, Cf, Hs, Ws = feats_deep.shape

        # graph pyramid per time step (share weights)
        g_list = []
        for t in range(Tf):
            g = self.pyramid(feats_deep[:, t])            # (B,C,Hs,Ws)
            g_list.append(g)
        g_seq = torch.stack(g_list, dim=1)                # (B,T,C,Hs,Ws)

        # mean-pool → per-step vector → Attention-LSTM
        g_vec = g_seq.mean(dim=[-1, -2])                  # (B,T,C)
        lstm_out, (h, c) = self.attn_lstm(g_vec)          # lstm_out (B,T,hid)

        # latest spatial map + LSTM condition
        spatial_last = g_seq[:, -1]                        # (B,C,Hs,Ws)
        cond = self.proj(lstm_out[:, -1])                  # (B,C)
        spatial = spatial_last + cond.view(Bf, Cf, 1, 1)

        # decoder rolls horizon
        env_use = env_fut[:, :H_out]
        out = self.decoder(
            spatial=spatial,
            lstm_state=(h[-1], c[-1]),    # last-layer hidden
            env_fut=env_use,
            skip1=skip1,
            skip2=skip2,
        )
        return out


# Quick smoke test
def _smoke():
    m = FireSTGraphNetV2(horizon=5).eval()
    B = 2; T = 10; H = W = 64
    batch = dict(
        af_hist = torch.randn(B, T, 8, H, W),
        aux_hist= torch.randn(B, T, 19, H, W),
        env_fut = torch.randn(B, 5, 4, H, W),
        ba_init = torch.zeros(B, 1, H, W),
        static  = torch.zeros(B, 1, H, W),
        source  = ['firepred']*B,
    )
    with torch.no_grad():
        out = m(batch)
    assert out['af_pred'].shape == (B, 5, 1, H, W), out['af_pred'].shape
    assert out['ba_pred'].shape == (B, 5, 1, H, W)
    # monotonic check
    ba = out['ba_pred'][0, :, 0, 0, 0]
    assert torch.all(ba[1:] >= ba[:-1] - 1e-5), 'BA must be non-decreasing'
    print('FireSTGraphNetV2 smoke test PASSED ✓')
    print(f'  af_pred: {out["af_pred"].shape}')
    print(f'  ba_pred: {out["ba_pred"].shape}  (monotonic ✓)')
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'  params : {n_params:.2f}M')

_smoke()
'''
cells.append(code_cell(MODEL_CELL))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 6 — Section 5: Config
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 5 · Training configuration (`config_run27.yaml`)

Written to disk inline — no separate upload required.  Keeps all of run25's
proven settings (EMA 0.999, label smoothing 0.05, Lovász, cosine LR) and
adds the V2 curriculum + wider threshold scan + pos_weight rebalance."""))

CONFIG_CELL = r'''# ─── Section 5 · Write config_run27.yaml ───────────────────────────────────
CONFIG_STR = """
data:
  T_in: 9
  T_out: 1
  img_size: 256
  label_channel: 1
  num_bands: 8
  num_workers: 0
  pin_memory: true
  root: /content
  root_train: /content/dataset_train
  root_val: /content/dataset_val
  val_split: 0.1

device: cuda

forecast:
  T_in: 10
  enabled: true
  future_channels: 4
  future_emb_dim: 64
  horizon: 15
  in_channels: 28
  inference_mode: firepred
  weather_channels: 4
  weather_emb_dim: 64

model:
  decoder_channels: [128, 64, 32]
  encoder_channels: [32, 64, 128]
  gnn_hidden: 128
  gnn_layers: 2
  in_channels: 8
  k_neighbors: 8
  lstm_dropout: 0.4
  lstm_hidden: 256
  lstm_layers: 2
  out_channels: 1
  patch_size: 8
  use_cbam: true
  use_input_norm: true
  use_spatial_feedback: true
  decoder_dropout: 0.15
  v2_enabled: true
  v2_gat_heads: 4
  v2_graph_pyramid_patches: [8, 16]
  v2_unet_skips: true
  v2_cross_attn_heads: 4
  v2_lstm_attn_heads: 4
  v2_aux_ba_head: true

output_dir: /content/outputs/run27

pinn:
  beta_max: 0.005
  eta_max: 0.08
  gamma: 0.25
  pde_D: 0.01
  pde_mu: 0.12
  start_epoch: 10
  warmup_epochs: 15
  xi_max: 0.0

seed: 42

training:
  augmentation: true
  augmentation_noise_std: 0.01
  batch_size: 8
  boundary_weight: 0.12
  dice_weight: 0.15
  ema_decay: 0.999
  ema_start_epoch: 10
  epochs: 100
  focal_gamma: 2.0
  grad_clip: 0.5
  horizon_decay: 0.10
  label_smoothing: 0.05
  lovasz_weight: 0.25
  lr: 0.00015
  metric_every: 5
  mixed_precision: true
  patience: 25
  pos_weight: 4.5
  save_every: 5
  scheduler: cosine
  temporal_weight: 0.10
  tversky_weight: 0.25
  tversky_alpha: 0.4
  tversky_beta: 0.6
  val_every: 1
  val_threshold: 0.50
  val_threshold_scan: true
  val_threshold_grid: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
  warmup_epochs_lr: 5
  warmup_start_factor: 0.1
  weight_decay: 0.015
  v2_curriculum_horizon: true
  v2_horizon_start: 5
  v2_horizon_full: 15
  v2_horizon_ramp_end_epoch: 30
  v2_mixup_prob: 0.2
  v2_mixup_alpha: 0.4
  v2_aux_ba_weight: 0.05
"""

CFG_PATH = Path('/content/config_run27.yaml')
CFG_PATH.write_text(CONFIG_STR)
cfg = yaml.safe_load(CONFIG_STR)
print(f'Config written to {CFG_PATH}')
print(f'  epochs       : {cfg["training"]["epochs"]}')
print(f'  batch_size   : {cfg["training"]["batch_size"]}')
print(f'  horizon      : {cfg["forecast"]["horizon"]}  (curriculum {cfg["training"]["v2_horizon_start"]} -> {cfg["training"]["v2_horizon_full"]} by ep {cfg["training"]["v2_horizon_ramp_end_epoch"]})')
print(f'  pos_weight   : {cfg["training"]["pos_weight"]}')
print(f'  lovasz       : {cfg["training"]["lovasz_weight"]}')
print(f'  thr grid     : {cfg["training"]["val_threshold_grid"]}')
print(f'  EMA          : decay={cfg["training"]["ema_decay"]}, start=ep{cfg["training"]["ema_start_epoch"]}')
'''
cells.append(code_cell(CONFIG_CELL))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 7 — Section 6: Training loop
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 6 · Training loop (curriculum horizon + mixup + aux BA)

Rather than subclass the full project `ForecastTrainer`, we run a lean,
self-contained training loop here.  It reuses `ForecastPINNLoss` for the
main supervised term and adds:

1. **Curriculum horizon** — `H_cur = min(H_full, 5 + (ep-1)*10/30)` for ep ≤ 30
2. **Mixup** (prob=0.2, α=0.4) — mix two samples' VIIRS + AUX + labels
3. **Aux BA loss** — BCE between predicted cumulative BA and `cumsum(af_fut) > 0`
4. **EMA** of model weights (decay=0.999) starting epoch 10, used only at val
5. **Wider threshold scan** — 7 thresholds [0.35..0.65] accumulating global TP/FP/FN"""))

TRAINER_CELL = r'''# ─── Section 6 · Training loop ─────────────────────────────────────────────
import copy, time
import torch
from torch.cuda.amp import autocast, GradScaler

# ── helpers ────────────────────────────────────────────────────────────────
def curriculum_horizon(epoch, cfg):
    """Linear ramp from H_start (ep1) to H_full (ramp_end)."""
    tr = cfg['training']
    if not tr.get('v2_curriculum_horizon', False):
        return cfg['forecast']['horizon']
    s = tr['v2_horizon_start']
    f = tr['v2_horizon_full']
    e = tr['v2_horizon_ramp_end_epoch']
    if epoch >= e:
        return f
    frac = (epoch - 1) / max(e - 1, 1)
    return max(s, int(round(s + frac * (f - s))))


def mixup_batch(batch, alpha=0.4, prob=0.2):
    """Mixup pairs within the batch on img + aux + env + labels."""
    if torch.rand(1).item() >= prob:
        return batch
    B = batch['af_hist'].shape[0]
    if B < 2: return batch
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B, device=batch['af_hist'].device)
    out = {}
    for k in ('af_hist', 'aux_hist', 'env_fut', 'af_fut', 'ba_init', 'static'):
        if k not in batch: continue
        out[k] = lam * batch[k] + (1 - lam) * batch[k][perm]
    out['source'] = batch.get('source', None)
    return out


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def swap_in(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow})

    def swap_out(self, model):
        model.load_state_dict({**model.state_dict(), **self.backup})


def iou_pr_at_threshold(logits, target, thr, eps=1e-7):
    """Returns dict of tp/fp/fn/pred_pos/gt_pos for later accumulation."""
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    tgt  = (target > 0.5).float()
    tp = float((pred * tgt).sum().item())
    fp = float((pred * (1 - tgt)).sum().item())
    fn = float(((1 - pred) * tgt).sum().item())
    return dict(tp=tp, fp=fp, fn=fn,
                pred_pos=float(pred.sum().item()),
                gt_pos=float(tgt.sum().item()))


def aux_ba_loss(ba_pred, af_fut, eps=1e-7):
    """BCE between predicted cumulative BA and cumsum(af_fut) > 0."""
    # af_fut (B,H,1,H,W)  ->  ba_target cumulative
    with torch.no_grad():
        ba_target = (af_fut.cumsum(dim=1) > 0.5).float()
    return F.binary_cross_entropy(
        ba_pred.clamp(1e-7, 1 - 1e-7), ba_target, reduction='mean'
    )


# ── main training loop ────────────────────────────────────────────────────
def train_v2(cfg, out_dir='/content/outputs/run27'):
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = Path(out_dir) / 'metrics.csv'

    # Dataloaders
    train_loader, val_loader, _, _ = build_loaders(cfg)
    print(f'  train batches: {len(train_loader)}   val batches: {len(val_loader)}')

    # Model
    mcfg = cfg['model']
    model = FireSTGraphNetV2(
        in_ch_img=8, in_ch_aux=19, in_ch_static=1,
        encoder_channels=tuple(mcfg['encoder_channels']),
        gnn_patches=tuple(mcfg['v2_graph_pyramid_patches']),
        gat_heads=mcfg['v2_gat_heads'],
        lstm_hidden=mcfg['lstm_hidden'],
        lstm_layers=mcfg['lstm_layers'],
        lstm_dropout=mcfg['lstm_dropout'],
        lstm_attn_heads=mcfg['v2_lstm_attn_heads'],
        weather_ch=4, weather_emb=64,
        horizon=cfg['forecast']['horizon'],
        dropout=mcfg['decoder_dropout'],
        aux_ba=mcfg['v2_aux_ba_head'],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  model params : {n_params:.2f}M')

    # Loss — project ForecastPINNLoss for supervised + Lovász + PINN
    # Real API: kwargs are NOT prefixed `pinn_` — they live at top level.
    tcfg = cfg['training']; pcfg = cfg['pinn']
    criterion = ForecastPINNLoss(
        pos_weight      = tcfg['pos_weight'],
        focal_gamma     = tcfg['focal_gamma'],
        boundary_weight = tcfg['boundary_weight'],
        temporal_weight = tcfg['temporal_weight'],
        horizon_decay   = tcfg['horizon_decay'],
        dice_weight     = tcfg['dice_weight'],
        tversky_weight  = tcfg['tversky_weight'],
        tversky_alpha   = tcfg['tversky_alpha'],
        tversky_beta    = tcfg['tversky_beta'],
        lovasz_weight   = tcfg['lovasz_weight'],
        label_smoothing = tcfg['label_smoothing'],
        gamma           = pcfg['gamma'],
        pde_D           = pcfg['pde_D'],
        pde_mu          = pcfg['pde_mu'],
        beta_max        = pcfg['beta_max'],
        eta_max         = pcfg['eta_max'],
        xi_max          = pcfg['xi_max'],
        warmup_epochs   = pcfg['warmup_epochs'],
        start_epoch     = pcfg['start_epoch'],
    ).to(DEVICE)

    # Optimiser + scheduler + scaler
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg['training']['lr'],
                            weight_decay=cfg['training']['weight_decay'],
                            betas=(0.9, 0.995))
    total_ep = cfg['training']['epochs']
    warm = cfg['training']['warmup_epochs_lr']
    def lr_lambda(ep):
        if ep < warm:
            return cfg['training']['warmup_start_factor'] + (
                1 - cfg['training']['warmup_start_factor']) * ep / max(warm, 1)
        t = (ep - warm) / max(total_ep - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = cfg['training']['mixed_precision']
    scaler = GradScaler(enabled=use_amp)

    ema = EMA(model, decay=cfg['training']['ema_decay'])
    ema_start = cfg['training']['ema_start_epoch']

    # Header
    col_names = [
        'epoch', 'horizon_cur', 'train_loss', 'train_sup', 'train_ba',
        'val_loss', 'val_iou_fixed', 'val_iou_best_thr', 'val_thr_best',
        'val_prec', 'val_rec', 'val_f1', 'lr',
    ]
    with metrics_path.open('w', newline='') as f:
        csv.writer(f).writerow(col_names)

    best_val_iou = -1.0
    best_epoch = 0

    thr_grid = cfg['training']['val_threshold_grid']
    fixed_thr = cfg['training']['val_threshold']
    aux_ba_w = cfg['training'].get('v2_aux_ba_weight', 0.05)
    mixup_p = cfg['training'].get('v2_mixup_prob', 0.2)
    mixup_a = cfg['training'].get('v2_mixup_alpha', 0.4)
    use_aux_ba = mcfg['v2_aux_ba_head']

    def _to_dev(batch):
        """Move all tensor values in batch dict to DEVICE (skip meta/env_source)."""
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(DEVICE, non_blocking=True)
            else:
                out[k] = v
        return out

    for epoch in range(1, total_ep + 1):
        ep_start = time.time()
        H_cur = curriculum_horizon(epoch, cfg)
        criterion.set_epoch(epoch)

        # ───── TRAIN ─────
        model.train()
        train_losses = []; train_sup = []; train_ba = []
        pbar = tqdm(train_loader, desc=f'ep {epoch:03d}/{total_ep}  H={H_cur}',
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = _to_dev(batch)
            # Truncate labels + env to curriculum horizon
            if H_cur < batch['af_fut'].shape[1]:
                batch = dict(batch)
                batch['af_fut']  = batch['af_fut'][:, :H_cur]
                batch['env_fut'] = batch['env_fut'][:, :H_cur]

            # Mixup
            batch = mixup_batch(batch, alpha=mixup_a, prob=mixup_p)

            # env_source is str or list[str] after collate
            env_source = batch.get('env_source', 'firepred')
            if isinstance(env_source, (list, tuple)):
                env_source = env_source[0]

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                out = model(batch, horizon_override=H_cur)
                af_logits = out['af_pred']                         # (B, H, 1, Hh, Ww)
                af_targets = batch['af_fut']                       # (B, H, 1, Hh, Ww)
                # wind for PDE (zeros if torch scalar doesn't match — extract_wind handles)
                try:
                    wind = extract_wind(batch['env_fut'], source=env_source)
                except Exception:
                    wind = None
                losses_dict = criterion(
                    af_logits, af_targets,
                    ba_init=batch.get('ba_init'), wind=wind,
                )
                loss_main = losses_dict['total']
                sup_val   = float(losses_dict.get('sup', loss_main).item())

                # aux BA
                if use_aux_ba and 'ba_pred' in out:
                    l_ba = aux_ba_loss(out['ba_pred'], batch['af_fut'])
                else:
                    l_ba = torch.tensor(0.0, device=DEVICE)
                loss = loss_main + aux_ba_w * l_ba

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
            scaler.step(opt)
            scaler.update()

            if epoch >= ema_start:
                ema.update(model)

            train_losses.append(float(loss.item()))
            train_sup.append(sup_val)
            train_ba.append(float(l_ba.item()))
            pbar.set_postfix(loss=f'{loss.item():.3f}', sup=f'{sup_val:.3f}')

        sched.step()

        # ───── VAL (with EMA if active) ─────
        if epoch >= ema_start:
            ema.swap_in(model)
        model.eval()

        acc = {thr: dict(tp=0, fp=0, fn=0) for thr in thr_grid}
        fixed_acc = dict(tp=0, fp=0, fn=0)
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_dev(batch)
                env_source = batch.get('env_source', 'firepred')
                if isinstance(env_source, (list, tuple)):
                    env_source = env_source[0]
                out = model(batch)                              # full horizon at val
                af_logits = out['af_pred']
                af_targets = batch['af_fut']
                try:
                    wind = extract_wind(batch['env_fut'], source=env_source)
                except Exception:
                    wind = None
                val_losses_d = criterion(af_logits, af_targets,
                                         ba_init=batch.get('ba_init'), wind=wind)
                val_losses.append(float(val_losses_d['total'].item()))
                # threshold scan uses af_logits squeezed to (B,H,Hh,Ww) for simpler math
                afl = af_logits.squeeze(2)
                aft = af_targets.squeeze(2)
                for thr in thr_grid:
                    d = iou_pr_at_threshold(afl, aft, thr)
                    for k in ('tp','fp','fn'):
                        acc[thr][k] += d[k]
                d0 = iou_pr_at_threshold(afl, aft, fixed_thr)
                for k in ('tp','fp','fn'):
                    fixed_acc[k] += d0[k]

        def iou_of(a):
            denom = a['tp'] + a['fp'] + a['fn']
            return a['tp'] / denom if denom > 0 else 0.0

        def pr_of(a):
            p = a['tp'] / (a['tp'] + a['fp']) if (a['tp']+a['fp'])>0 else 0.0
            r = a['tp'] / (a['tp'] + a['fn']) if (a['tp']+a['fn'])>0 else 0.0
            f = 2*p*r / (p+r) if (p+r)>0 else 0.0
            return p, r, f

        ious = {thr: iou_of(a) for thr, a in acc.items()}
        thr_best = max(ious, key=ious.get)
        val_iou_best = ious[thr_best]
        val_iou_fixed = iou_of(fixed_acc)
        p_best, r_best, f1_best = pr_of(acc[thr_best])

        if epoch >= ema_start:
            ema.swap_out(model)

        # log row
        row = [
            epoch, H_cur,
            float(np.mean(train_losses)), float(np.mean(train_sup)), float(np.mean(train_ba)),
            float(np.mean(val_losses)),
            val_iou_fixed, val_iou_best, thr_best,
            p_best, r_best, f1_best,
            opt.param_groups[0]['lr'],
        ]
        with metrics_path.open('a', newline='') as f:
            csv.writer(f).writerow(row)

        # save & track best
        if val_iou_best > best_val_iou:
            best_val_iou = val_iou_best
            best_epoch = epoch
            # save EMA weights if active, else plain
            state = dict(model.state_dict())
            if epoch >= ema_start:
                state = {**state, **ema.shadow}
            torch.save({
                'epoch': epoch,
                'model_state': state,
                'val_iou_best_thr': val_iou_best,
                'thr_best': thr_best,
                'cfg': cfg,
            }, Path(out_dir)/'best_model.pt')
        if epoch % cfg['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_shadow': ema.shadow if epoch >= ema_start else None,
            }, Path(out_dir)/f'checkpoint_epoch{epoch:03d}.pt')

        dt = time.time() - ep_start
        print(f'ep{epoch:03d}  H={H_cur:02d}  '
              f'train={np.mean(train_losses):.3f}  val={np.mean(val_losses):.3f}  '
              f'IoU@fixed={val_iou_fixed:.4f}  IoU@best={val_iou_best:.4f} '
              f'(thr={thr_best:.2f})  P={p_best:.3f} R={r_best:.3f} F1={f1_best:.3f}  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  {dt:.0f}s')

    print(f'\n=== BEST: val_iou_best_thr = {best_val_iou:.4f} @ep {best_epoch} ===')
    return str(Path(out_dir)/'best_model.pt')


# Run!
BEST_CKPT = train_v2(cfg, out_dir='/content/outputs/run27')
print(f'\nBest checkpoint: {BEST_CKPT}')
'''
cells.append(code_cell(TRAINER_CELL))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 8 — Section 7: Validation curves
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 7 · Post-training validation metrics

Plot IoU (fixed vs best-thr), Precision/Recall/F1, best-threshold trajectory
(proof we escaped run26's 0.5 ceiling), overfit gap, and a per-threshold
IoU sweep from the best epoch."""))

VAL_CELL = r'''# ─── Section 7 · Validation curves ─────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path('/content/outputs/run27')
df = pd.read_csv(OUT_DIR / 'metrics.csv')
print(f'Loaded {len(df)} epochs of metrics.')
print(df.tail(5).to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle('FireST-GraphNet V2 (run27) — training curves', fontsize=14)

# 1. IoU — fixed vs best-thr  + history
ax = axes[0,0]
ax.plot(df['epoch'], df['val_iou_fixed'],    label='Val IoU (fixed thr=0.5)', color='tomato', alpha=0.7)
ax.plot(df['epoch'], df['val_iou_best_thr'], label='Val IoU (best thr)',     color='darkred', lw=2)
ax.axhline(0.598, color='grey',  ls=':', alpha=0.5, label='run23 0.598')
ax.axhline(0.580, color='brown', ls=':', alpha=0.5, label='run25 0.580')
ax.axhline(0.525, color='teal',  ls=':', alpha=0.5, label='run26 0.525')
b = df.loc[df['val_iou_best_thr'].idxmax()]
ax.axhline(b['val_iou_best_thr'], ls='--', color='red', alpha=0.6,
           label=f"V2 best {b['val_iou_best_thr']:.4f} @ep{int(b['epoch'])}")
ax.set_title('Val IoU — V2 vs history')
ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

# 2. Threshold trajectory (proof we escaped 0.5 ceiling)
ax = axes[0,1]
ax.plot(df['epoch'], df['val_thr_best'], marker='.', color='purple')
ax.axhline(0.50, ls='--', color='grey', alpha=0.5, label='run26 pegged @0.5')
ax.set_ylim(0.30, 0.70)
ax.set_title('Best threshold per epoch (should NOT peg @0.5 or ends)')
ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

# 3. Curriculum horizon
ax = axes[0,2]
ax.plot(df['epoch'], df['horizon_cur'], marker='.', color='teal')
ax.set_title('Curriculum horizon')
ax.set_xlabel('epoch'); ax.set_ylabel('H_cur'); ax.grid(True)

# 4. Precision / Recall / F1
ax = axes[1,0]
ax.plot(df['epoch'], df['val_prec'], label='Precision', color='C0', lw=2)
ax.plot(df['epoch'], df['val_rec'],  label='Recall',    color='C1', lw=2)
ax.plot(df['epoch'], df['val_f1'],   label='F1',        color='C2', lw=2)
ax.axhline(0.65, ls=':', color='grey', alpha=0.4, label='run26 prec=0.65')
ax.set_title('Val P / R / F1 (at best-thr)')
ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

# 5. Train vs Val loss
ax = axes[1,1]
ax.plot(df['epoch'], df['train_loss'], label='train', color='steelblue')
ax.plot(df['epoch'], df['val_loss'],   label='val',   color='darkorange')
ax.set_title('Loss')
ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

# 6. Aux BA loss trajectory
ax = axes[1,2]
ax.plot(df['epoch'], df['train_ba'], marker='.', color='darkgreen')
ax.set_title('Auxiliary monotonic BA loss')
ax.set_xlabel('epoch'); ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves_v2.png', dpi=140)
plt.show()

# ── summary ─────────────────────────────────────────────────
best = df.loc[df['val_iou_best_thr'].idxmax()]
thrs = df['val_thr_best'].value_counts().sort_index()
print('\n=== Best epoch ===')
print(f"  epoch {int(best['epoch']):3d}   H={int(best['horizon_cur']):2d}")
print(f"  Val IoU (best thr): {best['val_iou_best_thr']:.4f}  at thr={best['val_thr_best']:.2f}")
print(f"  Val IoU (fixed=0.5): {best['val_iou_fixed']:.4f}")
print(f"  P / R / F1         : {best['val_prec']:.4f} / {best['val_rec']:.4f} / {best['val_f1']:.4f}")
print('\nThreshold distribution across epochs:')
for t, n in thrs.items():
    marker = '  ← pegged at 0.5' if abs(t-0.5)<0.01 and n==len(df) else ''
    print(f'  thr={t:.2f}: {n:3d} epochs {marker}')

if len(df) >= 20:
    print(f"\nVal IoU stability (last 20 ep std): {df['val_iou_best_thr'].tail(20).std():.4f}  "
          f"(target <= 0.005 with EMA)")
'''
cells.append(code_cell(VAL_CELL))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 9 — Section 8: Inference visualisation
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 8 · Inference visualisation

Load `best_model.pt` and plot 15 horizon frames side-by-side (prediction vs
ground-truth) for 3 random validation events.  This is the most important
qualitative check — real fires should spread realistically; the prediction
should follow the spread direction/shape."""))

VIZ_CELL = r'''# ─── Section 8 · Inference visualisation ───────────────────────────────────
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
print(f"Loaded best_model.pt — epoch {ckpt['epoch']}  val_iou_best_thr={ckpt['val_iou_best_thr']:.4f}  thr={ckpt['thr_best']:.2f}")

# Rebuild model and load state (non-strict to tolerate EMA keys)
mcfg = cfg['model']
model_inf = FireSTGraphNetV2(
    in_ch_img=8, in_ch_aux=19, in_ch_static=1,
    encoder_channels=tuple(mcfg['encoder_channels']),
    gnn_patches=tuple(mcfg['v2_graph_pyramid_patches']),
    gat_heads=mcfg['v2_gat_heads'],
    lstm_hidden=mcfg['lstm_hidden'],
    lstm_layers=mcfg['lstm_layers'],
    lstm_dropout=mcfg['lstm_dropout'],
    lstm_attn_heads=mcfg['v2_lstm_attn_heads'],
    weather_ch=4, weather_emb=64,
    horizon=cfg['forecast']['horizon'],
    dropout=mcfg['decoder_dropout'],
    aux_ba=mcfg['v2_aux_ba_head'],
).to(DEVICE)
missing, unexpected = model_inf.load_state_dict(ckpt['model_state'], strict=False)
print(f'  missing keys: {len(missing)}   unexpected: {len(unexpected)}')
model_inf.eval()

# Pull a few val samples
val_base = TSSatFireForecastDataset(
    root=cfg['data']['root_val'], split='val',
    T_in=cfg['forecast']['T_in'], horizon=cfg['forecast']['horizon'],
    img_size=cfg['data']['img_size'],
)
import numpy as np
np.random.seed(0)
pick = np.random.choice(len(val_base), size=min(3, len(val_base)), replace=False)

cmap_bin = ListedColormap(['#1b2631', '#ff7e33'])   # dark / fire-orange
thr = float(ckpt['thr_best'])
H = cfg['forecast']['horizon']

for p_idx, s_idx in enumerate(pick):
    sample = val_base[int(s_idx)]
    batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else [v])
             for k, v in sample.items()}
    with torch.no_grad():
        out = model_inf(batch)
    af_logit = out['af_pred'][0, :, 0].cpu().numpy()   # (H, Hh, Ww)
    af_prob  = 1/(1+np.exp(-af_logit))
    af_pred  = (af_prob > thr).astype(np.uint8)
    af_gt    = sample['af_fut'].squeeze(1).numpy().astype(np.uint8)

    rows = 2; cols = H
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.4, rows*1.6))
    fig.suptitle(f'Sample {p_idx+1}  (val idx={int(s_idx)})  thr={thr:.2f}',
                 fontsize=11, y=1.02)
    for t in range(H):
        axes[0, t].imshow(af_pred[t], cmap=cmap_bin, vmin=0, vmax=1, interpolation='nearest')
        axes[0, t].set_title(f't+{t+1}', fontsize=7)
        axes[0, t].set_xticks([]); axes[0, t].set_yticks([])
        axes[1, t].imshow(af_gt[t], cmap=cmap_bin, vmin=0, vmax=1, interpolation='nearest')
        axes[1, t].set_xticks([]); axes[1, t].set_yticks([])
    axes[0, 0].set_ylabel('Pred',  fontsize=8)
    axes[1, 0].set_ylabel('GT',    fontsize=8)
    plt.tight_layout()
    outp = OUT_DIR / f'inference_sample{p_idx+1}.png'
    plt.savefig(outp, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  saved {outp}')

# Per-horizon IoU breakdown on a small val slice
print('\n=== Per-horizon IoU on 16 val samples ===')
per_h_tp = np.zeros(H); per_h_fp = np.zeros(H); per_h_fn = np.zeros(H)
count = 0
with torch.no_grad():
    for i, idx in enumerate(np.random.choice(len(val_base), size=min(16, len(val_base)), replace=False)):
        sample = val_base[int(idx)]
        batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else [v])
                 for k, v in sample.items()}
        out = model_inf(batch)
        af_prob = torch.sigmoid(out['af_pred'][0, :, 0]).cpu().numpy()
        gt = sample['af_fut'].squeeze(1).numpy()
        pred = (af_prob > thr).astype(np.float32)
        gt_b = (gt > 0.5).astype(np.float32)
        for t in range(H):
            per_h_tp[t] += ((pred[t]*gt_b[t]).sum())
            per_h_fp[t] += ((pred[t]*(1-gt_b[t])).sum())
            per_h_fn[t] += (((1-pred[t])*gt_b[t]).sum())
        count += 1
ious_h = per_h_tp / np.clip(per_h_tp + per_h_fp + per_h_fn, 1e-6, None)
for t in range(H):
    print(f'  t+{t+1:2d}:  IoU={ious_h[t]:.4f}')

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(range(1, H+1), ious_h, marker='o', color='C3')
ax2.set_xlabel('horizon'); ax2.set_ylabel('IoU'); ax2.grid(True)
ax2.set_title('Per-horizon IoU (16 val samples, best-thr)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'per_horizon_iou.png', dpi=140)
plt.show()
'''
cells.append(code_cell(VIZ_CELL))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 10 — Section 9: Export to Drive
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md_cell(r"""## Section 9 · Export best checkpoint to Drive

Copy `best_model.pt`, `metrics.csv`, and training curve PNGs back into your
Google Drive project folder so they survive the Colab session."""))

EXPORT_CELL = r'''# ─── Section 9 · Export to Drive ───────────────────────────────────────────
import shutil

DRIVE_OUT = Path(PROJECT_PATH) / 'output' / 'run27'
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

LOCAL_OUT = Path('/content/outputs/run27')

exported = []
for fname in ['best_model.pt', 'metrics.csv',
              'training_curves_v2.png', 'per_horizon_iou.png',
              'inference_sample1.png', 'inference_sample2.png', 'inference_sample3.png']:
    src = LOCAL_OUT / fname
    if src.exists():
        dst = DRIVE_OUT / fname
        shutil.copy2(src, dst)
        sz_mb = dst.stat().st_size / 1024 / 1024
        exported.append((fname, sz_mb))

# Also copy the config used
shutil.copy2('/content/config_run27.yaml', DRIVE_OUT / 'config_run27.yaml')

print(f'\n✓ Exported to Drive: {DRIVE_OUT}')
for name, sz in exported:
    print(f'    {name:35s}  {sz:7.2f} MB')
print(f'    config_run27.yaml')

# Final summary
import pandas as pd
df = pd.read_csv(LOCAL_OUT / 'metrics.csv')
best = df.loc[df['val_iou_best_thr'].idxmax()]
print(f'\n════════════════════════════════════════════════════════════')
print(f'  FireST-GraphNet V2 (run27) — FINAL RESULTS')
print(f'════════════════════════════════════════════════════════════')
print(f'  best epoch       : {int(best["epoch"])}   H_cur={int(best["horizon_cur"])}')
print(f'  val IoU (best)   : {best["val_iou_best_thr"]:.4f}   @thr={best["val_thr_best"]:.2f}')
print(f'  val IoU (fixed)  : {best["val_iou_fixed"]:.4f}')
print(f'  P / R / F1       : {best["val_prec"]:.4f} / {best["val_rec"]:.4f} / {best["val_f1"]:.4f}')
print(f'  vs history       : run23=0.598  run25=0.580  run26=0.525')
print(f'  Δ vs run26 (+)   : {best["val_iou_best_thr"]-0.525:+.4f}')
print(f'  Δ vs run23 best  : {best["val_iou_best_thr"]-0.598:+.4f}')
print(f'  thr_best spread  : '
      f'{df["val_thr_best"].min():.2f} .. {df["val_thr_best"].max():.2f}'
      f'  (run26 was pegged @0.50)')
if len(df) >= 20:
    print(f'  last-20-ep std   : {df["val_iou_best_thr"].tail(20).std():.4f}  (EMA target <=0.005)')
print(f'════════════════════════════════════════════════════════════')
'''
cells.append(code_cell(EXPORT_CELL))

# ═══════════════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════════════
nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language':     'python',
            'name':         'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.11',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

Path(DST).parent.mkdir(parents=True, exist_ok=True)
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Wrote {DST}')
print(f'  cells: {len(cells)}  (expected 11: 1 report + 10 sections)')
