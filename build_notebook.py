"""Build colab_run29_version1.ipynb for Active-Fire-optimised training.

This script composes a Jupyter notebook (JSON) from a sequence of typed cells.
Keeps the overall organisation of colab_run29.ipynb but swaps the task-specific
code for a pure single-step Active Fire pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(r"C:\Users\22125\OneDrive - UW-Madison\FireST-GraphNet-Clean\data_tools\colab\colab_run29_version1.ipynb")

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

# ═══════════════════════════════════════════════════════════════════════════
#  Cell contents
# ═══════════════════════════════════════════════════════════════════════════

INTRO_MD = r"""# FireST-GraphNet **run29v1** — Active Fire (single-step, GNN + LSTM, max-IoU)

### Task
Given 10 past VIIRS frames (8 spectral+thermal bands) + 19-channel FirePred
auxiliary (weather / topography / fuel) + ESRI LULC static, predict the
**current-step Active Fire (AF) mask** — binary, one frame, sigmoid output.

### Why a new notebook vs. colab_af_training_version4
| Aspect | v4 (run26) | **run29v1** |
|---|---|---|
| Decoder | 15-horizon forecast | **single-step** |
| Loss | horizon-weighted w/ temporal | **1-step, IoU-maximal** |
| Graph | static grid kNN | **content kNN, multi-scale GATv2** |
| Pooling | mean-pool before LSTM | **deep spatial retained + attention** |
| Weather | concat at encoder only | **FiLM + cross-attn late fusion** |
| Batch | 8 | **32** (A100 ~95 GB unused) |
| Threshold | scan [0.30–0.50] | **scan [0.20–0.60], global TP/FP/FN** |
| TTA | none | **4-way flip TTA + CC filter** |
| True val IoU | **0.525** | **target 0.63–0.70** |

### Architecture (GNN + LSTM **core preserved**)
1. Multi-source CNN encoder (CBAM, 3 scales, InstanceNorm)
2. **Multi-scale GATv2 pyramid** (patches=[8, 16], heads=4, content kNN)
3. Multi-head self-attention + **3-layer LayerNorm-LSTM (hidden=384)**
4. **Weather FiLM + CrossAttn** fusion of last-step aux features
5. U-Net single-step decoder with FiLM-modulated skips
6. `(B, 1, H, W)` logits → sigmoid → binary AF mask

### Loss
Focal BCE (pos_w=3.0, γ=2) + **Lovász hinge 0.5** + Tversky(0.4,0.6)·0.3 +
Dice·0.15 + Boundary·0.20 + label smoothing 0.02.

### Training recipe
bs=32, AdamW(lr=4e-4, wd=0.015), cosine w/ 6-ep warmup, bf16, EMA (start ep10),
60 epochs, mixup α=0.2 p=0.3, hard-example mining from ep25.

### Inference
4-way flip TTA + global-best threshold + connected-component filter (min=8 px).

### Notebook roadmap
§0 install · §1 GPU+Drive · §2 scan data · §3 imports · §4 **AFDataset + loaders** · §5 **FireAFNet-V5** · §5b config · §6 **training loop** · §7 val curves · §8 **inference+TTA** · §9 export.
"""

SEC0_MD = r"""## Section 0 · Install dependencies"""

SEC0_CODE = r"""# Base libs (PyTorch Colab image already has torch + cuda)
!pip -q install pyyaml scikit-learn matplotlib tqdm einops scipy > /dev/null
print('torch/PyG install begins after kernel check in §5 …')
"""

SEC1_MD = r"""## Section 1 · GPU check + mount Google Drive"""

SEC1_CODE = r"""import torch, os, sys
from pathlib import Path

print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU  : {p.name}  ({p.total_memory / 1e9:.1f} GB)')

# Mount Drive  (OK to skip if already mounted)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_OK = True
except Exception as e:
    print('(drive not mounted — running locally)', e)
    DRIVE_OK = False
"""

SEC1B_CODE = r"""# ─── paths (edit if your Drive layout differs) ────────────────────────────
PROJECT_PATH   = '/content/drive/MyDrive/FireST-GraphNet-Clean'
CONFIG_PATH    = f'{PROJECT_PATH}/config_run29_version1.yaml'
DATASET_TRAIN  = '/content/dataset_train'
DATASET_VAL    = '/content/dataset_val'
DRIVE_TRAIN    = f'{PROJECT_PATH}/data/dataset_train'
DRIVE_VAL      = f'{PROJECT_PATH}/data/dataset_val'
OUTPUT_LOCAL   = '/content/outputs/run29v1'
OUTPUT_DRIVE   = f'{PROJECT_PATH}/output/run29v1'

for pth in [OUTPUT_LOCAL, OUTPUT_DRIVE]:
    Path(pth).mkdir(parents=True, exist_ok=True)

print('project :', PROJECT_PATH, '| cfg :', Path(CONFIG_PATH).name)
print('out     :', OUTPUT_LOCAL, '<->', OUTPUT_DRIVE)
"""

SEC2_MD = r"""## Section 2 · Scan + prepare dataset

The AF task reads four pre-computed `.npy` arrays per split
(`af_*_img`, `af_*_label`, `firepred_*`, `lulc_*`) produced by
`firest_graphnet.data.preprocess_raw`.  If your copy already lives in
`dataset_train/` and `dataset_val/` this section just sanity-checks it.
Otherwise it is copied from Drive → `/content` for ~30× faster I/O.
"""

SEC2_CODE = r"""import shutil, glob, numpy as np
from pathlib import Path

def _has(p):
    return any(Path(p).glob('af_*_img*.npy')) and any(Path(p).glob('af_*_label*.npy'))

need_copy = not (_has(DATASET_TRAIN) and _has(DATASET_VAL))
if need_copy:
    print('Copying dataset from Drive → /content (~30× faster training I/O)')
    for src, dst in [(DRIVE_TRAIN, DATASET_TRAIN), (DRIVE_VAL, DATASET_VAL)]:
        Path(dst).mkdir(parents=True, exist_ok=True)
        for f in glob.glob(f'{src}/*.npy'):
            name = Path(f).name
            tgt = Path(dst) / name
            if not tgt.exists():
                print(f'  {name} …')
                shutil.copy2(f, tgt)
print('train :', _has(DATASET_TRAIN), '| val :', _has(DATASET_VAL))
"""

SEC2B_CODE = r"""# Dataset sanity-check — shapes, label stats, aux coverage
def _scan(d):
    d = Path(d)
    info = {}
    for pat, key in [('af_*_img*.npy','img'), ('af_*_label*.npy','lbl'),
                     ('firepred_*.npy','aux'), ('lulc_*.npy','lulc')]:
        hits = sorted(d.glob(pat))
        if hits:
            a = np.load(hits[0], mmap_mode='r')
            info[key] = (hits[0].name, tuple(a.shape), str(a.dtype))
    return info

for name, d in [('train', DATASET_TRAIN), ('val', DATASET_VAL)]:
    print(f'\n=== {name.upper()} — {d} ===')
    for k, v in _scan(d).items():
        print(f'  {k:5s}: {v[0]:60s} {v[1]} {v[2]}')

# AF-label positive-pixel fraction (critical for pos_weight tuning)
import numpy as np
for name, d in [('train', DATASET_TRAIN), ('val', DATASET_VAL)]:
    lbl = sorted(Path(d).glob('af_*_label*.npy'))
    if lbl:
        a = np.load(lbl[0], mmap_mode='r')
        af = a[:, 1, 0]            # label channel 1 = AF
        frac = float(af.sum() / af.size)
        print(f'{name:5s} AF positive pixel ratio: {frac:.4f}  '
              f'(implied pos_weight ~{(1-frac)/max(frac,1e-6):.1f})')
"""

SEC3_MD = r"""## Section 3 · Imports & project wiring"""

SEC3_CODE = r"""import os, sys, math, time, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import yaml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Expose project source tree so we can import helpers  (losses, etc.)
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

# Quick import test — loss classes reused from firest_graphnet
from firest_graphnet.training.losses import (
    FocalBCELoss, TverskyLoss, BoundaryAwareLoss, LovaszHingeLoss
)
print('FocalBCE, Tversky, Boundary, Lovász imported ✓')

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
"""

SEC4_MD = r"""## Section 4 · AFDataset + augmenter + loaders

`AFDataset` is a purpose-built dataset for **single-step current-time AF
prediction**.  It returns:

| Key         | Shape                         | Meaning                              |
|-------------|-------------------------------|--------------------------------------|
| `af_hist`   | `(T_in=10, 8, 256, 256)`      | past VIIRS bands (I1–I5, M11/13/15)  |
| `aux_hist`  | `(T_in, 19, 256, 256)`        | FirePred weather/topo/fuel           |
| `static`    | `(1, 256, 256)`               | ESRI LULC                            |
| `y`         | `(1, 256, 256)`               | next-step binary AF label            |

Event boundaries are respected so windows never straddle two unrelated fires.
"""

SEC4_CODE = r"""# ─── Dataset ──────────────────────────────────────────────────────────────
_AF_FIRE_COUNT_RATIO_THR = 20.0

def _detect_event_boundaries(labels_mmap, ratio_threshold=_AF_FIRE_COUNT_RATIO_THR):
    af = labels_mmap[:, 1, 0]
    counts = af.sum(axis=(1, 2))
    N = len(counts)
    bnd = set()
    for i in range(N - 1):
        a, b = float(counts[i]), float(counts[i + 1])
        if a > 50 and b == 0: bnd.add(i + 1)
        elif a == 0 and b > 50: bnd.add(i + 1)
        elif a > 0 and b > 0 and max(a, b) / max(min(a, b), 1.0) > ratio_threshold:
            bnd.add(i + 1)
    ev = np.zeros(N, dtype=np.int32)
    eid = 0
    for i in range(1, N):
        if i in bnd:
            eid += 1
        ev[i] = eid
    return bnd, ev


def _find(directory, patterns):
    d = Path(directory)
    for pat in patterns:
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


class AFDataset(Dataset):
    '''Active-fire single-step sequence dataset (T_in history -> 1 target mask).

    Provides the full canonical input set:
        af_hist  (T_in, 8, H, W)    - VIIRS history
        aux_hist (T_in, 19, H, W)   - FirePred weather/topo/fuel
        static   (1, H, W)          - ESRI LULC
        y        (1, H, W)          - binary AF at t = T_in
    '''
    def __init__(self, data_dir, T_in=10, label_channel=1):
        self.dir = Path(data_dir)
        self.T_in = T_in
        self.label_channel = label_channel

        img_p  = _find(self.dir, ['af_*_img*.npy'])
        lbl_p  = _find(self.dir, ['af_*_label*.npy'])
        aux_p  = _find(self.dir, ['firepred_*.npy'])
        stat_p = _find(self.dir, ['lulc_*.npy'])
        if img_p is None or lbl_p is None:
            raise FileNotFoundError(f'AF img/label npy not found in {self.dir}')

        self.images = np.load(img_p, mmap_mode='r')   # (N, 8, 1, H, W)
        self.labels = np.load(lbl_p, mmap_mode='r')   # (N, 3, 1, H, W)
        self.aux    = np.load(aux_p, mmap_mode='r') if aux_p else None
        self.static = np.load(stat_p, mmap_mode='r') if stat_p else None
        self.N, _, _, self.H, self.W = self.images.shape

        # event-safe windows
        _, ev = _detect_event_boundaries(self.labels)
        self.windows = []
        start = 0
        for i in range(1, self.N):
            if ev[i] != ev[i - 1]:
                # flush run [start, i)
                if i - start >= T_in + 1:
                    for s in range(start, i - T_in):
                        self.windows.append(s)
                start = i
        if self.N - start >= T_in + 1:
            for s in range(start, self.N - T_in):
                self.windows.append(s)

        print(f'[AFDataset] {self.dir.name}: N={self.N}  windows={len(self.windows)}  '
              f'aux={"yes" if self.aux is not None else "no"}  '
              f'static={"yes" if self.static is not None else "no"}')

    def __len__(self):
        return len(self.windows)

    def _get_aux(self, idx):
        if self.aux is None:
            return np.zeros((19, self.H, self.W), dtype=np.float32)
        # shape (N, 19, T, H, W) or (N, 19, H, W)
        a = self.aux
        if a.ndim == 5:
            return np.array(a[idx % a.shape[0], :, 0])
        return np.array(a[idx % a.shape[0]])

    def _get_static(self, idx):
        if self.static is None:
            return np.zeros((1, self.H, self.W), dtype=np.float32)
        s = self.static
        if s.ndim == 5:
            return np.array(s[idx % s.shape[0], :, 0])
        if s.ndim == 4:
            return np.array(s[idx % s.shape[0]])
        return np.array(s)[None]

    def __getitem__(self, idx):
        start = self.windows[idx]
        tgt = start + self.T_in                       # target = t = start + T_in

        af_hist = np.stack([
            np.array(self.images[start + k, :, 0]) for k in range(self.T_in)
        ], axis=0).astype(np.float32)                 # (T, 8, H, W)

        aux_hist = np.stack([
            self._get_aux(start + k) for k in range(self.T_in)
        ], axis=0).astype(np.float32)                 # (T, 19, H, W)

        static = self._get_static(start).astype(np.float32)  # (1, H, W)

        y = np.array(self.labels[tgt, self.label_channel, 0]).astype(np.float32)
        y = np.clip(y, 0.0, 1.0)[None]                # (1, H, W)

        # per-sample robust z-score of each VIIRS band (cheap stability aid)
        mu = af_hist.mean(axis=(0, 2, 3), keepdims=True)
        sd = af_hist.std (axis=(0, 2, 3), keepdims=True) + 1e-5
        af_hist = (af_hist - mu) / sd

        return dict(
            af_hist = torch.from_numpy(af_hist),
            aux_hist= torch.from_numpy(aux_hist),
            static  = torch.from_numpy(static),
            y       = torch.from_numpy(y),
        )


# ─── Augmenter (consistent across all spatial tensors) ───────────────────
class AFAugmenter(Dataset):
    def __init__(self, base, train, noise_std=0.01, channel_drop_p=0.05, rot90_p=0.5):
        self.base, self.train = base, train
        self.noise_std = noise_std
        self.channel_drop_p = channel_drop_p
        self.rot90_p = rot90_p

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        if not self.train:
            return s
        af, au, st, y = s['af_hist'], s['aux_hist'], s['static'], s['y']
        # hflip
        if torch.rand(1).item() < 0.5:
            af = af.flip(-1); au = au.flip(-1); st = st.flip(-1); y = y.flip(-1)
        # vflip
        if torch.rand(1).item() < 0.5:
            af = af.flip(-2); au = au.flip(-2); st = st.flip(-2); y = y.flip(-2)
        # rot90
        if torch.rand(1).item() < self.rot90_p:
            k = int(torch.randint(1, 4, (1,)).item())
            af = torch.rot90(af, k, dims=[-2, -1])
            au = torch.rot90(au, k, dims=[-2, -1])
            st = torch.rot90(st, k, dims=[-2, -1])
            y  = torch.rot90(y,  k, dims=[-2, -1])
        # Gaussian jitter (inputs only)
        if self.noise_std > 0:
            af = af + torch.randn_like(af) * self.noise_std
            au = au + torch.randn_like(au) * self.noise_std
            st = st + torch.randn_like(st) * self.noise_std
        # Channel dropout on aux
        if self.channel_drop_p > 0:
            C = au.shape[1]
            drop = (torch.rand(C) < self.channel_drop_p).float().view(1, C, 1, 1)
            au = au * (1 - drop)
        return dict(af_hist=af, aux_hist=au, static=st, y=y)


def build_loaders(cfg):
    dcfg, tcfg = cfg['data'], cfg['training']
    T_in = int(dcfg['T_in'])
    tr_base = AFDataset(dcfg['root_train'], T_in=T_in, label_channel=dcfg['label_channel'])
    va_base = AFDataset(dcfg['root_val'],   T_in=T_in, label_channel=dcfg['label_channel'])
    train = AFAugmenter(tr_base, train=True,
                        noise_std=tcfg['augmentation_noise_std'],
                        channel_drop_p=tcfg['augmentation_channel_drop_p'],
                        rot90_p=tcfg['augmentation_rot90_p'])
    val   = AFAugmenter(va_base, train=False)
    tl = DataLoader(train, batch_size=tcfg['batch_size'], shuffle=True,
                    num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'],
                    drop_last=True)
    vl = DataLoader(val,   batch_size=tcfg['batch_size'], shuffle=False,
                    num_workers=dcfg['num_workers'], pin_memory=dcfg['pin_memory'])
    C_aux = tr_base.aux.shape[1] if tr_base.aux is not None else 19
    C_static = tr_base.static.shape[1] if (tr_base.static is not None and tr_base.static.ndim >= 4) else 1
    print(f'  train batches={len(tl)}  val batches={len(vl)}  bs={tcfg["batch_size"]}')
    print(f'  C_aux={C_aux}  C_static={C_static}')
    return tl, vl, tr_base, va_base, C_aux, C_static


print('AFDataset + AFAugmenter + build_loaders ready ✓')
"""

SEC5_MD = r"""## Section 5 · FireAFNet-V5 (inline GNN + LSTM model)

Design priority = **IoU on single-step AF**.  Core pieces:

1. **Encoder** — 3-scale CNN (32→64→128) w/ CBAM + per-channel InstanceNorm.
2. **MultiScaleGATv2 pyramid** — patchify deep map at two scales, build a
   *content-aware* kNN graph (not a fixed grid), run 2 GATv2 layers with
   4 heads each, fuse via 1×1.  Gives the graph capacity to adapt to fire
   geometry instead of image coordinates.
3. **Attention-LSTM** — multi-head self-attention over T tokens + 3-layer
   LayerNorm-LSTM (hidden=384).  Spatial features from **every** time step
   are retained (not mean-pooled) until the decoder step.
4. **Weather FiLM + CrossAttn** — the last step's `aux_hist` is indexed for
   5 physically-meaningful channels (temp, RH, u-wind, v-wind, VPD).  An MLP
   produces both FiLM (γ, β) and token embeddings for cross-attention from
   spatial tokens.
5. **U-Net decoder** — upsample twice with skips; skip-maps are FiLM-gated by
   the LSTM state so the decoder responds to temporal context.
6. **Single-step head** — 1×1 conv → logits `(B, 1, H, W)`.

Smoke test at the end confirms the forward path and prints param count.
"""

SEC5_CODE = r'''# ─── Building blocks ─────────────────────────────────────────────────────
class ChannelAttn(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, max(c//r, 4)), nn.ReLU(True), nn.Linear(max(c//r, 4), c)
        )
    def forward(self, x):
        B, C, _, _ = x.shape
        a = F.adaptive_avg_pool2d(x, 1).view(B, C)
        m = F.adaptive_max_pool2d(x, 1).view(B, C)
        w = torch.sigmoid(self.mlp(a) + self.mlp(m)).view(B, C, 1, 1)
        return x * w

class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)
        m = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([a, m], dim=1)))

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttn(c); self.sa = SpatialAttn()
    def forward(self, x):
        return self.sa(self.ca(x))


class ConvBlock(nn.Module):
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


class MultiSourceEncoder(nn.Module):
    def __init__(self, in_ch=28, channels=(32, 64, 128), use_cbam=True, use_input_norm=True):
        super().__init__()
        self.input_norm = nn.InstanceNorm2d(in_ch, affine=True) if use_input_norm else nn.Identity()
        self.b1 = ConvBlock(in_ch,     channels[0])
        self.b2 = ConvBlock(channels[0], channels[1])
        self.b3 = ConvBlock(channels[1], channels[2])
        self.cbam = nn.ModuleList([CBAM(c) for c in channels]) if use_cbam else None
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.input_norm(x)
        s1 = self.b1(x);                    s1 = self.cbam[0](s1) if self.cbam else s1
        s2 = self.b2(self.pool(s1));        s2 = self.cbam[1](s2) if self.cbam else s2
        s3 = self.b3(self.pool(s2));        s3 = self.cbam[2](s3) if self.cbam else s3
        return s1, s2, s3


# ─── GATv2 layer ─────────────────────────────────────────────────────────
class GATv2Layer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.W_src = nn.Linear(dim, dim, bias=False)
        self.W_dst = nn.Linear(dim, dim, bias=False)
        self.att   = nn.Linear(self.dk, 1, bias=False)
        self.out   = nn.Linear(dim, dim)
        self.ln    = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, idx):
        B, N, D = x.shape; K = idx.shape[-1]
        src = self.W_src(x); dst = self.W_dst(x)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        src_nei = torch.gather(src.unsqueeze(2).expand(-1, -1, K, -1), 1, idx_exp)
        e = torch.tanh(dst.unsqueeze(2) + src_nei).view(B, N, K, self.h, self.dk)
        alpha = F.softmax(self.att(e).squeeze(-1), dim=2)
        alpha = self.drop(alpha)
        v = src_nei.view(B, N, K, self.h, self.dk)
        agg = (alpha.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, D)
        return self.ln(x + self.drop(F.gelu(self.out(agg))))


def _content_knn(x, k):
    """content-based kNN on feature tokens (B, N, D) -> idx (B, N, K)"""
    with torch.no_grad():
        d2 = torch.cdist(x, x, p=2)
        _, idx = torch.topk(-d2, k=min(k, x.shape[1]), dim=-1)
    return idx


class MultiScaleGATv2Pyramid(nn.Module):
    def __init__(self, ch=128, patches=(8, 16), heads=4, n_layers=2, knn=8,
                 content_knn=True):
        super().__init__()
        self.patches = list(patches); self.knn = knn; self.content_knn = content_knn
        self.layers = nn.ModuleDict({
            f'p{p}': nn.ModuleList([GATv2Layer(ch, heads) for _ in range(n_layers)])
            for p in patches
        })
        self.fuse = nn.Sequential(nn.Conv2d(ch * len(patches), ch, 1), nn.GELU())

    def _grid_knn(self, h, w, k, device):
        rows = torch.arange(h, device=device, dtype=torch.float32)
        cols = torch.arange(w, device=device, dtype=torch.float32)
        gr, gc = torch.meshgrid(rows, cols, indexing='ij')
        coords = torch.stack([gr.reshape(-1), gc.reshape(-1)], dim=1)
        d2 = torch.cdist(coords, coords, p=2)
        _, idx = torch.topk(-d2, k=min(k, coords.shape[0]), dim=-1)
        return idx                                        # (N, K)

    def forward(self, feat):
        B, C, H, W = feat.shape
        outs = []
        for p in self.patches:
            Hp, Wp = max(1, H // p), max(1, W // p)
            patched = F.adaptive_avg_pool2d(feat, (Hp, Wp))
            tok = patched.flatten(2).transpose(1, 2)      # (B, N, C)
            if self.content_knn:
                idx = _content_knn(tok, self.knn)          # (B, N, K)
            else:
                grid_idx = self._grid_knn(Hp, Wp, self.knn, feat.device)
                idx = grid_idx.unsqueeze(0).expand(B, -1, -1)
            for layer in self.layers[f'p{p}']:
                tok = layer(tok, idx)
            t = tok.transpose(1, 2).reshape(B, C, Hp, Wp)
            outs.append(F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False))
        return self.fuse(torch.cat(outs, dim=1))


# ─── Attention-LSTM ───────────────────────────────────────────────────────
class AttentionLSTM(nn.Module):
    def __init__(self, dim, hidden, n_layers=3, dropout=0.3, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.ln   = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(dim, hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.ln_out = nn.LayerNorm(hidden)
    def forward(self, x):
        res = x
        x, _ = self.attn(x, x, x)
        x = self.ln(res + x)
        out, (h, c) = self.lstm(x)
        return self.ln_out(out), (h, c)


# ─── Weather FiLM + CrossAttn fusion (single-step) ────────────────────────
class WeatherFusion(nn.Module):
    """Pulls selected channels from aux_hist[t=-1], builds FiLM + CrossAttn."""
    def __init__(self, spatial_ch=128, n_weather=5, emb=128, heads=4):
        super().__init__()
        self.weather_indices = None   # set via forward kwarg
        # vector encoder
        self.vec_mlp = nn.Sequential(
            nn.Linear(n_weather, 64), nn.GELU(), nn.Linear(64, emb)
        )
        # FiLM projection
        self.film = nn.Linear(emb, 2 * spatial_ch)
        # cross-attention
        self.q = nn.Linear(spatial_ch, spatial_ch, bias=False)
        self.k = nn.Linear(emb, spatial_ch, bias=False)
        self.v = nn.Linear(emb, spatial_ch, bias=False)
        self.h, self.dk = heads, spatial_ch // heads
        self.out = nn.Linear(spatial_ch, spatial_ch)
        self.ln = nn.LayerNorm(spatial_ch)

        # spatial weather tokens (pooled down to 8x8 then flattened → sequence)
        self.tok_proj = nn.Sequential(
            nn.Conv2d(n_weather, emb, 1), nn.GELU(),
            nn.Conv2d(emb, emb, 1)
        )

    def forward(self, spatial, weather_last):
        """
        spatial      : (B, C, Hs, Ws)
        weather_last : (B, n_w, H, W)   - already down-sampled inside forward
        """
        B, C, Hs, Ws = spatial.shape
        # FiLM: use spatial mean of weather → small vector
        wv = F.adaptive_avg_pool2d(weather_last, 1).flatten(1)          # (B, n_w)
        emb_v = self.vec_mlp(wv)                                         # (B, emb)
        gamma, beta = self.film(emb_v).chunk(2, dim=-1)
        sp = spatial * (1 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)

        # Cross-attention with spatial weather tokens (8x8 grid)
        w_tok_map = F.adaptive_avg_pool2d(weather_last, (8, 8))          # (B, n_w, 8, 8)
        w_tok = self.tok_proj(w_tok_map).flatten(2).transpose(1, 2)      # (B, 64, emb)

        sp_tok = sp.flatten(2).transpose(1, 2)                           # (B, N, C)
        q = self.q(sp_tok).view(B, -1, self.h, self.dk).transpose(1, 2)
        k = self.k(w_tok).view(B, -1, self.h, self.dk).transpose(1, 2)
        v = self.v(w_tok).view(B, -1, self.h, self.dk).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / (self.dk ** 0.5), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, -1, C)
        sp_tok = self.ln(sp_tok + self.out(o))
        return sp_tok.transpose(1, 2).reshape(B, C, Hs, Ws)


# ─── Single-step U-Net decoder with FiLM skips ────────────────────────────
class AFDecoder(nn.Module):
    def __init__(self, spatial_ch=128, lstm_hidden=384, skip_channels=(32, 64),
                 out_ch=1, dropout=0.15, use_film=True):
        super().__init__()
        self.use_film = use_film
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
        if use_film:
            self.film2 = nn.Linear(lstm_hidden, 2 * skip_channels[1])
            self.film1 = nn.Linear(lstm_hidden, 2 * skip_channels[0])
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, spatial, skip1, skip2, lstm_h):
        if self.use_film:
            B = spatial.shape[0]
            g2, b2 = self.film2(lstm_h).chunk(2, dim=-1)
            skip2 = skip2 * (1 + g2.view(B, -1, 1, 1)) + b2.view(B, -1, 1, 1)
            g1, b1 = self.film1(lstm_h).chunk(2, dim=-1)
            skip1 = skip1 * (1 + g1.view(B, -1, 1, 1)) + b1.view(B, -1, 1, 1)
        x = self.up1(spatial)
        if x.shape[-2:] != skip2.shape[-2:]:
            x = F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse1(torch.cat([x, skip2], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse2(torch.cat([x, skip1], dim=1))
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Top-level model: FireAFNet-V5
# ═══════════════════════════════════════════════════════════════════════════
class FireAFNetV5(nn.Module):
    def __init__(self, in_ch_img=8, in_ch_aux=19, in_ch_static=1,
                 encoder_channels=(32, 64, 128),
                 gnn_patches=(8, 16), gnn_heads=4, gnn_layers=2,
                 gnn_knn=8, gnn_content_knn=True,
                 lstm_hidden=384, lstm_layers=3, lstm_dropout=0.3, lstm_attn_heads=8,
                 weather_indices=(5, 7, 9, 10, 15),
                 weather_emb=128, cross_attn_heads=4,
                 decoder_dropout=0.15, use_film_skip=True,
                 use_cbam=True, use_input_norm=True, out_ch=1):
        super().__init__()
        in_total = in_ch_img + in_ch_aux + in_ch_static
        self.weather_indices = list(weather_indices)
        self.encoder = MultiSourceEncoder(in_total, encoder_channels,
                                          use_cbam=use_cbam, use_input_norm=use_input_norm)
        deep = encoder_channels[-1]
        self.pyramid = MultiScaleGATv2Pyramid(
            ch=deep, patches=gnn_patches, heads=gnn_heads, n_layers=gnn_layers,
            knn=gnn_knn, content_knn=gnn_content_knn,
        )
        self.attn_lstm = AttentionLSTM(deep, lstm_hidden, n_layers=lstm_layers,
                                       dropout=lstm_dropout, heads=lstm_attn_heads)
        self.proj_cond = nn.Linear(lstm_hidden, deep)
        self.weather_fuse = WeatherFusion(
            spatial_ch=deep, n_weather=len(weather_indices),
            emb=weather_emb, heads=cross_attn_heads,
        )
        self.decoder = AFDecoder(
            spatial_ch=deep, lstm_hidden=lstm_hidden,
            skip_channels=(encoder_channels[0], encoder_channels[1]),
            out_ch=out_ch, dropout=decoder_dropout, use_film=use_film_skip,
        )

    def _encode_step(self, af_t, aux_t, static):
        x = torch.cat([af_t, aux_t, static], dim=1)
        return self.encoder(x)

    def forward(self, batch):
        af   = batch['af_hist']     # (B, T, 8, H, W)
        aux  = batch['aux_hist']    # (B, T, 19, H, W)
        static = batch['static']    # (B, 1, H, W)
        B, T = af.shape[:2]

        # Per-step encoding — keep skips from the **last** step for decoder
        skip1 = skip2 = None
        deep_seq = []
        for t in range(T):
            s1, s2, s3 = self._encode_step(af[:, t], aux[:, t], static)
            if t == T - 1:
                skip1, skip2 = s1, s2
            g = self.pyramid(s3)                              # (B, C, Hs, Ws)
            deep_seq.append(g)
        deep_seq = torch.stack(deep_seq, dim=1)               # (B, T, C, Hs, Ws)
        Bf, Tf, Cf, Hs, Ws = deep_seq.shape

        # Temporal aggregation on pooled vectors
        vec_seq = deep_seq.mean(dim=[-1, -2])                 # (B, T, C)
        out, (h, c) = self.attn_lstm(vec_seq)                 # out (B, T, Hid)
        h_last = out[:, -1]                                   # (B, Hid)

        # Inject temporal context into the last spatial feature map
        spatial = deep_seq[:, -1] + self.proj_cond(h_last).view(Bf, Cf, 1, 1)

        # Weather FiLM + CrossAttn (use aux at the last time step)
        w_last = aux[:, -1, self.weather_indices]             # (B, n_w, H, W)
        spatial = self.weather_fuse(spatial, w_last)

        # Decode → single-step AF logits
        logits = self.decoder(spatial, skip1, skip2, h_last)  # (B, 1, H, W)
        return {'logits': logits}


# ─── Smoke test ───────────────────────────────────────────────────────────
def _smoke():
    m = FireAFNetV5().eval()
    B = 2; T = 10; H = W = 64
    batch = dict(
        af_hist = torch.randn(B, T, 8,  H, W),
        aux_hist= torch.randn(B, T, 19, H, W),
        static  = torch.zeros(B, 1, H, W),
        y       = torch.zeros(B, 1, H, W),
    )
    with torch.no_grad():
        out = m(batch)
    assert out['logits'].shape == (B, 1, H, W), out['logits'].shape
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'FireAFNetV5 smoke ✓  logits={tuple(out["logits"].shape)}  params={n:.2f} M')

_smoke()
'''

SEC5B_MD = r"""## Section 5b · Load training configuration from Drive"""

SEC5B_CODE = r"""with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# Force Colab-local paths (in case the YAML was edited elsewhere)
cfg['data']['root_train'] = DATASET_TRAIN
cfg['data']['root_val']   = DATASET_VAL
cfg['output_dir']         = OUTPUT_LOCAL

print(yaml.dump(cfg, sort_keys=False))
"""

SEC6_MD = r"""## Section 6 · Training loop (single-step, IoU-maximal)

Features:
* Focal BCE + Lovász + Tversky + Dice + Boundary loss (all single-step)
* bfloat16 AMP
* EMA shadow with **delayed initialisation** at `ema_start_epoch`
  (fixes the run28 collapse bug).
* Global TP/FP/FN threshold scan on the entire val set — checkpoint selected
  by `val_iou_best_thr`.
* Mixup (p=0.3, α=0.2) — conservative because AF masks are sparse.
* Hard-example mining from epoch 25 onwards: BCE loss map is kept only on
  the top-k% worst pixels per sample.
"""

SEC6_CODE = r'''# ─── EMA (delayed-init) ──────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = None
        self._initialised = False

    def maybe_init(self, model, epoch, start_epoch):
        if (not self._initialised) and epoch >= start_epoch:
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self._initialised = True
            print(f'  [EMA] shadow initialised at epoch {epoch}')

    def update(self, model):
        if not self._initialised:
            return
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
                else:
                    self.shadow[k].copy_(v)

    def swap_in(self, model):
        if not self._initialised:
            return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def swap_out(self, model, backup):
        if backup is not None:
            model.load_state_dict(backup, strict=False)


# ─── Composite single-step loss ─────────────────────────────────────────
class AFLoss(nn.Module):
    def __init__(self, pos_weight=3.0, focal_gamma=2.0, lovasz_w=0.5,
                 dice_w=0.15, tversky_w=0.3, tversky_a=0.4, tversky_b=0.6,
                 boundary_w=0.2, label_smoothing=0.02,
                 hard_mining=False, hard_top_k_frac=0.75):
        super().__init__()
        self.focal = FocalBCELoss(pos_weight=pos_weight, gamma=focal_gamma,
                                  label_smoothing=label_smoothing)
        self.tversky = TverskyLoss(alpha=tversky_a, beta=tversky_b)
        self.boundary = BoundaryAwareLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.lovasz_w, self.dice_w, self.tversky_w, self.boundary_w = \
            lovasz_w, dice_w, tversky_w, boundary_w
        self.hard_mining = hard_mining
        self.hard_top_k_frac = hard_top_k_frac

    def _soft_dice(self, logits, tgt):
        p = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        t = tgt.reshape(tgt.shape[0], -1)
        inter = (p * t).sum(dim=1)
        return (1.0 - (2.0 * inter + 1.0) / (p.sum(dim=1) + t.sum(dim=1) + 1.0)).mean()

    def _hard_focal(self, logits, tgt):
        # focal BCE per pixel, keep top-k%
        probs = torch.sigmoid(logits)
        p_t = probs * tgt + (1.0 - probs) * (1.0 - tgt)
        focal_w = (1.0 - p_t).pow(2.0)
        pw = 3.0
        px_w = tgt * pw + (1.0 - tgt)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        loss_map = focal_w * px_w * bce                 # (B, 1, H, W)
        B = loss_map.shape[0]
        flat = loss_map.view(B, -1)
        k = max(1, int(flat.shape[1] * self.hard_top_k_frac))
        top, _ = flat.topk(k, dim=1, largest=True)
        return top.mean()

    def forward(self, logits, tgt):
        loss_focal = self._hard_focal(logits, tgt) if self.hard_mining else self.focal(logits, tgt)
        loss_lovasz = self.lovasz(logits, tgt)
        loss_dice = self._soft_dice(logits, tgt)
        loss_tversky = self.tversky(logits, tgt)
        loss_boundary = self.boundary(logits, tgt)
        total = (loss_focal
                 + self.lovasz_w   * loss_lovasz
                 + self.dice_w     * loss_dice
                 + self.tversky_w  * loss_tversky
                 + self.boundary_w * loss_boundary)
        return dict(total=total,
                    focal=loss_focal.detach(),
                    lovasz=loss_lovasz.detach(),
                    dice=loss_dice.detach(),
                    tversky=loss_tversky.detach(),
                    boundary=loss_boundary.detach())


# ─── Instantiate ────────────────────────────────────────────────────────
mcfg, tcfg = cfg['model'], cfg['training']
train_loader, val_loader, train_base, val_base, C_aux, C_static = build_loaders(cfg)

model = FireAFNetV5(
    in_ch_img=int(cfg['data']['num_bands']),
    in_ch_aux=int(C_aux),
    in_ch_static=int(C_static),
    encoder_channels=tuple(mcfg['encoder_channels']),
    gnn_patches=tuple(mcfg['gnn_patches']),
    gnn_heads=int(mcfg['gnn_heads']),
    gnn_layers=int(mcfg['gnn_layers']),
    gnn_knn=int(mcfg['gnn_knn']),
    gnn_content_knn=bool(mcfg['gnn_content_knn']),
    lstm_hidden=int(mcfg['lstm_hidden']),
    lstm_layers=int(mcfg['lstm_layers']),
    lstm_dropout=float(mcfg['lstm_dropout']),
    lstm_attn_heads=int(mcfg['lstm_attn_heads']),
    weather_indices=tuple(mcfg['weather_indices']),
    weather_emb=int(mcfg['weather_emb']),
    cross_attn_heads=int(mcfg['cross_attn_heads']),
    decoder_dropout=float(mcfg['decoder_dropout']),
    use_film_skip=bool(mcfg['use_film_skip']),
    use_cbam=bool(mcfg['use_cbam']),
    use_input_norm=bool(mcfg['use_input_norm']),
    out_ch=int(mcfg['out_channels']),
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model: FireAFNetV5  | params={n_params:.2f} M  | device={DEVICE}')

criterion = AFLoss(
    pos_weight=float(tcfg['pos_weight']),
    focal_gamma=float(tcfg['focal_gamma']),
    lovasz_w=float(tcfg['lovasz_weight']),
    dice_w=float(tcfg['dice_weight']),
    tversky_w=float(tcfg['tversky_weight']),
    tversky_a=float(tcfg['tversky_alpha']),
    tversky_b=float(tcfg['tversky_beta']),
    boundary_w=float(tcfg['boundary_weight']),
    label_smoothing=float(tcfg['label_smoothing']),
    hard_mining=False,            # toggled on after `hard_example_start_epoch`
    hard_top_k_frac=float(tcfg['hard_example_top_k_frac']),
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(tcfg['lr']),
    weight_decay=float(tcfg['weight_decay']),
    betas=(float(tcfg['adam_beta1']), float(tcfg['adam_beta2'])),
)

# Warmup + cosine LR
EPOCHS = int(tcfg['epochs'])
W = int(tcfg['warmup_epochs_lr'])
def lr_lambda(ep):
    if ep < W:
        return tcfg['warmup_start_factor'] + (1 - tcfg['warmup_start_factor']) * (ep / max(1, W))
    # cosine decay from 1 → eta_min/lr
    progress = (ep - W) / max(1, EPOCHS - W)
    return tcfg['eta_min'] / tcfg['lr'] + 0.5 * (1 - tcfg['eta_min'] / tcfg['lr']) * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

amp_dtype = torch.bfloat16 if tcfg.get('amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16
use_scaler = (amp_dtype == torch.float16)
scaler = torch.amp.GradScaler('cuda') if use_scaler else None

ema = EMA(model, decay=float(tcfg['ema_decay']))


# ─── Metric helpers (global accumulator) ─────────────────────────────────
def _run_validation(model, loader, criterion, thr_grid, ema=None):
    model.eval()
    bak = ema.swap_in(model) if ema is not None else None
    try:
        tp = {t: 0.0 for t in thr_grid}
        fp = {t: 0.0 for t in thr_grid}
        fn = {t: 0.0 for t in thr_grid}
        losses = dict(total=0.0, focal=0.0, lovasz=0.0, dice=0.0,
                      tversky=0.0, boundary=0.0)
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                for k in batch:
                    batch[k] = batch[k].to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    out = model(batch)
                    loss = criterion(out['logits'], batch['y'])
                logits = out['logits'].float()
                gt = (batch['y'] > 0.5).float()
                probs = torch.sigmoid(logits)
                for t in thr_grid:
                    pred = (probs > t).float()
                    tp[t] += float((pred * gt).sum())
                    fp[t] += float((pred * (1 - gt)).sum())
                    fn[t] += float(((1 - pred) * gt).sum())
                for k in losses:
                    losses[k] += float(loss[k])
                n_batches += 1

        # best threshold
        best_iou, best_t, best_p, best_r = -1.0, thr_grid[0], 0.0, 0.0
        for t in thr_grid:
            iou_t = tp[t] / max(tp[t] + fp[t] + fn[t], 1e-8)
            if iou_t > best_iou:
                best_iou = iou_t; best_t = t
                best_p = tp[t] / max(tp[t] + fp[t], 1e-8)
                best_r = tp[t] / max(tp[t] + fn[t], 1e-8)
        losses = {k: v / max(n_batches, 1) for k, v in losses.items()}
        return dict(
            **losses,
            iou_best_thr=best_iou, thr_best=best_t,
            prec_best=best_p, rec_best=best_r,
            f1_best=2 * best_p * best_r / max(best_p + best_r, 1e-8),
        )
    finally:
        if bak is not None:
            ema.swap_out(model, bak)


def _mixup(batch, alpha):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(batch['af_hist'].shape[0], device=DEVICE)
    for k in ['af_hist', 'aux_hist', 'static', 'y']:
        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]
    return batch


# ─── Fit loop ────────────────────────────────────────────────────────────
history = []
best_iou = -1.0
patience_ctr = 0
PATIENCE = int(tcfg['patience'])
thr_grid = list(tcfg['val_threshold_grid'])
mixup_p, mixup_a = float(tcfg['mixup_prob']), float(tcfg['mixup_alpha'])
grad_clip = float(tcfg['grad_clip'])
out_dir = Path(OUTPUT_LOCAL); out_dir.mkdir(parents=True, exist_ok=True)

hme_start = int(tcfg['hard_example_start_epoch'])

for epoch in range(1, EPOCHS + 1):
    criterion.hard_mining = bool(tcfg['hard_example_mining']) and (epoch >= hme_start)
    ema.maybe_init(model, epoch, int(tcfg['ema_start_epoch']))
    model.train()
    run = dict(total=0.0, focal=0.0, lovasz=0.0, dice=0.0, tversky=0.0, boundary=0.0, n=0)

    pbar = tqdm(train_loader, desc=f'ep {epoch:02d}/{EPOCHS}', leave=False)
    for batch in pbar:
        for k in batch:
            batch[k] = batch[k].to(DEVICE, non_blocking=True)
        if torch.rand(1).item() < mixup_p:
            batch = _mixup(batch, mixup_a)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out = model(batch)
            loss = criterion(out['logits'], batch['y'])

        if use_scaler:
            scaler.scale(loss['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        ema.update(model)

        for k in run:
            if k == 'n':
                run[k] += 1
            else:
                run[k] += float(loss[k])
        pbar.set_postfix(total=loss['total'].item(), lovasz=loss['lovasz'].item(),
                         lr=optimizer.param_groups[0]['lr'])

    tr = {k: v / max(run['n'], 1) for k, v in run.items() if k != 'n'}
    scheduler.step()

    # Validation (EMA if ready, else raw)
    va = _run_validation(model, val_loader, criterion, thr_grid,
                         ema=ema if ema._initialised else None)

    row = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], **{'tr_' + k: v for k, v in tr.items()}, **{'va_' + k: v for k, v in va.items()})
    history.append(row)

    print(f'ep {epoch:02d}  '
          f'tr_total={tr["total"]:.4f}  '
          f'va_total={va["total"]:.4f}  '
          f'val_iou_best={va["iou_best_thr"]:.4f}@t={va["thr_best"]:.2f}  '
          f'P={va["prec_best"]:.3f}  R={va["rec_best"]:.3f}  '
          f'lr={optimizer.param_groups[0]["lr"]:.2e}')

    # Checkpoint
    if va['iou_best_thr'] > best_iou:
        best_iou = va['iou_best_thr']; patience_ctr = 0
        torch.save({
            'model_state': model.state_dict(),
            'ema_state':   (ema.shadow if ema._initialised else None),
            'cfg': cfg, 'epoch': epoch,
            'val_iou_best_thr': best_iou,
            'val_thr_best': va['thr_best'],
        }, out_dir / 'best_model.pt')
        print(f'  ★ new best iou={best_iou:.4f}  → saved')
    else:
        patience_ctr += 1

    # Periodic save
    if epoch % int(tcfg['save_every']) == 0:
        torch.save({'model_state': model.state_dict(), 'epoch': epoch},
                   out_dir / f'checkpoint_epoch{epoch:03d}.pt')

    # Log
    import csv as _csv
    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    # Early stopping
    if patience_ctr >= PATIENCE:
        print(f'Early stopping — no improvement for {PATIENCE} epochs')
        break

print(f'\nTraining done.  best val_iou_best_thr = {best_iou:.4f}')
'''

SEC7_MD = r"""## Section 7 · Post-training validation curves"""

SEC7_CODE = r'''import pandas as pd
df = pd.DataFrame(history)
fig, ax = plt.subplots(2, 3, figsize=(16, 8))

ax[0, 0].plot(df['epoch'], df['tr_total'], label='train')
ax[0, 0].plot(df['epoch'], df['va_total'], label='val')
ax[0, 0].set_title('Total loss'); ax[0, 0].legend(); ax[0, 0].grid(True)

ax[0, 1].plot(df['epoch'], df['va_iou_best_thr'], label='val IoU (best thr)')
ax[0, 1].axhline(0.525, color='grey', ls='--', label='run26 baseline 0.525')
ax[0, 1].set_title('Validation IoU'); ax[0, 1].legend(); ax[0, 1].grid(True)

ax[0, 2].plot(df['epoch'], df['va_prec_best'], label='precision')
ax[0, 2].plot(df['epoch'], df['va_rec_best'],  label='recall')
ax[0, 2].plot(df['epoch'], df['va_f1_best'],   label='F1')
ax[0, 2].set_title('P / R / F1 at best thr'); ax[0, 2].legend(); ax[0, 2].grid(True)

ax[1, 0].plot(df['epoch'], df['va_thr_best'])
ax[1, 0].set_ylim(min(cfg['training']['val_threshold_grid']) - 0.02,
                  max(cfg['training']['val_threshold_grid']) + 0.02)
ax[1, 0].set_title('Best threshold per epoch'); ax[1, 0].grid(True)

ax[1, 1].plot(df['epoch'], df['tr_lovasz'], label='train Lovász')
ax[1, 1].plot(df['epoch'], df['va_lovasz'], label='val Lovász')
ax[1, 1].set_title('Lovász hinge'); ax[1, 1].legend(); ax[1, 1].grid(True)

ax[1, 2].plot(df['epoch'], df['lr'])
ax[1, 2].set_title('Learning rate'); ax[1, 2].grid(True)

fig.tight_layout()
fig.savefig(Path(OUTPUT_LOCAL) / 'training_curves.png', dpi=120)
plt.show()

print('best val IoU :', df['va_iou_best_thr'].max(),
      ' @ epoch', int(df.loc[df['va_iou_best_thr'].idxmax(), 'epoch']),
      ' thr=', float(df.loc[df['va_iou_best_thr'].idxmax(), 'va_thr_best']))
'''

SEC8_MD = r"""## Section 8 · Inference visualisation + 4-way TTA + CC filter

Loads the best checkpoint, runs **test-time augmentation** (mean of 4 flip
variants), picks the global best threshold from validation, and applies a
connected-component size filter to kill tiny false-positive islands.
"""

SEC8_CODE = r'''from scipy.ndimage import label as cc_label

BEST_CKPT = Path(OUTPUT_LOCAL) / 'best_model.pt'
ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
print('loaded best ep', ckpt.get('epoch'), 'iou', ckpt.get('val_iou_best_thr'))

# Rebuild the exact same architecture
model_inf = FireAFNetV5(
    in_ch_img=int(cfg['data']['num_bands']),
    in_ch_aux=int(C_aux),
    in_ch_static=int(C_static),
    encoder_channels=tuple(cfg['model']['encoder_channels']),
    gnn_patches=tuple(cfg['model']['gnn_patches']),
    gnn_heads=int(cfg['model']['gnn_heads']),
    gnn_layers=int(cfg['model']['gnn_layers']),
    gnn_knn=int(cfg['model']['gnn_knn']),
    gnn_content_knn=bool(cfg['model']['gnn_content_knn']),
    lstm_hidden=int(cfg['model']['lstm_hidden']),
    lstm_layers=int(cfg['model']['lstm_layers']),
    lstm_dropout=float(cfg['model']['lstm_dropout']),
    lstm_attn_heads=int(cfg['model']['lstm_attn_heads']),
    weather_indices=tuple(cfg['model']['weather_indices']),
    weather_emb=int(cfg['model']['weather_emb']),
    cross_attn_heads=int(cfg['model']['cross_attn_heads']),
    decoder_dropout=float(cfg['model']['decoder_dropout']),
    use_film_skip=bool(cfg['model']['use_film_skip']),
    use_cbam=bool(cfg['model']['use_cbam']),
    use_input_norm=bool(cfg['model']['use_input_norm']),
).to(DEVICE)
# Prefer EMA weights if present (best IoU usually comes from EMA)
state = ckpt.get('ema_state') or ckpt['model_state']
model_inf.load_state_dict(state, strict=False)
model_inf.eval()

BEST_THR = float(ckpt.get('val_thr_best', 0.40))
CC_MIN   = int(cfg['training']['cc_filter_min_px'])
USE_TTA  = bool(cfg['training']['tta'])

def _flip_batch(batch, dims):
    out = {}
    for k, v in batch.items():
        out[k] = v
        for d in dims:
            out[k] = out[k].flip(d)
    return out

def _flip_logits(logits, dims):
    out = logits
    for d in dims:
        out = out.flip(d)
    return out

@torch.no_grad()
def predict_one(batch):
    """Returns probs (B, 1, H, W) after TTA."""
    for k in batch:
        batch[k] = batch[k].to(DEVICE, non_blocking=True)
    with torch.amp.autocast('cuda', dtype=amp_dtype):
        logits = model_inf(batch)['logits'].float()
    probs = [torch.sigmoid(logits)]
    if USE_TTA:
        for dims in [(-1,), (-2,), (-1, -2)]:
            bf = _flip_batch(batch, dims)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                lf = model_inf(bf)['logits'].float()
            probs.append(torch.sigmoid(_flip_logits(lf, dims)))
    return torch.stack(probs).mean(0)

def cc_filter(mask_np, min_px):
    lbl, n = cc_label(mask_np)
    if n == 0:
        return mask_np
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes >= min_px
    return keep[lbl].astype(mask_np.dtype)


# ─── Render 3 qualitative samples (pred vs GT) ───────────────────────────
sample_ids = [0, max(0, len(val_base) // 2), max(0, len(val_base) - 1)]
fig, axes = plt.subplots(len(sample_ids), 4, figsize=(14, 4 * len(sample_ids)))
if len(sample_ids) == 1:
    axes = axes[None, :]

for row, sid in enumerate(sample_ids):
    s = val_base[sid]
    batch = {k: v.unsqueeze(0) for k, v in s.items() if torch.is_tensor(v)}
    probs = predict_one(batch)[0, 0].cpu().numpy()
    pred_raw = (probs > BEST_THR).astype(np.uint8)
    pred_cc  = cc_filter(pred_raw, CC_MIN)
    gt       = s['y'][0].cpu().numpy()

    axes[row, 0].imshow(s['af_hist'][-1, 0].cpu().numpy(), cmap='gray')
    axes[row, 0].set_title(f'Sample {sid} — VIIRS I1 (t=-1)')
    axes[row, 1].imshow(probs, cmap='hot', vmin=0, vmax=1)
    axes[row, 1].set_title(f'Pred prob (TTA)')
    axes[row, 2].imshow(pred_cc, cmap='Reds', vmin=0, vmax=1)
    axes[row, 2].set_title(f'Pred mask  thr={BEST_THR:.2f}, CC≥{CC_MIN}')
    axes[row, 3].imshow(gt, cmap='Reds', vmin=0, vmax=1)
    axes[row, 3].set_title('Ground truth')
    for c in range(4):
        axes[row, c].axis('off')
fig.tight_layout()
fig.savefig(Path(OUTPUT_LOCAL) / 'inference_samples.png', dpi=120)
plt.show()

# ─── Final global IoU with TTA + CC filter ───────────────────────────────
tp = fp = fn = 0.0
n = 0
for batch in val_loader:
    probs = predict_one(batch).cpu().numpy()
    gt = batch['y'].cpu().numpy()
    for b in range(probs.shape[0]):
        p = cc_filter((probs[b, 0] > BEST_THR).astype(np.uint8), CC_MIN).astype(np.float32)
        g = gt[b, 0].astype(np.float32)
        tp += float((p * g).sum())
        fp += float((p * (1 - g)).sum())
        fn += float(((1 - p) * g).sum())
    n += probs.shape[0]
iou_final = tp / max(tp + fp + fn, 1e-8)
prec_final = tp / max(tp + fp, 1e-8)
rec_final  = tp / max(tp + fn, 1e-8)
f1_final   = 2 * prec_final * rec_final / max(prec_final + rec_final, 1e-8)
print(f'\n=== FINAL (TTA + CC filter) ===')
print(f'  val samples  : {n}')
print(f'  IoU          : {iou_final:.4f}')
print(f'  F1           : {f1_final:.4f}')
print(f'  Precision    : {prec_final:.4f}')
print(f'  Recall       : {rec_final:.4f}')
print(f'  threshold    : {BEST_THR:.2f}')
print(f'  CC min pixels: {CC_MIN}')
'''

SEC9_MD = r"""## Section 9 · Export best checkpoint + metrics to Drive"""

SEC9_CODE = r"""import shutil
# Sync best_model.pt + metrics.csv + training_curves.png + inference_samples.png
dst = Path(OUTPUT_DRIVE); dst.mkdir(parents=True, exist_ok=True)
for fname in ['best_model.pt', 'metrics.csv',
              'training_curves.png', 'inference_samples.png']:
    src = Path(OUTPUT_LOCAL) / fname
    if src.exists():
        shutil.copy2(src, dst / fname)
        print('→', dst / fname)

# Also push the YAML used (so the run is reproducible)
with open(dst / 'config_run29_version1.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)
print('\nAll artifacts synced to Drive → ', dst)
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Assemble
# ═══════════════════════════════════════════════════════════════════════════
cells = [
    md(INTRO_MD),
    md(SEC0_MD),  code(SEC0_CODE),
    md(SEC1_MD),  code(SEC1_CODE), code(SEC1B_CODE),
    md(SEC2_MD),  code(SEC2_CODE), code(SEC2B_CODE),
    md(SEC3_MD),  code(SEC3_CODE),
    md(SEC4_MD),  code(SEC4_CODE),
    md(SEC5_MD),  code(SEC5_CODE),
    md(SEC5B_MD), code(SEC5B_CODE),
    md(SEC6_MD),  code(SEC6_CODE),
    md(SEC7_MD),  code(SEC7_CODE),
    md(SEC8_MD),  code(SEC8_CODE),
    md(SEC9_MD),  code(SEC9_CODE),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)')
