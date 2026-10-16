# -*- coding: utf-8 -*-
"""Build the V2 AF batch-inference Colab notebook."""
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

# ---------------------------------------------------------------- CELL 0 ----
cells.append(
    md(
        """# FireST-GraphNet \u2014 AF V2 Batch Inference (Colab)

**Purpose** \u2014 Run **one-step Active-Fire (AF)** inference on every V2 run (`run27`,
`run28`, `run29`, \u2026) and every checkpoint available for each run, over the
**entire** `dataset_train` and `dataset_val`.

**Architecture** \u2014 V2 = `FireSTGraphNetV2` + `ForecastDecoderV3` (GATv2 pyramid +
Attention-LSTM + FiLM-skip U-Net decoder with per-horizon autoregressive feedback).
Ported inline from `data_tools/colab/colab_run29.ipynb` so this notebook does **not**
depend on the legacy/V1 `firest_graphnet.models.firest_graphnet` module.

**What the notebook does (top-to-bottom)**
1. Check GPU, mount Drive, set paths
2. Sync `dataset_train` / `dataset_val` from Drive to Colab local SSD (one-off)
3. Discover all V2 AF runs under `OUTPUT_ROOT`
4. For each run, discover all checkpoints (`best_model.pt`, `last.pt`, `epoch_XXX.pt`, \u2026)
5. Build the V2 model, load weights `strict=False`, run **one-step** inference over
   **all** train + val samples
6. Save raw `pred` / `gt` `.npy` arrays, per-sample PNG visualisations, `manifest.csv`,
   `inference_summary.json`
7. Display a representative inline grid, write a global summary across all
   runs \u00d7 checkpoints

**Output layout (Drive)**
```
<OUTPUT_ROOT>/inference_exports_af_colab/
  <RUN_NAME>/
    <CHECKPOINT_TAG>/
      checkpoint_info.txt
      train/  (predictions/, ground_truth/, visualizations/, manifest.csv, inference_summary.json)
      val/    (predictions/, ground_truth/, visualizations/, manifest.csv, inference_summary.json)
  all_runs_all_checkpoints_summary.csv
  all_runs_all_checkpoints_summary.json
```

**Strict naming** \u2014 every exported sample uses:
`<RUN>__<CKPT_TAG>__<SPLIT>__sample_<IDX:06d>__{pred,gt,viz}.{npy,png}`

> **Scope** \u2014 AF only, **one-step** only (`horizon_override=1`). BA / Pred /
> multi-step rollout are out of scope. Training is not performed.
"""
    )
)

# ---------------------------------------------------------------- CELL 1 ----
cells.append(md("## Section 0 \u00b7 GPU check"))

cells.append(
    code(
        """import torch, subprocess
print('torch         :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device        :', torch.cuda.get_device_name(0))
    print('capability    :', torch.cuda.get_device_capability(0))
    try:
        print(subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=name,memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits',
        ]).decode())
    except Exception:
        pass
else:
    print('WARNING - no CUDA. Inference will fall back to CPU (slow).')
"""
    )
)

# ---------------------------------------------------------------- CELL 2 ----
cells.append(md("## Section 1 \u00b7 Mount Google Drive"))

cells.append(
    code(
        """from google.colab import drive
drive.mount('/content/drive')
"""
    )
)

# ---------------------------------------------------------------- CELL 3 ----
cells.append(md("## Section 2 \u00b7 Project paths"))

cells.append(
    code(
        """import os, sys, shutil, json, time, csv, math, traceback, re
from pathlib import Path

# Paths (match the training notebook conventions)
PROJECT_ROOT = Path('/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean')
OUTPUT_ROOT  = PROJECT_ROOT / 'output'
DRIVE_TRAIN  = PROJECT_ROOT / 'dataset_train'
DRIVE_VAL    = PROJECT_ROOT / 'dataset_val'

# Local Colab cache (Drive I/O is ~30x slower - never iterate directly on Drive)
DATASET_TRAIN = Path('/content/dataset_train')
DATASET_VAL   = Path('/content/dataset_val')

# Where inference artefacts are written.  We write locally first, then mirror
# to Drive per-checkpoint to keep Drive I/O cheap.
INFER_ROOT_DRIVE = OUTPUT_ROOT / 'inference_exports_af_colab'
INFER_ROOT_LOCAL = Path('/content/inference_exports_af_colab')
INFER_ROOT_DRIVE.mkdir(parents=True, exist_ok=True)
INFER_ROOT_LOCAL.mkdir(parents=True, exist_ok=True)

# Expose project source tree for helper imports (providers, dataset)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print('PROJECT_ROOT    :', PROJECT_ROOT, '  exists =', PROJECT_ROOT.exists())
print('OUTPUT_ROOT     :', OUTPUT_ROOT,  '  exists =', OUTPUT_ROOT.exists())
print('DRIVE_TRAIN     :', DRIVE_TRAIN,  '  exists =', DRIVE_TRAIN.exists())
print('DRIVE_VAL       :', DRIVE_VAL,    '  exists =', DRIVE_VAL.exists())
print('INFER_ROOT (Dr) :', INFER_ROOT_DRIVE)
print('INFER_ROOT (Lc) :', INFER_ROOT_LOCAL)
"""
    )
)

# ---------------------------------------------------------------- CELL 4 ----
cells.append(md("## Section 3 \u00b7 Sync datasets from Drive \u2192 local SSD"))

cells.append(
    code(
        """import glob

def _has(p):
    p = Path(p)
    return any(p.glob('af_*_img*.npy')) and any(p.glob('af_*_label*.npy'))

def copy_datasets_to_local(drive_train, drive_val, local_train, local_val):
    need_copy = not (_has(local_train) and _has(local_val))
    if not need_copy:
        print('Datasets already staged locally - skipping copy.')
        return
    print('Copying datasets Drive -> /content (a few minutes the first time)')
    for src, dst in [(drive_train, local_train), (drive_val, local_val)]:
        Path(dst).mkdir(parents=True, exist_ok=True)
        for f in sorted(glob.glob(f'{src}/*.npy')):
            name = Path(f).name
            tgt  = Path(dst) / name
            if not tgt.exists():
                print(f'  {name} ...', end=' ', flush=True)
                t0 = time.time()
                shutil.copy2(f, tgt)
                print(f'{(time.time()-t0):.1f}s')

copy_datasets_to_local(DRIVE_TRAIN, DRIVE_VAL, DATASET_TRAIN, DATASET_VAL)
print('train staged :', _has(DATASET_TRAIN))
print('val   staged :', _has(DATASET_VAL))
"""
    )
)

# ---------------------------------------------------------------- CELL 5 ----
cells.append(md("## Section 4 \u00b7 Imports"))

cells.append(
    code(
        """import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm

SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print('device :', DEVICE,
      f'({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else '')

# Project-side helpers (dataset + providers).  These come from the existing
# firest_graphnet package - the V2 inference pipeline reuses them exactly as
# training did, so dataset indexing is guaranteed identical to training runs.
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider   import build_static_provider
from firest_graphnet.data.dataset_forecast  import TSSatFireForecastDataset
print('backend modules imported OK')
"""
    )
)

# ---------------------------------------------------------------- CELL 6 ----
cells.append(
    md(
        """## Section 5 \u00b7 Utility helpers

Run discovery, checkpoint discovery, V2 detection, filename sanitising \u2014 all the
small pieces that glue the pipeline together. Nothing model-specific lives here.
"""
    )
)

cells.append(
    code(
        r"""# Filename / tag sanitising
_SAFE_RE = re.compile(r'[^A-Za-z0-9_.-]+')
def sanitize_tag(name):
    return _SAFE_RE.sub('_', name).strip('._') or 'ckpt'

def checkpoint_tag(path):
    return sanitize_tag(Path(path).stem)

# Preferred ordering: best -> last -> epoch_N (numeric) -> others alpha
def _ckpt_sort_key(p):
    tag = Path(p).stem.lower()
    if tag == 'best_model':
        return (0, tag)
    if tag == 'last':
        return (1, tag)
    m = re.match(r'(?:epoch|checkpoint_epoch|ckpt_epoch|ep)[_-]?(\d+)', tag)
    if m:
        return (2, int(m.group(1)))
    return (3, tag)

# Run discovery
def _load_yaml(p):
    try:
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def discover_af_runs(output_root):
    runs = []
    for run_dir in sorted(Path(output_root).glob('*')):
        if not run_dir.is_dir() or run_dir.name.startswith('inference_exports'):
            continue
        cfgs = sorted(run_dir.glob('config*.yaml')) + sorted(run_dir.glob('config*.yml'))
        if not cfgs:
            continue
        cfg = _load_yaml(cfgs[0])
        if cfg is None:
            continue
        label_channel = int(cfg.get('data', {}).get('label_channel', -1))
        if label_channel != 1:
            continue
        ckpts = sorted(
            [p for p in run_dir.glob('*.pt') if p.is_file()],
            key=_ckpt_sort_key,
        )
        runs.append(dict(
            name=run_dir.name,
            dir=run_dir,
            config_path=cfgs[0],
            cfg=cfg,
            checkpoints=ckpts,
        ))
    return runs

# V2 detection (config flag OR checkpoint layout)
def is_v2_config(cfg):
    return bool(cfg.get('model', {}).get('v2_enabled', False))

_V2_MARKER_KEYS = (
    'encoder.b1.c1.weight',
    'decoder.feedback_conv.weight',
    'decoder.pos_emb',
    'pyramid.layers.p8.0.W_src.weight',
)
def is_v2_state_dict(sd):
    return any(k in sd for k in _V2_MARKER_KEYS)

def peek_checkpoint(ckpt_path):
    c = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = c.get('model_state') or c.get('model_state_dict') or {}
    return dict(
        epoch            = c.get('epoch'),
        val_iou_best_thr = c.get('val_iou_best_thr'),
        thr_best         = c.get('thr_best', c.get('val_thr_best', 0.5)),
        has_ema          = 'ema_state' in c and c['ema_state'] is not None,
        is_v2            = is_v2_state_dict(sd),
        top_keys         = list(c.keys()),
        n_tensors        = len(sd),
    )

print('Utility helpers ready')
"""
    )
)

# ---------------------------------------------------------------- CELL 7 ----
cells.append(
    md(
        """## Section 6 \u00b7 V2 architecture \u2014 `FireSTGraphNetV2` + `ForecastDecoderV3`

Ported verbatim from `data_tools/colab/colab_run29.ipynb` cell 16. This is the
architecture that trained `run27`, `run28`, and `run29`. The parameter names match
the keys in those checkpoints exactly (`encoder.b1.c1.weight`,
`decoder.feedback_conv.weight`, `decoder.pos_emb`,
`pyramid.layers.p8.0.W_src.weight`, \u2026), so `load_state_dict` succeeds with no
unexpected keys.
"""
    )
)

cells.append(
    code(
        """# CBAM
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
        self.ca = _ChannelAttn(c); self.sa = _SpatialAttn()
    def forward(self, x):
        return self.sa(self.ca(x))

# Encoder
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
    def __init__(self, in_ch=28, channels=(32, 64, 128), use_cbam=True):
        super().__init__()
        self.input_norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.b1 = _ConvBlock(in_ch,       channels[0])
        self.b2 = _ConvBlock(channels[0], channels[1])
        self.b3 = _ConvBlock(channels[1], channels[2])
        self.cbam = nn.ModuleList([CBAM(c) for c in channels]) if use_cbam else None
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.input_norm(x)
        s1 = self.b1(x)
        if self.cbam: s1 = self.cbam[0](s1)
        s2 = self.b2(self.pool(s1))
        if self.cbam: s2 = self.cbam[1](s2)
        s3 = self.b3(self.pool(s2))
        if self.cbam: s3 = self.cbam[2](s3)
        return s1, s2, s3

# GATv2 layer
class GATv2Layer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.h = heads; self.dk = dim // heads
        assert dim % heads == 0
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
        e = torch.tanh(dst.unsqueeze(2) + src_nei)
        e = e.view(B, N, K, self.h, self.dk)
        alpha = self.att(e).squeeze(-1)
        alpha = F.softmax(alpha, dim=2)
        alpha = self.drop(alpha)
        v = src_nei.view(B, N, K, self.h, self.dk)
        agg = (alpha.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, D)
        return self.ln(x + self.drop(F.gelu(self.out(agg))))

def _build_knn(x, k):
    with torch.no_grad():
        d2 = torch.cdist(x, x, p=2)
        _, idx = torch.topk(-d2, k=k, dim=-1)
    return idx

class MultiScaleGraphPyramid(nn.Module):
    def __init__(self, ch=128, patches=(8, 16), heads=4, n_layers=2):
        super().__init__()
        self.patches = patches
        self.layers = nn.ModuleDict()
        for p in patches:
            self.layers[f'p{p}'] = nn.ModuleList([
                GATv2Layer(ch, heads=heads) for _ in range(n_layers)
            ])
        self.fuse = nn.Sequential(nn.Conv2d(ch * len(patches), ch, 1), nn.GELU())
        self.k = 8
    def forward(self, feat):
        B, C, H, W = feat.shape
        outs = []
        for p in self.patches:
            Hp, Wp = H // p, W // p
            patched = (F.adaptive_avg_pool2d(feat, (Hp, Wp))
                       if Hp * p != H or Wp * p != W
                       else F.avg_pool2d(feat, p))
            B_, C_, hp, wp = patched.shape
            tok = patched.flatten(2).transpose(1, 2)
            idx = _build_knn(tok, min(self.k, tok.shape[1]))
            for layer in self.layers[f'p{p}']:
                tok = layer(tok, idx)
            tok = tok.transpose(1, 2).reshape(B_, C_, hp, wp)
            up  = F.interpolate(tok, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(up)
        return self.fuse(torch.cat(outs, dim=1))

# Attention-LSTM
class AttentionLSTM(nn.Module):
    def __init__(self, dim, hidden, num_layers=2, dropout=0.4, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.ln   = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(dim, hidden, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
    def forward(self, x):
        res = x
        x, _ = self.attn(x, x, x)
        x = self.ln(res + x)
        out, (h, c) = self.lstm(x)
        return out, (h, c)

# Cross-attn weather fusion
class CrossAttnWeatherFusion(nn.Module):
    def __init__(self, dim_spatial, dim_weather, heads=4):
        super().__init__()
        self.q = nn.Linear(dim_spatial, dim_spatial, bias=False)
        self.k = nn.Linear(dim_weather, dim_spatial, bias=False)
        self.v = nn.Linear(dim_weather, dim_spatial, bias=False)
        self.h = heads
        self.dk = dim_spatial // heads
        self.out = nn.Linear(dim_spatial, dim_spatial)
        self.ln  = nn.LayerNorm(dim_spatial)
    def forward(self, spatial, weather):
        B, Ns, Ds = spatial.shape
        _, Nw, _ = weather.shape
        q = self.q(spatial).view(B, Ns, self.h, self.dk).transpose(1, 2)
        k = self.k(weather).view(B, Nw, self.h, self.dk).transpose(1, 2)
        v = self.v(weather).view(B, Nw, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, Ns, Ds)
        return self.ln(spatial + self.out(out))

# Forecast decoder V3 (per-horizon pos_emb + autoregressive feedback + FiLM skips)
class ForecastDecoderV3(nn.Module):
    def __init__(self, spatial_ch, lstm_hidden, horizon,
                 skip_channels=(32, 64),
                 weather_ch=4, weather_emb=64,
                 dropout=0.15, heads=4, aux_ba=True,
                 use_pos_emb=True, use_feedback=True, use_skip_film=True):
        super().__init__()
        self.H = horizon
        self.spatial_ch = spatial_ch
        self.use_pos_emb   = use_pos_emb
        self.use_feedback  = use_feedback
        self.use_skip_film = use_skip_film

        self.weather_embed = nn.Sequential(
            nn.Linear(weather_ch, 32), nn.GELU(),
            nn.Linear(32, weather_emb),
        )
        self.cross_attn = CrossAttnWeatherFusion(spatial_ch, weather_emb, heads=heads)
        self.lstm_cell  = nn.LSTMCell(spatial_ch, lstm_hidden)
        self.proj_cond  = nn.Linear(lstm_hidden, spatial_ch)

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(horizon, spatial_ch) * 0.02)
        if use_feedback:
            self.feedback_conv = nn.Conv2d(1, spatial_ch, kernel_size=3, padding=1)
        if use_skip_film:
            self.film1 = nn.Linear(lstm_hidden, 2 * skip_channels[1])
            self.film2 = nn.Linear(lstm_hidden, 2 * skip_channels[0])

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
        B, C, Hs, Ws = spatial.shape
        H_out = env_fut.shape[1]
        h, c = lstm_state

        af_seq, ba_seq = [], []
        ba_cum  = torch.zeros(B, 1, skip1.shape[-2], skip1.shape[-1],
                              device=spatial.device, dtype=spatial.dtype)
        prev_af = torch.zeros(B, 1, skip1.shape[-2], skip1.shape[-1],
                              device=spatial.device, dtype=spatial.dtype)

        for t in range(H_out):
            sp_vec = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
            h, c = self.lstm_cell(sp_vec, (h, c))
            cond = self.proj_cond(h)
            spatial_t = spatial + cond.view(B, C, 1, 1)

            if self.use_pos_emb:
                t_idx = min(t, self.pos_emb.shape[0] - 1)
                spatial_t = spatial_t + self.pos_emb[t_idx].view(1, C, 1, 1)
            if self.use_feedback:
                fb = F.adaptive_avg_pool2d(prev_af, (Hs, Ws))
                spatial_t = spatial_t + self.feedback_conv(fb)

            w_now = env_fut[:, t]
            w_vec = F.adaptive_avg_pool2d(w_now, 1).flatten(1)
            w_tok = self.weather_embed(w_vec).unsqueeze(1)
            sp_tok = spatial_t.flatten(2).transpose(1, 2)
            sp_tok = self.cross_attn(sp_tok, w_tok)
            spatial_t = sp_tok.transpose(1, 2).reshape(B, C, Hs, Ws)

            if self.use_skip_film:
                g1, b1 = self.film1(h).chunk(2, dim=-1)
                skip2_t = skip2 * (1 + g1.view(B, -1, 1, 1)) + b1.view(B, -1, 1, 1)
                g2, b2 = self.film2(h).chunk(2, dim=-1)
                skip1_t = skip1 * (1 + g2.view(B, -1, 1, 1)) + b2.view(B, -1, 1, 1)
            else:
                skip1_t, skip2_t = skip1, skip2

            x = self.up1(spatial_t)
            if x.shape[-2:] != skip2_t.shape[-2:]:
                x = F.interpolate(x, size=skip2_t.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse1(torch.cat([x, skip2_t], dim=1))
            x = self.up2(x)
            if x.shape[-2:] != skip1_t.shape[-2:]:
                x = F.interpolate(x, size=skip1_t.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse2(torch.cat([x, skip1_t], dim=1))

            af_logit = self.head_af(x)
            af_seq.append(af_logit)
            if self.use_feedback:
                prev_af = torch.sigmoid(af_logit).detach()
            if self.aux_ba:
                ba_delta = torch.sigmoid(self.head_ba(x))
                ba_cum = torch.clamp(ba_cum + ba_delta, 0.0, 1.0)
                ba_seq.append(ba_cum)

        out = {'af_pred': torch.stack(af_seq, dim=1)}
        if self.aux_ba:
            out['ba_pred'] = torch.stack(ba_seq, dim=1)
        return out

# Top-level V2 model
class FireSTGraphNetV2(nn.Module):
    def __init__(self,
                 in_ch_img=8, in_ch_aux=19, in_ch_static=1,
                 encoder_channels=(32, 64, 128),
                 gnn_patches=(8, 16), gat_heads=4,
                 lstm_hidden=256, lstm_layers=2, lstm_dropout=0.4,
                 lstm_attn_heads=4,
                 weather_ch=4, weather_emb=64,
                 horizon=15,
                 dropout=0.15, aux_ba=True,
                 use_pos_emb=True, use_feedback=True, use_skip_film=True):
        super().__init__()
        in_total = in_ch_img + in_ch_aux + in_ch_static
        self.encoder = MultiSourceEncoderV2(in_ch=in_total,
                                            channels=encoder_channels, use_cbam=True)
        self.pyramid = MultiScaleGraphPyramid(
            ch=encoder_channels[2], patches=gnn_patches, heads=gat_heads)
        self.attn_lstm = AttentionLSTM(
            dim=encoder_channels[2], hidden=lstm_hidden,
            num_layers=lstm_layers, dropout=lstm_dropout, heads=lstm_attn_heads)
        self.proj = nn.Linear(lstm_hidden, encoder_channels[2])
        self.decoder = ForecastDecoderV3(
            spatial_ch=encoder_channels[2],
            lstm_hidden=lstm_hidden,
            horizon=horizon,
            skip_channels=(encoder_channels[0], encoder_channels[1]),
            weather_ch=weather_ch, weather_emb=weather_emb,
            dropout=dropout, heads=gat_heads, aux_ba=aux_ba,
            use_pos_emb=use_pos_emb, use_feedback=use_feedback,
            use_skip_film=use_skip_film,
        )
        self.horizon = horizon

    def _encode_step(self, img, aux, static):
        x = torch.cat([img, aux, static], dim=1)
        return self.encoder(x)

    def forward(self, batch, horizon_override=None):
        x_seq    = batch['af_hist']
        aux_hist = batch['aux_hist']
        env_fut  = batch['env_fut']
        static   = batch['static']

        B, T = x_seq.shape[:2]
        H_out = env_fut.shape[1] if horizon_override is None else horizon_override

        s1_list = []; s2_list = []; s3_list = []
        for t in range(T):
            s1, s2, s3 = self._encode_step(x_seq[:, t], aux_hist[:, t], static)
            s1_list.append(s1); s2_list.append(s2); s3_list.append(s3)
        skip1 = s1_list[-1]; skip2 = s2_list[-1]
        feats_deep = torch.stack(s3_list, dim=1)

        g_list = []
        for t in range(feats_deep.shape[1]):
            g_list.append(self.pyramid(feats_deep[:, t]))
        g_seq = torch.stack(g_list, dim=1)

        g_vec = g_seq.mean(dim=[-1, -2])
        lstm_out, (h, c) = self.attn_lstm(g_vec)

        spatial_last = g_seq[:, -1]
        cond = self.proj(lstm_out[:, -1])
        spatial = spatial_last + cond.view(*spatial_last.shape[:2], 1, 1)

        env_use = env_fut[:, :H_out]
        return self.decoder(
            spatial=spatial,
            lstm_state=(h[-1], c[-1]),
            env_fut=env_use,
            skip1=skip1, skip2=skip2,
        )

print('FireSTGraphNetV2 + ForecastDecoderV3 defined')
"""
    )
)

# ---------------------------------------------------------------- CELL 8 ----
cells.append(
    md(
        """## Section 7 \u00b7 Builders \u2014 V2 model, dataset, inference helpers

- `build_v2_model_from_config(cfg, aux_ch)` \u2014 rebuilds the exact V2 architecture used in
  training (reads v2_* keys from config).
- `build_dataset(data_dir, cfg)` \u2014 `TSSatFireForecastDataset` with auto-detected firepred +
  LULC providers (no augmentation, no shuffle, deterministic order).
- `run_one_step(model, sample)` \u2014 calls the model with `horizon_override=1`; returns
  sigmoid probabilities `(H, W)` and ground truth `(H, W)`.
- Export helpers write pred / GT / viz using the strict naming convention.
"""
    )
)

cells.append(
    code(
        r"""# V2 model builder (matches training notebook's Section 8 pattern)
def build_v2_model_from_config(cfg, aux_ch):
    m = cfg['model']
    fcfg = cfg.get('forecast', {})
    model = FireSTGraphNetV2(
        in_ch_img    = int(cfg['data'].get('num_bands', 8)),
        in_ch_aux    = int(aux_ch),
        in_ch_static = 1,
        encoder_channels = tuple(m['encoder_channels']),
        gnn_patches      = tuple(m.get('v2_graph_pyramid_patches',
                                       m.get('gnn_patches', [8, 16]))),
        gat_heads        = int(m.get('v2_gat_heads', m.get('gnn_heads', 4))),
        lstm_hidden      = int(m['lstm_hidden']),
        lstm_layers      = int(m['lstm_layers']),
        lstm_dropout     = float(m['lstm_dropout']),
        lstm_attn_heads  = int(m.get('v2_lstm_attn_heads',
                                     m.get('lstm_attn_heads', 4))),
        weather_ch       = int(fcfg.get('weather_channels', 4)),
        weather_emb      = int(fcfg.get('weather_emb_dim', 64)),
        horizon          = int(fcfg.get('horizon', 15)),
        dropout          = float(m.get('decoder_dropout', 0.15)),
        aux_ba           = bool(m.get('v2_aux_ba_head', True)),
        use_pos_emb      = bool(m.get('v3_pos_emb',        True)),
        use_feedback     = bool(m.get('v3_autoregressive', True)),
        use_skip_film    = bool(m.get('v3_skip_film',      True)),
    )
    return model

def load_state_into_model(model, ckpt):
    # Prefer EMA weights if present (matches training notebook behaviour)
    state = ckpt.get('ema_state') or ckpt.get('model_state') or ckpt.get('model_state_dict')
    if state is None:
        raise RuntimeError('no model_state / ema_state / model_state_dict in checkpoint')
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        'used_ema'          : bool(ckpt.get('ema_state')),
        'n_missing_keys'    : len(missing),
        'n_unexpected_keys' : len(unexpected),
        'missing_sample'    : list(missing)[:5],
        'unexpected_sample' : list(unexpected)[:5],
    }

# Dataset builder (no shuffle, no aug - deterministic iteration order)
def build_dataset(data_dir, cfg):
    img_sz = int(cfg['data'].get('img_size', 256))
    fp = build_firepred_provider(data_root=str(data_dir), spatial_size=(img_sz, img_sz))
    st = build_static_provider  (data_root=str(data_dir), spatial_size=(img_sz, img_sz))
    ds = TSSatFireForecastDataset(
        str(data_dir),
        firepred_provider = fp,
        static_provider   = st,
        T_in              = int(cfg['forecast']['T_in']),
        horizon           = int(cfg['forecast']['horizon']),
        label_channel     = int(cfg['data'].get('label_channel', 1)),
    )
    return ds, int(fp.channels)

# One-step inference over a single sample
@torch.no_grad()
def run_one_step(model, sample, device, use_amp):
    batch = {
        k: (v.unsqueeze(0).to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in sample.items()
    }
    if use_amp and device.type == 'cuda':
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(batch, horizon_override=1)
    else:
        out = model(batch, horizon_override=1)
    af_logit = out['af_pred'][0, 0, 0].float().cpu().numpy()
    af_prob  = 1.0 / (1.0 + np.exp(-af_logit))
    gt       = sample['af_fut'][0, 0].cpu().numpy().astype(np.float32)
    hist_last = sample['af_hist'][-1, 0].cpu().numpy().astype(np.float32)
    return dict(prob=af_prob.astype(np.float32), gt=gt, hist_last=hist_last)

# IoU helper
def iou_bin(pred_bin, gt_bin):
    pred_bin = (pred_bin > 0).astype(np.uint8)
    gt_bin   = (gt_bin   > 0).astype(np.uint8)
    inter = float((pred_bin & gt_bin).sum())
    union = float((pred_bin | gt_bin).sum())
    return inter / union if union > 0 else 0.0

# Export helpers
def sample_stem(run, ckpt_tag, split, idx):
    return f'{run}__{ckpt_tag}__{split}__sample_{idx:06d}'

def save_pred_gt_npy(out_dir, run, ckpt_tag, split, idx, prob, gt):
    pred_dir = out_dir / 'predictions'
    gt_dir   = out_dir / 'ground_truth'
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir  .mkdir(parents=True, exist_ok=True)
    stem = sample_stem(run, ckpt_tag, split, idx)
    p_pred = pred_dir / f'{stem}__pred.npy'
    p_gt   = gt_dir   / f'{stem}__gt.npy'
    np.save(p_pred, prob.astype(np.float32))
    np.save(p_gt,   gt.astype(np.float32))
    return p_pred, p_gt

def save_visualisation(out_dir, run, ckpt_tag, split, idx, prob, gt, hist_last, thr):
    viz_dir = out_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = sample_stem(run, ckpt_tag, split, idx)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    axes[0].imshow(hist_last, cmap='gray')
    axes[0].set_title(f'{run} | {ckpt_tag}\nInput VIIRS I1 (t) | {split}[{idx}]',
                      fontsize=9)
    axes[1].imshow(gt, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Ground truth t+1', fontsize=9)
    axes[2].imshow(prob, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Prediction probability', fontsize=9)
    pb = (prob > thr).astype(np.uint8)
    gb = (gt   > 0.5).astype(np.uint8)
    axes[3].imshow(np.abs(pb - gb), cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    axes[3].set_title(f'|pred>={thr:.2f} - GT|  IoU={iou_bin(pb, gb):.3f}', fontsize=9)
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    p = viz_dir / f'{stem}__viz.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return p

def write_manifest(out_dir, rows):
    mp = out_dir / 'manifest.csv'
    if not rows:
        mp.write_text('')
        return mp
    with open(mp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return mp

def mirror_to_drive(local_dir, drive_dir):
    drive_dir.parent.mkdir(parents=True, exist_ok=True)
    if drive_dir.exists():
        shutil.rmtree(drive_dir)
    shutil.copytree(local_dir, drive_dir)

print('Builders + exporters ready')
"""
    )
)

# ---------------------------------------------------------------- CELL 9 ----
cells.append(
    md(
        """## Section 8 \u00b7 Discover AF runs + checkpoints

Scans `OUTPUT_ROOT`, keeps runs with `label_channel == 1` **and** a V2-compatible
checkpoint. Lists every `.pt` file per run (ordered best -> last -> epoch_N ->
others) so you can see exactly what is about to be processed.
"""
    )
)

cells.append(
    code(
        r"""runs_all = discover_af_runs(OUTPUT_ROOT)
print(f'Discovered {len(runs_all)} AF run(s) under {OUTPUT_ROOT}')

v2_runs      = []
skipped_runs = []
for r in runs_all:
    reason = None
    if not is_v2_config(r['cfg']):
        reason = 'config has v2_enabled=False (V1 architecture - use V1 inference script)'
    elif not r['checkpoints']:
        reason = 'no .pt files found in run directory'
    else:
        ok = False
        for p in r['checkpoints']:
            try:
                info = peek_checkpoint(p)
                if info['is_v2']:
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            reason = 'no checkpoint exposes V2-style keys (encoder.b1.c1.weight ...)'
    if reason is None:
        v2_runs.append(r)
    else:
        skipped_runs.append((r['name'], reason))

print(f"\n-> V2 AF runs to process : {len(v2_runs)}")
for r in v2_runs:
    print(f"  {r['name']:10s}  ckpts={len(r['checkpoints'])}  config={r['config_path'].name}")
    for p in r['checkpoints']:
        info = peek_checkpoint(p)
        print(f"     - {p.name:30s}  epoch={info['epoch']}  "
              f"val_iou={info['val_iou_best_thr']}  thr={info['thr_best']}  "
              f"v2={info['is_v2']}  ema={info['has_ema']}")

if skipped_runs:
    print('\n-> Runs skipped upfront:')
    for name, why in skipped_runs:
        print(f'  {name:10s}  {why}')
"""
    )
)

# --------------------------------------------------------------- CELL 10 ----
cells.append(
    md(
        """## Section 9 \u00b7 Batch one-step inference over all train + val samples

For each V2 run \u00d7 every checkpoint \u00d7 {train, val}:

- rebuild model from the run's config
- load weights (`strict=False`, prefer `ema_state`)
- iterate the dataset in order (no shuffle) and export `.npy` pred + `.npy` GT for
  every sample
- render a per-sample PNG visualisation (configurable cap via `MAX_VIZ_PER_SPLIT`
  \u2014 raw `.npy` export is always uncapped)
- write `manifest.csv` and `inference_summary.json`
- mirror the run/checkpoint folder from local cache to Drive

All per-checkpoint failures are isolated \u2014 one bad checkpoint will not kill the loop.
"""
    )
)

cells.append(
    code(
        r"""# Tunable export knobs
MAX_VIZ_PER_SPLIT = 40     # cap expensive PNGs per split; .npy pred/gt always all
USE_AMP           = True   # bfloat16 autocast on GPU
MIRROR_TO_DRIVE   = True   # set False to stay local-only (debug)

agg_rows = []   # one row per (run, ckpt, split) triple

for r_info in v2_runs:
    run_name = r_info['name']
    run_dir  = r_info['dir']
    cfg      = r_info['cfg']
    ckpts    = r_info['checkpoints']

    # Remap Drive paths baked into the config -> local Colab SSD
    cfg['data']['root_train'] = str(DATASET_TRAIN)
    cfg['data']['root_val']   = str(DATASET_VAL)

    print('\n' + '=' * 78)
    print(f'  RUN  {run_name}   ({len(ckpts)} checkpoint(s))')
    print('=' * 78)

    # Build datasets ONCE per run (identical ordering across all checkpoints
    # of the same run - ensures fair comparison sample-by-sample).
    t0 = time.time()
    train_ds, aux_ch_tr = build_dataset(DATASET_TRAIN, cfg)
    val_ds,   aux_ch_va = build_dataset(DATASET_VAL,   cfg)
    assert aux_ch_tr == aux_ch_va, f'aux channel mismatch {aux_ch_tr} vs {aux_ch_va}'
    aux_ch = aux_ch_tr
    print(f'  train samples : {len(train_ds):5d}   val samples : {len(val_ds):5d}   '
          f'aux_ch={aux_ch}   ({time.time()-t0:.1f}s)')

    for ckpt_path in ckpts:
        ckpt_tag = checkpoint_tag(ckpt_path)
        print(f'\n  -- ckpt  {ckpt_path.name}   tag={ckpt_tag}')
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        except Exception as e:
            print(f'     !! load failed: {e}')
            agg_rows.append(dict(run=run_name, ckpt=ckpt_tag, split='-',
                                 status='load_failed', error=str(e)[:240]))
            continue

        info = dict(
            epoch            = ckpt.get('epoch'),
            val_iou_best_thr = ckpt.get('val_iou_best_thr'),
            thr_best         = ckpt.get('thr_best', ckpt.get('val_thr_best', 0.5)),
            has_ema          = 'ema_state' in ckpt and ckpt['ema_state'] is not None,
        )
        thr = float(info['thr_best']) if info['thr_best'] is not None else 0.5
        print(f"     epoch={info['epoch']}   val_iou={info['val_iou_best_thr']}   "
              f"thr={thr:.2f}   has_ema={info['has_ema']}")

        try:
            model = build_v2_model_from_config(cfg, aux_ch=aux_ch).to(DEVICE)
            load_info = load_state_into_model(model, ckpt)
            model.eval()
        except Exception as e:
            traceback.print_exc()
            agg_rows.append(dict(run=run_name, ckpt=ckpt_tag, split='-',
                                 status='build_failed', error=str(e)[:240]))
            continue
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"     built V2 model ({n_param:.2f} M params)   "
              f"missing={load_info['n_missing_keys']}  "
              f"unexpected={load_info['n_unexpected_keys']}  "
              f"used_ema={load_info['used_ema']}")
        if load_info['n_unexpected_keys']:
            print(f"     unexpected sample: {load_info['unexpected_sample']}")
        if load_info['n_missing_keys']:
            print(f"     missing    sample: {load_info['missing_sample']}")

        ckpt_local = INFER_ROOT_LOCAL / run_name / ckpt_tag
        ckpt_drive = INFER_ROOT_DRIVE / run_name / ckpt_tag
        ckpt_local.mkdir(parents=True, exist_ok=True)

        (ckpt_local / 'checkpoint_info.txt').write_text(
            'run            : ' + run_name
            + '\ncheckpoint     : ' + str(ckpt_path.relative_to(OUTPUT_ROOT))
            + '\ntag            : ' + ckpt_tag
            + f"\nepoch          : {info['epoch']}"
            + f"\nval_iou_best_thr: {info['val_iou_best_thr']}"
            + f'\nthr_best       : {thr}'
            + f"\nhas_ema        : {info['has_ema']}"
            + f"\nused_ema       : {load_info['used_ema']}"
            + f"\nmissing_keys   : {load_info['n_missing_keys']}"
            + f"\nunexpected_keys: {load_info['n_unexpected_keys']}"
            + f'\nn_parameters   : {n_param:.3f} M'
            + f'\ndevice         : {DEVICE}\n'
        )

        for split_name, ds in [('train', train_ds), ('val', val_ds)]:
            split_dir = ckpt_local / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            ious = []
            bar = tqdm(range(len(ds)),
                       desc=f'     {split_name:5s}  {run_name}/{ckpt_tag}',
                       ncols=90, leave=False)
            for idx in bar:
                try:
                    sample = ds[idx]
                    res    = run_one_step(model, sample, DEVICE, USE_AMP)
                except Exception as e:
                    rows.append(dict(run=run_name, ckpt=ckpt_tag, split=split_name,
                                     sample_idx=idx, pred='', gt='', viz='',
                                     iou_thr='', status=f'inference_error: {str(e)[:120]}'))
                    continue
                p_pred, p_gt = save_pred_gt_npy(
                    split_dir, run_name, ckpt_tag, split_name,
                    idx, res['prob'], res['gt'])
                pb = (res['prob'] > thr).astype(np.uint8)
                gb = (res['gt']   > 0.5).astype(np.uint8)
                this_iou = iou_bin(pb, gb)
                ious.append(this_iou)
                viz_rel = ''
                if idx < MAX_VIZ_PER_SPLIT:
                    vp = save_visualisation(
                        split_dir, run_name, ckpt_tag, split_name,
                        idx, res['prob'], res['gt'], res['hist_last'], thr=thr)
                    viz_rel = str(vp.relative_to(split_dir))
                rows.append(dict(
                    run=run_name, ckpt=ckpt_tag, split=split_name,
                    sample_idx=idx,
                    pred=str(p_pred.relative_to(split_dir)),
                    gt  =str(p_gt  .relative_to(split_dir)),
                    viz =viz_rel,
                    iou_thr=f'{this_iou:.4f}',
                    status='ok',
                ))

            write_manifest(split_dir, rows)
            mean_iou = float(np.mean(ious)) if ious else 0.0
            summary = dict(
                run=run_name, checkpoint=str(ckpt_path.name),
                checkpoint_tag=ckpt_tag, split=split_name,
                n_samples=len(ds), n_exported=len(rows),
                n_viz=min(len(ds), MAX_VIZ_PER_SPLIT),
                threshold=thr, mean_iou_thr=mean_iou,
                device=str(DEVICE), used_ema=load_info['used_ema'],
                n_missing_keys=load_info['n_missing_keys'],
                n_unexpected_keys=load_info['n_unexpected_keys'],
                inference_mode='one-step',
                horizon_override=1,
                label_channel=int(cfg['data'].get('label_channel', 1)),
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            )
            with open(split_dir / 'inference_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            agg_rows.append(dict(
                run=run_name, ckpt=ckpt_tag, split=split_name, status='ok',
                n_samples=len(ds), mean_iou_thr=mean_iou, threshold=thr,
                epoch=info['epoch'], val_iou_best_thr=info['val_iou_best_thr'],
            ))
            print(f'     {split_name:5s}  done  n={len(ds):5d}  '
                  f'mean_iou@{thr:.2f}={mean_iou:.4f}')

        # Free model before moving on
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        if MIRROR_TO_DRIVE:
            t0 = time.time()
            mirror_to_drive(ckpt_local, ckpt_drive)
            print(f'     mirrored -> {ckpt_drive}   ({time.time()-t0:.1f}s)')

print('\n' + '=' * 78)
print(f' Batch complete.  Rows collected: {len(agg_rows)}')
print('=' * 78)
"""
    )
)

# --------------------------------------------------------------- CELL 11 ----
cells.append(
    md(
        """## Section 10 \u00b7 Representative inline visualisations

Displays one PNG per (run, checkpoint, split) triple directly in the notebook so
you can eyeball the quality without opening Drive. Pulls the **first saved**
visualisation for each triple (deterministic \u2014 same sample across checkpoints of
the same run).
"""
    )
)

cells.append(
    code(
        r"""from IPython.display import Image, display, Markdown

for row in agg_rows:
    if row.get('status') != 'ok':
        continue
    run, ckpt, split = row['run'], row['ckpt'], row['split']
    viz_dir = INFER_ROOT_LOCAL / run / ckpt / split / 'visualizations'
    pngs = sorted(viz_dir.glob('*.png'))
    if not pngs:
        continue
    display(Markdown(f'**{run} / {ckpt} / {split}** - sample 0  '
                     f"(mean IoU @ thr={row['threshold']:.2f} = {row['mean_iou_thr']:.4f})"))
    display(Image(filename=str(pngs[0])))
"""
    )
)

# --------------------------------------------------------------- CELL 12 ----
cells.append(
    md(
        """## Section 11 \u00b7 Final report

Writes a consolidated summary CSV + JSON alongside the per-run folders, and prints
a tidy table showing which runs/checkpoints/splits ran, how many samples each
covers, and the mean IoU at the checkpoint's own best threshold.
"""
    )
)

cells.append(
    code(
        r"""csv_path  = INFER_ROOT_LOCAL / 'all_runs_all_checkpoints_summary.csv'
json_path = INFER_ROOT_LOCAL / 'all_runs_all_checkpoints_summary.json'

if agg_rows:
    fieldnames = sorted({k for r in agg_rows for k in r.keys()})
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in agg_rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
with open(json_path, 'w') as f:
    json.dump(dict(
        device=str(DEVICE),
        cuda_available=torch.cuda.is_available(),
        output_root=str(OUTPUT_ROOT),
        infer_root_drive=str(INFER_ROOT_DRIVE),
        infer_root_local=str(INFER_ROOT_LOCAL),
        n_v2_runs=len(v2_runs),
        skipped_runs=[{'run': n, 'reason': w} for n, w in skipped_runs],
        rows=agg_rows,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    ), f, indent=2)

if MIRROR_TO_DRIVE:
    shutil.copy2(csv_path,  INFER_ROOT_DRIVE / csv_path.name)
    shutil.copy2(json_path, INFER_ROOT_DRIVE / json_path.name)
    print('Summary mirrored to Drive:')
    print('  ', INFER_ROOT_DRIVE / csv_path.name)
    print('  ', INFER_ROOT_DRIVE / json_path.name)

print('\n' + '=' * 86)
print('  FINAL REPORT')
print('=' * 86)
print(f'  device            : {DEVICE}')
print(f'  V2 runs processed : {len(v2_runs)}')
print(f'  skipped upfront   : {len(skipped_runs)}')
for n, w in skipped_runs:
    print(f'      {n:12s} | {w}')
print()
print(f'  {"run":10s} {"ckpt":22s} {"split":6s} {"status":16s} '
      f'{"n":>6s} {"thr":>5s} {"mean_iou":>9s}')
print('  ' + '-' * 82)
for r in agg_rows:
    mean = r.get('mean_iou_thr', '-')
    if isinstance(mean, float):
        mean = f'{mean:.4f}'
    thr = r.get('threshold', '-')
    if isinstance(thr, float):
        thr = f'{thr:.2f}'
    print(f"  {r.get('run',''):10s} {r.get('ckpt',''):22s} "
          f"{r.get('split',''):6s} {r.get('status',''):16s} "
          f"{str(r.get('n_samples','-')):>6s} {str(thr):>5s} {str(mean):>9s}")
print()
print(f'  summary csv  : {csv_path}')
print(f'  summary json : {json_path}')
print('=' * 86)
"""
    )
)

# --------------------------------------------------------- assemble ----
nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = (
    r"C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/"
    r"data_tools/colab/colab_inference_af_v2_all_checkpoints.ipynb"
)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("WROTE", OUT)
print("num cells:", len(cells))

# Re-parse to validate JSON round-trip + cell counts
nb2 = json.load(open(OUT, encoding="utf-8"))
assert len(nb2["cells"]) == len(cells), "cell-count mismatch"
print("re-parsed OK   cells =", len(nb2["cells"]))
