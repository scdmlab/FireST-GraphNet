"""Builds the colab_af_swinunetr_feature48_baseline.ipynb notebook.

Reuses the structure / training pipeline of colab_af_baseline_comparison.ipynb but
trains ONLY SwinUNETR with feature_size=48 (vs the original 24), keeping every other
fairness setting identical. Saves outputs to output/baseline_swinunetr_feature48/.
"""
from pathlib import Path
import json

OUT = Path(r"C:\Users\22125\OneDrive - UW-Madison\FireST-GraphNet-Clean\data_tools\colab\colab_af_swinunetr_feature48_baseline.ipynb")


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


cells = []

# ============== Cell 0: Title + motivation ==============
cells.append(md("""# FireST-GraphNet — Stronger SwinUNETR AF Baseline (feature_size = 48)

This notebook re-runs the **SwinUNETR baseline only** for Active-Fire (AF) detection,
under the **same protocol** as the original baseline comparison
(`colab_af_baseline_comparison.ipynb`), but with a more reasonable transformer capacity:

| Setting | Original (baseline comparison) | This notebook |
|---|---|---|
| Architecture | MONAI `SwinUNETR` (2D) | MONAI `SwinUNETR` (2D) |
| `feature_size` | **24** (MONAI minimum) | **48** (standard medical-imaging size) |
| `depths` | (2, 2, 2, 2) | (2, 2, 2, 2) |
| `num_heads` | (3, 6, 12, 24) | (3, 6, 12, 24) |
| Total parameters | ≈ 6.40 M | ≈ ?? (printed live below) |
| Pretrained weights | none (from scratch) | none (from scratch) |
| Every other setting | unified | **identical** to the original baseline |

## Why
The previous bug-check diagnosis confirmed that the original SwinUNETR result was
**valid** (no bug in code, fairness, or parameter counting), but that the model was
configured at MONAI's **minimum** preset. The first patch embedding compressed 280
input channels to only 24, and the deepest stage had only 192 channels — narrower
than U-Net's 512-channel bottleneck. SwinUNETR also showed clear overfitting
(train-IoU 0.834 vs val-IoU 0.594).

This notebook tests whether a **more reasonable** SwinUNETR capacity (`feature_size=48`)
changes the conclusion. **All** other settings (dataset, splits, loss, optimizer,
schedule, threshold, metric function, best-checkpoint selection rule) are kept
**identical** to the original baseline comparison so the result is directly comparable.

## What this notebook does NOT do
- It does **not** retrain U-Net or any other baseline — the old U-Net result is read
  from `output/baseline/baseline_comparison/unet/metrics.csv`.
- It does **not** change the dataset, label, channels, T_in, horizon, or augmentation.
- It does **not** load any pretrained weights — fair comparison vs the from-scratch
  SwinUNETR-24 means SwinUNETR-48 also trains from scratch.

## Outputs (saved on Drive)
```
output/baseline_swinunetr_feature48/
  ├── best_model.pt              best-val-IoU checkpoint
  ├── last_model.pt              last-epoch checkpoint
  ├── metrics.csv                per-epoch train/val log
  ├── config.json                the CONFIG used
  ├── param_count.json           total / trainable / per-module
  ├── val_summary.json           best-epoch val metrics + train_iou + gap
  ├── comparison_table.csv       SwinUNETR-24 vs SwinUNETR-48 vs U-Net
  ├── swinunetr_feature48_baseline_report.md
  └── figures/
        ├── train_val_loss.png
        ├── train_val_iou.png
        ├── metric_comparison.png
        ├── param_count_comparison.png
        ├── precision_recall.png
        └── iou_vs_params.png
```
"""))

# ============== Cell 1: Mount Drive ==============
cells.append(md("## 1 · Environment setup"))

cells.append(code("""# 1.1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""))

cells.append(code("""# 1.2 — Install dependencies (same as the baseline comparison notebook)
!pip install -q torch torchvision torchaudio
!pip install -q "monai>=1.3.2" einops
!pip install -q torch-geometric
!pip install -q pyyaml tensorboard scikit-learn matplotlib pandas tqdm
print('\\nDependencies installed')"""))

cells.append(code("""# 1.3 — Project path + sys.path
import os, sys, shutil, glob, time, json, gc, math
from pathlib import Path

PROJECT_PATH = '/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean'
assert Path(PROJECT_PATH).exists(), f'Project folder not found: {PROJECT_PATH}'

os.chdir('/content')
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

print(f'Project: {PROJECT_PATH}')
try:
    import firest_graphnet  # noqa: F401
    print('firest_graphnet importable: yes')
except Exception as e:
    print(f'firest_graphnet importable: NO -- {e}')"""))

cells.append(code("""# 1.4 — Imports, seed, device
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from firest_graphnet.utils.seed import seed_everything
from firest_graphnet.utils.logging import MetricLogger
from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider import build_static_provider
from firest_graphnet.data.transforms import get_forecast_train_transforms

SEED = 42
seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True"""))

# ============== Cell: Configuration ==============
cells.append(md("""## 2 · Configuration

Identical to the original baseline comparison (`colab_af_baseline_comparison.ipynb` cell 2.1),
except:
- `output_root` / `local_output` now point at `baseline_swinunetr_feature48`.
- A new `swin_feature_size = 48` knob (the **only** modeling change).
- Optional `batch_size` fallback: if a memory test fails at the original `batch_size=8`,
  we fall back to 4 and document it in the report. (The note from a previous Colab session
  is that batch_size=8 used only ~13 GB on a high-memory GPU, so this fallback shouldn't trigger.)

`FAST_DEBUG = True` runs a 3-epoch smoke test before any long training."""))

cells.append(code("""# 2.1 — Master config (identical to baseline comparison, plus swin_feature_size=48)
FAST_DEBUG = False    # set True for a ~5-minute smoke test

CONFIG = {
    # Paths
    'project_path' : PROJECT_PATH,
    'drive_train'  : f'{PROJECT_PATH}/dataset_train',
    'drive_val'    : f'{PROJECT_PATH}/dataset_val',
    'local_train'  : '/content/dataset_train',
    'local_val'    : '/content/dataset_val',
    'output_root'  : f'{PROJECT_PATH}/output/baseline_swinunetr_feature48',
    'local_output' : '/content/outputs/baseline_swinunetr_feature48',

    # Data (UNCHANGED — must match original baseline)
    'T_in'         : 10,
    'horizon'      : 1,
    'label_channel': 1,
    'img_size'     : 256,
    'in_viirs'     : 8,
    'in_aux'       : 19,
    'in_static'    : 1,

    # Training (UNCHANGED unless memory forces a fallback)
    'batch_size'         : 8,
    'batch_size_fallback': 4,
    'epochs'             : 30,
    'lr'                 : 1.5e-4,
    'weight_decay'       : 5e-3,
    'grad_clip'          : 0.5,
    'mixed_precision'    : True,
    'val_threshold'      : 0.5,

    # Loss (UNCHANGED)
    'pos_weight' : 4.0,
    'focal_gamma': 2.0,
    'dice_weight': 0.3,

    # Augmentation (UNCHANGED)
    'aug_noise_std': 0.0,

    # Reproducibility (UNCHANGED)
    'seed'        : SEED,
    'num_workers' : 0,
    'pin_memory'  : True,

    # New knob: SwinUNETR feature size
    'swin_feature_size': 48,
    'swin_depths'      : (2, 2, 2, 2),
    'swin_num_heads'   : (3, 6, 12, 24),
    'swin_use_v2'      : False,
}
CONFIG['in_per_step'] = CONFIG['in_viirs'] + CONFIG['in_aux'] + CONFIG['in_static']  # 28

if FAST_DEBUG:
    CONFIG['epochs']     = 3
    CONFIG['batch_size'] = 2
    CONFIG['fast_debug_train_n'] = 64
    CONFIG['fast_debug_val_n']   = 16

Path(CONFIG['local_output']).mkdir(parents=True, exist_ok=True)
print(json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in CONFIG.items()
                  if not (k.endswith('_path') or k.startswith('local_') or k.startswith('drive_'))},
                 indent=2))"""))

# ============== Cell: Data preparation ==============
cells.append(md("""## 3 · Data preparation

Copies the same `dataset_train` / `dataset_val` from Drive to local SSD (Colab convention),
then builds the same providers and `TSSatFireForecastDataset` as the original baseline
comparison. **No change to data, split, or augmentation.**"""))

cells.append(code("""# 3.1 — Copy dataset to local SSD (one-time per Colab session)
def _copy_dataset(src_dir, dst_dir):
    src, dst = Path(src_dir), Path(dst_dir)
    if not src.exists():
        raise FileNotFoundError(f'Drive dataset missing: {src}')
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.suffix not in {'.npy', '.json', '.yaml', '.yml'}:
            continue
        target = dst / f.name
        if target.exists() and target.stat().st_size == f.stat().st_size:
            continue
        print(f'  copying {f.name} ({f.stat().st_size/1e6:.1f} MB)')
        shutil.copy2(f, target)

print('Train:'); _copy_dataset(CONFIG['drive_train'], CONFIG['local_train'])
print('Val  :'); _copy_dataset(CONFIG['drive_val'],   CONFIG['local_val'])"""))

cells.append(code("""# 3.2 — Build providers + datasets + loaders (identical to baseline comparison)
fp_train = build_firepred_provider(CONFIG['local_train'], spatial_size=(CONFIG['img_size'], CONFIG['img_size']))
st_train = build_static_provider  (CONFIG['local_train'], spatial_size=(CONFIG['img_size'], CONFIG['img_size']))
fp_val   = build_firepred_provider(CONFIG['local_val'],   spatial_size=(CONFIG['img_size'], CONFIG['img_size']))
st_val   = build_static_provider  (CONFIG['local_val'],   spatial_size=(CONFIG['img_size'], CONFIG['img_size']))

train_tf = get_forecast_train_transforms(noise_std=CONFIG['aug_noise_std'])

train_ds = TSSatFireForecastDataset(
    data_dir=CONFIG['local_train'],
    T_in=CONFIG['T_in'], horizon=CONFIG['horizon'], label_channel=CONFIG['label_channel'],
    firepred_provider=fp_train, static_provider=st_train,
    transform=train_tf,
)
val_ds = TSSatFireForecastDataset(
    data_dir=CONFIG['local_val'],
    T_in=CONFIG['T_in'], horizon=CONFIG['horizon'], label_channel=CONFIG['label_channel'],
    firepred_provider=fp_val, static_provider=st_val,
    transform=None,
)

if FAST_DEBUG:
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, list(range(min(CONFIG['fast_debug_train_n'], len(train_ds)))))
    val_ds   = Subset(val_ds,   list(range(min(CONFIG['fast_debug_val_n'],   len(val_ds)))))

def make_loaders(bs):
    tr = DataLoader(train_ds, batch_size=bs, shuffle=True,
                    num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
                    drop_last=True)
    va = DataLoader(val_ds, batch_size=bs, shuffle=False,
                    num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
    return tr, va

train_loader, val_loader = make_loaders(CONFIG['batch_size'])
print(f'Train: {len(train_ds)} samples -> {len(train_loader)} batches (bs={CONFIG[\"batch_size\"]})')
print(f'Val  : {len(val_ds)} samples -> {len(val_loader)} batches')"""))

cells.append(code("""# 3.3 — Sanity-check shapes from a single val batch
sample = next(iter(val_loader))
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f'{k:12s}: {tuple(v.shape)}  {v.dtype}')
    elif isinstance(v, dict):
        print(f'{k:12s}: dict({list(v.keys())})')
    else:
        print(f'{k:12s}: {type(v).__name__}={v}')

assert sample['af_hist'].shape[1:] == (CONFIG['T_in'], CONFIG['in_viirs'], CONFIG['img_size'], CONFIG['img_size'])
assert sample['aux_hist'].shape[1:] == (CONFIG['T_in'], CONFIG['in_aux'],   CONFIG['img_size'], CONFIG['img_size'])
assert sample['static'].shape[1:]   == (CONFIG['in_static'], CONFIG['img_size'], CONFIG['img_size'])
assert sample['af_fut'].shape[1:]   == (CONFIG['horizon'], 1, CONFIG['img_size'], CONFIG['img_size'])
print('\\nShapes match the unified contract.')"""))

# ============== Cell: Helpers (loss + input prep) ==============
cells.append(md("""## 4 · Shared helpers (loss, input prep)

Both functions are **byte-identical** to the original baseline comparison:
the same `FocalDiceLoss` and the same `prepare_inputs(want_seq=False)` that yields
`(B, T·C, H, W) = (B, 280, 256, 256)`. This is essential for fairness."""))

cells.append(code("""# 4.1 — FocalDiceLoss (identical to original baseline)
class FocalDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 4.0, gamma: float = 2.0, dice_weight: float = 0.3):
        super().__init__()
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.register_buffer('pw', torch.tensor(float(pos_weight)))

    def forward(self, logits, targets):
        while logits.ndim > 4:
            logits  = logits.flatten(0, 1)
            targets = targets.flatten(0, 1)
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        f_w   = (1 - p_t).pow(self.gamma)
        pw    = self.pw.to(logits.device)
        px_w  = targets * pw + (1 - targets)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal = (f_w * px_w * bce).mean()
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice  = 1 - ((2 * inter + 1.0) / (union + 1.0)).mean()
        return focal + self.dice_weight * dice


# 4.2 — Input preparation (identical to original baseline, channel-stack path)
def prepare_inputs(batch, device):
    af_hist  = batch['af_hist'].to(device, non_blocking=True)
    aux_hist = batch['aux_hist'].to(device, non_blocking=True)
    static   = batch['static'].to(device, non_blocking=True)
    af_fut   = batch['af_fut'].to(device, non_blocking=True)
    B, T, _, H, W = af_hist.shape
    static_t = static.unsqueeze(1).expand(-1, T, -1, -1, -1)
    x_seq    = torch.cat([af_hist, aux_hist, static_t], dim=2)   # (B, T, 28, H, W)
    x = x_seq.reshape(B, T * x_seq.shape[2], H, W)               # (B, 280, H, W)
    y = af_fut[:, 0]                                              # (B, 1, H, W)
    return x, y"""))

# ============== Cell: Model build + param analysis ==============
cells.append(md("""## 5 · SwinUNETR (feature_size = 48) — build + parameter analysis

Identical constructor to the original except `feature_size=48` (was 24).
Below the build we print:
- total / trainable parameters
- per-top-module breakdown
- patch-embed details (the layer that previously compressed 280 → 24)
- a dummy-batch forward to confirm input/output shapes match the AF target."""))

cells.append(code("""# 5.1 — Model factory
from monai.networks.nets import SwinUNETR

C_IN_STACK = CONFIG['T_in'] * CONFIG['in_per_step']   # 10 * 28 = 280
IMG_HW     = (CONFIG['img_size'], CONFIG['img_size'])


class StackedWrapper(nn.Module):
    \"\"\"Channel-stack: (B, T, C, H, W) -> (B, T*C, H, W) -> backbone -> (B, 1, H, W).\"\"\"
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
    def forward(self, batch):
        x, _ = prepare_inputs(batch, next(self.parameters()).device)
        return self.backbone(x)


def build_swin_unetr(feature_size: int):
    net = SwinUNETR(
        in_channels=C_IN_STACK, out_channels=1,
        feature_size=feature_size,
        depths=tuple(CONFIG['swin_depths']),
        num_heads=tuple(CONFIG['swin_num_heads']),
        spatial_dims=2,
        use_v2=CONFIG['swin_use_v2'],
        use_checkpoint=False,
    )
    return StackedWrapper(net)"""))

cells.append(code("""# 5.2 — Try feature_size=48 first; if MONAI rejects it, try 36 and document.
# Construction-time check only. The full GPU shape-check is done in 5.4 (cheap)
# and the OOM check in cell 6.1 (smoke test). A full CPU forward of SwinUNETR with
# 280 input channels is too slow to be a useful gate.
import traceback

CANDIDATE_FEATURE_SIZES = [48, 36]   # MONAI requires feature_size divisible by 12
swin_model = None
chosen_fs = None
for fs in CANDIDATE_FEATURE_SIZES:
    try:
        m = build_swin_unetr(fs)
        # Probe with a tiny synthetic tensor on CPU (1 x C x 256 x 256, the same H/W
        # used in training — needed because SwinUNETR's window_size=7 expects
        # adequately-sized feature maps).
        with torch.no_grad():
            probe = torch.zeros(1, C_IN_STACK, CONFIG['img_size'], CONFIG['img_size'])
            out = m.backbone(probe)
        assert tuple(out.shape) == (1, 1, CONFIG['img_size'], CONFIG['img_size']), \\
            f'unexpected backbone out: {tuple(out.shape)}'
        swin_model = m
        chosen_fs = fs
        print(f'OK  feature_size={fs} built, backbone out shape = {tuple(out.shape)}')
        break
    except Exception as e:
        print(f'feature_size={fs} failed: {type(e).__name__}: {e}')
        traceback.print_exc()
        try:
            del m
        except NameError:
            pass
        gc.collect()

assert swin_model is not None, 'No feature_size candidate worked. Stop here.'
CONFIG['swin_feature_size'] = chosen_fs
print(f'\\nUsing SwinUNETR with feature_size = {chosen_fs}')"""))

cells.append(code("""# 5.3 — Parameter analysis (total / trainable / per-module / patch embed)
def per_module_breakdown(m, top_only=True):
    rows = []
    for n, mod in m.named_children():
        p = sum(pp.numel() for pp in mod.parameters())
        rows.append((n, p))
    return rows


total    = sum(p.numel() for p in swin_model.parameters())
trainable= sum(p.numel() for p in swin_model.parameters() if p.requires_grad)
print(f'SwinUNETR (feature_size={chosen_fs})')
print(f'  total     = {total:,}  ({total/1e6:.3f} M)')
print(f'  trainable = {trainable:,}  ({trainable/1e6:.3f} M)')
print(f'  frozen    = {total - trainable}')

# Drill into the actual SwinUNETR (the backbone inside StackedWrapper)
backbone = swin_model.backbone
print('\\nTop-level modules of SwinUNETR backbone:')
top_rows = per_module_breakdown(backbone)
for n, p in top_rows:
    print(f'  {n:18s} -> {p/1e6:7.3f} M')

# Patch embed details
pe = backbone.swinViT.patch_embed
print('\\nPatch embed (the layer that previously compressed 280 -> 24):')
print(f'  {pe}')
patch_embed_params = sum(p.numel() for p in pe.parameters())
print(f'  params = {patch_embed_params:,}')

# Save param info
PARAM_INFO = {
    'feature_size': chosen_fs,
    'depths'      : list(CONFIG['swin_depths']),
    'num_heads'   : list(CONFIG['swin_num_heads']),
    'total_params'    : total,
    'trainable_params': trainable,
    'frozen_params'   : total - trainable,
    'patch_embed_params': patch_embed_params,
    'patch_embed_repr'  : repr(pe),
    'per_top_module_M'  : {n: p/1e6 for n, p in top_rows},
}
with open(Path(CONFIG['local_output']) / 'param_count.json', 'w') as f:
    json.dump(PARAM_INFO, f, indent=2)
print(f'\\nSaved param_count.json')"""))

cells.append(code("""# 5.4 — Dummy forward to confirm input / output / target shapes line up
swin_model = swin_model.to(device).eval()
b = next(iter(val_loader))
print(f'Input  af_hist : {tuple(b[\"af_hist\"].shape)}')
print(f'Input  aux_hist: {tuple(b[\"aux_hist\"].shape)}')
print(f'Input  static  : {tuple(b[\"static\"].shape)}')
print(f'Target af_fut  : {tuple(b[\"af_fut\"].shape)}  -> y after [:,0] = (B, 1, H, W)')
with torch.no_grad():
    logits = swin_model(b)
print(f'Output logits  : {tuple(logits.shape)}')
y = b['af_fut'].to(device)[:, 0]
print(f'Label shape    : {tuple(y.shape)}')
assert logits.shape == y.shape, f'logits {logits.shape} != y {y.shape}'
print('\\nShape check OK.')"""))

# ============== Cell: Memory smoke test ==============
cells.append(md("""## 6 · Memory smoke test

Before committing to a full 30-epoch run, do **one full step** (forward + loss + backward +
optimizer.step) at the chosen `batch_size` and print peak VRAM. If this fails with OOM,
fall back to `batch_size_fallback=4` and document it."""))

cells.append(code("""# 6.1 — One full step at the chosen batch size; fall back to 4 on OOM
def one_full_step(bs):
    tr_loader, va_loader = make_loaders(bs)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    crit = FocalDiceLoss(pos_weight=CONFIG['pos_weight'], gamma=CONFIG['focal_gamma'],
                         dice_weight=CONFIG['dice_weight']).to(device)
    opt = torch.optim.AdamW(swin_model.parameters(), lr=CONFIG['lr'],
                            weight_decay=CONFIG['weight_decay'])
    use_amp = CONFIG['mixed_precision'] and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    swin_model.train()
    batch = next(iter(tr_loader))
    y = batch['af_fut'].to(device)[:, 0]
    with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        logits = swin_model(batch)
        loss = crit(logits, y)
    opt.zero_grad(set_to_none=True)
    if use_amp and amp_dtype == torch.float16:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(swin_model.parameters(), CONFIG['grad_clip'])
        scaler.step(opt); scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(swin_model.parameters(), CONFIG['grad_clip'])
        opt.step()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return float(loss.item()), peak, tr_loader, va_loader


peak_gb_used = None
try:
    loss_val, peak_gb_used, train_loader, val_loader = one_full_step(CONFIG['batch_size'])
    print(f'OK  bs={CONFIG[\"batch_size\"]}  loss={loss_val:.4f}  peak VRAM={peak_gb_used:.2f} GB')
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache(); gc.collect()
    print(f'OOM at bs={CONFIG[\"batch_size\"]}, falling back to {CONFIG[\"batch_size_fallback\"]}')
    CONFIG['batch_size'] = CONFIG['batch_size_fallback']
    loss_val, peak_gb_used, train_loader, val_loader = one_full_step(CONFIG['batch_size'])
    print(f'OK  bs={CONFIG[\"batch_size\"]}  loss={loss_val:.4f}  peak VRAM={peak_gb_used:.2f} GB')

# fresh state for real training
torch.cuda.empty_cache(); gc.collect()
seed_everything(SEED)"""))

# ============== Cell: train_model ==============
cells.append(md("""## 7 · Unified training pipeline

This `train_model` is the same trainer used in the original baseline comparison. We pass
the (re-seeded) SwinUNETR-48 through it. Cosine LR, AdamW, AMP, grad-clip, early-stop
patience=8, best-by-val-IoU — all identical to the original run."""))

cells.append(code("""# 7.1 — Unified trainer (identical to baseline comparison)
def _aggregate_metrics(probs_buf, targets_buf, threshold):
    p = torch.cat(probs_buf)
    t = torch.cat(targets_buf)
    pred = (p >= threshold).float()
    tp = (pred * t).sum().item()
    fp = (pred * (1 - t)).sum().item()
    fn = ((1 - pred) * t).sum().item()
    eps = 1e-7
    iou  = tp / max(tp + fp + fn, eps)
    prec = tp / max(tp + fp,      eps)
    rec  = tp / max(tp + fn,      eps)
    f1   = 2 * prec * rec / max(prec + rec, eps)
    return {'iou': iou, 'f1': f1, 'precision': prec, 'recall': rec}


def train_model(name: str, model: nn.Module, cfg: dict, train_loader, val_loader,
                output_dir: Path, patience: int = 8):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = FocalDiceLoss(pos_weight=cfg['pos_weight'], gamma=cfg['focal_gamma'],
                              dice_weight=cfg['dice_weight']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    use_amp = cfg['mixed_precision'] and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    logger = MetricLogger(output_dir / 'metrics.csv')
    best_iou = -1.0
    best_epoch = 0
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_probs, tr_tgts = [], []
        for batch in tqdm(train_loader, desc=f'{name} ep{epoch} train', leave=False):
            y = batch['af_fut'].to(device, non_blocking=True)[:, 0]
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits = model(batch)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                if cfg['grad_clip'] > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if cfg['grad_clip'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                optimizer.step()
            tr_loss += loss.item()
            with torch.no_grad():
                tr_probs.append(torch.sigmoid(logits.detach()).float().cpu().reshape(-1))
                tr_tgts.append(y.detach().float().cpu().reshape(-1))
        scheduler.step()
        tr_loss /= max(1, len(train_loader))
        tr_m = _aggregate_metrics(tr_probs, tr_tgts, cfg['val_threshold'])

        # ---- val ----
        model.eval()
        v_loss = 0.0
        v_probs, v_tgts = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'{name} ep{epoch} val  ', leave=False):
                y = batch['af_fut'].to(device, non_blocking=True)[:, 0]
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    logits = model(batch)
                    loss = criterion(logits, y)
                v_loss += loss.item()
                v_probs.append(torch.sigmoid(logits.float()).cpu().reshape(-1))
                v_tgts.append(y.float().cpu().reshape(-1))
        v_loss /= max(1, len(val_loader))
        v_m = _aggregate_metrics(v_probs, v_tgts, cfg['val_threshold'])

        row = {
            'epoch': epoch,
            'lr'   : optimizer.param_groups[0]['lr'],
            'train_loss': tr_loss, 'val_loss': v_loss,
            'train_iou': tr_m['iou'], 'val_iou': v_m['iou'],
            'train_f1':  tr_m['f1'],  'val_f1':  v_m['f1'],
            'train_precision': tr_m['precision'], 'val_precision': v_m['precision'],
            'train_recall':    tr_m['recall'],    'val_recall':    v_m['recall'],
            'elapsed_sec': time.time() - t0,
        }
        logger.log(row)
        print(f'  {name} ep{epoch:03d}  train_loss={tr_loss:.4f}  val_loss={v_loss:.4f}  '
              f'val_IoU={v_m[\"iou\"]:.4f}  val_F1={v_m[\"f1\"]:.4f}  '
              f'val_P={v_m[\"precision\"]:.4f}  val_R={v_m[\"recall\"]:.4f}')

        # best checkpoint by val IoU
        if v_m['iou'] > best_iou:
            best_iou = v_m['iou']; best_epoch = epoch; patience_cnt = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_iou,
                'config': cfg,
                'model_name': name,
            }, output_dir / 'best_model.pt')
        else:
            patience_cnt += 1

        # last checkpoint every epoch (extra to the original)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': v_m['iou'],
            'config': cfg,
            'model_name': name,
        }, output_dir / 'last_model.pt')

        if patience > 0 and patience_cnt >= patience:
            print(f'  Early stopping at epoch {epoch} (best ep {best_epoch}, IoU {best_iou:.4f})')
            break

    elapsed = time.time() - t0
    print(f'  done {name} -- best val IoU={best_iou:.4f} at epoch {best_epoch}  ({elapsed/60:.1f} min)')
    return {
        'name': name,
        'best_val_iou': best_iou,
        'best_epoch'  : best_epoch,
        'elapsed_sec' : elapsed,
        'output_dir'  : str(output_dir),
        'num_params'  : sum(p.numel() for p in model.parameters() if p.requires_grad),
        'stopped_early': patience_cnt >= patience and patience > 0,
    }"""))

# ============== Run training ==============
cells.append(md("""## 8 · Train SwinUNETR-48

Single training run, same protocol as the original baseline comparison. Outputs go to
`/content/outputs/baseline_swinunetr_feature48/` and are mirrored to Drive at the end."""))

cells.append(code("""# 8.1 — Save the final CONFIG before training (so report has the ground-truth settings)
out_dir = Path(CONFIG['local_output']); out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'config.json', 'w') as f:
    safe = {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()}
    json.dump(safe, f, indent=2)
print(f'Saved {out_dir / \"config.json\"}')"""))

cells.append(code("""# 8.2 — Train
seed_everything(SEED)
result = train_model(
    name=f'swin_unetr_fs{CONFIG[\"swin_feature_size\"]}',
    model=swin_model,
    cfg=CONFIG,
    train_loader=train_loader, val_loader=val_loader,
    output_dir=Path(CONFIG['local_output']),
    patience=8 if not FAST_DEBUG else 999,
)
print(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in result.items()}, indent=2))"""))

# ============== Evaluation ==============
cells.append(md("""## 9 · Evaluation + comparison vs SwinUNETR-24 and U-Net

Load the best checkpoint, recompute the val metrics using the same `_aggregate_metrics`
function used during training (sanity), and assemble a comparison table against the
**existing** SwinUNETR-24 and U-Net runs read directly from
`output/baseline/baseline_comparison/`. **U-Net is not retrained.**"""))

cells.append(code("""# 9.1 — Load best checkpoint and recompute val metrics (sanity)
best_ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
swin_model.load_state_dict(best_ckpt['model_state_dict'])
swin_model = swin_model.to(device).eval()

v_probs, v_tgts = [], []
v_loss_sum = 0.0
crit = FocalDiceLoss(pos_weight=CONFIG['pos_weight'], gamma=CONFIG['focal_gamma'],
                     dice_weight=CONFIG['dice_weight']).to(device)
with torch.no_grad():
    for batch in tqdm(val_loader, desc='final val'):
        y = batch['af_fut'].to(device)[:, 0]
        logits = swin_model(batch)
        v_loss_sum += crit(logits, y).item()
        v_probs.append(torch.sigmoid(logits.float()).cpu().reshape(-1))
        v_tgts.append(y.float().cpu().reshape(-1))
v_loss = v_loss_sum / max(1, len(val_loader))
v_m = _aggregate_metrics(v_probs, v_tgts, CONFIG['val_threshold'])
print(json.dumps({**v_m, 'val_loss': v_loss,
                  'best_epoch': int(best_ckpt['epoch'])}, indent=2))"""))

cells.append(code("""# 9.2 — Combine into a summary JSON (best-epoch row from metrics.csv too, for train_iou & gap)
mdf = pd.read_csv(out_dir / 'metrics.csv')
best_row = mdf.loc[mdf['val_iou'].idxmax()]
val_summary = {
    'feature_size'  : CONFIG['swin_feature_size'],
    'best_epoch'    : int(best_row['epoch']),
    'last_epoch'    : int(mdf['epoch'].max()),
    'stopped_early' : bool(int(mdf['epoch'].max()) < CONFIG['epochs']),
    'train_iou'     : float(best_row['train_iou']),
    'val_iou'       : float(best_row['val_iou']),
    'val_f1'        : float(best_row['val_f1']),
    'val_precision' : float(best_row['val_precision']),
    'val_recall'    : float(best_row['val_recall']),
    'val_loss'      : float(best_row['val_loss']),
    'train_loss'    : float(best_row['train_loss']),
    'train_val_gap' : float(best_row['train_iou'] - best_row['val_iou']),
    'num_params_M'  : result['num_params'] / 1e6,
    'patch_embed_params': PARAM_INFO['patch_embed_params'],
    'final_recompute_val_iou' : v_m['iou'],
    'final_recompute_val_f1'  : v_m['f1'],
    'final_recompute_val_loss': v_loss,
}
with open(out_dir / 'val_summary.json', 'w') as f:
    json.dump(val_summary, f, indent=2)
print(json.dumps(val_summary, indent=2))"""))

cells.append(code("""# 9.3 — Read old SwinUNETR-24 and U-Net metrics from baseline_comparison (no retraining)
# Path detection: the ORIGINAL colab_af_baseline_comparison.ipynb mirrored to
#   {PROJECT}/outputs/baseline_comparison   (plural 'outputs')
# but on the local Windows mirror the user re-organized them under
#   {PROJECT}/output/baseline/baseline_comparison   (singular 'output')
# So we probe both, plus a couple of other reasonable spots.
_candidates = [
    Path(PROJECT_PATH) / 'outputs' / 'baseline_comparison',
    Path(PROJECT_PATH) / 'output'  / 'baseline' / 'baseline_comparison',
    Path(PROJECT_PATH) / 'output'  / 'baseline_comparison',
]

def _has_old_baselines(p: Path) -> bool:
    return (p / 'swin_unetr' / 'metrics.csv').exists() and (p / 'unet' / 'metrics.csv').exists()

OLD_BASELINE = next((c for c in _candidates if _has_old_baselines(c)), None)
assert OLD_BASELINE is not None, (
    'Old baseline metrics not found. Tried:\\n  - '
    + '\\n  - '.join(str(c) for c in _candidates)
    + '\\nUpdate the candidates list above with the actual location.'
)
print(f'Old baselines at: {OLD_BASELINE}')


def _best_from_csv(p):
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'val_iou' not in df.columns:
        return None
    return df.loc[df['val_iou'].idxmax()]


old_swin = _best_from_csv(OLD_BASELINE / 'swin_unetr' / 'metrics.csv')
old_unet = _best_from_csv(OLD_BASELINE / 'unet'       / 'metrics.csv')
assert old_swin is not None, 'Could not find old SwinUNETR-24 metrics.'
assert old_unet is not None, 'Could not find old U-Net metrics.'

# Try to read parameter counts from the original comparison_summary.csv
summary_csv = OLD_BASELINE / 'comparison_summary.csv'
old_params = {}
if summary_csv.exists():
    sdf = pd.read_csv(summary_csv)
    for _, r in sdf.iterrows():
        old_params[r['model']] = float(r['num_params_M'])

def _row(name, row, params_M, fs=None):
    return {
        'model'         : name,
        'feature_size'  : fs,
        'params_M'      : params_M,
        'val_iou'       : float(row['val_iou']),
        'val_f1'        : float(row['val_f1']),
        'val_precision' : float(row['val_precision']),
        'val_recall'    : float(row['val_recall']),
        'val_loss'      : float(row['val_loss']),
        'train_iou'     : float(row['train_iou']),
        'train_loss'    : float(row['train_loss']),
        'train_val_gap' : float(row['train_iou'] - row['val_iou']),
        'best_epoch'    : int(row['epoch']),
    }

rows = [
    _row('SwinUNETR-24 (old)', old_swin, old_params.get('swin_unetr'), 24),
    _row('SwinUNETR-%d (new)' % CONFIG['swin_feature_size'], best_row,
         result['num_params'] / 1e6, CONFIG['swin_feature_size']),
    _row('U-Net (old, ref)',  old_unet, old_params.get('unet')),
]
comparison = pd.DataFrame(rows)
comparison.to_csv(out_dir / 'comparison_table.csv', index=False)
with pd.option_context('display.max_columns', None, 'display.width', 160,
                       'display.float_format', '{:.4f}'.format):
    print(comparison)"""))

# ============== Figures ==============
cells.append(md("""## 10 · Figures"""))

cells.append(code("""# 10.1 — Train / val loss + IoU curves for SwinUNETR-48
FIG = out_dir / 'figures'; FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(mdf['epoch'], mdf['train_loss'], '--', c='C3', label='train')
ax[0].plot(mdf['epoch'], mdf['val_loss'],   '-',  c='C3', label='val', lw=2)
ax[0].set_title(f'SwinUNETR-{CONFIG[\"swin_feature_size\"]}: Loss')
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('FocalDice loss')
ax[0].grid(True, alpha=0.3); ax[0].legend()

ax[1].plot(mdf['epoch'], mdf['train_iou'], '--', c='C3', label='train')
ax[1].plot(mdf['epoch'], mdf['val_iou'],   '-',  c='C3', label='val', lw=2)
ax[1].set_title(f'SwinUNETR-{CONFIG[\"swin_feature_size\"]}: IoU')
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('IoU')
ax[1].grid(True, alpha=0.3); ax[1].legend()

plt.tight_layout()
plt.savefig(FIG / 'train_val_loss.png', dpi=140)
plt.savefig(FIG / 'train_val_iou.png',  dpi=140)
plt.show()"""))

cells.append(code("""# 10.2 — IoU / F1 / Precision / Recall comparison vs SwinUNETR-24 and U-Net
metrics = ['val_iou', 'val_f1', 'val_precision', 'val_recall']
labels  = ['Val IoU', 'Val F1', 'Val Precision', 'Val Recall']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
xs = np.arange(len(comparison))
colors = ['#aec7e8', '#1f77b4', '#ff7f0e']
for ax, m, lbl in zip(axes, metrics, labels):
    vals = comparison[m].values
    ax.bar(xs, vals, color=colors)
    ax.set_xticks(xs); ax.set_xticklabels(comparison['model'], rotation=15, ha='right', fontsize=8)
    ax.set_title(lbl); ax.set_ylabel(lbl); ax.grid(True, alpha=0.3, axis='y')
    for xi, yi in zip(xs, vals):
        ax.text(xi, yi + 0.005, f'{yi:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIG / 'metric_comparison.png', dpi=140)
plt.show()"""))

cells.append(code("""# 10.3 — Parameter-count comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(xs, comparison['params_M'], color=colors)
ax.set_xticks(xs); ax.set_xticklabels(comparison['model'], rotation=15, ha='right')
ax.set_ylabel('Trainable parameters (M)')
ax.set_title('Parameter count comparison')
for xi, yi in zip(xs, comparison['params_M']):
    ax.text(xi, yi + 0.4, f'{yi:.2f}M', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG / 'param_count_comparison.png', dpi=140)
plt.show()"""))

cells.append(code("""# 10.4 — Precision-Recall scatter
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for i, r in comparison.iterrows():
    ax.scatter(r['val_recall'], r['val_precision'], s=160, c=colors[i])
    ax.annotate(r['model'], (r['val_recall'], r['val_precision']),
                xytext=(8, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel('Validation recall')
ax.set_ylabel('Validation precision')
ax.set_title('Precision vs Recall (best epoch, threshold=0.5)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / 'precision_recall.png', dpi=140)
plt.show()"""))

cells.append(code("""# 10.5 — IoU vs parameter count scatter
fig, ax = plt.subplots(figsize=(7, 5))
for i, r in comparison.iterrows():
    ax.scatter(r['params_M'], r['val_iou'], s=160, c=colors[i])
    ax.annotate(r['model'], (r['params_M'], r['val_iou']),
                xytext=(8, 5), textcoords='offset points', fontsize=9)
ax.set_xscale('log')
ax.set_xlabel('Trainable parameters (M, log)')
ax.set_ylabel('Validation IoU')
ax.set_title('IoU vs parameter count')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / 'iou_vs_params.png', dpi=140)
plt.show()"""))

# ============== Report ==============
cells.append(md("""## 11 · Write final Markdown report

The conclusion sentence is chosen automatically based on whether SwinUNETR-48
(i) beats U-Net, (ii) closes the gap to within 0.01 IoU, or (iii) still loses.
The wording follows the constraints in the task brief."""))

cells.append(code("""# 11.1 — Decide the conclusion wording
new_iou  = float(val_summary['val_iou'])
old_iou  = float(comparison.iloc[0]['val_iou'])   # SwinUNETR-24
unet_iou = float(comparison.iloc[2]['val_iou'])

CLOSE_TOL = 0.01

if new_iou > unet_iou + CLOSE_TOL:
    case = 'beats'
    conclusion_line = (
        'The original SwinUNETR result should not be used as evidence against '
        'transformer-based baselines. The paper/thesis should update the baseline '
        'table using the stronger SwinUNETR configuration.'
    )
elif new_iou >= unet_iou - CLOSE_TOL:
    case = 'closes'
    conclusion_line = (
        'The original SwinUNETR underperformance was partly due to an under-capacity '
        'configuration. The baseline comparison should report the stronger SwinUNETR '
        'setting or explicitly state that the original SwinUNETR was a compact configuration.'
    )
else:
    case = 'still_loses'
    conclusion_line = (
        \"Under this AF baseline protocol, even a higher-capacity SwinUNETR remains \"
        \"below U-Net, suggesting that U-Net's convolutional inductive bias and \"
        \"preservation of local fire signals are better matched to sparse active-fire \"
        \"segmentation in this setting.\"
    )

print(f'case = {case}')
print(f'new_iou={new_iou:.4f}  old_iou={old_iou:.4f}  unet_iou={unet_iou:.4f}')"""))

cells.append(code("""# 11.2 — Generate the Markdown report
def fmt(x, n=4):
    try:
        return f'{float(x):.{n}f}'
    except Exception:
        return str(x)

cmp_rows = comparison.to_dict('records')
def _row_to_md(r):
    return (f\"| {r['model']} | {r['feature_size'] if r['feature_size'] else '—'} | \"
            f\"{fmt(r['params_M'], 3)} M | {fmt(r['val_iou'])} | {fmt(r['val_f1'])} | \"
            f\"{fmt(r['val_precision'])} | {fmt(r['val_recall'])} | {fmt(r['val_loss'])} | \"
            f\"{fmt(r['train_iou'])} | {fmt(r['train_val_gap'])} | {int(r['best_epoch'])} |\")

new_row = comparison.iloc[1]; old_row = comparison.iloc[0]; unet_row = comparison.iloc[2]
delta_iou_swin = new_row['val_iou']    - old_row['val_iou']
delta_iou_unet = new_row['val_iou']    - unet_row['val_iou']
delta_rec_swin = new_row['val_recall'] - old_row['val_recall']
delta_gap_swin = new_row['train_val_gap'] - old_row['train_val_gap']

ans_A = 'YES — val IoU improved' if delta_iou_swin > 0 else 'NO — val IoU did not improve'
ans_A += f' (Δ = {delta_iou_swin:+.4f}).'
ans_B = 'YES — recall improved' if delta_rec_swin > 0 else 'NO — recall did not improve'
ans_B += f' (Δ = {delta_rec_swin:+.4f}).'
ans_C = ('Gap **shrank** (better generalization).'
         if delta_gap_swin < 0
         else 'Gap **grew** (more overfitting at higher capacity).')
ans_C += f' (Δ gap = {delta_gap_swin:+.4f})'
ans_D = ('YES — closer to U-Net.' if abs(new_row['val_iou'] - unet_row['val_iou']) <
         abs(old_row['val_iou'] - unet_row['val_iou']) else 'NO — not closer to U-Net.')
ans_D += f' (|Δ to U-Net|: old {abs(old_row[\"val_iou\"]-unet_row[\"val_iou\"]):.4f} -> new {abs(new_row[\"val_iou\"]-unet_row[\"val_iou\"]):.4f})'
ans_E = 'YES.' if delta_iou_unet > 0 else 'NO.'
ans_E += f' (Δ vs U-Net = {delta_iou_unet:+.4f})'
ans_F = ('YES — the stronger conclusion is supported.' if case == 'still_loses'
         else 'NO — the conclusion no longer holds at higher capacity.')
ans_G = ('YES — update the baseline framing to disclose the compact SwinUNETR-24.'
         if case in ('closes', 'beats')
         else 'No revision strictly required, but the disclosure of capacity choice still helps.')

md_path = out_dir / 'swinunetr_feature48_baseline_report.md'

md_text = f\"\"\"# SwinUNETR (feature_size = {CONFIG['swin_feature_size']}) — Stronger AF Baseline Report

**Project:** FireST-GraphNet
**Notebook:** `colab_af_swinunetr_feature48_baseline.ipynb`
**Output folder:** `output/baseline_swinunetr_feature48/`
**Compared against:** SwinUNETR-24 and U-Net from `output/baseline/baseline_comparison/`.

---

## Motivation

A previous bug-check diagnosis (`swinunetr_vs_unet_bugcheck_diagnosis.md`) confirmed that the
original AF baseline-comparison SwinUNETR result was **valid** — no bugs in code, fairness,
or parameter counting — but used MONAI's **minimum** configuration (`feature_size=24`,
total ≈ 6.40 M params, transformer encoder only ≈ 1.62 M). The first patch embedding
compressed the 280-channel stacked input down to 24 channels at stride 2, the deepest stage
had only 192 channels (vs U-Net's 512), and the model showed clear overfitting (train IoU
0.834 vs val IoU 0.594).

This notebook tests whether a **more reasonable** capacity (`feature_size=48`, the standard
medical-imaging size for MONAI SwinUNETR) changes the conclusion, while keeping every other
training and evaluation knob **identical** to the original baseline comparison.

## Experiment setup

- **Dataset / split:** identical `TSSatFireForecastDataset` over `dataset_train` /
  `dataset_val` from the original protocol.
- **Input contract:** `(B, 10, 28, 256, 256)` channel-stacked to `(B, 280, 256, 256)`
  (T_in=10, 8 VIIRS + 19 FirePred + 1 LULC = 28 per step).
- **Target:** binary AF mask of the next frame, `(B, 1, 256, 256)` (horizon=1).
- **Loss:** `FocalDiceLoss(pos_weight=4, γ=2, dice_w=0.3)` (unchanged).
- **Optimizer:** AdamW(lr=1.5e-4, wd=5e-3), Cosine LR schedule (T_max=30) (unchanged).
- **Batch size:** {CONFIG['batch_size']}  (target 8; falls back to {CONFIG['batch_size_fallback']} only on OOM).
- **Epochs:** {CONFIG['epochs']}, early stopping patience = 8 (unchanged).
- **Threshold:** 0.5 (unchanged).
- **Seed:** 42, re-seeded before training (unchanged).
- **Peak VRAM at smoke test:** {peak_gb_used:.2f} GB.

## Model configuration

| Setting | SwinUNETR-24 (old) | SwinUNETR-{CONFIG['swin_feature_size']} (new) |
|---|---|---|
| feature_size | 24 | **{CONFIG['swin_feature_size']}** |
| depths | (2,2,2,2) | (2,2,2,2) |
| num_heads | (3,6,12,24) | (3,6,12,24) |
| use_v2 | False | False |
| pretrained | none | none |
| spatial_dims | 2 | 2 |
| in_channels | 280 | 280 |
| out_channels | 1 | 1 |
| total params | {fmt(old_row['params_M'], 3) if old_row['params_M'] else 'n/a'} M | **{fmt(new_row['params_M'], 3)} M** |
| patch_embed params | ~6.7 k | {PARAM_INFO['patch_embed_params']:,} |
| patch_embed | Conv2d(280, 24, k=2, s=2) | {PARAM_INFO['patch_embed_repr']} |

### Per-top-module breakdown (this run)

```
\"\"\"

for n, mB in PARAM_INFO['per_top_module_M'].items():
    md_text += f'  {n:18s} -> {mB:7.3f} M\\n'

md_text += f\"\"\"```

## Parameter analysis

- SwinUNETR-{CONFIG['swin_feature_size']} has **{fmt(new_row['params_M'], 3)} M** trainable parameters
  — {(new_row['params_M']/max(old_row['params_M'],1e-6)):.2f}× more than SwinUNETR-24
  ({fmt(old_row['params_M'], 3)} M) and {(new_row['params_M']/max(unet_row['params_M'],1e-6)):.2f}× U-Net's
  {fmt(unet_row['params_M'], 3)} M.
- The first patch embedding now compresses 280 input channels to **{CONFIG['swin_feature_size']}**
  channels (vs 24 before), losing relatively less spectral capacity at the very first layer.
- The deepest Swin stage now has **{CONFIG['swin_feature_size']*8}** channels
  (= feature_size × 2³), vs 192 for SwinUNETR-24 and 512 for U-Net's bottleneck.

## Results

| Model | feature_size | Params | Val IoU | Val F1 | Val P | Val R | Val Loss | Train IoU | Train-Val Gap | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|---|
\"\"\"
for r in cmp_rows:
    md_text += _row_to_md(r) + '\\n'

md_text += f\"\"\"
**Key deltas (SwinUNETR-{CONFIG['swin_feature_size']} vs SwinUNETR-24):**

- Δ val IoU       : {delta_iou_swin:+.4f}
- Δ val recall    : {delta_rec_swin:+.4f}
- Δ train-val gap : {delta_gap_swin:+.4f}  ({'smaller is better — less overfitting' if delta_gap_swin < 0 else 'larger means more overfitting'})

**SwinUNETR-{CONFIG['swin_feature_size']} vs U-Net:**

- Δ val IoU       : {delta_iou_unet:+.4f}
- |Δ to U-Net| dropped from {abs(old_row['val_iou']-unet_row['val_iou']):.4f} to {abs(new_row['val_iou']-unet_row['val_iou']):.4f}.

## Training dynamics

- Best epoch: {int(new_row['best_epoch'])} / {CONFIG['epochs']}.
- Final train loss: {fmt(new_row['train_loss'])}, val loss: {fmt(new_row['val_loss'])}.
- Train-val IoU gap at best epoch: {fmt(new_row['train_val_gap'])}.
- Compared with SwinUNETR-24 (gap {fmt(old_row['train_val_gap'])}) and U-Net (gap {fmt(unet_row['train_iou'] - unet_row['val_iou'])}).

## Answers to diagnosis questions

- **A. Did feature_size={CONFIG['swin_feature_size']} improve SwinUNETR over feature_size=24?**  {ans_A}
- **B. Did recall improve?**  {ans_B}
- **C. Did the train/val gap shrink or get worse?**  {ans_C}
- **D. Is the stronger SwinUNETR now closer to U-Net?**  {ans_D}
- **E. Does it outperform U-Net?**  {ans_E}
- **F. If it still underperforms U-Net, is the conclusion stronger that U-Net is better for this AF baseline setting?**  {ans_F}
- **G. If it closes the gap or beats U-Net, should the thesis/paper framing be revised?**  {ans_G}

## Conclusion

{conclusion_line}

## Files written

- `best_model.pt`, `last_model.pt` — checkpoints
- `metrics.csv` — per-epoch train/val log
- `config.json` — exact CONFIG used
- `param_count.json` — total / trainable / per-module / patch-embed details
- `val_summary.json` — best-epoch summary (re-computed on val_loader)
- `comparison_table.csv` — SwinUNETR-24 vs SwinUNETR-{CONFIG['swin_feature_size']} vs U-Net
- `figures/` — train/val curves + comparison plots
- `swinunetr_feature48_baseline_report.md` — this report
\"\"\"

md_path.write_text(md_text, encoding='utf-8')
print(f'Saved report to {md_path}')"""))

# ============== Final mirror to Drive ==============
cells.append(md("""## 12 · Mirror to Drive"""))

cells.append(code("""# 12.1 — Mirror everything from /content to Drive so it survives the session
src_root = Path(CONFIG['local_output'])
dst_root = Path(CONFIG['output_root'])
dst_root.mkdir(parents=True, exist_ok=True)
for src in src_root.rglob('*'):
    if src.is_file():
        rel = src.relative_to(src_root)
        tgt = dst_root / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if not tgt.exists() or tgt.stat().st_size != src.stat().st_size:
            shutil.copy2(src, tgt)
print(f'Mirrored to: {dst_root}')
print('\\nContents of Drive output folder:')
for p in sorted(dst_root.rglob('*')):
    if p.is_file():
        print(' ', p.relative_to(dst_root), f'({p.stat().st_size/1e3:.1f} KB)')"""))


# =================== Assemble notebook ===================
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
print(f"Cell count: {len(cells)}")
