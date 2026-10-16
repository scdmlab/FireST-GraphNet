"""
Build the BA (Burned Area) training notebook for the V2 architecture.
Mirrors the format of colab_inference_af_v2_all_checkpoints.ipynb and the
structure of colab_run29.ipynb (AF V2 training).

Output:
    C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/
        colab_train_ba_v2.ipynb
"""
import json
from pathlib import Path

OUT = Path("C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_train_ba_v2.ipynb")
MODEL_SRC = Path(r"C:\Users\22125\Desktop\FireST-GraphNet\_nb29_extract\cell_16.py")


def md(text: str):
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines if lines else [""],
    }


def code(text: str):
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines if lines else [""],
    }


cells = []

# ===================================================================
# Cell 0 — Title + roadmap
# ===================================================================
cells.append(md("""# FireST-GraphNet V2 — **Burned Area (BA) training**
### Multi-step cumulative burned-area forecasting  (T_in=10, horizon=15)

> **Task**: binary **future cumulative BA mask** prediction — direct multi-horizon forecast.
> **Inputs**: 8-band VIIRS active-fire history + 19-band FirePred auxiliary + 1-band LULC static + 4-channel future weather (temp, u-wind, v-wind, RH) for 15 horizons.
> **Output**: per-pixel BA probability for each of the next 15 time steps.
> **Architecture**: reuses the **FireSTGraphNetV2 + ForecastDecoderV3** backbone from `colab_run29` — CNN encoder → GATv2 multi-scale graph pyramid → Attention-LSTM → cross-attention weather decoder with U-Net skips. The `head_af` branch is retargeted as the BA logit head; the auxiliary monotonic `head_ba` is disabled by default (redundant with the primary task).

---

### Key differences vs. the AF runs (run27 / run28 / run29)
| Aspect | AF runs | **BA (this notebook)** |
|---|---|---|
| `data.label_channel` | `1` (active fire) | **`0` (burned area)** |
| Primary target semantics | per-frame AF mask | **cumulative BA mask** |
| `model.v2_aux_ba_head` | `true` | **`false`** (primary head IS BA) |
| Temporal constraint | none | **monotonic consistency loss** |
| Threshold grid | `[0.35 … 0.65]` | **`[0.25 … 0.60]`** (BA is denser) |
| `training.pos_weight` | 4.5 | **2.0** (more positive pixels) |
| Mixup | ON (p=0.2) | **OFF** (breaks cumulative structure) |
| Output root | `output/run29/` | **`output/run_ba_01/`** |

---

### Notebook roadmap
| Section | Content |
|---|---|
| 0 | Install dependencies |
| 1 | Mount Google Drive + set `CONFIG_PATH` |
| 2 | Scan / generate / sync data to local SSD |
| 3 | Imports + env check |
| 4 | Dataloader (`label_channel=0`) |
| 5 | **Inline `FireSTGraphNetV2`** |
| 6 | Load or auto-generate `config_<RUN_NAME>.yaml` |
| 7 | Training loop (BA-specific: no mixup, monotonic loss) |
| 8 | Validation curves |
| 9 | Inference visualisation — 15 horizon frames for 3 val events |
| 10 | Export `best_model.pt` + metrics to Drive |

The resulting `best_model.pt` layout matches `output/run29/best_model.pt` verbatim, so the BA checkpoint is directly consumable by `colab_inference_af_v2_all_checkpoints.ipynb` (which auto-discovers V2 runs under `OUTPUT_ROOT`).

---
"""))

# ===================================================================
# Section 0 — Install dependencies
# ===================================================================
cells.append(md("## Section 0 · Install dependencies"))
cells.append(code("""# Same dependency set as colab_run29.ipynb (A100 tested)
!pip install -q torch torchvision torchaudio
!pip install -q torch-geometric
!pip install -q pyyaml scikit-learn matplotlib tqdm einops

import torch
pt_version = torch.__version__
print(f"PyTorch  : {pt_version}")
print(f"CUDA     : {torch.version.cuda}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"GPU mem   : {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")

import subprocess
pyg_url = f"https://data.pyg.org/whl/torch-{pt_version}.html"
subprocess.run(["pip", "install", "-q", "torch-cluster", "-f", pyg_url])
print("\\nAll dependencies ready.")"""))

cells.append(md("### GPU + disk check"))
cells.append(code("""import os, shutil
!nvidia-smi
print(f"\\nFree disk : {shutil.disk_usage('/').free/(1024**3):.1f} GB")
print(f"CWD       : {os.getcwd()}")"""))

# ===================================================================
# Section 1 — Mount Drive + paths
# ===================================================================
cells.append(md("## Section 1 · Mount Google Drive + paths"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil
from pathlib import Path

# ── ONLY LINE YOU NEED TO CHANGE BETWEEN RUNS ─────────────────────────────
CONFIG_PATH = "/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean/config_run_ba_01.yaml"
# ──────────────────────────────────────────────────────────────────────────
# If the file does not exist on Drive, Section 6 will auto-write a default
# BA config to this location.  Override the RUN_NAME by pointing CONFIG_PATH
# at a different `config_run_ba_XX.yaml`.

RUN_NAME  = Path(CONFIG_PATH).stem.replace('config_', '')    # e.g. 'run_ba_01'
LOCAL_OUT = f'/content/outputs/{RUN_NAME}'

PROJECT_PATH = str(Path(CONFIG_PATH).parent)
DRIVE_TRAIN  = f"{PROJECT_PATH}/dataset_train"
DRIVE_VAL    = f"{PROJECT_PATH}/dataset_val"

assert Path(PROJECT_PATH).exists(), f"Project folder missing: {PROJECT_PATH}"
print(f"Run name : {RUN_NAME}")
print(f"Config   : {CONFIG_PATH}   (auto-create if missing)")
print(f"Local out: {LOCAL_OUT}")
print(f"Project  : {PROJECT_PATH}")
print(f"Train dir: {DRIVE_TRAIN}  (exists={Path(DRIVE_TRAIN).exists()})")
print(f"Val dir  : {DRIVE_VAL}    (exists={Path(DRIVE_VAL).exists()})")

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)
os.chdir('/content')

# ── RESUME CONTROLS ────────────────────────────────────────────────────────
# Flip these to control how the next run starts.  All three are inspected
# by the training loop in Section 7.
RESUME_TRAINING      = True    # False → always start fresh (ignores any checkpoint)
AUTO_RESUME_LATEST   = True    # True  → auto-detect latest.pt in LOCAL_OUT then DRIVE_OUT
RESUME_CKPT_PATH     = None    # Explicit override, e.g. f"{DRIVE_OUT}/latest.pt"
# ───────────────────────────────────────────────────────────────────────────

DRIVE_OUT = f"{PROJECT_PATH}/output/{RUN_NAME}"
print(f"\\nResume config:")
print(f"  RESUME_TRAINING    : {RESUME_TRAINING}")
print(f"  AUTO_RESUME_LATEST : {AUTO_RESUME_LATEST}")
print(f"  RESUME_CKPT_PATH   : {RESUME_CKPT_PATH}")
print(f"  DRIVE_OUT          : {DRIVE_OUT}")"""))

# ===================================================================
# Section 2 — Data (scan / gen / copy / verify)
# ===================================================================
cells.append(md("## Section 2 · Scan Drive data files"))
cells.append(code("""import numpy as np
from pathlib import Path

print("=" * 60)
print("Scanning Google Drive data files")
print("=" * 60 + "\\n")

check_files = {
    "train": {
        "af_img":   f"{DRIVE_TRAIN}/af_train_img_seqtoseq_alll_1i_1.npy",
        "af_label": f"{DRIVE_TRAIN}/af_train_label_seqtoseq_alll_1i_1.npy",
        "firepred": f"{DRIVE_TRAIN}/firepred_train.npy",
        "lulc":     f"{DRIVE_TRAIN}/lulc_train.npy",
    },
    "val": {
        "af_img":   f"{DRIVE_VAL}/af_val_img_seqtoseq_alll_1i_1.npy",
        "af_label": f"{DRIVE_VAL}/af_val_label_seqtoseq_alll_1i_1.npy",
        "firepred": f"{DRIVE_VAL}/firepred_val.npy",
        "lulc":     f"{DRIVE_VAL}/lulc_val.npy",
    },
}
missing_critical, missing_aux = [], []
for split, files in check_files.items():
    print(f"--- {split.upper()} ---")
    for key, path in files.items():
        p = Path(path)
        if p.exists():
            try:
                arr = np.load(path, mmap_mode='r')
                print(f"  OK {key:12s}  shape={str(arr.shape):30s}  {p.stat().st_size/1024**2:8.1f} MB")
            except Exception:
                print(f"  OK {key:12s}  {p.stat().st_size/1024**2:8.1f} MB")
        else:
            print(f"  !! {key:12s}  NOT FOUND")
            if key in ("af_img", "af_label"):
                missing_critical.append(f"{split}/{key}")
            if key in ("firepred", "lulc"):
                missing_aux.append(f"{split}/{key}")
    print()

# Peek channel layout so we can confirm BA lives at channel 0
try:
    lbl = np.load(check_files['train']['af_label'], mmap_mode='r')
    if lbl.ndim == 5:
        print(f"Label tensor   : N={lbl.shape[0]}  T={lbl.shape[1]}  C={lbl.shape[2]}  H={lbl.shape[3]}  W={lbl.shape[4]}")
    else:
        print(f"Label tensor   : {lbl.shape}")
    print("NOTE: channel 0 = BA (cumulative), channel 1 = AF (per-frame).")
    print("      If your convention differs, override data.label_channel in Section 6.")
except Exception as e:
    print(f"Could not inspect label channels: {e}")

if missing_critical:
    print("\\nSTOP: af_img / af_label missing. Cannot train.")
elif missing_aux:
    print(f"\\nAux data missing: {missing_aux}  -> next cell will fill with placeholders.")
else:
    print("\\nAll files present.")"""))

cells.append(md("### Generate missing aux data (zero placeholders for firepred / lulc)"))
cells.append(code("""import numpy as np, shutil
from pathlib import Path

fp_train = Path(DRIVE_TRAIN) / "firepred_train.npy"
fp_val   = Path(DRIVE_VAL)   / "firepred_val.npy"
lulc_tr  = Path(DRIVE_TRAIN) / "lulc_train.npy"
lulc_va  = Path(DRIVE_VAL)   / "lulc_val.npy"

need_fp   = not (fp_train.exists() and fp_val.exists())
need_lulc = not (lulc_tr.exists() and lulc_va.exists())

if not need_fp and not need_lulc:
    print("firepred + lulc already exist. Skipping.")
else:
    af_tr = Path(DRIVE_TRAIN) / "af_train_img_seqtoseq_alll_1i_1.npy"
    af_va = Path(DRIVE_VAL)   / "af_val_img_seqtoseq_alll_1i_1.npy"
    arr_tr = np.load(str(af_tr), mmap_mode='r')
    arr_va = np.load(str(af_va), mmap_mode='r')
    N_tr, N_va = arr_tr.shape[0], arr_va.shape[0]
    H, W = arr_tr.shape[3], arr_tr.shape[4]
    print(f"N_train={N_tr}  N_val={N_va}  H={H}  W={W}")

    def _try_copy_pred_as_fp(src_file, dst_file, N, label):
        if not Path(src_file).exists():
            return False
        arr = np.load(str(src_file), mmap_mode='r')
        if arr.shape[0] == N and arr.shape[1] == 19:
            shutil.copy2(str(src_file), str(dst_file))
            print(f"  copied {label}  ({Path(dst_file).stat().st_size/1024**2:.1f} MB)")
            return True
        return False

    if need_fp:
        ok1 = _try_copy_pred_as_fp(
            f"{DRIVE_TRAIN}/pred_train_img_seqtoseq_alll_1i_1.npy", fp_train, N_tr, "firepred_train")
        ok2 = _try_copy_pred_as_fp(
            f"{DRIVE_VAL}/pred_val_img_seqtoseq_alll_1i_1.npy", fp_val, N_va, "firepred_val")
        if not ok1 or not ok2:
            print("  Fallback: generating zero firepred placeholders (19ch)")
            np.save(str(fp_train), np.zeros((N_tr, 19, 1, H, W), dtype=np.float32))
            np.save(str(fp_val),   np.zeros((N_va, 19, 1, H, W), dtype=np.float32))

    if need_lulc:
        print("  Generating zero LULC placeholders (1ch)")
        np.save(str(lulc_tr), np.zeros((N_tr, 1, H, W), dtype=np.float32))
        np.save(str(lulc_va), np.zeros((N_va, 1, H, W), dtype=np.float32))

print("\\nAux data ready.")"""))

cells.append(md("### Copy data to Colab local SSD  (~30× faster I/O than Drive)"))
cells.append(code("""import shutil, os
from pathlib import Path

DATASET_TRAIN = "/content/dataset_train"
DATASET_VAL   = "/content/dataset_val"
os.makedirs(DATASET_TRAIN, exist_ok=True)
os.makedirs(DATASET_VAL,   exist_ok=True)

files_to_copy = {
    "train": [
        ("af_train_img_seqtoseq_alll_1i_1.npy",   True),
        ("af_train_label_seqtoseq_alll_1i_1.npy", True),
        ("firepred_train.npy",                      True),
        ("lulc_train.npy",                          False),
    ],
    "val": [
        ("af_val_img_seqtoseq_alll_1i_1.npy",     True),
        ("af_val_label_seqtoseq_alll_1i_1.npy",   True),
        ("firepred_val.npy",                        True),
        ("lulc_val.npy",                            False),
    ],
}

print("Copying data to Colab local SSD...")
missing = []
for split, flist in files_to_copy.items():
    src_dir  = DRIVE_TRAIN if split == "train" else DRIVE_VAL
    dst_dir  = DATASET_TRAIN if split == "train" else DATASET_VAL
    for fname, required in flist:
        src  = Path(src_dir) / fname
        dst  = Path(dst_dir) / fname
        if src.exists():
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
            print(f"  OK  {split}/{fname}  {dst.stat().st_size/1024**2:.1f} MB")
        elif required:
            print(f"  !! MISSING (required): {split}/{fname}")
            missing.append(f"{split}/{fname}")
        else:
            print(f"  -- {split}/{fname}  (optional, using zeros)")

if missing:
    raise FileNotFoundError("Required files missing:\\n" + "\\n".join(missing))
print("\\nData ready on local SSD.")"""))

cells.append(md("### Verify data alignment + BA / AF densities"))
cells.append(code("""import numpy as np
from pathlib import Path

def _check_split(data_dir, mode):
    af_img = np.load(f"{data_dir}/af_{mode}_img_seqtoseq_alll_1i_1.npy", mmap_mode='r')
    af_lbl = np.load(f"{data_dir}/af_{mode}_label_seqtoseq_alll_1i_1.npy", mmap_mode='r')
    N = af_img.shape[0]
    print(f"  af_img  : {af_img.shape}")
    print(f"  af_label: {af_lbl.shape}  N match: {af_img.shape[0]==af_lbl.shape[0]}")
    # Channel-wise density on 10 random samples
    if af_lbl.ndim == 5 and af_lbl.shape[2] >= 2:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, N, size=min(10, N))
        sub = np.asarray(af_lbl[idx])
        ba_dens = float((sub[:, :, 0] > 0.5).mean())
        af_dens = float((sub[:, :, 1] > 0.5).mean())
        print(f"  densities  BA(ch0)={ba_dens:.4f}   AF(ch1)={af_dens:.4f}")
        if ba_dens < af_dens:
            print('  !! WARNING: BA density < AF density -> channels may be swapped in your data.')
    fp = Path(data_dir) / f"firepred_{mode}.npy"
    if fp.exists():
        a = np.load(str(fp), mmap_mode='r')
        print(f"  firepred: {a.shape}  N match: {a.shape[0]==N}  C=19: {a.shape[1]==19}")
    lulc = Path(data_dir) / f"lulc_{mode}.npy"
    if lulc.exists():
        a = np.load(str(lulc), mmap_mode='r')
        print(f"  lulc    : {a.shape}  N match: {a.shape[0]==N}")

print("TRAIN:"); _check_split(DATASET_TRAIN, "train")
print("\\nVAL:");   _check_split(DATASET_VAL,   "val")
print("\\nData verified.")"""))

# ===================================================================
# Section 3 — Imports
# ===================================================================
cells.append(md("## Section 3 · Imports + environment check"))
cells.append(code("""import math, json, time, random, warnings, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else '-'})")
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark  = True

from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
from firest_graphnet.training.losses import ForecastPINNLoss
print("Backend modules imported OK")"""))

# ===================================================================
# Section 4 — Dataloader (label_channel=0)
# ===================================================================
cells.append(md("""## Section 4 · Dataloader — `label_channel=0` for BA

Same `V2Augmenter` wrapper as the AF notebook (consistent flips + gaussian jitter + aux channel-dropout) with a single crucial change:

- **`label_channel` defaults to `0`** so `sample['af_fut']` is the cumulative BA mask.
- **Batch-level mixup is NOT used** in the trainer (Section 7) — mixing two cumulative masks produces fractional "half-burned" pixels that break monotonicity. The wrapper itself stays mixup-free; the trainer keeps `v2_mixup_prob = 0`.
"""))
cells.append(code("""from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider   import build_static_provider


class V2Augmenter(torch.utils.data.Dataset):
    # Wrap TSSatFireForecastDataset with consistent flips + jitter + channel dropout.
    # BA variant: no mixup inside __getitem__.
    def __init__(self, base, train: bool, noise_std: float = 0.01,
                 channel_drop_p: float = 0.05):
        self.base = base
        self.train = train
        self.noise_std = noise_std
        self.channel_drop_p = channel_drop_p

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        if not self.train:
            return s

        af_hist  = s['af_hist']
        aux_hist = s['aux_hist']
        env_fut  = s['env_fut']
        af_fut   = s['af_fut']          # BA mask when label_channel=0
        ba_init  = s['ba_init']
        static   = s['static']

        if torch.rand(1).item() < 0.5:
            af_hist  = torch.flip(af_hist,  dims=[-1])
            aux_hist = torch.flip(aux_hist, dims=[-1])
            env_fut  = torch.flip(env_fut,  dims=[-1])
            af_fut   = torch.flip(af_fut,   dims=[-1])
            ba_init  = torch.flip(ba_init,  dims=[-1])
            static   = torch.flip(static,   dims=[-1])
        if torch.rand(1).item() < 0.5:
            af_hist  = torch.flip(af_hist,  dims=[-2])
            aux_hist = torch.flip(aux_hist, dims=[-2])
            env_fut  = torch.flip(env_fut,  dims=[-2])
            af_fut   = torch.flip(af_fut,   dims=[-2])
            ba_init  = torch.flip(ba_init,  dims=[-2])
            static   = torch.flip(static,   dims=[-2])

        af_hist  = af_hist  + torch.randn_like(af_hist)  * self.noise_std
        aux_hist = aux_hist + torch.randn_like(aux_hist) * self.noise_std
        env_fut  = env_fut  + torch.randn_like(env_fut)  * self.noise_std

        if self.channel_drop_p > 0:
            C = aux_hist.shape[1]
            drop = (torch.rand(C) < self.channel_drop_p).float()
            keep_mask = (1 - drop).view(1, C, 1, 1)
            aux_hist = aux_hist * keep_mask

        return dict(
            af_hist=af_hist, aux_hist=aux_hist, env_fut=env_fut,
            af_fut=af_fut, ba_init=ba_init, static=static,
            env_source=s.get('env_source', 'firepred'),
        )


def _build_providers(data_dir, img_size):
    fp = build_firepred_provider(data_root=data_dir, spatial_size=(img_size, img_size))
    st = build_static_provider(data_root=data_dir, spatial_size=(img_size, img_size))
    print(f'    [{Path(data_dir).name}]  aux={type(fp).__name__}  C_aux={fp.channels}  |  '
          f'static={type(st).__name__}  C_static={st.channels}')
    return fp, st


def build_loaders(cfg, batch_size=None):
    # All config fields are validated in Section 6 — NO defaults here.
    dcfg   = cfg['data']
    fcfg   = cfg['forecast']
    tcfg   = cfg['training']
    img_sz = int(dcfg['img_size'])
    T_in   = int(fcfg['T_in'])
    H_fut  = int(fcfg['horizon'])
    bs     = int(batch_size if batch_size is not None else tcfg['batch_size'])

    train_dir = Path(dcfg['root_train'])
    val_dir   = Path(dcfg['root_val'])
    assert train_dir.is_dir(), f'train dir missing: {train_dir}'
    assert val_dir.is_dir(),   f'val dir missing: {val_dir}'

    print('Building providers:')
    fp_tr, st_tr = _build_providers(train_dir, img_sz)
    fp_va, st_va = _build_providers(val_dir,   img_sz)

    # KEY: label_channel=0 for BA, =1 for AF
    lbl_ch = int(dcfg['label_channel'])
    base_kw = dict(T_in=T_in, horizon=H_fut, label_channel=lbl_ch)
    print(f'  label_channel = {lbl_ch}   (0 = BA, 1 = AF)')

    train_base = TSSatFireForecastDataset(
        str(train_dir), firepred_provider=fp_tr, static_provider=st_tr, **base_kw)
    val_base   = TSSatFireForecastDataset(
        str(val_dir),   firepred_provider=fp_va, static_provider=st_va, **base_kw)

    train_ds = V2Augmenter(train_base, train=True,
                           noise_std=float(tcfg['augmentation_noise_std']))
    val_ds   = V2Augmenter(val_base, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=int(dcfg['num_workers']),
        pin_memory=bool(dcfg['pin_memory']),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=int(dcfg['num_workers']),
        pin_memory=bool(dcfg['pin_memory']),
        drop_last=False,
    )

    aux_ch = int(fp_tr.channels)
    print(f'  train={len(train_ds)}  val={len(val_ds)}  batch={bs}  aux_ch={aux_ch}')
    return train_loader, val_loader, train_ds, val_ds, aux_ch

print('V2Augmenter + build_loaders ready (BA mode)')"""))

# ===================================================================
# Section 5 — inline V2 model (verbatim from cell_16.py)
# ===================================================================
cells.append(md("## Section 5 · Inline `FireSTGraphNetV2` model"))
model_src = MODEL_SRC.read_text(encoding="utf-8")
# Replace the leading section comment to reflect the BA focus
model_src = model_src.replace(
    "# ─── Section 4 · FireSTGraphNetV2 (inline) ─────────────────────────────────",
    "# ─── FireSTGraphNetV2 (inline) — used here for BA logit prediction ─────────",
    1,
)
cells.append(code(model_src.rstrip()))

# ===================================================================
# Section 6 — Load/auto-generate config
# ===================================================================
cells.append(md("""## Section 6 · Load BA config — **strict mode**

The YAML at `CONFIG_PATH` is the **single source of truth** for every training setting.
The notebook does **not** inject any defaults.

- If the file is missing → `FileNotFoundError` with a clear message.
- If any required key is missing → `KeyError` listing every missing key.
- Only `output_dir` is overridden (to `LOCAL_OUT`) because that path is Colab-infrastructure-specific.

If you need to change a hyperparameter, edit the YAML on Drive and re-run this cell — don't patch it from inside the notebook.

A reference config lives at `FireST-GraphNet-Clean/config_run_ba_01.yaml`; copy and rename (e.g. `config_run_ba_02.yaml`) for a new run, then point `CONFIG_PATH` at it in Section 1.
"""))
cells.append(code("""import yaml
from pathlib import Path

# ── STRICT CONFIG LOADING — no notebook-side defaults ────────────────────
cfg_path = Path(CONFIG_PATH)
if not cfg_path.exists():
    raise FileNotFoundError(
        f"\\nConfig file not found: {cfg_path}\\n"
        f"The notebook no longer writes default configs. Create the YAML "
        f"on Drive first (e.g. copy config_run_ba_01.yaml into the project "
        f"root and point CONFIG_PATH at it)."
    )

# Copy Drive -> local for portability + fast re-reads
local_cfg_path = Path(f'/content/config_{RUN_NAME}.yaml')
shutil.copy2(cfg_path, local_cfg_path)
print(f'Loaded config : {cfg_path}')
print(f'Local copy    : {local_cfg_path}')

with open(local_cfg_path) as f:
    cfg = yaml.safe_load(f)
if not isinstance(cfg, dict):
    raise ValueError(f'Config did not parse as a dict: {type(cfg)}')

# ── Required keys — every training setting must be explicit in the YAML ──
REQUIRED = {
    'data':     ['root_train', 'root_val', 'img_size', 'num_workers',
                 'pin_memory', 'label_channel'],
    'forecast': ['T_in', 'horizon', 'weather_channels', 'weather_emb_dim'],
    'model':    ['encoder_channels', 'lstm_hidden', 'lstm_layers',
                 'lstm_dropout', 'decoder_dropout',
                 'v2_enabled', 'v2_graph_pyramid_patches', 'v2_gat_heads',
                 'v2_cross_attn_heads', 'v2_lstm_attn_heads', 'v2_aux_ba_head',
                 'v3_pos_emb', 'v3_autoregressive', 'v3_skip_film'],
    'training': ['epochs', 'batch_size', 'lr', 'eta_min', 'weight_decay',
                 'grad_clip', 'warmup_epochs_lr', 'warmup_start_factor',
                 'mixed_precision', 'pos_weight', 'focal_gamma', 'dice_weight',
                 'tversky_weight', 'tversky_alpha', 'tversky_beta',
                 'lovasz_weight', 'boundary_weight', 'temporal_weight',
                 'horizon_decay', 'label_smoothing',
                 'v2_curriculum_horizon', 'v2_horizon_start', 'v2_horizon_full',
                 'v2_horizon_ramp_end_epoch', 'v2_mixup_prob', 'v2_mixup_alpha',
                 'v2_aux_ba_weight', 'v2_monotonic_weight',
                 'ema_start_epoch', 'ema_decay',
                 'val_threshold', 'val_threshold_grid',
                 'augmentation_noise_std', 'save_every'],
    'pinn':     ['gamma', 'pde_D', 'pde_mu', 'beta_max', 'eta_max', 'xi_max',
                 'warmup_epochs', 'start_epoch'],
}

missing = []
for section, keys in REQUIRED.items():
    if section not in cfg or not isinstance(cfg[section], dict):
        missing.append(f'{section}  (whole section)')
        continue
    for k in keys:
        if k not in cfg[section]:
            missing.append(f'{section}.{k}')
if missing:
    raise KeyError(
        'Config is missing required keys — notebook has no defaults to fill them:\\n  - '
        + '\\n  - '.join(missing)
        + f'\\n\\nEdit the YAML ({cfg_path}) and re-run this cell.'
    )

# The only field the notebook is allowed to override is output_dir, because
# that path is Colab-infrastructure-dependent and not a training choice.
# Every other setting is taken verbatim from the YAML.
cfg['output_dir'] = LOCAL_OUT

# Sanity check on dataset paths from the YAML
assert Path(cfg['data']['root_train']).is_dir(), \\
    f\"data.root_train does not exist: {cfg['data']['root_train']}\"
assert Path(cfg['data']['root_val']).is_dir(), \\
    f\"data.root_val does not exist: {cfg['data']['root_val']}\"

print(f\"\\nConfig summary ({RUN_NAME}):\")
print(f\"  label_channel : {cfg['data']['label_channel']}  (0 = BA)\")
print(f\"  root_train    : {cfg['data']['root_train']}\")
print(f\"  root_val      : {cfg['data']['root_val']}\")
print(f\"  epochs        : {cfg['training']['epochs']}\")
print(f\"  batch_size    : {cfg['training']['batch_size']}\")
print(f\"  pos_weight    : {cfg['training']['pos_weight']}\")
print(f\"  lr / eta_min  : {cfg['training']['lr']} / {cfg['training']['eta_min']}\")
print(f\"  weight_decay  : {cfg['training']['weight_decay']}\")
print(f\"  mixup p/a     : {cfg['training']['v2_mixup_prob']} / {cfg['training']['v2_mixup_alpha']}\")
print(f\"  monotonic w   : {cfg['training']['v2_monotonic_weight']}\")
print(f\"  v2_aux_ba_head: {cfg['model']['v2_aux_ba_head']}\")
print(f\"  thr grid      : {cfg['training']['val_threshold_grid']}\")"""))

# ===================================================================
# Section 7 — Training loop (BA-specific)
# ===================================================================
cells.append(md("""## Section 7 · Training loop (BA) — **resumable**

Self-contained training loop derived from `colab_run29` cell 20, with BA-specific changes and full resume support.

### BA-specific behaviour
1. **Target is the BA mask** (because `label_channel=0`) — same name `af_fut` in the batch, different semantics.
2. **Mixup is disabled** (`v2_mixup_prob = 0`) to preserve monotonic structure.
3. **Monotonic consistency loss**: `mean(relu(p_t - p_{t+1}))` between consecutive-horizon probabilities.  Weight `v2_monotonic_weight`.
4. **Auxiliary BA head is OFF** — primary head already predicts BA, so the `l_ba` term is zero by construction.
5. **Threshold grid widened** to `[0.25, 0.60]` — BA masks are denser than per-frame AF so the optimal threshold sits lower.

### Resume mechanism
At the end of **every** epoch the loop saves a **full-state** `latest.pt` to both `LOCAL_OUT` and `DRIVE_OUT`. Contents:

- `epoch`, `global_step`
- `model_state`, `opt_state`, `sched_state`, `scaler_state`
- `ema_shadow`, `ema_initialized`
- `best_val_iou`, `best_epoch`, `best_thr`
- `rng_state`, `cuda_rng_state`, `np_rng_state` *(for reproducibility of augmentation draws)*
- `cfg` *(embedded snapshot for audit)*

On the next Colab session:

- `RESUME_TRAINING = False` → loop always starts fresh (old checkpoints ignored).
- `RESUME_TRAINING = True` and `RESUME_CKPT_PATH` set → loads that exact checkpoint.
- `RESUME_TRAINING = True` and `AUTO_RESUME_LATEST = True` → searches `LOCAL_OUT/latest.pt`, then `DRIVE_OUT/latest.pt`.
- `RESUME_TRAINING = True` but nothing found → prints a clear warning and starts fresh.

When a checkpoint is loaded, `metrics.csv` is **appended** instead of truncated, `best_val_iou` carries over so `best_model.pt` is only overwritten on genuine improvement, and the next epoch number equals `ckpt['epoch'] + 1`.

### Invariants (unchanged from before)
`ForecastPINNLoss` supervised term, AdamW + cosine LR with linear warmup, bf16 autocast on CUDA, EMA(0.999) from `ema_start_epoch`, curriculum horizon, wind extracted from `env_fut` for PINN, `grad_clip = 0.5`.
"""))
cells.append(code("""import copy, time, csv, math, shutil
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np

from firest_graphnet.training.losses import ForecastPINNLoss
from firest_graphnet.data.future_providers import extract_wind


# ── helpers ────────────────────────────────────────────────────────────────
def curriculum_horizon(epoch, cfg):
    tr = cfg['training']
    if not bool(tr['v2_curriculum_horizon']):
        return int(cfg['forecast']['horizon'])
    s = int(tr['v2_horizon_start'])
    f = int(tr['v2_horizon_full'])
    e = int(tr['v2_horizon_ramp_end_epoch'])
    if epoch >= e:
        return f
    frac = (epoch - 1) / max(e - 1, 1)
    return max(s, int(round(s + frac * (f - s))))


def monotonic_penalty(logits):
    # logits (B,H,1,Hp,Wp); penalise cases where prob decreases along horizon.
    if logits.shape[1] < 2:
        return logits.new_zeros(())
    prob = torch.sigmoid(logits)
    diff = prob[:, :-1] - prob[:, 1:]     # (B,H-1,...)
    return F.relu(diff).mean()


class EMA:
    # Delayed-init EMA (shadow captured at ema_start_epoch).
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.initialized = False

    def maybe_init(self, model, epoch, start_epoch):
        if not self.initialized and epoch >= start_epoch:
            self.shadow = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if v.dtype.is_floating_point
            }
            self.initialized = True

    def update(self, model):
        if not self.initialized:
            return
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().to(self.shadow[k].dtype),
                                                alpha=1.0 - d)

    def swap_in(self, model):
        if not self.initialized:
            return False
        live = model.state_dict()
        self.backup = {k: live[k].detach().clone() for k in self.shadow}
        model.load_state_dict({**live, **self.shadow}, strict=False)
        return True

    def swap_out(self, model):
        if not self.initialized:
            return
        live = model.state_dict()
        model.load_state_dict({**live, **self.backup}, strict=False)


def _accum_thr(logits, target, thr_grid, fixed_thr, acc, fixed_acc):
    prob = torch.sigmoid(logits.detach())
    gt   = (target.detach() > 0.5).float()
    for t in thr_grid:
        pred = prob.gt(t).float()
        acc[t]['tp'] += float((pred * gt).sum().item())
        acc[t]['fp'] += float((pred * (1 - gt)).sum().item())
        acc[t]['fn'] += float(((1 - pred) * gt).sum().item())
    pred = prob.gt(fixed_thr).float()
    fixed_acc['tp'] += float((pred * gt).sum().item())
    fixed_acc['fp'] += float((pred * (1 - gt)).sum().item())
    fixed_acc['fn'] += float(((1 - pred) * gt).sum().item())


def _iou(a):
    denom = a['tp'] + a['fp'] + a['fn']
    return a['tp'] / denom if denom > 0 else 0.0


def _pr(a):
    p = a['tp'] / (a['tp'] + a['fp']) if (a['tp'] + a['fp']) > 0 else 0.0
    r = a['tp'] / (a['tp'] + a['fn']) if (a['tp'] + a['fn']) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ── resume infrastructure ──────────────────────────────────────────────────
def save_full_checkpoint(path, *, epoch, global_step, model, opt, sched,
                         scaler, ema, best_val_iou, best_epoch, best_thr,
                         cfg):
    \"\"\"Write a full-state checkpoint atomically (tmp + rename).\"\"\"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch':           int(epoch),
        'global_step':     int(global_step),
        'model_state':     {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'opt_state':       opt.state_dict(),
        'sched_state':     sched.state_dict(),
        'scaler_state':    (scaler.state_dict() if scaler is not None else None),
        'ema_shadow':      ({k: v.detach().cpu() for k, v in ema.shadow.items()}
                            if ema.initialized else None),
        'ema_initialized': bool(ema.initialized),
        'best_val_iou':    float(best_val_iou),
        'best_epoch':      int(best_epoch),
        'best_thr':        float(best_thr),
        'cfg':             cfg,
        'rng_state':       torch.get_rng_state(),
        'cuda_rng_state':  (torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None),
        'np_rng_state':    np.random.get_state(),
    }
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, str(tmp))
    tmp.replace(path)        # atomic on POSIX, best-effort elsewhere


def find_resume_checkpoint(local_dir, drive_dir, explicit_path, auto):
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    if not auto:
        return None
    for d in (local_dir, drive_dir):
        if d is None:
            continue
        p = Path(d) / 'latest.pt'
        if p.exists():
            return p
    return None


def try_resume(ckpt_path, model, opt, sched, scaler, ema):
    \"\"\"Load full state. Returns dict(start_epoch, global_step, best_val_iou,
    best_epoch, best_thr, resumed_from) or None on failure.\"\"\"
    if ckpt_path is None or not Path(ckpt_path).exists():
        return None
    sd = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

    model.load_state_dict(sd['model_state'], strict=True)
    opt.load_state_dict(sd['opt_state'])
    sched.load_state_dict(sd['sched_state'])
    if scaler is not None and sd.get('scaler_state') is not None:
        scaler.load_state_dict(sd['scaler_state'])

    if sd.get('ema_shadow') is not None:
        ema.shadow = {k: v.to(DEVICE) for k, v in sd['ema_shadow'].items()}
        ema.initialized = bool(sd.get('ema_initialized', True))
    else:
        ema.shadow = {}
        ema.initialized = False

    # RNG restoration (best-effort; never fatal)
    try:
        rs = sd.get('rng_state')
        if rs is not None:
            torch.set_rng_state(rs.cpu() if hasattr(rs, 'cpu') else rs)
    except Exception as e:
        print(f'  [resume] could not restore torch rng: {e}')
    try:
        crs = sd.get('cuda_rng_state')
        if crs is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, 'cpu') else s
                                          for s in crs])
    except Exception as e:
        print(f'  [resume] could not restore cuda rng: {e}')
    try:
        if sd.get('np_rng_state') is not None:
            np.random.set_state(sd['np_rng_state'])
    except Exception as e:
        print(f'  [resume] could not restore numpy rng: {e}')

    return dict(
        start_epoch  = int(sd['epoch']) + 1,
        global_step  = int(sd.get('global_step', 0)),
        best_val_iou = float(sd.get('best_val_iou', -1.0)),
        best_epoch   = int(sd.get('best_epoch', 0)),
        best_thr     = float(sd.get('best_thr', 0.5)),
        resumed_from = str(ckpt_path),
        ckpt_epoch   = int(sd['epoch']),
    )


# ── main training loop ────────────────────────────────────────────────────
def train_v2_ba(cfg, out_dir=LOCAL_OUT, drive_out=DRIVE_OUT,
                resume=RESUME_TRAINING, auto_resume=AUTO_RESUME_LATEST,
                resume_path=RESUME_CKPT_PATH):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_out_p = Path(drive_out) if drive_out else None
    if drive_out_p is not None:
        drive_out_p.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.csv'

    # ── build dataset, model, loss, opt, sched, scaler, ema ──────────────
    train_loader, val_loader, _, _, aux_ch = build_loaders(cfg)
    print(f'  train batches: {len(train_loader)}   val batches: {len(val_loader)}')

    mcfg = cfg['model']
    model = FireSTGraphNetV2(
        in_ch_img=8, in_ch_aux=aux_ch, in_ch_static=1,
        encoder_channels=tuple(mcfg['encoder_channels']),
        gnn_patches=tuple(mcfg['v2_graph_pyramid_patches']),
        gat_heads=int(mcfg['v2_gat_heads']),
        lstm_hidden=int(mcfg['lstm_hidden']),
        lstm_layers=int(mcfg['lstm_layers']),
        lstm_dropout=float(mcfg['lstm_dropout']),
        lstm_attn_heads=int(mcfg['v2_lstm_attn_heads']),
        weather_ch=int(cfg['forecast']['weather_channels']),
        weather_emb=int(cfg['forecast']['weather_emb_dim']),
        horizon=int(cfg['forecast']['horizon']),
        dropout=float(mcfg['decoder_dropout']),
        aux_ba=bool(mcfg['v2_aux_ba_head']),
        use_pos_emb   = bool(mcfg['v3_pos_emb']),
        use_feedback  = bool(mcfg['v3_autoregressive']),
        use_skip_film = bool(mcfg['v3_skip_film']),
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  model params : {n_params:.2f}M  (aux_ba_head={mcfg[\"v2_aux_ba_head\"]})')

    tr = cfg['training']; pn = cfg['pinn']
    loss_fn = ForecastPINNLoss(
        pos_weight      = float(tr['pos_weight']),
        focal_gamma     = float(tr['focal_gamma']),
        boundary_weight = float(tr['boundary_weight']),
        temporal_weight = float(tr['temporal_weight']),
        horizon_decay   = float(tr['horizon_decay']),
        dice_weight     = float(tr['dice_weight']),
        tversky_weight  = float(tr['tversky_weight']),
        tversky_alpha   = float(tr['tversky_alpha']),
        tversky_beta    = float(tr['tversky_beta']),
        lovasz_weight   = float(tr['lovasz_weight']),
        label_smoothing = float(tr['label_smoothing']),
        gamma           = float(pn['gamma']),
        pde_D           = float(pn['pde_D']),
        pde_mu           = float(pn['pde_mu']),
        beta_max        = float(pn['beta_max']),
        eta_max         = float(pn['eta_max']),
        xi_max          = float(pn['xi_max']),
        warmup_epochs   = int(pn['warmup_epochs']),
        start_epoch     = int(pn['start_epoch']),
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr['lr']),
        weight_decay=float(tr['weight_decay']),
        betas=(0.9, 0.995),
    )
    total_ep = int(tr['epochs'])
    warm = int(tr['warmup_epochs_lr'])

    def lr_lambda(ep):
        if ep < warm:
            return float(tr['warmup_start_factor']) + (
                1 - float(tr['warmup_start_factor'])) * ep / max(warm, 1)
        t = (ep - warm) / max(total_ep - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(tr['mixed_precision'])
    amp_dtype = torch.bfloat16 if (DEVICE.type == 'cuda'
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    ema = EMA(decay=float(tr['ema_decay']))
    ema_start = int(tr['ema_start_epoch'])

    thr_grid  = [float(t) for t in tr['val_threshold_grid']]
    fixed_thr = float(tr['val_threshold'])
    mono_w    = float(tr['v2_monotonic_weight'])

    # ── RESUME ───────────────────────────────────────────────────────────
    start_epoch   = 1
    global_step   = 0
    best_val_iou  = -1.0
    best_epoch    = 0
    best_thr      = fixed_thr
    resumed_from  = None

    if resume:
        ckpt_path = find_resume_checkpoint(out_dir, drive_out_p,
                                           resume_path, auto_resume)
        if ckpt_path is not None:
            info = try_resume(ckpt_path, model, opt, sched, scaler, ema)
            if info is not None:
                start_epoch   = info['start_epoch']
                global_step   = info['global_step']
                best_val_iou  = info['best_val_iou']
                best_epoch    = info['best_epoch']
                best_thr      = info['best_thr']
                resumed_from  = info['resumed_from']
                print(f\"\\n>>> RESUMED from    : {resumed_from}\")
                print(f\"    ckpt epoch      : {info['ckpt_epoch']}\")
                print(f\"    next epoch      : {start_epoch}\")
                print(f\"    best so far     : val_iou_best_thr={best_val_iou:.4f}  \"
                      f\"@ep{best_epoch}  thr={best_thr:.2f}\")
                print(f\"    global_step     : {global_step}\")
                print(f\"    ema initialized : {ema.initialized}\")
            else:
                print(f\"\\n>>> Resume checkpoint unreadable: {ckpt_path} — starting FRESH.\")
        else:
            print(\"\\n>>> Resume requested but NO checkpoint found — starting FRESH.\")
    else:
        print(\"\\n>>> Fresh training run (resume disabled).\")

    if start_epoch > total_ep:
        print(f\"\\nNothing to do: start_epoch ({start_epoch}) > epochs ({total_ep}). \"
              f\"Increase training.epochs in the YAML or set RESUME_TRAINING=False.\")
        return str(out_dir / 'best_model.pt')

    # ── CSV: append when resuming, otherwise create with header ──────────
    col_names = [
        'epoch', 'horizon_cur', 'train_loss', 'train_sup', 'train_mono',
        'val_loss', 'val_iou_fixed', 'val_iou_best_thr', 'val_thr_best',
        'val_prec', 'val_rec', 'val_f1', 'lr',
    ]
    if resumed_from is not None and metrics_path.exists() and metrics_path.stat().st_size > 0:
        print(f\"    metrics.csv     : appending (existing {metrics_path.stat().st_size/1024:.1f} KB)\")
    else:
        with metrics_path.open('w', newline='') as f:
            csv.writer(f).writerow(col_names)

    # ── MAIN EPOCH LOOP ──────────────────────────────────────────────────
    for epoch in range(start_epoch, total_ep + 1):
        ep_start = time.time()
        H_cur = curriculum_horizon(epoch, cfg)
        loss_fn.set_epoch(epoch)

        # TRAIN
        model.train()
        train_loss, train_sup_v, train_mono_v = [], [], []
        pbar = tqdm(train_loader, desc=f'ep {epoch:03d}/{total_ep}  H={H_cur}',
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if H_cur < batch['af_fut'].shape[1]:
                batch = dict(batch)
                batch['af_fut']  = batch['af_fut'][:, :H_cur]
                batch['env_fut'] = batch['env_fut'][:, :H_cur]

            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                out = model(batch, horizon_override=H_cur)
                af_logit = out['af_pred']
                tgt      = batch['af_fut']
                wind = extract_wind(batch['env_fut'],
                                    source=batch.get('env_source', 'firepred')
                                    if not torch.is_tensor(batch.get('env_source'))
                                    else 'firepred')
                losses = loss_fn(af_logit, tgt,
                                 ba_init=batch['ba_init'], wind=wind)
                loss_main = losses['total']
                l_mono    = monotonic_penalty(af_logit)

            loss = loss_main + mono_w * l_mono

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(tr['grad_clip']))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(tr['grad_clip']))
                opt.step()

            ema.maybe_init(model, epoch, ema_start)
            if epoch >= ema_start:
                ema.update(model)

            global_step += 1
            train_loss.append(float(loss.item()))
            train_sup_v.append(float(losses['sup'].item()))
            train_mono_v.append(float(l_mono.item()))
            pbar.set_postfix(loss=f'{loss.item():.3f}',
                             sup=f'{float(losses[\"sup\"].item()):.3f}',
                             mono=f'{float(l_mono.item()):.4f}')

        sched.step()

        # VAL (EMA weights if warmed up)
        swap_ctx = False
        if epoch >= ema_start and ema.initialized:
            swap_ctx = ema.swap_in(model)
        model.eval()

        acc = {t: dict(tp=0.0, fp=0.0, fn=0.0) for t in thr_grid}
        fixed_acc = dict(tp=0.0, fp=0.0, fn=0.0)
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    out = model(batch)
                    af_logit = out['af_pred']
                    tgt      = batch['af_fut']
                    wind = extract_wind(batch['env_fut'], source='firepred')
                    vlosses = loss_fn(af_logit, tgt,
                                      ba_init=batch['ba_init'], wind=wind)
                val_losses.append(float(vlosses['total'].item()))
                _accum_thr(af_logit, tgt, thr_grid, fixed_thr, acc, fixed_acc)

        ious = {t: _iou(acc[t]) for t in thr_grid}
        thr_best_ep   = max(ious, key=ious.get)
        val_iou_best  = ious[thr_best_ep]
        val_iou_fixed = _iou(fixed_acc)
        p_best, r_best, f1_best = _pr(acc[thr_best_ep])

        if swap_ctx:
            ema.swap_out(model)

        # APPEND to CSV
        row = [
            epoch, H_cur,
            float(np.mean(train_loss)),
            float(np.mean(train_sup_v)),
            float(np.mean(train_mono_v)),
            float(np.mean(val_losses)),
            val_iou_fixed, val_iou_best, thr_best_ep,
            p_best, r_best, f1_best,
            opt.param_groups[0]['lr'],
        ]
        with metrics_path.open('a', newline='') as f:
            csv.writer(f).writerow(row)

        # BEST (with EMA shadow merged if warmed up)
        if val_iou_best > best_val_iou:
            best_val_iou = val_iou_best
            best_epoch   = epoch
            best_thr     = thr_best_ep
            state = dict(model.state_dict())
            if epoch >= ema_start and ema.initialized:
                state = {**state, **ema.shadow}
            torch.save({
                'epoch': epoch,
                'model_state': state,
                'val_iou_best_thr': val_iou_best,
                'thr_best': thr_best_ep,
                'cfg': cfg,
            }, str(out_dir / 'best_model.pt'))
            if drive_out_p is not None:
                shutil.copy2(out_dir / 'best_model.pt',
                             drive_out_p / 'best_model.pt')

        # PERIODIC snapshot (preserves old behaviour)
        if epoch % int(tr['save_every']) == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_shadow': ema.shadow if epoch >= ema_start else None,
            }, str(out_dir / f'checkpoint_epoch{epoch:03d}.pt'))
            if drive_out_p is not None:
                shutil.copy2(out_dir / f'checkpoint_epoch{epoch:03d}.pt',
                             drive_out_p / f'checkpoint_epoch{epoch:03d}.pt')

        # LATEST (full state) — every epoch, mirrored to Drive for crash safety
        save_full_checkpoint(
            out_dir / 'latest.pt',
            epoch=epoch, global_step=global_step,
            model=model, opt=opt, sched=sched, scaler=scaler, ema=ema,
            best_val_iou=best_val_iou, best_epoch=best_epoch, best_thr=best_thr,
            cfg=cfg,
        )
        if drive_out_p is not None:
            try:
                shutil.copy2(out_dir / 'latest.pt',
                             drive_out_p / 'latest.pt')
            except Exception as e:
                print(f'  [latest] drive mirror failed (non-fatal): {e}')

        dt = time.time() - ep_start
        print(f'ep{epoch:03d}  H={H_cur:02d}  '
              f'train={np.mean(train_loss):.3f}  val={np.mean(val_losses):.3f}  '
              f'IoU@fixed={val_iou_fixed:.4f}  IoU@best={val_iou_best:.4f} '
              f'(thr={thr_best_ep:.2f})  P={p_best:.3f} R={r_best:.3f} F1={f1_best:.3f}  '
              f'mono={np.mean(train_mono_v):.4f}  '
              f'lr={opt.param_groups[0][\"lr\"]:.2e}  {dt:.0f}s')

    print(f'\\n=== BEST: val_iou_best_thr = {best_val_iou:.4f} @ep {best_epoch} '
          f'(thr={best_thr:.2f}) ===')
    return str(out_dir / 'best_model.pt')


# Run!
BEST_CKPT = train_v2_ba(cfg, out_dir=LOCAL_OUT, drive_out=DRIVE_OUT,
                        resume=RESUME_TRAINING,
                        auto_resume=AUTO_RESUME_LATEST,
                        resume_path=RESUME_CKPT_PATH)
print(f'\\nBest checkpoint: {BEST_CKPT}')"""))

# ===================================================================
# Section 8 — Validation curves
# ===================================================================
cells.append(md("## Section 8 · Validation curves"))
cells.append(code("""import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(LOCAL_OUT)
df = pd.read_csv(OUT_DIR / 'metrics.csv')
print(f'Loaded {len(df)} epochs of metrics.')
print(df.tail(5).to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle(f'FireST-GraphNet V2 BA ({RUN_NAME}) — training curves', fontsize=14)

ax = axes[0,0]
ax.plot(df['epoch'], df['val_iou_fixed'],    label='Val IoU (fixed thr=0.5)', color='tomato', alpha=0.7)
ax.plot(df['epoch'], df['val_iou_best_thr'], label='Val IoU (best thr)',     color='darkred', lw=2)
b = df.loc[df['val_iou_best_thr'].idxmax()]
ax.axhline(b['val_iou_best_thr'], ls='--', color='red', alpha=0.6,
           label=f"BA best {b['val_iou_best_thr']:.4f} @ep{int(b['epoch'])}")
ax.set_title('Val IoU — BA'); ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

ax = axes[0,1]
ax.plot(df['epoch'], df['val_thr_best'], marker='.', color='purple')
ax.set_ylim(0.20, 0.70)
ax.set_title('Best threshold per epoch'); ax.set_xlabel('epoch'); ax.grid(True)

ax = axes[0,2]
ax.plot(df['epoch'], df['horizon_cur'], marker='.', color='teal')
ax.set_title('Curriculum horizon'); ax.set_xlabel('epoch'); ax.set_ylabel('H_cur'); ax.grid(True)

ax = axes[1,0]
ax.plot(df['epoch'], df['val_prec'], label='Precision', color='C0', lw=2)
ax.plot(df['epoch'], df['val_rec'],  label='Recall',    color='C1', lw=2)
ax.plot(df['epoch'], df['val_f1'],   label='F1',        color='C2', lw=2)
ax.set_title('Val P / R / F1 (at best-thr)'); ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

ax = axes[1,1]
ax.plot(df['epoch'], df['train_loss'], label='train', color='steelblue')
ax.plot(df['epoch'], df['val_loss'],   label='val',   color='darkorange')
ax.set_title('Loss'); ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

ax = axes[1,2]
ax.plot(df['epoch'], df['train_mono'], marker='.', color='darkgreen')
ax.set_title('Monotonic penalty (train)'); ax.set_xlabel('epoch'); ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves_v2_ba.png', dpi=140)
plt.show()

best = df.loc[df['val_iou_best_thr'].idxmax()]
thrs = df['val_thr_best'].value_counts().sort_index()
print('\\n=== Best epoch ===')
print(f"  epoch {int(best['epoch']):3d}   H={int(best['horizon_cur']):2d}")
print(f"  Val IoU (best thr): {best['val_iou_best_thr']:.4f}  at thr={best['val_thr_best']:.2f}")
print(f"  Val IoU (fixed=0.5): {best['val_iou_fixed']:.4f}")
print(f"  P / R / F1         : {best['val_prec']:.4f} / {best['val_rec']:.4f} / {best['val_f1']:.4f}")
print('\\nThreshold distribution across epochs:')
for t, n in thrs.items():
    print(f'  thr={t:.2f}: {n:3d} epochs')

if len(df) >= 20:
    print(f"\\nVal IoU stability (last 20 ep std): {df['val_iou_best_thr'].tail(20).std():.4f}  (target <= 0.005)")"""))

# ===================================================================
# Section 9 — Inference visualisation
# ===================================================================
cells.append(md("## Section 9 · Inference visualisation — 3 val samples × 15 horizons"))
cells.append(code("""from matplotlib.colors import ListedColormap
from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider   import build_static_provider

ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
print(f"Loaded best_model.pt — epoch {ckpt['epoch']}  "
      f"val_iou_best_thr={ckpt['val_iou_best_thr']:.4f}  thr={ckpt['thr_best']:.2f}")

val_dir = cfg['data']['root_val']
fp_va = build_firepred_provider(data_root=val_dir,
                                spatial_size=(cfg['data']['img_size'],)*2)
st_va = build_static_provider(data_root=val_dir,
                              spatial_size=(cfg['data']['img_size'],)*2)

mcfg = cfg['model']
model_inf = FireSTGraphNetV2(
    in_ch_img=8, in_ch_aux=int(fp_va.channels), in_ch_static=1,
    encoder_channels=tuple(mcfg['encoder_channels']),
    gnn_patches=tuple(mcfg['v2_graph_pyramid_patches']),
    gat_heads=mcfg['v2_gat_heads'],
    lstm_hidden=mcfg['lstm_hidden'],
    lstm_layers=mcfg['lstm_layers'],
    lstm_dropout=mcfg['lstm_dropout'],
    lstm_attn_heads=mcfg['v2_lstm_attn_heads'],
    weather_ch=cfg['forecast']['weather_channels'],
    weather_emb=cfg['forecast']['weather_emb_dim'],
    horizon=cfg['forecast']['horizon'],
    dropout=mcfg['decoder_dropout'],
    aux_ba=mcfg['v2_aux_ba_head'],
    use_pos_emb   = bool(mcfg['v3_pos_emb']),
    use_feedback  = bool(mcfg['v3_autoregressive']),
    use_skip_film = bool(mcfg['v3_skip_film']),
).to(DEVICE)
missing, unexpected = model_inf.load_state_dict(ckpt['model_state'], strict=False)
print(f'  missing keys: {len(missing)}   unexpected: {len(unexpected)}')
model_inf.eval()

val_base = TSSatFireForecastDataset(
    val_dir,
    firepred_provider=fp_va, static_provider=st_va,
    T_in=cfg['forecast']['T_in'],
    horizon=cfg['forecast']['horizon'],
    label_channel=int(cfg['data']['label_channel']),
)

np.random.seed(0)
pick = np.random.choice(len(val_base), size=min(3, len(val_base)), replace=False)

cmap_bin = ListedColormap(['#1b2631', '#c0392b'])   # BA palette — dark bg / burnt red
thr = float(ckpt['thr_best'])
H = cfg['forecast']['horizon']

for p_idx, s_idx in enumerate(pick):
    sample = val_base[int(s_idx)]
    batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
             for k, v in sample.items()}
    with torch.no_grad():
        out = model_inf(batch)
    ba_logit = out['af_pred'][0, :, 0].cpu().numpy()
    ba_prob  = 1 / (1 + np.exp(-ba_logit))
    # enforce monotonic cumulative at inference for cleaner visualisation
    ba_prob_mono = np.maximum.accumulate(ba_prob, axis=0)
    ba_pred = (ba_prob_mono > thr).astype(np.uint8)
    ba_gt   = sample['af_fut'].squeeze(1).numpy().astype(np.uint8)

    rows, cols = 2, H
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    fig.suptitle(f'BA sample {p_idx+1}  (val idx={int(s_idx)})  thr={thr:.2f}',
                 fontsize=11, y=1.02)
    for t in range(H):
        axes[0, t].imshow(ba_pred[t], cmap=cmap_bin, vmin=0, vmax=1,
                          interpolation='nearest')
        axes[0, t].set_title(f't+{t+1}', fontsize=7)
        axes[0, t].set_xticks([]); axes[0, t].set_yticks([])
        axes[1, t].imshow(ba_gt[t], cmap=cmap_bin, vmin=0, vmax=1,
                          interpolation='nearest')
        axes[1, t].set_xticks([]); axes[1, t].set_yticks([])
    axes[0, 0].set_ylabel('Pred', fontsize=8)
    axes[1, 0].set_ylabel('GT',   fontsize=8)
    plt.tight_layout()
    outp = OUT_DIR / f'ba_inference_sample{p_idx+1}.png'
    plt.savefig(outp, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'  saved {outp}')

print('\\n=== Per-horizon IoU on 16 val samples ===')
per_h_tp = np.zeros(H); per_h_fp = np.zeros(H); per_h_fn = np.zeros(H)
with torch.no_grad():
    sel = np.random.choice(len(val_base), size=min(16, len(val_base)), replace=False)
    for idx in sel:
        sample = val_base[int(idx)]
        batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
                 for k, v in sample.items()}
        out = model_inf(batch)
        ba_prob = torch.sigmoid(out['af_pred'][0, :, 0]).cpu().numpy()
        ba_prob = np.maximum.accumulate(ba_prob, axis=0)
        gt = sample['af_fut'].squeeze(1).numpy()
        pred = (ba_prob > thr).astype(np.float32)
        gt_b = (gt > 0.5).astype(np.float32)
        for t in range(H):
            per_h_tp[t] += (pred[t] * gt_b[t]).sum()
            per_h_fp[t] += (pred[t] * (1 - gt_b[t])).sum()
            per_h_fn[t] += ((1 - pred[t]) * gt_b[t]).sum()

ious_h = per_h_tp / np.clip(per_h_tp + per_h_fp + per_h_fn, 1e-6, None)
for t in range(H):
    print(f'  t+{t+1:2d}:  IoU={ious_h[t]:.4f}')

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(range(1, H + 1), ious_h, marker='o', color='C3')
ax2.set_xlabel('horizon'); ax2.set_ylabel('IoU'); ax2.grid(True)
ax2.set_title('BA per-horizon IoU (16 val samples, best-thr, monotonic)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'ba_per_horizon_iou.png', dpi=140)
plt.show()"""))

# ===================================================================
# Section 10 — Export to Drive
# ===================================================================
cells.append(md("## Section 10 · Export results to Google Drive"))
cells.append(code("""import shutil, pandas as pd
from pathlib import Path

LOCAL_OUT  = Path(LOCAL_OUT)
DRIVE_OUT  = Path(DRIVE_OUT)        # already set in Section 1
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

# Copy key artefacts. `latest.pt` is mirrored every epoch by the trainer,
# but we copy it once more here to pick up any final-epoch state.
artifacts = [
    'best_model.pt', 'latest.pt', 'metrics.csv',
    'training_curves_v2_ba.png', 'ba_per_horizon_iou.png',
    'ba_inference_sample1.png', 'ba_inference_sample2.png', 'ba_inference_sample3.png',
]
# Also copy any checkpoint_epoch*.pt files
for p in LOCAL_OUT.glob('checkpoint_epoch*.pt'):
    artifacts.append(p.name)

for fname in artifacts:
    src = LOCAL_OUT / fname
    if src.exists():
        dst = DRIVE_OUT / fname
        shutil.copy2(src, dst)
        print(f"  {fname:40s}  {dst.stat().st_size/1024**2:7.2f} MB")
    else:
        print(f"  {fname:40s}  (not present — skipped)")

shutil.copy2(str(local_cfg_path), DRIVE_OUT / local_cfg_path.name)
print(f"\\nExported to: {DRIVE_OUT}")

if (LOCAL_OUT / 'metrics.csv').exists():
    df = pd.read_csv(LOCAL_OUT / 'metrics.csv')
    best = df.loc[df['val_iou_best_thr'].idxmax()]
    print(f"\\n{'='*55}")
    print(f"  FireST-GraphNet V2 BA ({RUN_NAME}) FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  best epoch     : {int(best['epoch'])}  H={int(best['horizon_cur'])}")
    print(f"  val IoU (best) : {best['val_iou_best_thr']:.4f}  @thr={best['val_thr_best']:.2f}")
    print(f"  P / R / F1     : {best['val_prec']:.4f} / {best['val_rec']:.4f} / {best['val_f1']:.4f}")
    print(f"{'='*55}")
    print('\\nNext step: run colab_inference_af_v2_all_checkpoints.ipynb to batch-export')
    print(f'  predictions for every checkpoint of {RUN_NAME}.')"""))


# ===================================================================
# Serialise
# ===================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language":     "python",
            "name":         "python3",
        },
        "language_info": {
            "name":           "python",
            "version":        "3.11",
            "file_extension": ".py",
            "mimetype":       "text/x-python",
        },
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Round-trip validation
with OUT.open(encoding="utf-8") as f:
    nb2 = json.load(f)
assert len(nb2["cells"]) == len(cells), "cell count mismatch after round-trip"
print(f"WROTE {OUT}")
print(f"num cells: {len(cells)}")
print(f"re-parsed OK   cells = {len(nb2['cells'])}")
