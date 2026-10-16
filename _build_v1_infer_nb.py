# -*- coding: utf-8 -*-
"""Build the V1 AF batch-inference Colab notebook.

This is the V1 counterpart of `_build_v2_infer_nb.py` -> `colab_inference_af_v2_all_checkpoints.ipynb`.
Produces `colab_inference_af_v1_all_checkpoints.ipynb`.

Key V1-vs-V2 differences baked into this notebook:
  * uses `FireSTGraphNet` imported from the installed `firest_graphnet` package
    (no inlined architecture) — supports both *legacy* (8-ch AF only) and
    *forecast* (28-ch = 8 AF + 19 aux + 1 static) modes, auto-detected from
    the checkpoint state_dict.
  * run discovery accepts runs where `v2_enabled` is false / missing AND
    rejects any checkpoint exposing V2 marker keys (`encoder.b1.c1.*`).
  * inference call uses V1 positional/keyword signature:
        model(af_hist, aux_hist=..., env_fut=..., env_source=..., static=...)
    and handles both (B, H, 1, H, W) and (B, 1, H, W) return shapes.
  * output layout matches V2 exactly so a single postprocess pipeline works:
        inference_exports_af_v1_colab/<run>/<ckpt>/<split>/{predictions,ground_truth,visualizations,manifest.csv,inference_summary.json}
"""
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
        """# FireST-GraphNet \u2014 AF **V1** Batch Inference (Colab)

**Purpose** \u2014 Run **one-step Active-Fire (AF)** inference on every **V1** run
(`run12`, `run13`, \u2026, `run26`) and every checkpoint available for each run,
over the **entire** `dataset_train` and `dataset_val`.

**Architecture** \u2014 V1 = `FireSTGraphNet` (legacy + forecast unified class) imported
from the project's `firest_graphnet` package.  No architecture is inlined in this
notebook \u2014 we rely on the code at `PROJECT_ROOT/firest_graphnet/models/firest_graphnet.py`
that was used to train these checkpoints in the first place, so `load_state_dict`
matches exactly.

Two V1 sub-architectures are supported, auto-detected from checkpoint keys:
  * **Legacy** (run12) \u2014 8-ch AF-only encoder + single-step `Decoder` with FiLM
  * **Forecast** (run13+) \u2014 28-ch encoder (AF 8 + aux 19 + static 1) + multi-step
    `ForecastDecoder` (we still run **one-step** inference by slicing `env_fut[:, :1]`)

**What the notebook does (top-to-bottom)**
1. Check GPU, mount Drive, set paths (`outputs/` plural, matching the Drive folder
   name the user specified: `/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean/outputs`)
2. Sync `dataset_train` / `dataset_val` from Drive to Colab local SSD (one-off)
3. Discover all V1 AF runs under `OUTPUT_ROOT` (rejects V2 runs / V2-style
   checkpoints so there is never an architecture mismatch)
4. For each run, discover all checkpoints (`best_model.pt`, `last.pt`,
   `checkpoint_epoch_XXX.pt`, \u2026)
5. Build the V1 model matching the run's checkpoint, load weights, run **one-step**
   inference on **all** train + val samples
6. Save raw `pred` / `gt` `.npy` arrays, per-sample PNG visualisations,
   `manifest.csv`, `inference_summary.json`
7. Display a representative inline grid, write a global summary across all
   runs \u00d7 checkpoints

**Output layout (Drive)**
```
<OUTPUT_ROOT>/inference_exports_af_v1_colab/
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

This layout is **identical to the V2 inference notebook's** so the V1 postprocess
notebook (`postprocess_af_inference_exports_v1.ipynb`) can read it uniformly with
the V2 layout via the V2-style filename branch of `_parse_run_ckpt_split`.

> **Scope** \u2014 AF only, **one-step** only.  BA / PA / multi-step rollout are out
> of scope.  Training is not performed.
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

# Paths
# NOTE: the user's Drive folder is the PLURAL `outputs/` (not `output/`) --
# this differs from the V2 notebook which used `output/`.  Both are supported
# below via a first-existing-wins probe.
PROJECT_ROOT = Path('/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean')

_OUTPUT_CANDIDATES = [
    PROJECT_ROOT / 'outputs',   # plural, matches user's folder
    PROJECT_ROOT / 'output',    # singular, fallback
]
OUTPUT_ROOT = next((p for p in _OUTPUT_CANDIDATES if p.exists()), _OUTPUT_CANDIDATES[0])

DRIVE_TRAIN  = PROJECT_ROOT / 'dataset_train'
DRIVE_VAL    = PROJECT_ROOT / 'dataset_val'

# Local Colab cache (Drive I/O is ~30x slower - never iterate directly on Drive)
DATASET_TRAIN = Path('/content/dataset_train')
DATASET_VAL   = Path('/content/dataset_val')

# Where inference artefacts are written.  Local first, mirror to Drive per-checkpoint.
INFER_ROOT_DRIVE = OUTPUT_ROOT / 'inference_exports_af_v1_colab'
INFER_ROOT_LOCAL = Path('/content/inference_exports_af_v1_colab')
INFER_ROOT_DRIVE.mkdir(parents=True, exist_ok=True)
INFER_ROOT_LOCAL.mkdir(parents=True, exist_ok=True)

# Expose project source tree for helper imports (firest_graphnet package,
# providers, dataset).
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

# Project-side helpers -- V1 model + V1 dataset + V1 providers.
# Evict any pre-cached older copy before importing from PROJECT_ROOT.
for mod in list(sys.modules):
    if mod == 'firest_graphnet' or mod.startswith('firest_graphnet.'):
        del sys.modules[mod]

from firest_graphnet.data.firepred_provider import build_firepred_provider
from firest_graphnet.data.static_provider   import build_static_provider
from firest_graphnet.data.dataset_forecast  import TSSatFireForecastDataset
from firest_graphnet.models.firest_graphnet import FireSTGraphNet
print('backend modules imported OK  (V1 FireSTGraphNet)')
"""
    )
)

# ---------------------------------------------------------------- CELL 6 ----
cells.append(
    md(
        """## Section 5 \u00b7 Utility helpers

Run discovery, checkpoint discovery, V1/V2 detection, filename sanitising \u2014 all the
small pieces that glue the pipeline together.
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

def _load_yaml(p):
    try:
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def discover_af_runs(output_root):
    # Return all directories under OUTPUT_ROOT that look like AF training runs.
    # Filter criteria:
    #   - has at least one YAML config (config*.yaml / config*.yml)
    #   - config.data.label_channel == 1  (AF)
    #   - has at least one .pt file
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

# V1 / V2 detection ---------------------------------------------------------
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

# V1 layout detection from state_dict -- distinguishes legacy vs forecast mode
def probe_v1_state_dict(sd):
    # Inspect the state_dict and report which V1 sub-architecture it matches.
    # Returns a dict with: has_input_norm, has_spatial_feedback,
    # has_legacy_decoder, has_forecast_decoder, first_conv_in, is_v2_layout.
    keys = list(sd.keys())
    first_conv_in = None
    for k in keys:
        if k.endswith('encoder.0.conv.weight') or k.endswith('encoder.encoder.0.conv.weight'):
            t = sd[k]
            if hasattr(t, 'shape') and len(t.shape) == 4:
                first_conv_in = int(t.shape[1])
                break
    return {
        'has_input_norm'      : any('encoder.input_norm' in k for k in keys),
        'has_spatial_feedback': any('spatial_feedback' in k or 'spatial_gate' in k for k in keys),
        'has_legacy_decoder'  : any(k.startswith('decoder.blocks.') for k in keys),
        'has_forecast_decoder': any(k.startswith('decoder.future_enc') for k in keys),
        'first_conv_in'       : first_conv_in,
        'is_v2_layout'        : is_v2_state_dict(sd),
    }

def peek_checkpoint(ckpt_path):
    c = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = c.get('model_state') or c.get('model_state_dict') or {}
    if not sd and isinstance(c, dict):
        # older V1 checkpoints sometimes stored the raw state dict at the top
        tensor_like = all(hasattr(v, 'shape') for v in c.values()) if c else False
        if tensor_like:
            sd = c
    probe = probe_v1_state_dict(sd)
    return dict(
        epoch            = c.get('epoch') if isinstance(c, dict) else None,
        val_iou_best_thr = c.get('val_iou_best_thr') if isinstance(c, dict) else None,
        thr_best         = c.get('thr_best', c.get('val_thr_best', 0.5)) if isinstance(c, dict) else 0.5,
        has_ema          = isinstance(c, dict) and 'ema_state' in c and c['ema_state'] is not None,
        is_v2            = probe['is_v2_layout'],
        forecast_mode    = probe['has_forecast_decoder'],
        legacy_mode      = probe['has_legacy_decoder'] and not probe['has_forecast_decoder'],
        first_conv_in    = probe['first_conv_in'],
        has_input_norm   = probe['has_input_norm'],
        has_spatial_fb   = probe['has_spatial_feedback'],
        top_keys         = list(c.keys()) if isinstance(c, dict) else [],
        n_tensors        = len(sd),
    )

print('Utility helpers ready')
"""
    )
)

# ---------------------------------------------------------------- CELL 7 ----
cells.append(
    md(
        """## Section 6 \u00b7 V1 architecture \u2014 `FireSTGraphNet` (legacy + forecast)

Nothing is inlined here \u2014 the V1 architecture is imported directly from
`firest_graphnet.models.firest_graphnet` (exactly the module that trained every
V1 run), so `load_state_dict` matches without strict-mode surprises.

This cell simply verifies the class is importable and has the expected forward
signature.  Two V1 checkpoints layouts are supported:

  | checkpoint class      | encoder in-channels | decoder                |
  | --------------------- | ------------------- | ---------------------- |
  | **legacy** (run12)    | 8 (AF only)         | `Decoder` (single step)|
  | **forecast** (run13+) | 28 (AF + aux + static) | `ForecastDecoder` |

V1 checkpoints that accidentally expose V2-style keys (`encoder.b1.c1.*`,
`pyramid.layers.p8.*`) will be rejected upfront in Section 8.
"""
    )
)

cells.append(
    code(
        r"""# Smoke-test import and forward signature
import inspect
print('FireSTGraphNet module :', inspect.getsourcefile(FireSTGraphNet))
fwd = inspect.signature(FireSTGraphNet.forward)
print('forward signature     :', fwd)
print('forecast_mode support :', 'forecast_mode' in inspect.signature(FireSTGraphNet.__init__).parameters)
"""
    )
)

# ---------------------------------------------------------------- CELL 8 ----
cells.append(
    md(
        """## Section 7 \u00b7 Builders \u2014 V1 model, dataset, inference helpers

- `build_v1_model_from_checkpoint(cfg, sd, device)` \u2014 builds the correct V1
  variant (legacy vs forecast) based on auto-detected state_dict keys, ported
  from `run_af_inference_exports.py::build_model_for_run`.
- `build_dataset(data_dir, cfg)` \u2014 `TSSatFireForecastDataset` + firepred +
  LULC providers.  Includes the *force-align* hack from the Windows script:
  if the `firepred_*.npy` / `lulc_*.npy` arrays contain more frames than the
  AF image/label arrays, we override the providers' internal `_N` to the AF
  frame count so the alignment guard doesn't silently zero out aux + static.
- `run_one_step_v1(model, sample, device, forecast_mode)` \u2014 calls the model
  in the right mode and returns sigmoid probabilities `(H, W)` + GT `(H, W)`.
- Export helpers write pred / GT / viz using the **V2-style strict naming**
  so a single postprocess pipeline handles both V1 and V2 outputs.
"""
    )
)

cells.append(
    code(
        r"""def build_v1_model_from_checkpoint(cfg, sd, device):
    # Construct a V1 FireSTGraphNet matching the checkpoint state_dict.
    # Rejects V2-layout checkpoints explicitly -- this notebook can't load them.
    mcfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}
    dcfg = cfg.get('data', {}) if isinstance(cfg, dict) else {}
    fcfg = cfg.get('forecast', {}) if isinstance(cfg, dict) else {}

    probe = probe_v1_state_dict(sd)
    if probe['is_v2_layout']:
        raise NotImplementedError(
            "checkpoint uses V2 architecture (encoder.b1.c1 keys). "
            "This is the V1 inference notebook -- use the V2 notebook "
            "`colab_inference_af_v2_all_checkpoints.ipynb` for V2 runs."
        )

    use_input_norm       = probe['has_input_norm']
    use_spatial_feedback = probe['has_spatial_feedback']

    if probe['has_legacy_decoder']:
        # Legacy run12 -- AF-only (first conv is 8-ch).
        in_channels = probe['first_conv_in'] or 8
        forecast_mode = False
    else:
        # Forecast mode -- AF(8) + aux(19) + static(1) = 28 channels.
        in_channels = probe['first_conv_in'] or (8 + 19 + 1)
        forecast_mode = True

    model = FireSTGraphNet(
        in_channels         = in_channels,
        encoder_channels    = mcfg.get('encoder_channels', [32, 64, 128]),
        use_cbam            = mcfg.get('use_cbam', True),
        patch_size          = mcfg.get('patch_size', 8),
        k_neighbors         = mcfg.get('k_neighbors', 8),
        gnn_hidden          = mcfg.get('gnn_hidden', 128),
        gnn_layers          = mcfg.get('gnn_layers', 2),
        lstm_hidden         = mcfg.get('lstm_hidden', 256),
        lstm_layers         = mcfg.get('lstm_layers', 2),
        lstm_dropout        = mcfg.get('lstm_dropout', 0.1),
        decoder_channels    = mcfg.get('decoder_channels', [128, 64, 32]),
        out_channels        = mcfg.get('out_channels', 1),
        img_h               = dcfg.get('img_size', 256),
        img_w               = dcfg.get('img_size', 256),
        forecast_mode       = forecast_mode,
        future_channels     = fcfg.get('future_channels', fcfg.get('weather_channels', 4)),
        future_emb_dim      = fcfg.get('future_emb_dim',   fcfg.get('weather_emb_dim',   64)),
        use_input_norm      = use_input_norm,
        use_spatial_feedback= use_spatial_feedback,
        decoder_dropout     = float(mcfg.get('decoder_dropout', 0.0)),
    )
    model = model.to(device)
    build_info = dict(
        forecast_mode       = forecast_mode,
        in_channels         = in_channels,
        use_input_norm      = use_input_norm,
        use_spatial_feedback= use_spatial_feedback,
    )
    return model, build_info

def load_state_into_model_v1(model, ckpt):
    # Load weights from the checkpoint, preferring EMA if present.
    # Returns a diagnostics dict.  Uses strict=False to tolerate the odd
    # use_input_norm / use_spatial_feedback mismatch seen in older runs.
    if isinstance(ckpt, dict):
        state = (ckpt.get('ema_state')
                 or ckpt.get('model_state')
                 or ckpt.get('model_state_dict'))
        if state is None:
            tensor_like = all(hasattr(v, 'shape') for v in ckpt.values()) if ckpt else False
            if tensor_like:
                state = ckpt
        if state is None:
            raise RuntimeError('no model_state / model_state_dict / ema_state in checkpoint')
    else:
        state = ckpt
    res = model.load_state_dict(state, strict=False)
    return {
        'used_ema'          : isinstance(ckpt, dict) and bool(ckpt.get('ema_state')),
        'n_missing_keys'    : len(res.missing_keys),
        'n_unexpected_keys' : len(res.unexpected_keys),
        'missing_sample'    : list(res.missing_keys)[:5],
        'unexpected_sample' : list(res.unexpected_keys)[:5],
    }

def build_dataset(data_dir, cfg):
    # Build TSSatFireForecastDataset with firepred + LULC providers.
    # Applies the provider force-alignment hack from run_af_inference_exports.py
    # so that extra firepred/LULC frames don't silently zero out aux+static.
    # Returns (dataset, aux_channels).
    img_sz = int(cfg['data'].get('img_size', 256))
    fp = build_firepred_provider(data_root=str(data_dir), spatial_size=(img_sz, img_sz))
    st = build_static_provider  (data_root=str(data_dir), spatial_size=(img_sz, img_sz))

    # Force-align providers to the AF frame count if they have more frames
    af_img_path = Path(data_dir) / 'af_train_img_seqtoseq_alll_1i_1.npy'
    if not af_img_path.exists():
        af_img_path = Path(data_dir) / 'af_val_img_seqtoseq_alll_1i_1.npy'
    if af_img_path.exists():
        N_af = int(np.load(str(af_img_path), mmap_mode='r').shape[0])
        if getattr(fp, '_N', None) is not None and fp._N != N_af:
            fp._N = N_af
        if getattr(st, '_N', None) is not None and st._N != N_af:
            st._N = N_af

    ds = TSSatFireForecastDataset(
        data_dir          = str(data_dir),
        T_in              = int(cfg['forecast']['T_in']),
        horizon           = int(cfg['forecast'].get('horizon', 20)),
        label_channel     = int(cfg['data'].get('label_channel', 1)),
        firepred_provider = fp,
        static_provider   = st,
    )
    aux_ch = int(getattr(fp, 'channels', 19))
    return ds, aux_ch

# One-step inference over a single sample (handles both V1 modes) -----------
@torch.no_grad()
def run_one_step_v1(model, sample, device, forecast_mode, use_amp):
    af_hist = sample['af_hist'].unsqueeze(0).to(device, non_blocking=True)

    if forecast_mode:
        aux_hist = sample['aux_hist'].unsqueeze(0).to(device, non_blocking=True)
        env_fut  = sample['env_fut' ].unsqueeze(0).to(device, non_blocking=True)
        static   = sample['static'  ].unsqueeze(0).to(device, non_blocking=True)
        env_fut_one = env_fut[:, :1]       # one-step future conditioning
        env_source = sample.get('env_source', 'firepred')
        if isinstance(env_source, (list, tuple)):
            env_source = env_source[0]

        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(af_hist, aux_hist=aux_hist, env_fut=env_fut_one,
                               env_source=env_source, static=static)
        else:
            logits = model(af_hist, aux_hist=aux_hist, env_fut=env_fut_one,
                           env_source=env_source, static=static)
    else:
        # Legacy: AF-only history, no aux / env / static
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(af_hist)
        else:
            logits = model(af_hist)

    # Normalise output shapes:
    #   forecast mode: (B, H, 1, H, W) -> first horizon step
    #   legacy   mode: (B, 1, H, W)    -> direct
    if logits.dim() == 5:
        prob = torch.sigmoid(logits[0, 0, 0]).float().cpu().numpy()
    elif logits.dim() == 4:
        prob = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
    else:
        prob = torch.sigmoid(logits.view(logits.shape[-2], logits.shape[-1])).float().cpu().numpy()

    gt_step0 = sample['af_fut'].numpy()[0, 0].astype(np.float32)
    ba_init = sample.get('ba_init')
    if ba_init is not None:
        ba_np = ba_init.numpy()
        hist_last = ba_np[0] if ba_np.ndim == 3 else ba_np
    else:
        hist_last = sample['af_hist'].numpy()[-1, -1]
    return dict(
        prob      = prob.astype(np.float32),
        gt        = gt_step0,
        hist_last = hist_last.astype(np.float32),
    )

# IoU helper
def iou_bin(pred_bin, gt_bin):
    pred_bin = (pred_bin > 0).astype(np.uint8)
    gt_bin   = (gt_bin   > 0).astype(np.uint8)
    inter = float((pred_bin & gt_bin).sum())
    union = float((pred_bin | gt_bin).sum())
    return inter / union if union > 0 else 0.0

# Export helpers (V2-style strict naming, identical layout) -----------------
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
    axes[1].imshow(gt,   cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Ground truth t+1', fontsize=9)
    axes[2].imshow(prob, cmap='hot',  vmin=0, vmax=1)
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

print('Builders + exporters ready (V1)')
"""
    )
)

# ---------------------------------------------------------------- CELL 9 ----
cells.append(
    md(
        """## Section 8 \u00b7 Discover AF runs + checkpoints

Scans `OUTPUT_ROOT`, keeps runs with `label_channel == 1` **and** a V1-compatible
checkpoint.  Specifically:

- rejects runs with `model.v2_enabled: true` in the config
- rejects checkpoints exposing V2 marker keys (`encoder.b1.c1.*`, \u2026)
- accepts V1 legacy (run12) and V1 forecast (run13+) alike -- the architecture
  is auto-selected at model-build time
"""
    )
)

cells.append(
    code(
        r"""runs_all = discover_af_runs(OUTPUT_ROOT)
print(f'Discovered {len(runs_all)} AF run(s) under {OUTPUT_ROOT}')

v1_runs      = []
skipped_runs = []
for r in runs_all:
    reason = None
    if is_v2_config(r['cfg']):
        reason = 'config has v2_enabled=True (V2 run - use the V2 inference notebook)'
    elif not r['checkpoints']:
        reason = 'no .pt files found in run directory'
    else:
        ok = False
        for p in r['checkpoints']:
            try:
                info = peek_checkpoint(p)
                if info['is_v2']:
                    continue
                if info['forecast_mode'] or info['legacy_mode']:
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            reason = 'no checkpoint exposes V1-style keys (decoder.blocks.* / decoder.future_enc*)'
    if reason is None:
        v1_runs.append(r)
    else:
        skipped_runs.append((r['name'], reason))

print(f"\n-> V1 AF runs to process : {len(v1_runs)}")
for r in v1_runs:
    print(f"  {r['name']:10s}  ckpts={len(r['checkpoints'])}  config={r['config_path'].name}")
    for p in r['checkpoints']:
        info = peek_checkpoint(p)
        mode = 'forecast' if info['forecast_mode'] else ('legacy' if info['legacy_mode'] else '?')
        print(f"     - {p.name:30s}  epoch={info['epoch']}  "
              f"val_iou={info['val_iou_best_thr']}  thr={info['thr_best']}  "
              f"mode={mode}  v2={info['is_v2']}  ema={info['has_ema']}")

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

For each V1 run \u00d7 every checkpoint \u00d7 {train, val}:

- peek the checkpoint to decide legacy vs forecast mode
- rebuild model, load weights (`strict=False`, prefer `ema_state`)
- iterate the dataset in order (no shuffle) and export `.npy` pred + `.npy` GT
  for every sample
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

for r_info in v1_runs:
    run_name = r_info['name']
    run_dir  = r_info['dir']
    cfg      = r_info['cfg']
    ckpts    = r_info['checkpoints']

    # Remap Drive paths baked into the config -> local Colab SSD
    cfg.setdefault('data', {})
    cfg['data']['root_train'] = str(DATASET_TRAIN)
    cfg['data']['root_val']   = str(DATASET_VAL)

    print('\n' + '=' * 78)
    print(f'  RUN  {run_name}   ({len(ckpts)} checkpoint(s))')
    print('=' * 78)

    # Build datasets ONCE per run
    t0 = time.time()
    try:
        train_ds, aux_ch_tr = build_dataset(DATASET_TRAIN, cfg)
    except Exception as e:
        print(f'  !! failed to build train dataset for {run_name}: {e}')
        traceback.print_exc()
        agg_rows.append(dict(run=run_name, ckpt='-', split='train',
                             status='dataset_build_failed', error=str(e)[:240]))
        continue
    try:
        val_ds,   aux_ch_va = build_dataset(DATASET_VAL,   cfg)
    except Exception as e:
        print(f'  !! failed to build val dataset for {run_name}: {e}')
        traceback.print_exc()
        agg_rows.append(dict(run=run_name, ckpt='-', split='val',
                             status='dataset_build_failed', error=str(e)[:240]))
        continue
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

        # Resolve state_dict and thr_best
        if isinstance(ckpt, dict):
            sd = (ckpt.get('ema_state')
                  or ckpt.get('model_state')
                  or ckpt.get('model_state_dict'))
            if sd is None:
                tensor_like = all(hasattr(v, 'shape') for v in ckpt.values()) if ckpt else False
                sd = ckpt if tensor_like else {}
            info = dict(
                epoch            = ckpt.get('epoch'),
                val_iou_best_thr = ckpt.get('val_iou_best_thr'),
                thr_best         = ckpt.get('thr_best', ckpt.get('val_thr_best', 0.5)),
                has_ema          = 'ema_state' in ckpt and ckpt['ema_state'] is not None,
            )
        else:
            sd = ckpt
            info = dict(epoch=None, val_iou_best_thr=None, thr_best=0.5, has_ema=False)

        thr = float(info['thr_best']) if info['thr_best'] is not None else 0.5
        print(f"     epoch={info['epoch']}   val_iou={info['val_iou_best_thr']}   "
              f"thr={thr:.2f}   has_ema={info['has_ema']}")

        try:
            model, build_info = build_v1_model_from_checkpoint(cfg, sd, DEVICE)
            load_info = load_state_into_model_v1(model, ckpt)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
        except NotImplementedError as e:
            print(f'     !! build skipped: {e}')
            agg_rows.append(dict(run=run_name, ckpt=ckpt_tag, split='-',
                                 status='v2_checkpoint_skipped', error=str(e)[:240]))
            continue
        except Exception as e:
            traceback.print_exc()
            agg_rows.append(dict(run=run_name, ckpt=ckpt_tag, split='-',
                                 status='build_failed', error=str(e)[:240]))
            continue
        forecast_mode = build_info['forecast_mode']
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"     built V1 {'forecast' if forecast_mode else 'legacy'} model "
              f'({n_param:.2f} M params)   '
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
            + f"\nforecast_mode  : {forecast_mode}"
            + f"\nin_channels    : {build_info['in_channels']}"
            + f"\nuse_input_norm : {build_info['use_input_norm']}"
            + f"\nuse_spatial_fb : {build_info['use_spatial_feedback']}"
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
                    res    = run_one_step_v1(model, sample, DEVICE,
                                             forecast_mode=forecast_mode, use_amp=USE_AMP)
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
                forecast_mode=forecast_mode,
                label_channel=int(cfg['data'].get('label_channel', 1)),
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            )
            with open(split_dir / 'inference_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            agg_rows.append(dict(
                run=run_name, ckpt=ckpt_tag, split=split_name, status='ok',
                n_samples=len(ds), mean_iou_thr=mean_iou, threshold=thr,
                epoch=info['epoch'], val_iou_best_thr=info['val_iou_best_thr'],
                forecast_mode=forecast_mode,
            ))
            print(f'     {split_name:5s}  done  n={len(ds):5d}  '
                  f'mean_iou@{thr:.2f}={mean_iou:.4f}')

        # Free model before moving on
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        if MIRROR_TO_DRIVE:
            t0 = time.time()
            try:
                mirror_to_drive(ckpt_local, ckpt_drive)
                print(f'     mirrored -> {ckpt_drive}   ({time.time()-t0:.1f}s)')
            except Exception as e:
                print(f'     [warn] mirror to Drive failed (non-fatal): {e}')

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
you can eyeball the quality without opening Drive.
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
    mode_tag = 'forecast' if row.get('forecast_mode') else 'legacy'
    display(Markdown(f'**{run} / {ckpt} / {split}** ({mode_tag}) - sample 0  '
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
        n_v1_runs=len(v1_runs),
        skipped_runs=[{'run': n, 'reason': w} for n, w in skipped_runs],
        rows=agg_rows,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    ), f, indent=2)

if MIRROR_TO_DRIVE:
    try:
        shutil.copy2(csv_path,  INFER_ROOT_DRIVE / csv_path.name)
        shutil.copy2(json_path, INFER_ROOT_DRIVE / json_path.name)
        print('Summary mirrored to Drive:')
        print('  ', INFER_ROOT_DRIVE / csv_path.name)
        print('  ', INFER_ROOT_DRIVE / json_path.name)
    except Exception as e:
        print(f'[warn] summary mirror to Drive failed: {e}')

print('\n' + '=' * 86)
print('  FINAL REPORT  (V1)')
print('=' * 86)
print(f'  device            : {DEVICE}')
print(f'  V1 runs processed : {len(v1_runs)}')
print(f'  skipped upfront   : {len(skipped_runs)}')
for n, w in skipped_runs:
    print(f'      {n:12s} | {w}')
print()
print(f'  {"run":10s} {"ckpt":22s} {"split":6s} {"status":16s} '
      f'{"n":>6s} {"thr":>5s} {"mean_iou":>9s}  mode')
print('  ' + '-' * 90)
for r in agg_rows:
    mean = r.get('mean_iou_thr', '-')
    if isinstance(mean, float):
        mean = f'{mean:.4f}'
    thr = r.get('threshold', '-')
    if isinstance(thr, float):
        thr = f'{thr:.2f}'
    mode = 'forecast' if r.get('forecast_mode') else 'legacy' if r.get('forecast_mode') is False else '-'
    print(f"  {r.get('run',''):10s} {r.get('ckpt',''):22s} "
          f"{r.get('split',''):6s} {r.get('status',''):16s} "
          f"{str(r.get('n_samples','-')):>6s} {str(thr):>5s} {str(mean):>9s}  {mode}")
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
    r"data_tools/colab/colab_inference_af_v1_all_checkpoints.ipynb"
)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("WROTE", OUT)
print("num cells:", len(cells))

# Re-parse to validate JSON round-trip + cell counts
nb2 = json.load(open(OUT, encoding="utf-8"))
assert len(nb2["cells"]) == len(cells), "cell-count mismatch"
print("re-parsed OK   cells =", len(nb2["cells"]))
