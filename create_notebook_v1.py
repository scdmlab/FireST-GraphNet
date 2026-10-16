"""
Create colab_af_training_version1.ipynb
"""
import json, copy, sys, os

with open('C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def code_cell(src_lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src_lines}

def md_cell(src_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": src_lines}

new_cells = []

# Cell 0: Updated title
new_cells.append(md_cell([
    "# FireST-GraphNet **version1**: Active Fire Training (Improved Pipeline)\n",
    "\n",
    "**Structural improvements over run22** (not just hyperparameter tuning):\n",
    "\n",
    "| # | Fix | Root Cause Addressed |\n",
    "|---|-----|---------------------|\n",
    "| 1 | **Data augmentation** applied to all forecast tensors | Was MISSING - transforms only handled legacy x_seq/y keys |\n",
    "| 2 | **InstanceNorm2d** at encoder input | 28ch span wildly different ranges (DEM:0-4000m, T:250-320K) |\n",
    "| 3 | **Decoder h initialised** from encoder temporal summary | Was zeros - decoder had no historical context at step 0 |\n",
    "| 4 | **Auto-regressive h** at each horizon step | Fixed temporal_cond repeated for all steps = weak predictions |\n",
    "| 5 | **SpatialFeedback** module in decoder | Same spatial_feat used for all horizon steps |\n",
    "| 6 | **Tversky loss** (alpha=0.3, beta=0.7) | Penalises missed fires more than false alarms |\n",
    "| 7 | **Per-horizon Dice** (not globally flat) | Global Dice dilutes fire pixels across all B*H*1*H*W |\n",
    "| 8 | **AF-BA mask disabled** (xi_max=0) | Was suppressing fire predictions near burned boundaries |\n",
    "| 9 | **LR warmup** 5-epoch linear ramp | AdamW needs warmup to stabilise parameter estimates |\n",
    "| 10 | **Val threshold 0.35** (not 0.5) | Optimal threshold for 3-4% fire pixel rate |\n",
    "\n",
    "**Config**: `config_run23.yaml` | **Architecture**: GNN + LSTM (same paradigm, improved internals)\n"
]))

# Cells 1-19: keep original (env setup, drive mount, data scan, data gen, copy, verify, deps, config, path update, torch-cluster)
for i in range(1, 20):
    new_cells.append(cells[i])

# Cell 17 replacement: update config file to run23
new_cells[-3] = code_cell([
    "# Set up project and load configuration\n",
    "os.chdir('/content')\n",
    "sys.path.insert(0, PROJECT_PATH)\n",
    "\n",
    "# version1 uses config_run23.yaml with structural improvements\n",
    "CONFIG_FILE = f'{PROJECT_PATH}/config_run23.yaml'\n",
    "\n",
    "if Path(CONFIG_FILE).exists():\n",
    "    print(f'Config: {CONFIG_FILE}')\n",
    "    !cat {CONFIG_FILE}\n",
    "else:\n",
    "    print(f'config_run23.yaml not found, falling back to run22')\n",
    "    for alt in ['config_run22.yaml', 'config_run12.yaml',\n",
    "                'firest_graphnet/configs/base.yaml']:\n",
    "        alt_path = f'{PROJECT_PATH}/{alt}'\n",
    "        if Path(alt_path).exists():\n",
    "            CONFIG_FILE = alt_path\n",
    "            print(f'Using: {CONFIG_FILE}')\n",
    "            !cat {CONFIG_FILE}\n",
    "            break\n",
    "\n",
    "RUN_NAME = Path(CONFIG_FILE).stem.replace('config_', '')\n",
    "if RUN_NAME == 'base': RUN_NAME = 'run_base'\n",
    "print(f'Run name: {RUN_NAME}')\n"
])

# Cell 20: Improved training cell (remove now-unnecessary inline patches)
new_cells.append(code_cell([
    "import glob\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────\n",
    "# version1 NOTE: The following fixes are now BAKED INTO the backend code:\n",
    "#   - BatchNorm train mode during validation (trainer.py line ~370)\n",
    "#   - pos_weight config-override fix (train.py)\n",
    "#   - ForecastAugmentation applied to training (train.py)\n",
    "#   - InstanceNorm2d at encoder input (encoder.py)\n",
    "#   - Decoder h init + spatial feedback (decoder.py)\n",
    "#   - TverskyLoss + per-horizon Dice (losses.py)\n",
    "# No inline patches needed here.\n",
    "# ─────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "# Fix any residual hardcoded Windows paths\n",
    "import subprocess\n",
    "subprocess.run(\n",
    "    ['sed', '-i',\n",
    "     r's|C:\\\\Users\\\\22125\\\\Desktop\\\\FireST-GraphNet|/content|g',\n",
    "     f'{PROJECT_PATH}/firest_graphnet/scripts/train.py'],\n",
    "    capture_output=True\n",
    ")\n",
    "\n",
    "# Training task\n",
    "TASK = 'af'          # 'af' | 'ba' | 'frp'\n",
    "TASK_FLAG = f'--task {TASK}'\n",
    "print(f'Training task: {TASK.upper()}')\n",
    "\n",
    "# Checkpoint resumption\n",
    "LOCAL_OUTPUT = f'/content/outputs/{RUN_NAME}'\n",
    "DRIVE_OUTPUT_CKPT = f'{PROJECT_PATH}/outputs/{RUN_NAME}'\n",
    "os.makedirs(LOCAL_OUTPUT, exist_ok=True)\n",
    "\n",
    "RESUME_FLAG = ''\n",
    "\n",
    "def _latest_checkpoint(ckpt_dir):\n",
    "    best = Path(ckpt_dir) / 'best_model.pt'\n",
    "    if best.exists():\n",
    "        return str(best)\n",
    "    candidates = sorted(glob.glob(str(Path(ckpt_dir) / 'checkpoint_epoch*.pt')))\n",
    "    return candidates[-1] if candidates else None\n",
    "\n",
    "local_ckpt = _latest_checkpoint(LOCAL_OUTPUT)\n",
    "if local_ckpt is None:\n",
    "    drive_ckpt = _latest_checkpoint(DRIVE_OUTPUT_CKPT)\n",
    "    if drive_ckpt is not None:\n",
    "        dest = Path(LOCAL_OUTPUT) / Path(drive_ckpt).name\n",
    "        print(f'Copying checkpoint from Drive: {Path(drive_ckpt).name}')\n",
    "        shutil.copy2(drive_ckpt, dest)\n",
    "        drive_csv = Path(DRIVE_OUTPUT_CKPT) / 'metrics.csv'\n",
    "        if drive_csv.exists():\n",
    "            shutil.copy2(str(drive_csv), str(Path(LOCAL_OUTPUT) / 'metrics.csv'))\n",
    "        local_ckpt = str(dest)\n",
    "\n",
    "if local_ckpt:\n",
    "    RESUME_FLAG = f'--resume {local_ckpt}'\n",
    "    print(f'Resuming from: {Path(local_ckpt).name}')\n",
    "else:\n",
    "    print('No checkpoint found — training from scratch (version1 new architecture)')\n",
    "\n",
    "LOG_FILE = f'{LOCAL_OUTPUT}/train.log'\n",
    "print(f'\\nStarting version1 training... task={TASK.upper()}  log: {LOG_FILE}\\n')\n",
    "\n",
    "!python {PROJECT_PATH}/firest_graphnet/scripts/train.py \\\n",
    "    --config {temp_config} \\\n",
    "    --mode forecast \\\n",
    "    {TASK_FLAG} \\\n",
    "    {RESUME_FLAG} \\\n",
    "    2>&1 | tee {LOG_FILE}\n",
    "\n",
    "print('\\nTraining complete!')\n"
]))

# Save results cell header
new_cells.append(cells[21])

# Save results code
new_cells.append(cells[22])

# Visualization header
new_cells.append(cells[23])

# Updated visualization cell
new_cells.append(code_cell([
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "\n",
    "metrics_file = Path(LOCAL_OUTPUT) / 'metrics.csv'\n",
    "\n",
    "if not metrics_file.exists():\n",
    "    print(f'Metrics file not found: {metrics_file}')\n",
    "else:\n",
    "    df = pd.read_csv(metrics_file)\n",
    "    print(f'Loaded {len(df)} epochs of metrics')\n",
    "\n",
    "    fig, axes = plt.subplots(2, 3, figsize=(16, 9))\n",
    "    fig.suptitle(f'FireST-GraphNet version1 — {RUN_NAME}', fontsize=13)\n",
    "\n",
    "    # Plot 1: Total + supervised loss\n",
    "    for col, lbl, ls in [\n",
    "        ('train_total', 'Train total', '-'),\n",
    "        ('train_sup', 'Train sup', '--'),\n",
    "        ('val_total', 'Val total', '-'),\n",
    "    ]:\n",
    "        if col in df.columns:\n",
    "            axes[0,0].plot(df['epoch'], df[col], label=lbl, linestyle=ls)\n",
    "    axes[0,0].set_title('Loss (total + supervised)')\n",
    "    axes[0,0].legend(fontsize=8); axes[0,0].grid(True)\n",
    "\n",
    "    # Plot 2: Auxiliary losses (Dice, Tversky, Boundary, Temporal)\n",
    "    for col, lbl in [\n",
    "        ('train_dice','Dice'), ('train_tversky','Tversky'),\n",
    "        ('train_boundary','Boundary'), ('train_temporal','Temporal'),\n",
    "    ]:\n",
    "        if col in df.columns:\n",
    "            axes[0,1].plot(df['epoch'], df[col], label=lbl)\n",
    "    axes[0,1].set_title('Auxiliary losses (version1)')\n",
    "    axes[0,1].legend(fontsize=8); axes[0,1].grid(True)\n",
    "\n",
    "    # Plot 3: PINN losses\n",
    "    for col, lbl in [('train_pde','PDE'), ('train_mono','Mono'), ('train_mask','Mask')]:\n",
    "        if col in df.columns:\n",
    "            axes[0,2].plot(df['epoch'], df[col], label=lbl)\n",
    "    axes[0,2].set_title('PINN losses')\n",
    "    axes[0,2].legend(fontsize=8); axes[0,2].grid(True)\n",
    "\n",
    "    # Plot 4: IoU\n",
    "    if 'val_iou' in df.columns:\n",
    "        axes[1,0].plot(df['epoch'], df['val_iou'], label='Val IoU', color='royalblue')\n",
    "        if 'train_iou' in df.columns:\n",
    "            axes[1,0].plot(df['epoch'], df['train_iou'], label='Train IoU', ls='--')\n",
    "        best_iou = df['val_iou'].max()\n",
    "        best_ep  = df.loc[df['val_iou'].idxmax(), 'epoch']\n",
    "        axes[1,0].axhline(best_iou, color='red', ls=':', alpha=0.6,\n",
    "                          label=f'Best {best_iou:.4f} @ep{best_ep}')\n",
    "        axes[1,0].axhline(0.617, color='grey', ls=':', alpha=0.4, label='run22 baseline ~0.617')\n",
    "    axes[1,0].set_title('IoU')\n",
    "    axes[1,0].legend(fontsize=8); axes[1,0].grid(True)\n",
    "\n",
    "    # Plot 5: F1\n",
    "    if 'val_f1' in df.columns:\n",
    "        axes[1,1].plot(df['epoch'], df['val_f1'], label='Val F1', color='seagreen')\n",
    "        if 'train_f1' in df.columns:\n",
    "            axes[1,1].plot(df['epoch'], df['train_f1'], label='Train F1', ls='--')\n",
    "        axes[1,1].axhline(0.760, color='grey', ls=':', alpha=0.4, label='run22 baseline ~0.760')\n",
    "    axes[1,1].set_title('F1 Score')\n",
    "    axes[1,1].legend(fontsize=8); axes[1,1].grid(True)\n",
    "\n",
    "    # Plot 6: Precision & Recall\n",
    "    if 'val_precision' in df.columns:\n",
    "        axes[1,2].plot(df['epoch'], df['val_precision'], label='Precision')\n",
    "    if 'val_recall' in df.columns:\n",
    "        axes[1,2].plot(df['epoch'], df['val_recall'], label='Recall')\n",
    "    axes[1,2].set_title('Precision & Recall')\n",
    "    axes[1,2].legend(fontsize=8); axes[1,2].grid(True)\n",
    "\n",
    "    plt.tight_layout()\n",
    "    out_png = f'{LOCAL_OUTPUT}/training_curves_v1.png'\n",
    "    plt.savefig(out_png, dpi=150)\n",
    "    plt.show()\n",
    "    print(f'Saved: {out_png}')\n",
    "\n",
    "    print('\\n=== Best Validation Metrics (version1 vs run22 baseline) ===')\n",
    "    if 'val_iou' in df.columns:\n",
    "        best = df.loc[df['val_iou'].idxmax()]\n",
    "        delta = best['val_iou'] - 0.617\n",
    "        sign  = '+' if delta >= 0 else ''\n",
    "        print(f\"  Best IoU:    {best['val_iou']:.4f}  @epoch {int(best['epoch'])}  \"\n",
    "              f\"({sign}{delta:.4f} vs run22)\")\n",
    "    if 'val_f1' in df.columns:\n",
    "        best_f1 = df.loc[df['val_f1'].idxmax()]\n",
    "        delta_f1 = best_f1['val_f1'] - 0.760\n",
    "        sign = '+' if delta_f1 >= 0 else ''\n",
    "        print(f\"  Best F1:     {best_f1['val_f1']:.4f}  @epoch {int(best_f1['epoch'])}  \"\n",
    "              f\"({sign}{delta_f1:.4f} vs run22)\")\n",
    "    if 'val_recall' in df.columns:\n",
    "        best_r = df.loc[df['val_recall'].idxmax()]\n",
    "        print(f\"  Best Recall: {best_r['val_recall']:.4f}  @epoch {int(best_r['epoch'])}\")\n"
]))

# Completion markdown
new_cells.append(md_cell([
    "## Complete\n",
    "\n",
    "### Summary of version1 Changes\n",
    "\n",
    "| File | Change | Reason |\n",
    "|------|--------|--------|\n",
    "| `data/transforms.py` | Added `ForecastAugmentation` class | hflip/vflip/rot90 on all spatial forecast tensors |\n",
    "| `models/encoder.py` | Added `InstanceNorm2d(affine=True)` | Normalise mixed-scale 28ch input |\n",
    "| `models/decoder.py` | `h` init from encoder + auto-regressive + `SpatialFeedback` | True temporal propagation + spatial evolution |\n",
    "| `training/losses.py` | Added `TverskyLoss`; per-horizon Dice | Better recall; avoid fire pixel dilution |\n",
    "| `training/trainer.py` | LR warmup; val threshold 0.35; Tversky tracking | Stable training; appropriate threshold |\n",
    "| `scripts/train.py` | Pass `ForecastAugmentation` to training dataset | Actually apply augmentation |\n",
    "| `config_run23.yaml` | New config with all version1 settings | Clean config for reproducibility |\n",
    "\n",
    "### To Reproduce\n",
    "Run this notebook on Colab A100 with `config_run23.yaml` in your Drive.\n",
    "Training takes approximately 3-4 hours for 120 epochs.\n"
]))

nb['cells'] = new_cells

out_path = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training_version1.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {out_path}')
print(f'Total cells: {len(new_cells)}')
