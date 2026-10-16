"""
Create colab_af_training_version2.ipynb for run24.
Follows exact same structure as version1 (26 cells).
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training.ipynb'
DST = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training_version2.ipynb'

with open(SRC, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Original cells: {len(cells)}')


def code_cell(src_lines):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': src_lines,
    }


def md_cell(src_lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src_lines}


new_cells = []

# ── Cell 0: Updated title for version2 ──────────────────────────────────
new_cells.append(md_cell([
    '# FireST-GraphNet **version2**: Active Fire Training (Anti-Overfit Pipeline)\n',
    '\n',
    '**Run 23 result**: best val_iou = 0.598 at epoch 70  \n',
    '  `train_iou = 0.735` vs `val_iou = 0.598` — overfit gap = **0.137** (primary blocker)  \n',
    '  val_loss diverged from epoch 22 (min 0.269, rose to 0.314 at epoch 85)  \n',
    '  Train fire rate = 4.3%, val fire rate = 2.7% — distribution shift\n',
    '\n',
    '**Regularisation improvements in Run 24**:\n',
    '\n',
    '| # | Fix | Root Cause Addressed |\n',
    '|---|-----|---------------------|\n',
    '| 1 | **Label smoothing 0.05** in FocalBCELoss | Model overconfident on training fire pixels |\n',
    '| 2 | **Decoder dropout 0.30** on LSTM output h | No regularisation in decoder path |\n',
    '| 3 | **Weight decay 0.005->0.02** (4x) | Directly addresses train/val loss divergence |\n',
    '| 4 | **LSTM dropout 0.4->0.5** | Encoder inter-layer regularisation |\n',
    '| 5 | **SpatialFeedback disabled** | Removed 180K overfit-prone parameters |\n',
    '| 6 | **Gaussian noise std=0.02** on inputs | Prevents memorisation of exact satellite patterns |\n',
    '| 7 | **pos_weight 4.0->5.0** | Val fire rate 2.7% vs train 4.3% (dist. shift) |\n',
    '| 8 | **Tversky weight 0.25->0.30** | More false-negative emphasis |\n',
    '| 9 | **Patience 15->20** | val_iou has +/-0.02 epoch-to-epoch variance |\n',
    '|10 | **Epochs 120->100** | Best was at epoch 70; save Colab compute time |\n',
    '\n',
    '**Config**: `config_run24.yaml` | **Architecture**: GNN + LSTM with version1 structural improvements\n',
]))

# ── Cells 1-19: keep original (setup, drive mount, data, deps, config, etc.) ──
for i in range(1, 20):
    new_cells.append(cells[i])

# ── Cell 17 replacement: update config file to run24 ──────────────────────
# new_cells[-3] points to the cell at position 17 (0-indexed in new_cells = cells[17])
new_cells[-3] = code_cell([
    '# Set up project and load configuration\n',
    'os.chdir(\'/content\')\n',
    'sys.path.insert(0, PROJECT_PATH)\n',
    '\n',
    '# version2 uses config_run24.yaml with anti-overfit regularisation\n',
    'CONFIG_FILE = f\'{PROJECT_PATH}/config_run24.yaml\'\n',
    '\n',
    'if Path(CONFIG_FILE).exists():\n',
    '    print(f\'Config: {CONFIG_FILE}\')\n',
    '    !cat {CONFIG_FILE}\n',
    'else:\n',
    '    print(f\'config_run24.yaml not found, falling back to run23\')\n',
    '    for alt in [\'config_run23.yaml\', \'config_run22.yaml\',\n',
    '                \'firest_graphnet/configs/base.yaml\']:\n',
    '        alt_path = f\'{PROJECT_PATH}/{alt}\'\n',
    '        if Path(alt_path).exists():\n',
    '            CONFIG_FILE = alt_path\n',
    '            print(f\'Using: {CONFIG_FILE}\')\n',
    '            !cat {CONFIG_FILE}\n',
    '            break\n',
    '\n',
    'RUN_NAME = Path(CONFIG_FILE).stem.replace(\'config_\', \'\')\n',
    'if RUN_NAME == \'base\': RUN_NAME = \'run_base\'\n',
    'print(f\'Run name: {RUN_NAME}\')\n',
])

# ── Cell 20: Training cell for run24 ──────────────────────────────────────
new_cells.append(code_cell([
    'import glob\n',
    '\n',
    '# -------------------------------------------------------------------------\n',
    '# version2 NOTE: All fixes are baked into the backend code:\n',
    '#   - Label smoothing in FocalBCELoss (losses.py)\n',
    '#   - Decoder dropout on LSTM output h (decoder.py)\n',
    '#   - Gaussian noise augmentation on input features (transforms.py)\n',
    '#   - Higher weight_decay + LSTM dropout (config_run24.yaml)\n',
    '#   - SpatialFeedback disabled (config_run24.yaml)\n',
    '# No inline patches needed.\n',
    '# -------------------------------------------------------------------------\n',
    '\n',
    '# Fix any residual hardcoded Windows paths\n',
    'import subprocess\n',
    'subprocess.run(\n',
    '    [\'sed\', \'-i\',\n',
    '     r\'s|C:\\\\Users\\\\22125\\\\Desktop\\\\FireST-GraphNet|/content|g\',\n',
    '     f\'{PROJECT_PATH}/firest_graphnet/scripts/train.py\'],\n',
    '    capture_output=True\n',
    ')\n',
    '\n',
    '# Training task\n',
    'TASK = \'af\'          # \'af\' | \'ba\' | \'frp\'\n',
    'TASK_FLAG = f\'--task {TASK}\'\n',
    'print(f\'Training task: {TASK.upper()}\')\n',
    '\n',
    '# Checkpoint resumption — can warm-start from run23 best_model.pt\n',
    'LOCAL_OUTPUT = f\'/content/outputs/{RUN_NAME}\'\n',
    'DRIVE_OUTPUT_CKPT = f\'{PROJECT_PATH}/outputs/{RUN_NAME}\'\n',
    'os.makedirs(LOCAL_OUTPUT, exist_ok=True)\n',
    '\n',
    'RESUME_FLAG = \'\'\n',
    '\n',
    'def _latest_checkpoint(ckpt_dir):\n',
    '    best = Path(ckpt_dir) / \'best_model.pt\'\n',
    '    if best.exists():\n',
    '        return str(best)\n',
    '    candidates = sorted(glob.glob(str(Path(ckpt_dir) / \'checkpoint_epoch*.pt\')))\n',
    '    return candidates[-1] if candidates else None\n',
    '\n',
    'local_ckpt = _latest_checkpoint(LOCAL_OUTPUT)\n',
    'if local_ckpt is None:\n',
    '    drive_ckpt = _latest_checkpoint(DRIVE_OUTPUT_CKPT)\n',
    '    if drive_ckpt is not None:\n',
    '        dest = Path(LOCAL_OUTPUT) / Path(drive_ckpt).name\n',
    '        print(f\'Copying checkpoint from Drive: {Path(drive_ckpt).name}\')\n',
    '        shutil.copy2(drive_ckpt, dest)\n',
    '        drive_csv = Path(DRIVE_OUTPUT_CKPT) / \'metrics.csv\'\n',
    '        if drive_csv.exists():\n',
    '            shutil.copy2(str(drive_csv), str(Path(LOCAL_OUTPUT) / \'metrics.csv\'))\n',
    '        local_ckpt = str(dest)\n',
    '\n',
    'if local_ckpt:\n',
    '    RESUME_FLAG = f\'--resume {local_ckpt}\'\n',
    '    print(f\'Resuming from: {Path(local_ckpt).name}\')\n',
    'else:\n',
    '    print(\'No checkpoint found — training from scratch (version2 anti-overfit)\')\n',
    '\n',
    'LOG_FILE = f\'{LOCAL_OUTPUT}/train.log\'\n',
    'print(f\'\\nStarting version2 training... task={TASK.upper()}  log: {LOG_FILE}\\n\')\n',
    '\n',
    '!python {PROJECT_PATH}/firest_graphnet/scripts/train.py \\\n',
    '    --config {CONFIG_FILE} \\\n',
    '    --mode forecast \\\n',
    '    {TASK_FLAG} \\\n',
    '    {RESUME_FLAG} \\\n',
    '    2>&1 | tee {LOG_FILE}\n',
    '\n',
    'print(\'\\nTraining complete!\')\n',
]))

# ── Cells 21-23: keep original (save results) ─────────────────────────────
new_cells.append(cells[21])
new_cells.append(cells[22])
new_cells.append(cells[23])

# ── Cell 24: Updated visualisation for run24 ─────────────────────────────
new_cells.append(code_cell([
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'from pathlib import Path\n',
    '\n',
    'metrics_file = Path(LOCAL_OUTPUT) / \'metrics.csv\'\n',
    '\n',
    'if not metrics_file.exists():\n',
    '    print(f\'Metrics file not found: {metrics_file}\')\n',
    'else:\n',
    '    df = pd.read_csv(metrics_file)\n',
    '    print(f\'Loaded {len(df)} epochs of metrics\')\n',
    '\n',
    '    fig, axes = plt.subplots(2, 3, figsize=(16, 9))\n',
    '    fig.suptitle(f\'FireST-GraphNet version2 (run24) anti-overfit — {RUN_NAME}\', fontsize=13)\n',
    '\n',
    '    # Plot 1: Total + supervised loss\n',
    '    for col, lbl, ls in [\n',
    '        (\'train_total\', \'Train total\', \'-\'),\n',
    '        (\'train_sup\', \'Train sup\', \'--\'),\n',
    '        (\'val_total\', \'Val total\', \'-\'),\n',
    '    ]:\n',
    '        if col in df.columns:\n',
    '            axes[0,0].plot(df[\'epoch\'], df[col], label=lbl, linestyle=ls)\n',
    '    axes[0,0].set_title(\'Loss (total + supervised)\')\n',
    '    axes[0,0].legend(fontsize=8); axes[0,0].grid(True)\n',
    '\n',
    '    # Plot 2: IoU with overfit gap\n',
    '    if \'val_iou\' in df.columns:\n',
    '        axes[0,1].plot(df[\'epoch\'], df[\'val_iou\'],   label=\'Val IoU\', color=\'tomato\')\n',
    '        if \'train_iou\' in df.columns:\n',
    '            axes[0,1].plot(df[\'epoch\'], df[\'train_iou\'], label=\'Train IoU\', ls=\'--\', color=\'steelblue\')\n',
    '        best_iou = df[\'val_iou\'].max()\n',
    '        best_ep  = df.loc[df[\'val_iou\'].idxmax(), \'epoch\']\n',
    '        axes[0,1].axhline(best_iou, color=\'red\', ls=\':\', alpha=0.6,\n',
    '                          label=f\'Best {best_iou:.4f} @ep{best_ep}\')\n',
    '        axes[0,1].axhline(0.598, color=\'grey\', ls=\':\', alpha=0.4, label=\'run23 best 0.598\')\n',
    '    axes[0,1].set_title(\'IoU\')\n',
    '    axes[0,1].legend(fontsize=8); axes[0,1].grid(True)\n',
    '\n',
    '    # Plot 3: Overfit gap\n',
    '    if \'train_iou\' in df.columns and \'val_iou\' in df.columns:\n',
    '        axes[0,2].plot(df[\'epoch\'], df[\'train_iou\'] - df[\'val_iou\'], color=\'darkorange\')\n',
    '        axes[0,2].axhline(0, ls=\'--\', color=\'grey\')\n',
    '        axes[0,2].axhline(0.137, ls=\':\', color=\'grey\', alpha=0.5, label=\'run23 gap=0.137\')\n',
    '    axes[0,2].set_title(\'Overfit Gap (train_iou - val_iou)\')\n',
    '    axes[0,2].legend(fontsize=8); axes[0,2].grid(True)\n',
    '\n',
    '    # Plot 4: Tversky + Dice\n',
    '    for col, lbl in [(\'train_tversky\',\'Tversky\'), (\'train_dice\',\'Dice\'),\n',
    '                     (\'val_tversky\',\'val Tversky\'), (\'val_dice\',\'val Dice\')]:\n',
    '        if col in df.columns:\n',
    '            axes[1,0].plot(df[\'epoch\'], df[col], label=lbl)\n',
    '    axes[1,0].set_title(\'Tversky / Dice Loss\')\n',
    '    axes[1,0].legend(fontsize=8); axes[1,0].grid(True)\n',
    '\n',
    '    # Plot 5: F1 + Precision + Recall\n',
    '    if \'val_f1\' in df.columns:\n',
    '        axes[1,1].plot(df[\'epoch\'], df[\'val_f1\'],        label=\'Val F1\')\n',
    '        axes[1,1].plot(df[\'epoch\'], df[\'val_precision\'],  label=\'Val Prec\', ls=\'--\')\n',
    '        axes[1,1].plot(df[\'epoch\'], df[\'val_recall\'],     label=\'Val Rec\', ls=\'-.\')\n',
    '        best_f1 = df[\'val_f1\'].max()\n',
    '        axes[1,1].axhline(0.734, color=\'grey\', ls=\':\', alpha=0.4, label=\'run23 best f1=0.734\')\n',
    '    axes[1,1].set_title(\'F1 / Precision / Recall\')\n',
    '    axes[1,1].legend(fontsize=8); axes[1,1].grid(True)\n',
    '\n',
    '    # Plot 6: Pred vs GT positive ratio\n',
    '    if \'val_pred_pos_ratio\' in df.columns:\n',
    '        axes[1,2].plot(df[\'epoch\'], df[\'val_pred_pos_ratio\'],  label=\'val pred_pos\')\n',
    '        axes[1,2].plot(df[\'epoch\'], df[\'val_gt_pos_ratio\'],    label=\'val gt_pos\', ls=\'--\')\n',
    '        if \'train_pred_pos_ratio\' in df.columns:\n',
    '            axes[1,2].plot(df[\'epoch\'], df[\'train_pred_pos_ratio\'], label=\'train pred_pos\', ls=\':.\')\n',
    '    axes[1,2].set_title(\'Fire Pixel Ratio\')\n',
    '    axes[1,2].legend(fontsize=8); axes[1,2].grid(True)\n',
    '\n',
    '    plt.tight_layout()\n',
    '    out_png = f\'{LOCAL_OUTPUT}/training_curves_v2.png\'\n',
    '    plt.savefig(out_png, dpi=150)\n',
    '    plt.show()\n',
    '    print(f\'Saved: {out_png}\')\n',
    '\n',
    '    print(\'\\n=== Best Validation Metrics (version2 vs run23) ===\')\n',
    '    if \'val_iou\' in df.columns:\n',
    '        best = df.loc[df[\'val_iou\'].idxmax()]\n',
    '        delta = best[\'val_iou\'] - 0.598\n',
    '        sign  = \'+\' if delta >= 0 else \'\'\n',
    '        print(f"  Best IoU:    {best[\'val_iou\']:.4f}  @epoch {int(best[\'epoch\'])}  "\n',
    '              f"({sign}{delta:.4f} vs run23)")\n',
    '    if \'val_f1\' in df.columns:\n',
    '        best_f1 = df.loc[df[\'val_f1\'].idxmax()]\n',
    '        delta_f1 = best_f1[\'val_f1\'] - 0.734\n',
    '        sign = \'+\' if delta_f1 >= 0 else \'\'\n',
    '        print(f"  Best F1:     {best_f1[\'val_f1\']:.4f}  @epoch {int(best_f1[\'epoch\'])}  "\n',
    '              f"({sign}{delta_f1:.4f} vs run23)")\n',
    '    if \'train_iou\' in df.columns and \'val_iou\' in df.columns:\n',
    '        best_row = df.loc[df[\'val_iou\'].idxmax()]\n',
    '        print(f"  Overfit gap: {best_row[\'train_iou\']-best_row[\'val_iou\']:.4f}  (run23 was 0.137)")\n',
]))

# ── Cell 25: Completion markdown ─────────────────────────────────────────
new_cells.append(md_cell([
    '## Complete\n',
    '\n',
    '### Summary of version2 Changes (over version1/run23)\n',
    '\n',
    '| File | Change | Reason |\n',
    '|------|--------|--------|\n',
    '| `training/losses.py` | `label_smoothing` param in `FocalBCELoss` | Prevents overconfident fire predictions |\n',
    '| `models/decoder.py` | `decoder_dropout` on LSTM output h_dec | Regularises spatial head without corrupting LSTM state |\n',
    '| `data/transforms.py` | `noise_std` in `ForecastAugmentation` | Gaussian noise on inputs (af_hist/aux_hist/env_fut/static) |\n',
    '| `models/firest_graphnet.py` | Forward `decoder_dropout` to ForecastDecoder | Wire new param |\n',
    '| `training/trainer.py` | Pass `label_smoothing` to ForecastPINNLoss | Wire new param |\n',
    '| `scripts/train.py` | Pass `decoder_dropout`, `noise_std` from config | Wire new params |\n',
    '| `config_run24.yaml` | All version2 settings | weight_decay 0.02, lstm_dropout 0.5, decoder_dropout 0.3 |\n',
    '\n',
    '### To Reproduce\n',
    'Run this notebook on Colab A100 with `config_run24.yaml` in your Drive.\n',
    'Training takes approximately 2-3 hours for 100 epochs.\n',
]))

nb['cells'] = new_cells
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Written: {DST}')
print(f'Total cells: {len(new_cells)}')
