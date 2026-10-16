"""
Create colab_af_training_version3.ipynb for run25 (EMA + surgical rollback).
Follows exact same structure as version2 (26 cells).
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training.ipynb'
DST = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training_version3.ipynb'

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

# ── Cell 0: Updated title for version3 ──────────────────────────────────
new_cells.append(md_cell([
    '# FireST-GraphNet **version3**: Active Fire Training (EMA + Surgical Regularisation)\n',
    '\n',
    '**Run 23 (baseline)**: val_iou = **0.598** @ ep70 | overfit gap = 0.137 (overfit)  \n',
    '**Run 24 (v2 regularisation)**: val_iou = **0.579** @ ep52 | overfit gap = 0.077 (over-regularised)  \n',
    '\n',
    '### Run 24 diagnosis\n',
    '- train_iou only reached **0.682** (run23 hit 0.735) -> model UNDERFITS\n',
    '- val_iou std over ep30-50 = **0.045** -> very volatile checkpoint selection\n',
    '- val_pred_pos = 0.034 vs gt = 0.027 at thr=0.33 -> over-predicting\n',
    '- train_sup = 0.055 vs val_sup = 0.115 -> distribution-shift loss still doubles\n',
    '\n',
    '### Run 25 strategy: EMA + surgical rollback\n',
    '\n',
    '| # | Change | Value | Rationale |\n',
    '|---|--------|-------|-----------|\n',
    '| 1 | **EMA of weights (NEW)** | decay=0.999, start=ep10 | Smooths val-IoU variance, +0.5-2% typical gain |\n',
    '| 2 | SpatialFeedback | false -> **true** | Restore inductive bias removed in run24 |\n',
    '| 3 | decoder_dropout | 0.30 -> **0.15** | Halve (still regularises LSTM output) |\n',
    '| 4 | weight_decay | 0.02 -> **0.01** | Halve (still 2x run23) |\n',
    '| 5 | lstm_dropout | 0.5 -> **0.4** | Back to run23 level (encoder capacity) |\n',
    '| 6 | augmentation_noise_std | 0.02 -> **0.01** | Halve input noise |\n',
    '| 7 | val_threshold | 0.33 -> **0.38** | Fix over-prediction (pred 0.034 > gt 0.027) |\n',
    '| 8 | epochs | 100 -> **120** | Give EMA longer to converge |\n',
    '| 9 | patience | 20 -> **25** | EMA further reduces noise |\n',
    '|   | **KEEP** | | label_smoothing=0.05, pos_weight=5.0, tversky_weight=0.30 |\n',
    '\n',
    '**Config**: `config_run25.yaml` | **Architecture**: GNN + LSTM (+ SpatialFeedback restored) | **New**: EMA in `ForecastTrainer`\n',
]))

# ── Cells 1-19: keep original (setup, drive mount, data, deps, config, etc.) ──
for i in range(1, 20):
    new_cells.append(cells[i])

# ── Cell 17 replacement: update config file to run25 ──────────────────────
new_cells[-3] = code_cell([
    '# Set up project and load configuration\n',
    'os.chdir(\'/content\')\n',
    'sys.path.insert(0, PROJECT_PATH)\n',
    '\n',
    '# version3 uses config_run25.yaml with EMA + surgical regularisation rollback\n',
    'CONFIG_FILE = f\'{PROJECT_PATH}/config_run25.yaml\'\n',
    '\n',
    'if Path(CONFIG_FILE).exists():\n',
    '    print(f\'Config: {CONFIG_FILE}\')\n',
    '    !cat {CONFIG_FILE}\n',
    'else:\n',
    '    print(f\'config_run25.yaml not found, falling back\')\n',
    '    for alt in [\'config_run24.yaml\', \'config_run23.yaml\',\n',
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

# ── Cell 20: Training cell for run25 ──────────────────────────────────────
new_cells.append(code_cell([
    'import glob\n',
    '\n',
    '# -------------------------------------------------------------------------\n',
    '# version3 NOTE: All changes baked into backend code + config:\n',
    '#   * EMA of weights (NEW) -> trainer.py  (ForecastTrainer._ema_*)\n',
    '#   * SpatialFeedback re-enabled -> config_run25.yaml\n',
    '#   * Halved decoder_dropout / weight_decay / noise_std -> config_run25.yaml\n',
    '#   * val_threshold 0.38 -> config_run25.yaml\n',
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
    '# Checkpoint resumption — can warm-start from run24 best_model.pt\n',
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
    '    print(\'No checkpoint found — training version3 (EMA) from scratch\')\n',
    '\n',
    'LOG_FILE = f\'{LOCAL_OUTPUT}/train.log\'\n',
    'print(f\'\\nStarting version3 training... task={TASK.upper()}  log: {LOG_FILE}\\n\')\n',
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

# ── Cell 24: Updated visualisation for run25 ─────────────────────────────
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
    '    fig.suptitle(f\'FireST-GraphNet version3 (run25) EMA+rollback — {RUN_NAME}\', fontsize=13)\n',
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
    '    # Plot 2: IoU with run23/24 reference lines\n',
    '    if \'val_iou\' in df.columns:\n',
    '        axes[0,1].plot(df[\'epoch\'], df[\'val_iou\'],   label=\'Val IoU (EMA)\', color=\'tomato\')\n',
    '        if \'train_iou\' in df.columns:\n',
    '            axes[0,1].plot(df[\'epoch\'], df[\'train_iou\'], label=\'Train IoU\', ls=\'--\', color=\'steelblue\')\n',
    '        best_iou = df[\'val_iou\'].max()\n',
    '        best_ep  = df.loc[df[\'val_iou\'].idxmax(), \'epoch\']\n',
    '        axes[0,1].axhline(best_iou, color=\'red\', ls=\':\', alpha=0.6,\n',
    '                          label=f\'Best {best_iou:.4f} @ep{best_ep}\')\n',
    '        axes[0,1].axhline(0.598, color=\'grey\',  ls=\':\', alpha=0.4, label=\'run23 0.598\')\n',
    '        axes[0,1].axhline(0.579, color=\'brown\', ls=\':\', alpha=0.4, label=\'run24 0.579\')\n',
    '    axes[0,1].set_title(\'IoU (val uses EMA weights)\')\n',
    '    axes[0,1].legend(fontsize=8); axes[0,1].grid(True)\n',
    '\n',
    '    # Plot 3: Overfit gap\n',
    '    if \'train_iou\' in df.columns and \'val_iou\' in df.columns:\n',
    '        axes[0,2].plot(df[\'epoch\'], df[\'train_iou\'] - df[\'val_iou\'], color=\'darkorange\')\n',
    '        axes[0,2].axhline(0, ls=\'--\', color=\'grey\')\n',
    '        axes[0,2].axhline(0.137, ls=\':\', color=\'grey\',  alpha=0.5, label=\'run23 gap=0.137\')\n',
    '        axes[0,2].axhline(0.077, ls=\':\', color=\'brown\', alpha=0.5, label=\'run24 gap=0.077\')\n',
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
    '    axes[1,1].set_title(\'F1 / Precision / Recall\')\n',
    '    axes[1,1].legend(fontsize=8); axes[1,1].grid(True)\n',
    '\n',
    '    # Plot 6: Pred vs GT positive ratio\n',
    '    if \'val_pred_pos_ratio\' in df.columns:\n',
    '        axes[1,2].plot(df[\'epoch\'], df[\'val_pred_pos_ratio\'],  label=\'val pred_pos\')\n',
    '        axes[1,2].plot(df[\'epoch\'], df[\'val_gt_pos_ratio\'],    label=\'val gt_pos\', ls=\'--\')\n',
    '        if \'train_pred_pos_ratio\' in df.columns:\n',
    '            axes[1,2].plot(df[\'epoch\'], df[\'train_pred_pos_ratio\'], label=\'train pred_pos\', ls=\':\')\n',
    '    axes[1,2].set_title(\'Fire Pixel Ratio (val threshold=0.38)\')\n',
    '    axes[1,2].legend(fontsize=8); axes[1,2].grid(True)\n',
    '\n',
    '    plt.tight_layout()\n',
    '    out_png = f\'{LOCAL_OUTPUT}/training_curves_v3.png\'\n',
    '    plt.savefig(out_png, dpi=150)\n',
    '    plt.show()\n',
    '    print(f\'Saved: {out_png}\')\n',
    '\n',
    '    print(\'\\n=== Best Validation Metrics (version3 vs run23 / run24) ===\')\n',
    '    if \'val_iou\' in df.columns:\n',
    '        best = df.loc[df[\'val_iou\'].idxmax()]\n',
    '        d23 = best[\'val_iou\'] - 0.598\n',
    '        d24 = best[\'val_iou\'] - 0.579\n',
    '        print(f"  Best IoU:    {best[\'val_iou\']:.4f} @ep {int(best[\'epoch\'])}  "\n',
    '              f"({d23:+.4f} vs run23, {d24:+.4f} vs run24)")\n',
    '    if \'val_f1\' in df.columns:\n',
    '        bf = df.loc[df[\'val_f1\'].idxmax()]\n',
    '        print(f"  Best F1:     {bf[\'val_f1\']:.4f} @ep {int(bf[\'epoch\'])}")\n',
    '    if \'train_iou\' in df.columns and \'val_iou\' in df.columns:\n',
    '        br = df.loc[df[\'val_iou\'].idxmax()]\n',
    '        gap = br[\'train_iou\'] - br[\'val_iou\']\n',
    '        print(f"  Overfit gap: {gap:.4f}  (run23=0.137, run24=0.077)")\n',
    '    # val-IoU variance (last 20 epochs): expect EMA to reduce vs run24\n',
    '    if \'val_iou\' in df.columns and len(df) >= 20:\n',
    '        tail = df[\'val_iou\'].tail(20)\n',
    '        print(f"  Val IoU std (last 20 ep): {tail.std():.4f}  "\n',
    '              f"(run24 was ~0.030; EMA target <0.015)")\n',
]))

# ── Cell 25: Completion markdown ─────────────────────────────────────────
new_cells.append(md_cell([
    '## Complete\n',
    '\n',
    '### Summary of version3 Changes (over version2/run24)\n',
    '\n',
    '| File | Change | Reason |\n',
    '|------|--------|--------|\n',
    '| `training/trainer.py` | **EMA shadow + swap in/out at validation** | Smooths val variance, +generalisation |\n',
    '| `config_run25.yaml` | `ema_decay=0.999`, `ema_start_epoch=10` | Activate EMA after LR warmup |\n',
    '| `config_run25.yaml` | `use_spatial_feedback: true` | Restore inductive bias from run23 |\n',
    '| `config_run25.yaml` | `decoder_dropout 0.30 -> 0.15` | Halve over-regularisation |\n',
    '| `config_run25.yaml` | `weight_decay 0.02 -> 0.01` | Halve (still 2x run23) |\n',
    '| `config_run25.yaml` | `lstm_dropout 0.5 -> 0.4` | Back to run23 encoder capacity |\n',
    '| `config_run25.yaml` | `augmentation_noise_std 0.02 -> 0.01` | Halve input noise |\n',
    '| `config_run25.yaml` | `val_threshold 0.33 -> 0.38` | Fix over-prediction |\n',
    '| `config_run25.yaml` | `epochs 100 -> 120`, `patience 20 -> 25` | EMA needs more steps |\n',
    '\n',
    '### How EMA works here\n',
    '- After every optimiser step, `_ema_update()` does `shadow = 0.999 * shadow + 0.001 * current`.\n',
    '- At validation start we swap live params <-> EMA shadow, validate, then swap back.\n',
    '- Best-model checkpoint stores EMA weights (so downstream inference uses them).\n',
    '- Warmup (ep 1-10): EMA disabled — LR warmup updates are large/noisy, would pollute shadow.\n',
    '\n',
    '### Expected impact\n',
    '- val_iou epoch-to-epoch std: **~0.030 -> <0.015** (EMA)\n',
    '- val_iou peak: **0.579 -> ~0.61-0.62** (rollback + EMA; reclaim lost capacity)\n',
    '- overfit gap: **0.077 -> 0.09-0.11** (accept small increase for big val gain)\n',
    '\n',
    '### To Reproduce\n',
    'Run this notebook on Colab A100 with `config_run25.yaml` in your Drive.\n',
    'Training takes approximately 2.5-3.5 hours for 120 epochs.\n',
]))

nb['cells'] = new_cells
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Written: {DST}')
print(f'Total cells: {len(new_cells)}')
