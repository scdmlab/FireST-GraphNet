"""
Create colab_af_training_version4.ipynb for run26.
Adds: Lovász hinge loss, adaptive val threshold scan, precision rebalance.
Structure identical to v3 (26 cells).
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training.ipynb'
DST = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_af_training_version4.ipynb'

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

# ── Cell 0: Title for version4 ─────────────────────────────────────────
new_cells.append(md_cell([
    '# FireST-GraphNet **version4**: Active Fire (Lovász + Adaptive Threshold + Precision Rebalance)\n',
    '\n',
    '### History\n',
    '| Run | best val_iou | gap | val_iou std | Key change |\n',
    '|-----|-------------|-----|-------------|-------------|\n',
    '| 23  | 0.598 @ep70 | 0.137 | ~0.02  | baseline |\n',
    '| 24  | 0.579 @ep52 | 0.077 | ~0.03  | heavy regularisation (over-regularised) |\n',
    '| 25  | 0.580 @ep86 | 0.135 | **0.003** | EMA + surgical rollback |\n',
    '\n',
    '### Run 25 diagnosis -> Run 26 targets\n',
    '- EMA worked brilliantly (val_iou std 10x smaller) BUT val_iou ceiling still ~0.58\n',
    '- val_precision=0.68 vs recall=0.80 -> OVER-PREDICTING systematically\n',
    '- val_pred_pos=0.034 > gt=0.027 at thr=0.38 -> fixed threshold still wrong\n',
    '- Three runs stuck at same ceiling -> standard regularisation tuning exhausted\n',
    '\n',
    '### Run 26 changes — attack IoU directly\n',
    '\n',
    '| # | Change | Value | Reason |\n',
    '|---|--------|-------|--------|\n',
    '| 1 | **Lovász hinge loss (NEW)** | weight=0.30 | Differentiable IoU surrogate (Berman 2018) |\n',
    '| 2 | **Adaptive val threshold scan (NEW)** | 8 thresholds | Global TP/FP/FN accumulation per threshold |\n',
    '| 3 | pos_weight | 5.0 -> **3.0** | Rebalance toward precision |\n',
    '| 4 | tversky_alpha/beta | 0.3/0.7 -> **0.5/0.5** | Symmetric (stop recall push) |\n',
    '| 5 | tversky_weight | 0.30 -> **0.20** | Lovász takes over rare-class role |\n',
    '| 6 | epochs | 120 -> **90** | Run25 best was ep86 |\n',
    '| KEEP | EMA 0.999, label_smoothing 0.05, decoder_dropout 0.15, weight_decay 0.01, lstm_dropout 0.4, spatial_feedback=true |||\n',
    '\n',
    '**Config**: `config_run26.yaml` | **Architecture**: GNN + LSTM (+ SpatialFeedback) | **New losses**: Lovász hinge\n',
]))

# ── Cells 1-19: setup (unchanged) ─────────────────────────────────────
for i in range(1, 20):
    new_cells.append(cells[i])

# ── Cell 17 replacement: config_run26.yaml ────────────────────────────
new_cells[-3] = code_cell([
    '# Set up project and load configuration\n',
    'os.chdir(\'/content\')\n',
    'sys.path.insert(0, PROJECT_PATH)\n',
    '\n',
    '# version4 uses config_run26.yaml (Lovász + adaptive thr + precision rebalance)\n',
    'CONFIG_FILE = f\'{PROJECT_PATH}/config_run26.yaml\'\n',
    '\n',
    'if Path(CONFIG_FILE).exists():\n',
    '    print(f\'Config: {CONFIG_FILE}\')\n',
    '    !cat {CONFIG_FILE}\n',
    'else:\n',
    '    print(f\'config_run26.yaml not found, falling back\')\n',
    '    for alt in [\'config_run25.yaml\', \'config_run24.yaml\', \'config_run23.yaml\',\n',
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

# ── Cell 20: Training cell for run26 ──────────────────────────────────
new_cells.append(code_cell([
    'import glob\n',
    '\n',
    '# -------------------------------------------------------------------------\n',
    '# version4 NOTE:  All changes baked into backend + config:\n',
    '#   * Lovász hinge loss (NEW)  -> training/losses.py :: LovaszHingeLoss\n',
    '#   * Adaptive val threshold scan (NEW) -> training/trainer.py\n',
    '#   * Precision rebalance (pos_weight, tversky) -> config_run26.yaml\n',
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
    'TASK = \'af\'\n',
    'TASK_FLAG = f\'--task {TASK}\'\n',
    'print(f\'Training task: {TASK.upper()}\')\n',
    '\n',
    '# Checkpoint resumption — can warm-start from run25 best_model.pt\n',
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
    '    print(\'No checkpoint found — training version4 (Lovász) from scratch\')\n',
    '\n',
    'LOG_FILE = f\'{LOCAL_OUTPUT}/train.log\'\n',
    'print(f\'\\nStarting version4 training... task={TASK.upper()}  log: {LOG_FILE}\\n\')\n',
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

# ── Cells 21-23: save results (unchanged) ─────────────────────────────
new_cells.append(cells[21])
new_cells.append(cells[22])
new_cells.append(cells[23])

# ── Cell 24: Visualisation for run26 ─────────────────────────────────
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
    '    fig.suptitle(f\'FireST-GraphNet version4 (run26) Lovász+AdaptiveThr — {RUN_NAME}\', fontsize=13)\n',
    '\n',
    '    # Plot 1: Loss components\n',
    '    for col, lbl, ls in [\n',
    '        (\'train_sup\',    \'train sup\',    \'-\'),\n',
    '        (\'train_lovasz\', \'train Lovász\', \'-\'),\n',
    '        (\'val_sup\',      \'val sup\',      \'--\'),\n',
    '        (\'val_lovasz\',   \'val Lovász\',   \'--\'),\n',
    '    ]:\n',
    '        if col in df.columns:\n',
    '            axes[0,0].plot(df[\'epoch\'], df[col], label=lbl, linestyle=ls)\n',
    '    axes[0,0].set_title(\'Supervised + Lovász loss\')\n',
    '    axes[0,0].legend(fontsize=8); axes[0,0].grid(True)\n',
    '\n',
    '    # Plot 2: IoU — fixed thr vs best-thr (NEW) + history lines\n',
    '    if \'val_iou\' in df.columns:\n',
    '        axes[0,1].plot(df[\'epoch\'], df[\'val_iou\'], label=\'Val IoU (fixed 0.40)\', color=\'tomato\')\n',
    '    if \'val_iou_best_thr\' in df.columns:\n',
    '        axes[0,1].plot(df[\'epoch\'], df[\'val_iou_best_thr\'], label=\'Val IoU (best thr)\', color=\'darkred\', lw=2)\n',
    '        b = df.loc[df[\'val_iou_best_thr\'].idxmax()]\n',
    '        axes[0,1].axhline(b[\'val_iou_best_thr\'], ls=\':\', color=\'red\', alpha=0.6,\n',
    '                          label=f"Best {b[\'val_iou_best_thr\']:.4f} @ep{int(b[\'epoch\'])}")\n',
    '    if \'train_iou\' in df.columns:\n',
    '        axes[0,1].plot(df[\'epoch\'], df[\'train_iou\'], label=\'Train IoU\', ls=\'--\', color=\'steelblue\')\n',
    '    axes[0,1].axhline(0.598, color=\'grey\',  ls=\':\', alpha=0.4, label=\'run23 0.598\')\n',
    '    axes[0,1].axhline(0.580, color=\'brown\', ls=\':\', alpha=0.4, label=\'run25 0.580\')\n',
    '    axes[0,1].set_title(\'IoU (fixed vs adaptive threshold)\')\n',
    '    axes[0,1].legend(fontsize=8); axes[0,1].grid(True)\n',
    '\n',
    '    # Plot 3: best threshold over time\n',
    '    if \'val_thr_best\' in df.columns:\n',
    '        axes[0,2].plot(df[\'epoch\'], df[\'val_thr_best\'], marker=\'.\', color=\'purple\')\n',
    '        axes[0,2].axhline(0.40, ls=\'--\', color=\'grey\', alpha=0.5, label=\'fixed thr 0.40\')\n',
    '        axes[0,2].set_ylim(0.25, 0.55)\n',
    '    axes[0,2].set_title(\'Best threshold per epoch\')\n',
    '    axes[0,2].legend(fontsize=8); axes[0,2].grid(True)\n',
    '\n',
    '    # Plot 4: Precision / Recall at best thr vs fixed thr\n',
    '    if \'val_prec_best_thr\' in df.columns:\n',
    '        axes[1,0].plot(df[\'epoch\'], df[\'val_prec_best_thr\'], label=\'Prec (best thr)\', color=\'C0\', lw=2)\n',
    '        axes[1,0].plot(df[\'epoch\'], df[\'val_rec_best_thr\'],  label=\'Rec  (best thr)\', color=\'C1\', lw=2)\n',
    '    if \'val_precision\' in df.columns:\n',
    '        axes[1,0].plot(df[\'epoch\'], df[\'val_precision\'], label=\'Prec (fixed)\', color=\'C0\', ls=\':\')\n',
    '        axes[1,0].plot(df[\'epoch\'], df[\'val_recall\'],    label=\'Rec  (fixed)\', color=\'C1\', ls=\':\')\n',
    '    axes[1,0].axhline(0.68, ls=\':\', color=\'grey\', alpha=0.4, label=\'run25 prec=0.68\')\n',
    '    axes[1,0].set_title(\'Precision / Recall\')\n',
    '    axes[1,0].legend(fontsize=8); axes[1,0].grid(True)\n',
    '\n',
    '    # Plot 5: Overfit gap\n',
    '    if \'train_iou\' in df.columns and \'val_iou_best_thr\' in df.columns:\n',
    '        axes[1,1].plot(df[\'epoch\'], df[\'train_iou\'] - df[\'val_iou_best_thr\'], color=\'darkorange\',\n',
    '                       label=\'gap (best thr)\')\n',
    '    if \'train_iou\' in df.columns and \'val_iou\' in df.columns:\n',
    '        axes[1,1].plot(df[\'epoch\'], df[\'train_iou\'] - df[\'val_iou\'], color=\'goldenrod\', ls=\':\',\n',
    '                       label=\'gap (fixed thr)\')\n',
    '    axes[1,1].axhline(0, ls=\'--\', color=\'grey\')\n',
    '    axes[1,1].axhline(0.137, ls=\':\', color=\'grey\',  alpha=0.5, label=\'run23 gap=0.137\')\n',
    '    axes[1,1].set_title(\'Overfit Gap\')\n',
    '    axes[1,1].legend(fontsize=8); axes[1,1].grid(True)\n',
    '\n',
    '    # Plot 6: Pred/GT positive ratio\n',
    '    if \'val_pred_pos_ratio\' in df.columns:\n',
    '        axes[1,2].plot(df[\'epoch\'], df[\'val_pred_pos_ratio\'], label=\'val pred_pos (fixed)\')\n',
    '        axes[1,2].plot(df[\'epoch\'], df[\'val_gt_pos_ratio\'],   label=\'val gt_pos\', ls=\'--\')\n',
    '        if \'train_pred_pos_ratio\' in df.columns:\n',
    '            axes[1,2].plot(df[\'epoch\'], df[\'train_pred_pos_ratio\'], label=\'train pred_pos\', ls=\':\')\n',
    '    axes[1,2].set_title(\'Fire Pixel Ratio\')\n',
    '    axes[1,2].legend(fontsize=8); axes[1,2].grid(True)\n',
    '\n',
    '    plt.tight_layout()\n',
    '    out_png = f\'{LOCAL_OUTPUT}/training_curves_v4.png\'\n',
    '    plt.savefig(out_png, dpi=150)\n',
    '    plt.show()\n',
    '    print(f\'Saved: {out_png}\')\n',
    '\n',
    '    print(\'\\n=== Best Validation Metrics (version4 vs history) ===\')\n',
    '    iou_col = \'val_iou_best_thr\' if \'val_iou_best_thr\' in df.columns else \'val_iou\'\n',
    '    if iou_col in df.columns:\n',
    '        best = df.loc[df[iou_col].idxmax()]\n',
    '        print(f"  Best IoU ({iou_col}): {best[iou_col]:.4f} @ep {int(best[\'epoch\'])}")\n',
    '        print(f"    run23: 0.598  |  run25: 0.580  |  delta_run23: {best[iou_col]-0.598:+.4f}")\n',
    '        if \'val_thr_best\' in df.columns:\n',
    '            print(f"    best threshold: {best[\'val_thr_best\']:.2f}")\n',
    '        if \'val_prec_best_thr\' in df.columns:\n',
    '            print(f"    P/R at best thr: {best[\'val_prec_best_thr\']:.4f} / {best[\'val_rec_best_thr\']:.4f}")\n',
    '    if \'val_iou\' in df.columns and len(df) >= 20:\n',
    '        tail = df[iou_col].tail(20)\n',
    '        print(f"  Val IoU std (last 20 ep): {tail.std():.4f}  (run25 EMA: 0.003)")\n',
]))

# ── Cell 25: Completion markdown ─────────────────────────────────────
new_cells.append(md_cell([
    '## Complete\n',
    '\n',
    '### Summary of version4 Changes (over version3/run25)\n',
    '\n',
    '| File | Change | Reason |\n',
    '|------|--------|--------|\n',
    '| `training/losses.py` | **NEW `LovaszHingeLoss`** class + wired into `ForecastPINNLoss` | Direct IoU surrogate (Berman 2018) |\n',
    '| `training/trainer.py` | **Adaptive val threshold scan** — 8 thresholds, global TP/FP/FN | Fixes persistent over-prediction |\n',
    '| `training/trainer.py` | Use `iou_best_thr` for checkpoint selection | Best-model now at the optimal operating point |\n',
    '| `config_run26.yaml` | `lovasz_weight=0.30`, `pos_weight 5.0->3.0`, symmetric Tversky | Rebalance toward precision |\n',
    '| `config_run26.yaml` | `val_threshold_scan: true`, 8-point grid | Enable adaptive threshold |\n',
    '\n',
    '### Why Lovász hinge\n',
    '- All prior losses (BCE, Focal, Dice, Tversky) are *approximations* of IoU.\n',
    '- Lovász hinge is the **exact** sub-gradient of the Jaccard extension — i.e. it *is* IoU optimisation.\n',
    '- Per-image mode: computes loss per sample, averages — empty-label samples contribute 0 rather than collapsing.\n',
    '- Cast to fp32 internally: cumsum/sort need precision that bf16/fp16 lack.\n',
    '\n',
    '### Why adaptive threshold\n',
    '- Run25 showed pred_pos=0.034 > gt=0.027 even at thr=0.38 → fixed threshold was wrong.\n',
    '- Scan accumulates TP/FP/FN **across the whole val set** at 8 thresholds (correct way).\n',
    '- Picks max-IoU threshold each epoch; best_model.pt now stores weights at the true optimum.\n',
    '\n',
    '### Expected outcome\n',
    '- val_iou (best thr): **0.58 -> 0.62+**\n',
    '- val_precision:     **0.68 -> 0.75+** (rebalance + adaptive threshold)\n',
    '- val_iou std:       **~0.003** (EMA keeps variance low)\n',
    '- Lovász loss converges in 5-10 epochs then stays roughly flat\n',
    '\n',
    '### To Reproduce\n',
    'Run this notebook on Colab A100 with `config_run26.yaml` in your Drive.\n',
    'Training takes approximately 2-2.5 hours for 90 epochs (Lovász adds ~5% per-step cost).\n',
]))

nb['cells'] = new_cells
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Written: {DST}')
print(f'Total cells: {len(new_cells)}')
