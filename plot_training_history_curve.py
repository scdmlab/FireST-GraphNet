import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
T_train   = [1, 4, 6, 8, 10]
IoU       = [0.633, 0.701, 0.708, 0.729, 0.736]
F1        = [0.775, 0.824, 0.829, 0.843, 0.848]
Precision = [0.752, 0.800, 0.792, 0.810, 0.820]
Recall    = [0.800, 0.850, 0.870, 0.880, 0.877]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.4))

lw = 1.8
ms = 7

ax.plot(T_train, IoU,       color='#d62728', marker='o', markersize=ms,
        linewidth=lw, linestyle='-',  label='IoU',       zorder=3)
ax.plot(T_train, F1,        color='#1f77b4', marker='s', markersize=ms,
        linewidth=lw, linestyle='-',  label='F1',        zorder=3)
ax.plot(T_train, Precision, color='#2ca02c', marker='^', markersize=ms,
        linewidth=lw, linestyle='--', label='Precision', zorder=3)
ax.plot(T_train, Recall,    color='#ff7f0e', marker='v', markersize=ms,
        linewidth=lw, linestyle='--', label='Recall',    zorder=3)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.2, 10.8)
ax.set_ylim(0.60, 0.90)
ax.set_xticks(T_train)
ax.tick_params(axis='both', labelsize=11)

ax.set_xlabel(
    r'Training history length $T_{\mathrm{in}}^{\mathrm{train}}$ (days)',
    fontsize=13,
)
ax.set_ylabel('Test-set metric', fontsize=13)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.grid(True, linestyle='-', linewidth=0.6, alpha=0.4, color='gray')

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(
    loc='lower right',
    ncol=2,
    fontsize=11,
    framealpha=0.9,
    edgecolor='#cccccc',
    handlelength=2.0,
    columnspacing=1.2,
    handletextpad=0.5,
)

# ── Layout & export ───────────────────────────────────────────────────────────
fig.tight_layout(pad=0.5)

out_png = r'C:\Users\22125\Desktop\FireST-GraphNet\training_history_curve.png'
out_pdf = r'C:\Users\22125\Desktop\FireST-GraphNet\training_history_curve.pdf'

fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf,           bbox_inches='tight')

print(f'Saved:\n  {out_png}\n  {out_pdf}')
