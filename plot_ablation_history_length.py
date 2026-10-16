import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
T_eval        = [1, 4, 6, 8, 10]
IoU           = [0.5122, 0.6704, 0.7046, 0.7214, 0.7262]
F1            = [0.6774, 0.8027, 0.8267, 0.8382, 0.8414]
Precision     = [0.7599, 0.8009, 0.8045, 0.8093, 0.8109]
Recall        = [0.6111, 0.8045, 0.8502, 0.8692, 0.8741]
IoU_retention = [70.5,   92.3,   97.0,   99.3,   100.0]

# ── Figure ────────────────────────────────────────────────────────────────────
# ~half the height of a typical (7, 5.5) ablation figure
fig, ax = plt.subplots(figsize=(7, 3.2))

lw = 1.8
ms = 6

ax.plot(T_eval, IoU,       color='#d62728', marker='o', markersize=ms,
        linewidth=lw, linestyle='-',  label='IoU',       zorder=3)
ax.plot(T_eval, F1,        color='#1f77b4', marker='s', markersize=ms,
        linewidth=lw, linestyle='-',  label='F1',        zorder=3)
ax.plot(T_eval, Precision, color='#2ca02c', marker='^', markersize=ms,
        linewidth=lw, linestyle='--', label='Precision', zorder=3)
ax.plot(T_eval, Recall,    color='#ff7f0e', marker='v', markersize=ms,
        linewidth=lw, linestyle='--', label='Recall',    zorder=3)

# ── IoU retention annotations ─────────────────────────────────────────────────
# nudge the first label left so it clears the y-axis spine
x_offsets = [-6, 0, 0, 0, 0]   # points in x direction for first label
for i, (x, y, ret) in enumerate(zip(T_eval, IoU, IoU_retention)):
    ax.annotate(
        f'{ret}%',
        xy=(x, y),
        xytext=(x_offsets[i], 6),
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=7.5, color='#555555',
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.2, 10.8)
ax.set_ylim(0.46, 0.935)
ax.set_xticks(T_eval)
ax.tick_params(axis='both', labelsize=8.5)

ax.set_xlabel(
    r'Inference history length $T_{\mathrm{in}}^{\mathrm{eval}}$ (days)',
    fontsize=9.5,
)
ax.set_ylabel('Validation metric', fontsize=9.5)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.35, color='gray')

# ── Legend ────────────────────────────────────────────────────────────────────
# two-column, lower-right; keep away from the annotation cluster (upper-right)
ax.legend(
    loc='lower right',
    ncol=2,
    fontsize=8,
    framealpha=0.85,
    edgecolor='#cccccc',
    handlelength=2.0,
    columnspacing=1.0,
    handletextpad=0.5,
)

# ── Layout & export ───────────────────────────────────────────────────────────
fig.tight_layout(pad=0.5)

out_png = r'C:\Users\22125\Desktop\FireST-GraphNet\ablation_history_length.png'
out_pdf = r'C:\Users\22125\Desktop\FireST-GraphNet\ablation_history_length.pdf'

fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf,           bbox_inches='tight')

print(f'Saved:\n  {out_png}\n  {out_pdf}')
plt.show()
