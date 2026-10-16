"""
Physics-informed calibration trade-off figure  (fig_af_physics_tradeoff)
X-axis : FP saved vs. baseline (derived from ΔPrecision, N=10 000 test pixels)
Y-axis : TP lost / FP saved ratio r  (lower = more efficient FP suppression)
"""
import matplotlib.pyplot as plt

# ── Derived FP-saved values ────────────────────────────────────────────────────
# Baseline: Precision=0.7879, Recall=0.8831
# FP_baseline / N  = Recall_base * (1/Precision_base - 1) = 0.8831 * 0.2694 = 0.2380
# FP_method / N    = Recall_method * (1/Precision_method - 1)
# FP_saved / N     = FP_baseline/N - FP_method/N   ;  scaled by N=10 000
#
#   Component filter     : 0.8831*(1/0.7879-1/0.7882)*10000 ≈  6
#   Multiscale context   : 0.8831*(1/0.7879-1/0.7944)*10000 ≈ 91
#   Recent-fire adaptive : 0.8787*(1/0.7879-1/0.8048)*10000 ≈ 247
#   Distance-adaptive    : 0.8786*(1/0.7879-1/0.8057)*10000 ≈ 264

fp_saved = [6,   91,    247,   264]
r_values = [0.04, 0.004, 0.18, 0.18]   # TP lost / FP saved

COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
MARKERS = ['o',       's',       'D',       '*']
MSIZES  = [8,          8,         8,         14]   # star slightly larger

# ── Figure — compressed height ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 2.2))

# r = 1 equal-trade-off reference line
ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.2, zorder=1)
# annotation sits just below the dashed line so it stays inside the axes
ax.text(298, 0.968,
        r'$r=1$ (equal trade-off)',
        ha='right', va='top', fontsize=8, color='#555555', style='italic')

# ── Scatter points ────────────────────────────────────────────────────────────
for fp, r, col, mk, ms in zip(fp_saved, r_values, COLORS, MARKERS, MSIZES):
    ax.scatter(fp, r, color=col, marker=mk,
               s=ms ** 2, zorder=4, linewidths=0.5,
               edgecolors='white' if mk != '*' else col)

# ── Method labels — placed directly beside markers, no arrows ─────────────────
# With figsize=(8, 2.2) and ylim=[0, 1.08], axis height ≈ 1.5 in.
# One text line at 8 pt ≈ 0.111 in ≈ 0.074 y-units; two lines ≈ 0.155 y-units.

# 1. Component filter  (6, 0.04)  ── right of marker; avoids left-edge clipping
ax.text(14, 0.064,
        'Component\nfilter',
        fontsize=8, color='#1f77b4', ha='left', va='bottom')

# 2. Multiscale context threshold  (91, 0.004)  ── right and above marker
ax.text(100, 0.026,
        'Multiscale context\nthreshold',
        fontsize=8, color='#ff7f0e', ha='left', va='bottom')

# 3. Recent-fire adaptive threshold  (247, 0.18)  ── lower-left of diamond
#    va='top'   → text hangs downward from y=0.160 (below the marker centre)
#    ha='right' → right edge at x=237, left of the diamond at x=247
#    y=0.160 chosen so the two-line block clears y=0 (bottom ≈ 0.005)
ax.text(237, 0.160,
        'Recent-fire\nadaptive threshold',
        fontsize=8, color='#2ca02c', ha='right', va='top')

# 4. Distance-adaptive threshold  (264, 0.18)  ── centred above the red star
#    ha='center' → text is centred on x=264 (same x as the star)
#    va='bottom' → text grows upward from y=0.210
#    right edge ≈ 264 + 26 = 290, well inside xlim=308
ax.text(264, 0.210,
        'Distance-adaptive\nthreshold',
        fontsize=8, color='#d62728', fontweight='bold', ha='center', va='bottom')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-18, 308)
ax.set_ylim(0.0, 1.08)
ax.tick_params(axis='both', labelsize=8.5)

ax.set_xlabel(
    r'FP saved vs. baseline ($\tau=0.5$, held-out test set)',
    fontsize=9.5,
)
ax.set_ylabel(r'TP lost / FP saved ratio $r$', fontsize=9.5)

# ── Grid ──────────────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.grid(True, linestyle='-', linewidth=0.45, alpha=0.35, color='gray')

# ── Layout & export ───────────────────────────────────────────────────────────
fig.tight_layout(pad=0.5)

IMGDIR = (r'C:\Users\22125\Desktop\FireST-GraphNet'
          r'\paper\University_of_Wisconsin_Madison_Dissertation_Template__new_\images')

out_png = IMGDIR + r'\fig_af_physics_tradeoff.png'
out_pdf = IMGDIR + r'\fig_af_physics_tradeoff.pdf'

fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf,           bbox_inches='tight')
print(f'Saved:\n  {out_png}\n  {out_pdf}')
plt.show()
