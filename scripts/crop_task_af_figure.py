"""Crop and rebuild a 3x5 task_af comparison figure, zoomed into the main fire cluster."""
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = r"C:\Users\22125\Desktop\FireST-GraphNet\paper\University_of_Wisconsin_Madison_Dissertation_Template__new_\images\task_af\event_0000_AC_ConvLSTM.png"
DST = r"C:\Users\22125\Desktop\FireST-GraphNet\paper\University_of_Wisconsin_Madison_Dissertation_Template__new_\images\task_af\event_0000_AC_ConvLSTM_Cropped.png"

# Panel grid detected from source (x0,x1 per col; y0,y1 per row).
COLS = [(91, 494), (530, 933), (964, 1378), (1399, 1809), (1844, 2247)]
ROWS = [(45, 448), (481, 883), (917, 1319)]
ROW_LABELS = ["Ground Truth", "Prediction", "Overlap"]
COL_TITLES = ["T+0", "T+1", "T+2", "T+3", "T+4"]
IOU_VALUES = [0.75, 0.59, 0.64, 0.58, 0.54]


def compute_crop_box(panel_bgr, margin=20, min_component=20):
    """Find tight bbox around bright (fire) pixels, ignoring small isolated noise."""
    gray = panel_bgr.mean(axis=2)
    mask = gray > 200
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep_ids = np.where(sizes > min_component)[0] + 1
    big = np.isin(labels, keep_ids)
    ys, xs = np.where(big)
    h, w = panel_bgr.shape[:2]
    y0 = max(0, ys.min() - margin)
    y1 = min(h, ys.max() + margin)
    x0 = max(0, xs.min() - margin)
    x1 = min(w, xs.max() + margin)
    return x0, y0, x1, y1


def main():
    full = np.array(Image.open(SRC).convert("RGB"))
    # Panel Row0 Col4 = Ground Truth T+4 -> determine crop box
    rx0, rx1 = COLS[4]
    ry0, ry1 = ROWS[0]
    ref_panel = full[ry0:ry1, rx0:rx1]
    cx0, cy0, cx1, cy1 = compute_crop_box(ref_panel)
    ph, pw = ref_panel.shape[:2]
    # Fractional box (same crop applied to all panels regardless of slight size diff)
    fx0, fy0 = cx0 / pw, cy0 / ph
    fx1, fy1 = cx1 / pw, cy1 / ph
    print(f"Reference panel size {pw}x{ph}, crop px=({cx0},{cy0})-({cx1},{cy1})")
    print(f"Fractional crop: x=[{fx0:.3f},{fx1:.3f}] y=[{fy0:.3f},{fy1:.3f}]")

    # Extract and crop each of 15 panels
    cropped = [[None] * 5 for _ in range(3)]
    for r, (y0, y1) in enumerate(ROWS):
        for c, (x0, x1) in enumerate(COLS):
            panel = full[y0:y1, x0:x1]
            h, w = panel.shape[:2]
            px0 = int(round(fx0 * w)); px1 = int(round(fx1 * w))
            py0 = int(round(fy0 * h)); py1 = int(round(fy1 * h))
            cropped[r][c] = panel[py0:py1, px0:px1]

    # Build new figure
    ch, cw = cropped[0][0].shape[:2]
    aspect = cw / ch
    panel_w = 2.2
    panel_h = panel_w / aspect
    fig_w = panel_w * 5 + 1.1   # extra for row labels
    fig_h = panel_h * 3 + 1.1   # extra for col titles + legend
    fig, axes = plt.subplots(3, 5, figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.055, right=0.995,
                        top=0.93, bottom=0.09)

    for r in range(3):
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(cropped[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=20, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=18, fontweight="bold",
                              labelpad=10)
            if r == 1:  # prediction -> IoU text
                ax.text(0.97, 0.04, f"IoU {IOU_VALUES[c]:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=13, color="white",
                        bbox=dict(facecolor="black", alpha=0.45,
                                  edgecolor="none", pad=2))

    # Legend
    handles = [
        Patch(facecolor=(0, 0.8, 0), edgecolor="black", label="True Positive"),
        Patch(facecolor=(0.9, 0.1, 0.1), edgecolor="black", label="False Positive"),
        Patch(facecolor=(0.1, 0.2, 0.95), edgecolor="black", label="False Negative"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=16,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.savefig(DST, dpi=300, bbox_inches=None)
    print(f"Saved: {DST}")


if __name__ == "__main__":
    main()
