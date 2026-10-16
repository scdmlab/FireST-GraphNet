"""Batch crop and rebuild 3x5 comparison figures across task_af/task_ba/task_pred.

Group by Event ID within each folder; compute one crop box per event from the
representative image's Row-1/Col-5 panel; apply that crop to every sibling
figure and rebuild with tighter spacing, enlarged labels, preserved IoU/MAE
text and legend.

Design:
- Panel grid is detected adaptively from each image (not hard-coded) by
  connected components on the non-white mask.
- Row labels, column titles and legend are extracted as bitmaps from the
  original image and scaled up -> enlarged without needing OCR.
- IoU/MAE text in the middle row is preserved by slicing a bottom-right
  strip from each original prediction panel and overlaying it onto the
  cropped+resized panel at the same relative position.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

# -------------------- config --------------------
ROOT = Path(
    r"C:\Users\22125\Desktop\FireST-GraphNet\paper"
    r"\University_of_Wisconsin_Madison_Dissertation_Template__new_\images"
)
FOLDERS = ["task_af", "task_ba", "task_pred"]
EVENT_RE = re.compile(r"^event_(\d{4})_", re.IGNORECASE)
REP_PREFERENCE = ["AC_ConvLSTM", "BA_ConvLSTM", "Pred_ConvLSTM", "ConvLSTM"]

# Rebuild style
PANEL_H = 340
GAP = 6
MARGIN = 14
ROW_LABEL_W = 190
COL_TITLE_H = 95
LEGEND_H = 100

# -------------------- helpers --------------------

def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def trim_white(arr: np.ndarray, pad: int = 2, threshold: int = 240) -> np.ndarray:
    if arr.size == 0:
        return arr
    mask = arr.max(axis=2) < threshold
    if not mask.any():
        return arr
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad); y1 = min(arr.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad); x1 = min(arr.shape[1], int(xs.max()) + 1 + pad)
    return arr[y0:y1, x0:x1]


def fit_img(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return img
    s = min(max_w / w, max_h / h)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


# -------------------- detection --------------------

def detect_panel_grid(img: np.ndarray) -> Optional[list[list[tuple[int, int, int, int]]]]:
    """Return list of 3 rows of 5 panel boxes (x0,y0,x1,y1) in left-to-right order."""
    not_white = img.max(axis=2) < 240
    closed = ndimage.binary_closing(not_white, iterations=3)
    labels, n = ndimage.label(closed)
    if n == 0:
        return None
    sizes = ndimage.sum(closed, labels, range(1, n + 1))
    candidates: list[tuple[int, int, int, int, float]] = []
    for i, s in enumerate(sizes, start=1):
        if s < 8000:
            continue
        ys, xs = np.where(labels == i)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w < 150 or h < 150:
            continue
        aspect = w / h
        if not (0.5 < aspect < 2.2):
            continue
        candidates.append((x0, y0, x1, y1, float(s)))
    if len(candidates) < 15:
        return None
    # Largest 15
    candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    top = candidates[:15]
    boxes = [(x0, y0, x1, y1) for x0, y0, x1, y1, _ in top]
    # Cluster into 3 rows by y-center, then sort each row by x
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    rows = [boxes[i * 5:(i + 1) * 5] for i in range(3)]
    for r in rows:
        r.sort(key=lambda b: b[0])
    return rows


def compute_crop_box(panel: np.ndarray, margin_frac: float = 0.06,
                     min_component: int = 15,
                     diff_threshold: float = 50.0) -> tuple[float, float, float, float]:
    """Return fractional (fx0, fy0, fx1, fy1) crop around main content cluster."""
    h, w = panel.shape[:2]
    # Background estimated from the four corners
    corners = np.stack([panel[0, 0], panel[0, -1], panel[-1, 0], panel[-1, -1]]).astype(float)
    bg = corners.mean(axis=0)
    diff = np.linalg.norm(panel.astype(float) - bg, axis=2)
    mask = diff > diff_threshold
    if not mask.any():
        return 0.0, 0.0, 1.0, 1.0
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    # Adaptive: drop components that are small relative to the largest cluster
    # (removes isolated noise dots near the panel edges that would inflate bbox).
    thresh = max(float(min_component), 0.10 * float(sizes.max()))
    big = np.isin(labels, np.where(sizes > thresh)[0] + 1)
    if not big.any():
        big = mask
    ys, xs = np.where(big)
    my = int(round(margin_frac * h)); mx = int(round(margin_frac * w))
    y0 = max(0, int(ys.min()) - my); y1 = min(h, int(ys.max()) + 1 + my)
    x0 = max(0, int(xs.min()) - mx); x1 = min(w, int(xs.max()) + 1 + mx)
    return x0 / w, y0 / h, x1 / w, y1 / h


# -------------------- rebuild --------------------

def rebuild_figure(img: np.ndarray, grid: list[list[tuple[int, int, int, int]]],
                   crop_frac: tuple[float, float, float, float]) -> Image.Image:
    H, W = img.shape[:2]
    fx0, fy0, fx1, fy1 = crop_frac
    grid_x0 = min(p[0] for row in grid for p in row)
    grid_x1 = max(p[2] for row in grid for p in row)
    grid_y0 = min(p[1] for row in grid for p in row)
    grid_y1 = max(p[3] for row in grid for p in row)

    # Extract decoration bitmaps
    col_title_strips = []
    for c in range(5):
        x0, x1 = grid[0][c][0], grid[0][c][2]
        strip = img[0:grid_y0, x0:x1]
        col_title_strips.append(trim_white(strip))

    row_label_strips = []
    for r in range(3):
        y0, y1 = grid[r][0][1], grid[r][0][3]
        strip = img[y0:y1, 0:grid_x0]
        row_label_strips.append(trim_white(strip))

    legend = trim_white(img[grid_y1:H, :])

    # Crop panels and collect middle-row IoU/MAE strips (bottom-right)
    cropped: list[list[np.ndarray]] = [[None] * 5 for _ in range(3)]  # type: ignore
    orig_panel_h = [[0] * 5 for _ in range(3)]
    iou_strips: list[Optional[np.ndarray]] = [None] * 5

    for r in range(3):
        for c in range(5):
            x0, y0, x1, y1 = grid[r][c]
            panel = img[y0:y1, x0:x1]
            ph, pw = panel.shape[:2]
            px0 = int(round(fx0 * pw)); px1 = int(round(fx1 * pw))
            py0 = int(round(fy0 * ph)); py1 = int(round(fy1 * ph))
            cropped[r][c] = panel[py0:py1, px0:px1]
            orig_panel_h[r][c] = ph
            if r == 1:
                sy0 = int(round(0.82 * ph))
                sx0 = int(round(0.52 * pw))
                iou_strips[c] = panel[sy0:ph, sx0:pw]

    # Layout dimensions
    ch, cw = cropped[0][0].shape[:2]
    target_h = PANEL_H
    target_w = max(1, int(round(target_h * cw / ch)))
    grid_w = 5 * target_w + 4 * GAP
    grid_h = 3 * target_h + 2 * GAP
    total_w = MARGIN + ROW_LABEL_W + GAP + grid_w + MARGIN
    total_h = MARGIN + COL_TITLE_H + GAP + grid_h + GAP + LEGEND_H + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # Column titles
    for c in range(5):
        title = Image.fromarray(col_title_strips[c])
        title = fit_img(title, target_w, COL_TITLE_H)
        x = MARGIN + ROW_LABEL_W + GAP + c * (target_w + GAP) + (target_w - title.width) // 2
        y = MARGIN + (COL_TITLE_H - title.height) // 2
        canvas.paste(title, (x, y))

    # Row labels
    grid_top = MARGIN + COL_TITLE_H + GAP
    for r in range(3):
        lbl = Image.fromarray(row_label_strips[r])
        lbl = fit_img(lbl, ROW_LABEL_W, target_h)
        x = MARGIN + (ROW_LABEL_W - lbl.width) // 2
        y = grid_top + r * (target_h + GAP) + (target_h - lbl.height) // 2
        canvas.paste(lbl, (x, y))

    # Panels + IoU overlay
    for r in range(3):
        for c in range(5):
            panel_img = Image.fromarray(cropped[r][c]).resize((target_w, target_h), Image.LANCZOS)
            if r == 1 and iou_strips[c] is not None and iou_strips[c].size > 0:
                strip = iou_strips[c]
                # scale strip by the same factor as panel display / original-panel-height
                scale = target_h / orig_panel_h[r][c]
                sw = max(1, int(round(strip.shape[1] * scale)))
                sh = max(1, int(round(strip.shape[0] * scale)))
                strip_img = Image.fromarray(strip).resize((sw, sh), Image.LANCZOS)
                # Clip so it fits inside the panel
                sw = min(sw, target_w); sh = min(sh, target_h)
                if (sw, sh) != strip_img.size:
                    strip_img = strip_img.crop((strip_img.width - sw, strip_img.height - sh,
                                                strip_img.width, strip_img.height))
                panel_img.paste(strip_img, (target_w - sw, target_h - sh))
            x = MARGIN + ROW_LABEL_W + GAP + c * (target_w + GAP)
            y = grid_top + r * (target_h + GAP)
            canvas.paste(panel_img, (x, y))

    # Legend
    if legend.size > 0:
        lg = Image.fromarray(legend)
        lg = fit_img(lg, grid_w, LEGEND_H)
        lg_x = MARGIN + ROW_LABEL_W + GAP + (grid_w - lg.width) // 2
        lg_y = grid_top + grid_h + GAP + (LEGEND_H - lg.height) // 2
        canvas.paste(lg, (lg_x, lg_y))

    return canvas


# -------------------- pipeline --------------------

def pick_representative(files: list[Path]) -> Path:
    for key in REP_PREFERENCE:
        for f in files:
            if key.lower() in f.stem.lower():
                return f
    return files[0]


def group_by_event(folder: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if "_Cropped" in p.stem:
            continue
        m = EVENT_RE.match(p.name)
        if not m:
            continue
        groups.setdefault(m.group(1), []).append(p)
    return groups


def process_folder(folder: Path) -> dict:
    print(f"\n=== Folder: {folder.name} ===")
    groups = group_by_event(folder)
    print(f"  event groups: {len(groups)}")
    meta: dict = {"folder": folder.name, "events": {}, "errors": []}
    processed = skipped = 0

    for event_id, files in sorted(groups.items()):
        rep = pick_representative(files)
        try:
            rep_img = load_rgb(rep)
            rep_grid = detect_panel_grid(rep_img)
        except Exception as e:
            meta["errors"].append({"file": str(rep), "error": f"rep load failed: {e}"})
            skipped += len(files)
            continue
        if rep_grid is None:
            meta["errors"].append({"file": str(rep), "error": "panel grid not detected"})
            skipped += len(files)
            continue
        ref_panel = rep_img[rep_grid[0][4][1]:rep_grid[0][4][3],
                            rep_grid[0][4][0]:rep_grid[0][4][2]]
        crop_frac = compute_crop_box(ref_panel)
        event_meta = {
            "representative": rep.name,
            "crop_fractional": [round(v, 4) for v in crop_frac],
            "files_processed": [],
            "files_skipped": [],
        }
        for f in files:
            try:
                img = load_rgb(f)
                grid = detect_panel_grid(img)
                if grid is None:
                    raise RuntimeError("panel grid not detected")
                out = rebuild_figure(img, grid, crop_frac)
                out_path = f.with_name(f.stem + "_Cropped" + f.suffix)
                out.save(out_path)
                event_meta["files_processed"].append(f.name)
                processed += 1
            except Exception as e:
                event_meta["files_skipped"].append({"file": f.name, "error": str(e)})
                meta["errors"].append({"file": str(f), "error": str(e)})
                skipped += 1
        meta["events"][event_id] = event_meta
        print(f"  event_{event_id}: rep={rep.name}  crop={event_meta['crop_fractional']}  "
              f"ok={len(event_meta['files_processed'])}  skip={len(event_meta['files_skipped'])}")

    meta["summary"] = {
        "event_groups": len(groups),
        "processed": processed,
        "skipped": skipped,
    }
    out_json = folder / f"{folder.name}_event_crop_boxes.json"
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"  -> processed={processed}  skipped={skipped}  meta={out_json.name}")
    return meta


def main() -> None:
    for name in FOLDERS:
        folder = ROOT / name
        if not folder.is_dir():
            print(f"missing folder: {folder}")
            continue
        process_folder(folder)


if __name__ == "__main__":
    main()
