from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

ROI = Tuple[int, int, int, int]

TITLE_FONTSIZE = 16
ROW_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13
METRIC_FONTSIZE = 11
LEGEND_HANDLELENGTH = 1.6
LEGEND_HANDLEHEIGHT = 1.2

WSPACE = 0.015
HSPACE = 0.015
LEFT = 0.065
RIGHT = 0.995
TOP = 0.93
BOTTOM = 0.085

TITLE_PAD = 4
ROW_LABEL_X = -0.10
LEGEND_Y = 0.018


def format_event_id(event_idx: int) -> str:
    return f"event_{event_idx:04d}"


def _coerce_roi(roi: object) -> ROI:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError(f"Invalid ROI payload: {roi!r}")
    y0, y1, x0, x1 = (int(v) for v in roi)
    return y0, y1, x0, x1


def _sanitize_roi(roi: ROI, image_shape: Tuple[int, int]) -> ROI:
    height, width = image_shape
    y0, y1, x0, x1 = roi
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    return y0, y1, x0, x1


def load_roi_cache(roi_cache_path: str | Path) -> dict[str, list[int]]:
    path = Path(roi_cache_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"ROI cache must be a JSON object: {path}")
    return data


def load_roi(event_id: str, roi_cache_path: str | Path) -> Optional[ROI]:
    cache = load_roi_cache(roi_cache_path)
    roi = cache.get(event_id)
    return None if roi is None else _coerce_roi(roi)


def save_roi(event_id: str, roi: ROI, roi_cache_path: str | Path) -> ROI:
    path = Path(roi_cache_path)
    cache = load_roi_cache(path)
    clean_roi = [int(v) for v in roi]
    cache[event_id] = clean_roi
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    return tuple(clean_roi)  # type: ignore[return-value]


def apply_roi(sequence: np.ndarray, roi: ROI) -> np.ndarray:
    y0, y1, x0, x1 = roi
    return sequence[:, y0:y1, x0:x1]


def compute_auto_roi(
    gt_seq: np.ndarray,
    ref_seq: np.ndarray,
    blur_fn: Callable[[np.ndarray, float], np.ndarray],
    threshold: float = 0.5,
    blur_sigma: float = 3.0,
    margin: float = 0.25,
    min_size: int = 80,
) -> ROI:
    combined = np.maximum(gt_seq.max(axis=0), ref_seq.max(axis=0))
    blurred = blur_fn((combined >= threshold).astype(np.float32), blur_sigma)
    ys, xs = np.where(blurred > 0.05)
    if len(ys) == 0:
        ys, xs = np.where(combined >= threshold)
    if len(ys) == 0:
        height, width = gt_seq.shape[1], gt_seq.shape[2]
        return 0, height, 0, width

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    height, width = gt_seq.shape[1], gt_seq.shape[2]
    pad_y = max(int((y1 - y0) * margin), 15)
    pad_x = max(int((x1 - x0) * margin), 15)
    y0 = max(0, y0 - pad_y)
    y1 = min(height, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(width, x1 + pad_x)
    if y1 - y0 < min_size:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_size // 2)
        y1 = min(height, y0 + min_size)
    if x1 - x0 < min_size:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_size // 2)
        x1 = min(width, x0 + min_size)
    return y0, y1, x0, x1


def select_roi_interactive(image: np.ndarray) -> ROI:
    import matplotlib

    # Some Windows scientific Python stacks load multiple OpenMP runtimes
    # when interactive backends are imported. This keeps ROI selection usable.
    if "KMP_DUPLICATE_LIB_OK" not in os.environ:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        print("Set KMP_DUPLICATE_LIB_OK=TRUE for interactive ROI selection.")

    backend = os.environ.get("FIREST_ROI_BACKEND")
    candidates = [backend] if backend else ["TkAgg", "WebAgg"]

    last_error: Optional[Exception] = None
    plt = None
    RectangleSelector = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            matplotlib.use(candidate, force=True)
            import matplotlib.pyplot as _plt
            from matplotlib.widgets import RectangleSelector as _RectangleSelector

            test_fig = _plt.figure()
            _plt.close(test_fig)
            plt = _plt
            RectangleSelector = _RectangleSelector
            if candidate.lower() == "webagg":
                print("ROI selection is using the WebAgg backend.")
                print("Open the local Matplotlib URL shown below in your browser if it does not open automatically.")
            break
        except Exception as exc:
            last_error = exc

    if plt is None or RectangleSelector is None:
        raise RuntimeError(
            "No interactive Matplotlib backend is available for ROI selection. "
            "Set FIREST_ROI_BACKEND to a working backend and try again."
        ) from last_error

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("ROI selection expects a 2D image")

    state: dict[str, Optional[ROI]] = {"roi": None}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="inferno", interpolation="nearest")
    ax.set_title("Drag ROI on final GT frame, then press Enter to confirm")
    ax.set_axis_off()

    def on_select(eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        roi = (
            int(np.floor(y0)),
            int(np.ceil(y1)),
            int(np.floor(x0)),
            int(np.ceil(x1)),
        )
        state["roi"] = _sanitize_roi(roi, image.shape)

    def on_key(event) -> None:
        if event.key == "enter":
            if state["roi"] is None:
                print("Select a rectangle before pressing Enter.")
                return
            plt.close(fig)
        elif event.key == "escape":
            state["roi"] = None
            plt.close(fig)

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
        drag_from_anywhere=True,
    )
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    selector.set_active(False)

    if state["roi"] is None:
        raise RuntimeError("ROI selection cancelled or no ROI was confirmed.")
    return _sanitize_roi(state["roi"], image.shape)
