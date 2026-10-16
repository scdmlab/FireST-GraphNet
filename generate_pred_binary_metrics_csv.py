from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Keep batch generation stable in Conda/OpenMP-heavy environments on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from visualize_illustration_pred import build_figure, load_multihorizon_gt
from visualize_roi_utils import apply_roi, format_event_id, load_roi


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f1_iou_per_frame(gt_seq: np.ndarray, pred_seq: np.ndarray) -> tuple[list[float], list[float]]:
    f1s: list[float] = []
    ious: list[float] = []
    for t in range(gt_seq.shape[0]):
        g = gt_seq[t] >= 0.5
        p = pred_seq[t] >= 0.5
        tp = int((g & p).sum())
        fp = int((~g & p).sum())
        fn = int((g & ~p).sum())
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        f1s.append(float(f1))
        ious.append(float(iou))
    return f1s, ious


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute binary F1/IoU metrics for cropped Pred figures from a CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="figures/task_pred/pred_results.csv",
        help="CSV file describing Pred figure settings.",
    )
    parser.add_argument(
        "--roi-cache",
        type=str,
        default="roi_cache.json",
        help="Shared ROI cache JSON file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset_val",
        help="Dataset directory for loading Pred events.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon passed to Pred loader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the Pred figure generator.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="figures/crop/task_pred/pred_binary_metrics.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    event_ids = OrderedDict()
    for row in rows:
        event_idx = int(row["event_idx"])
        event_ids[event_idx] = None

    gt_by_event: dict[int, np.ndarray] = {}
    roi_by_event: dict[int, tuple[int, int, int, int]] = {}

    for event_idx in event_ids:
        event_id = format_event_id(event_idx)
        roi = load_roi(event_id, args.roi_cache)
        if roi is None:
            raise FileNotFoundError(
                f"ROI for {event_id} not found in {args.roi_cache}. "
                "Select ROI first before computing binary metrics."
            )
        roi_by_event[event_idx] = roi
        print(f"Loading {event_id} with ROI {roi} ...")
        gt_by_event[event_idx] = load_multihorizon_gt(args.data_dir, event_idx, args.horizon)

    out_rows: list[dict[str, object]] = []
    for row in rows:
        event_idx = int(row["event_idx"])
        filename = row["filename"]
        roi = roi_by_event[event_idx]
        gt_full = np.clip(gt_by_event[event_idx][:5].copy(), 0, 1)
        gt_crop = apply_roi((gt_full >= 0.5).astype(np.float32), roi)

        _, pred_full, _ = build_figure(
            gt_by_event[event_idx],
            seed=args.seed,
            quality=float(row["quality"]),
            fragmentation=float(row["fragmentation"]),
            save_path=None,
            show=False,
            roi=roi,
            auto_crop=False,
        )
        pred_crop = apply_roi(pred_full[:5], roi)

        f1s, ious = f1_iou_per_frame(gt_crop, pred_crop)
        out_rows.append(
            {
                "event_idx": event_idx,
                "event_rank": int(row["event_rank"]),
                "quality": float(row["quality"]),
                "fragmentation": float(row["fragmentation"]),
                "mean_iou": round(float(np.mean(ious)), 4),
                "iou_t0": round(ious[0], 4),
                "iou_t1": round(ious[1], 4),
                "iou_t2": round(ious[2], 4),
                "iou_t3": round(ious[3], 4),
                "iou_t4": round(ious[4], 4),
                "mean_f1": round(float(np.mean(f1s)), 4),
                "f1_t0": round(f1s[0], 4),
                "f1_t1": round(f1s[1], 4),
                "f1_t2": round(f1s[2], 4),
                "f1_t3": round(f1s[3], 4),
                "f1_t4": round(f1s[4], 4),
                "filename": filename,
            }
        )
        print(f"Computed metrics for {filename}")

    fieldnames = [
        "event_idx",
        "event_rank",
        "quality",
        "fragmentation",
        "mean_iou",
        "iou_t0",
        "iou_t1",
        "iou_t2",
        "iou_t3",
        "iou_t4",
        "mean_f1",
        "f1_t0",
        "f1_t1",
        "f1_t2",
        "f1_t3",
        "f1_t4",
        "filename",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved binary Pred metrics CSV to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
