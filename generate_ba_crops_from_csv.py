from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path

# Keep batch generation stable in Conda/OpenMP-heavy environments on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from visualize_illustration_ba import build_figure, load_multihorizon_gt
from visualize_roi_utils import format_event_id, load_roi


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate cropped BA figures from a results CSV using cached ROIs."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="figures/task_ba/ba_results.csv",
        help="CSV file describing BA figure settings.",
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
        help="Dataset directory for loading BA events.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon passed to BA loader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the BA figure generator.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/crop/task_ba",
        help="Directory for cropped BA figures.",
    )
    parser.add_argument(
        "--skip-missing-roi",
        action="store_true",
        help="Skip rows whose event ROI is not cached instead of failing.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    event_ids = OrderedDict()
    for row in rows:
        event_idx = int(row["event_idx"])
        event_ids[event_idx] = None

    gt_by_event = {}
    roi_by_event = {}
    missing_events: list[str] = []

    for event_idx in event_ids:
        event_id = format_event_id(event_idx)
        roi = load_roi(event_id, args.roi_cache)
        if roi is None:
            missing_events.append(event_id)
            continue
        roi_by_event[event_idx] = roi
        print(f"Loading {event_id} with ROI {roi} ...")
        gt_by_event[event_idx] = load_multihorizon_gt(args.data_dir, event_idx, args.horizon)

    if missing_events and not args.skip_missing_roi:
        missing = ", ".join(missing_events)
        raise FileNotFoundError(
            f"Missing cached ROI for: {missing}. "
            "Select ROI for those events first or rerun with --skip-missing-roi."
        )

    written = 0
    skipped = 0
    for row in rows:
        event_idx = int(row["event_idx"])
        filename = row["filename"]
        if event_idx not in gt_by_event:
            skipped += 1
            print(f"Skipping {filename}: ROI for {format_event_id(event_idx)} is not cached.")
            continue

        output_path = out_dir / filename
        print(
            f"Generating {filename} "
            f"(event={event_idx}, quality={row['quality']}, fragmentation={row['fragmentation']})"
        )
        build_figure(
            gt_by_event[event_idx],
            seed=args.seed,
            quality=float(row["quality"]),
            fragmentation=float(row["fragmentation"]),
            save_path=str(output_path),
            dpi=args.dpi,
            show=False,
            roi=roi_by_event[event_idx],
            auto_crop=False,
        )
        written += 1

    print(f"\nWrote {written} cropped BA figure(s) to: {out_dir.resolve()}")
    if skipped:
        print(f"Skipped {skipped} row(s) because their ROI is not cached yet.")


if __name__ == "__main__":
    main()
