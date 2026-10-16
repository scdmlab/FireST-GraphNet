"""
compute_pred_stats.py — Build training-set global min/max for the Pred task.

This CLI pre-computes the `GlobalProbStats` JSON consumed by
`firest_graphnet.training.normalization` to normalize the Pred probability
field into [0, 1] before nMAE is computed.

Because the FireST-GraphNet Pred output IS a probability field (sigmoid of
logits), the canonical range is [0, 1].  Running this script on the training
split documents the empirically observed (x_min, x_max) once, and that JSON
is then reused at all eval points (code, figures, paper tables) — ensuring a
single reproducible normalization across the entire experimental protocol.

Usage:
    python -m firest_graphnet.scripts.compute_pred_stats \
        --data-dir dataset_train \
        --out firest_graphnet/configs/pred_stats_train.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from firest_graphnet.training.normalization import (
    GlobalProbStats,
    compute_global_stats,
)


def _iter_train_gt_prob(data_dir: Path):
    """Yield ground-truth probability frames from the training split.

    The training label file contains binary AF masks; for the Pred task these
    are converted to a soft probability field via the same `af_to_prob` used
    at evaluation time.  We import lazily so this module remains usable
    even when the visualization stack (scipy/matplotlib) is unavailable.
    """
    from visualize_illustration_pred import af_to_prob  # lazy

    for name in [
        "af_train_label_seqtoseq_alll_1i_1.npy",
        "af_val_label_seqtoseq_alll_1i_1.npy",
    ]:
        p = data_dir / name
        if p.exists():
            labels = np.load(str(p), mmap_mode="r")
            n = labels.shape[0]
            for i in range(n):
                if labels.ndim == 5:
                    obs = np.array(labels[i, 1, 0], dtype=np.float32)
                elif labels.ndim == 4:
                    obs = np.array(labels[i, 1],    dtype=np.float32)
                else:
                    continue
                obs = np.clip(obs, 0, 1)
                if (obs >= 0.5).sum() < 50:
                    continue
                yield af_to_prob(obs)
            return
    raise FileNotFoundError(f"No label file found in {data_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Build training-set GlobalProbStats JSON for nMAE."
    )
    p.add_argument("--data-dir", type=str, default="dataset_train",
                   help="Training-split directory (leakage-free).")
    p.add_argument("--out", type=str,
                   default="firest_graphnet/configs/pred_stats_train.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    print(f"[stats] Scanning {data_dir} ...")
    stats = compute_global_stats(
        _iter_train_gt_prob(data_dir),
        source_desc=f"train split ({data_dir.name}), af_to_prob(gt_af)",
    )
    print(f"[stats] n_events={stats.n_events}  "
          f"x_min={stats.x_min:.6f}  x_max={stats.x_max:.6f}")

    stats.save(out_path)
    print(f"[stats] saved -> {out_path}")


if __name__ == "__main__":
    main()
