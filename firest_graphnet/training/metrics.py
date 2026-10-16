"""
metrics.py — Pixel-level evaluation metrics for fire masks and probability fields.

Supports:
    - Single-step   : compute_metrics(logits, targets)  → aggregate dict
    - Multi-horizon : compute_metrics_per_horizon(logits, targets) → {h: dict}
    - Pred task     : compute_nmae / compute_nmae_per_horizon
                      — normalized MAE on the continuous probability field,
                        rescaled with training-set global statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from firest_graphnet.training.normalization import (
    DEFAULT_PROB_STATS,
    GlobalProbStats,
    nmae as _nmae,
    normalize_to_unit,
)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute IoU, F1, Precision, Recall from raw logits and binary targets.

    Accepts both (B, 1, H, W) and (B, H, 1, Hp, Wp) shapes —
    the latter is flattened across the horizon dimension.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + eps)).item()

    intersection = tp
    union = tp + fp + fn
    iou = (intersection / (union + eps)).item()

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def compute_metrics_per_horizon(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[int, Dict[str, float]]:
    """Per-horizon metrics.

    Parameters
    ----------
    logits  : (B, H, 1, Hp, Wp)
    targets : (B, H, 1, Hp, Wp)

    Returns
    -------
    dict  {horizon_step: {iou, f1, precision, recall}}
    """
    H = logits.shape[1]
    result = {}
    for h in range(H):
        result[h] = compute_metrics(
            logits[:, h], targets[:, h], threshold=threshold, eps=eps
        )
    return result


def compute_nmae(
    logits: torch.Tensor,
    targets: torch.Tensor,
    stats: Optional[GlobalProbStats] = None,
) -> float:
    """Normalized MAE on the Pred continuous probability field.

    Both prediction (sigmoid of `logits`) and `targets` are rescaled to
    [0, 1] with training-set global min/max before the absolute error is
    averaged.  Shapes: any broadcastable pair (B, 1, H, W) or (B, H, 1, Hp, Wp).
    """
    probs = torch.sigmoid(logits)
    return _nmae(probs, targets, stats or DEFAULT_PROB_STATS)


def compute_nmae_per_horizon(
    logits: torch.Tensor,
    targets: torch.Tensor,
    stats: Optional[GlobalProbStats] = None,
) -> Dict[int, float]:
    """Per-horizon nMAE for (B, H, 1, Hp, Wp) logits / targets."""
    H = logits.shape[1]
    probs = torch.sigmoid(logits)
    s = stats or DEFAULT_PROB_STATS
    return {h: _nmae(probs[:, h], targets[:, h], s) for h in range(H)}


def metrics_summary_str(
    per_horizon: Dict[int, Dict[str, float]],
    max_show: int = 5,
) -> str:
    """Pretty-print per-horizon metrics (first/last few steps)."""
    lines = []
    keys = sorted(per_horizon.keys())
    show = keys[:max_show] + (["..."] if len(keys) > max_show * 2 else []) + keys[-max_show:]
    for k in show:
        if k == "...":
            lines.append("  ...")
            continue
        m = per_horizon[k]
        lines.append(
            f"  h={k:2d}  IoU={m['iou']:.4f}  F1={m['f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}"
        )
    return "\n".join(lines)
