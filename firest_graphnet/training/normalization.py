"""
normalization.py — Global dataset-level normalization utilities for the Pred task.

Design decision (documented, reproducible):
    We use *global* (dataset-level) min-max normalization on the training-set
    ground-truth probability field, then rescale BOTH ground truth and
    prediction into [0, 1] with the SAME learned statistics before computing
    any error metric.  This approach is preferred over per-event or per-frame
    normalization because:

      1. Global statistics are stored once; the same transform is applied at
         train / val / test / deployment time — avoiding data leakage.
      2. Values on [0, 1] are comparable across events, horizons, and
         weather-conditioning regimes — fire events have very different peak
         probability magnitudes (a small grass fire vs. a megafire), so
         per-event scaling would hide that heterogeneity.
      3. Per-frame normalization couples the metric scale to the prediction
         itself (the predicted max enters the divisor), which is unstable when
         a prediction collapses toward zero and makes ablations comparable
         only on matched event sets.

For the FireST-GraphNet probability task the canonical range is [0, 1]
(pre-sigmoid logits are passed through torch.sigmoid).  The learned training
statistics typically yield x_min ≈ 0.0 and x_max ≈ 1.0; this module makes
that explicit, and generalises cleanly if the output is ever expressed in a
non-probability scale (e.g. brightness temperature, radiative power).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ═══════════════════════════════════════════════════════════════════════════
#  Global statistics container
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class GlobalProbStats:
    """Training-set statistics used for the Pred-task min-max normalization.

    Attributes
    ----------
    x_min : float   training-set minimum of the GT probability field.
    x_max : float   training-set maximum of the GT probability field.
    source : str    description of how the statistics were derived
                    (e.g. "train split, af_to_prob(gt_af)" or "probability-range default").
    n_events : int  number of training events used to derive statistics.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    source: str = "probability-range default ([0,1])"
    n_events: int = 0

    @property
    def span(self) -> float:
        span = float(self.x_max) - float(self.x_min)
        return span if span > 1e-8 else 1.0  # guard against degenerate span

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalProbStats":
        return cls(**d)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GlobalProbStats":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# Canonical defaults for the probability-map case.  These are the values used
# throughout the thesis when per-dataset training statistics have not been
# explicitly computed; they make the normalization an identity on [0, 1].
DEFAULT_PROB_STATS = GlobalProbStats(
    x_min=0.0,
    x_max=1.0,
    source="probability-range default ([0,1])",
    n_events=0,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Core normalization
# ═══════════════════════════════════════════════════════════════════════════
def normalize_to_unit(x: ArrayLike, stats: Optional[GlobalProbStats] = None) -> ArrayLike:
    """Apply global min-max normalization to [0, 1].

    x_norm = clip((x - x_min) / (x_max - x_min), 0, 1)

    Accepts NumPy arrays or PyTorch tensors; returns the same type.
    """
    if stats is None:
        stats = DEFAULT_PROB_STATS

    if _HAS_TORCH and hasattr(x, "clamp"):
        return ((x - stats.x_min) / stats.span).clamp(0.0, 1.0)
    x_arr = np.asarray(x, dtype=np.float32)
    return np.clip((x_arr - stats.x_min) / stats.span, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Normalized MAE (primary Pred-task metric)
# ═══════════════════════════════════════════════════════════════════════════
def nmae(
    pred: ArrayLike,
    target: ArrayLike,
    stats: Optional[GlobalProbStats] = None,
) -> float:
    """Normalized Mean Absolute Error on the Pred task.

    Both `pred` and `target` are first rescaled to [0, 1] using the SAME
    training-set global statistics, then the pixel-wise absolute error is
    averaged.  Output is a scalar float in [0, 1].

    Formally
    --------
        x̂ = clip((pred   - x_min) / (x_max - x_min), 0, 1)
        x  = clip((target - x_min) / (x_max - x_min), 0, 1)
        nMAE = (1 / N) * Σ_i |x̂_i - x_i|

    For the probability-map case (default stats) this reduces to the usual
    MAE on clipped [0, 1] probability maps, but the normalization step is
    now explicit, documented, and reproducible — enabling consistent metric
    comparisons across tasks, events, and weather-conditioning regimes.
    """
    p = normalize_to_unit(pred, stats)
    t = normalize_to_unit(target, stats)
    if _HAS_TORCH and hasattr(p, "abs"):
        return float((p - t).abs().mean().item())
    return float(np.mean(np.abs(p - t)))


def nmae_per_frame(
    pred_seq: ArrayLike,
    target_seq: ArrayLike,
    stats: Optional[GlobalProbStats] = None,
) -> list:
    """Per-frame normalized MAE for a (T, H, W) sequence."""
    pred_seq = np.asarray(pred_seq, dtype=np.float32)
    target_seq = np.asarray(target_seq, dtype=np.float32)
    T = pred_seq.shape[0]
    return [nmae(pred_seq[t], target_seq[t], stats) for t in range(T)]


def nmae_per_horizon_batch(
    pred_b: "torch.Tensor",
    target_b: "torch.Tensor",
    stats: Optional[GlobalProbStats] = None,
) -> dict:
    """Per-horizon nMAE for batched forecast output.

    Parameters
    ----------
    pred_b   : (B, H, 1, Hp, Wp)  predicted probabilities (post-sigmoid) or logits
    target_b : (B, H, 1, Hp, Wp)  ground-truth probability field
    stats    : GlobalProbStats    training-set normalization statistics

    Note: `pred_b` is assumed to be probabilities (sigmoid applied).  If
    logits are passed the caller MUST apply sigmoid first; nMAE is only
    meaningful on the probability scale.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for nmae_per_horizon_batch")
    import torch as _T
    H = pred_b.shape[1]
    out = {}
    for h in range(H):
        out[h] = nmae(pred_b[:, h], target_b[:, h], stats)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics builder  (training-set only — leakage-free)
# ═══════════════════════════════════════════════════════════════════════════
def compute_global_stats(
    prob_iterable: Iterable[np.ndarray],
    source_desc: str = "train split, GT probability field",
    clip_range: tuple = (0.0, 1.0),
) -> GlobalProbStats:
    """Compute global (dataset-level) min / max statistics from an iterable of
    training ground-truth probability frames.

    Parameters
    ----------
    prob_iterable : iterable of np.ndarray
        Training GT probability frames (any shape).  Only finite pixels are
        considered; NaNs are ignored.
    source_desc : str    human-readable description of the source.
    clip_range : tuple   clamp final (x_min, x_max) into this range so that
                         numerical noise does not push statistics outside
                         the canonical [0,1] probability interval.

    Returns
    -------
    GlobalProbStats
    """
    x_min_acc = +np.inf
    x_max_acc = -np.inf
    n = 0
    for frame in prob_iterable:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.size == 0:
            continue
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        x_min_acc = min(x_min_acc, float(np.min(arr[finite])))
        x_max_acc = max(x_max_acc, float(np.max(arr[finite])))
        n += 1

    if n == 0:
        return DEFAULT_PROB_STATS

    x_min = float(np.clip(x_min_acc, clip_range[0], clip_range[1]))
    x_max = float(np.clip(x_max_acc, clip_range[0], clip_range[1]))
    if x_max - x_min < 1e-6:
        x_min, x_max = clip_range[0], clip_range[1]

    return GlobalProbStats(
        x_min=x_min,
        x_max=x_max,
        source=source_desc,
        n_events=n,
    )
