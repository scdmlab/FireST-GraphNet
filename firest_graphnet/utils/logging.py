"""
logging.py — Lightweight CSV / console metric logger.
"""

from __future__ import annotations

import csv
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricLogger:
    """Accumulates per-epoch metrics and writes them to a CSV file."""

    def __init__(self, save_path: str | Path, resume: bool = False):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        self._fieldnames: List[str] = []

        if resume and self.save_path.exists():
            with open(self.save_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                self._fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    self.rows.append(row)

    def log(self, metrics: Dict[str, Any]) -> None:
        """Append one row of metrics and flush to disk."""
        for k in metrics:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self.rows.append(metrics)
        self._flush()

    def _flush(self) -> None:
        with open(self.save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def last(self) -> Dict[str, Any]:
        return self.rows[-1] if self.rows else {}

    def __len__(self) -> int:
        return len(self.rows)


def print_metrics(epoch: int, metrics: Dict[str, Any], prefix: str = "") -> None:
    """Pretty-print metrics for one epoch."""
    parts = [f"{prefix}Epoch {epoch:>4d}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    print("  |  ".join(parts))
