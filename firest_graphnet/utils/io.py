"""
io.py — I/O and configuration helpers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: str | Path) -> None:
    """Save configuration dict to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    """Save object as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def find_dataset_dirs(
    root: str | Path, max_depth: int = 4
) -> tuple[Optional[Path], Optional[Path]]:
    """Locate dataset_train and dataset_val under *root*.

    Search priority:
      1. root/dataset_train
      2. root/dataset/dataset_train
      3. root/TS-SatFire/dataset/dataset_train
      4. Recursive walk up to *max_depth*
    """
    root = Path(root).resolve()
    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None

    candidates = [
        root,
        root / "dataset",
        root / "TS-SatFire" / "dataset",
    ]
    for base in candidates:
        t = base / "dataset_train"
        v = base / "dataset_val"
        if t.is_dir():
            train_dir = t
        if v.is_dir():
            val_dir = v
        if train_dir and val_dir:
            return train_dir, val_dir

    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        for d in dirnames:
            full = Path(dirpath) / d
            if d == "dataset_train" and train_dir is None:
                train_dir = full
            elif d == "dataset_val" and val_dir is None:
                val_dir = full
        if train_dir and val_dir:
            break

    return train_dir, val_dir


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
