"""
Build `postprocess_af_inference_exports_v1.ipynb` by cloning the V2 notebook
structure cell-for-cell and patching only the V1-specific cells:

  - Cell 0  (title markdown)             — V1 focus
  - Cell 6  (path configuration code)    — V1 candidate roots
  - Cell 10 (scanning code)              — V1 filename + path depth + summary
                                           lookup for ckpt / split

All other cells are copied verbatim so the two notebooks stay visually
identical and the same CSV schema / PNG layout is produced.
"""
import json
from pathlib import Path

SRC = Path(r"C:\Users\22125\Desktop\FireST-GraphNet\postprocess_af_inference_exports.ipynb")
OUT = Path(r"C:\Users\22125\Desktop\FireST-GraphNet\postprocess_af_inference_exports_v1.ipynb")

nb = json.loads(SRC.read_text(encoding="utf-8"))


def _set_code(cell_idx: int, new_src: str) -> None:
    assert nb["cells"][cell_idx]["cell_type"] == "code", \
        f"cell {cell_idx} is not code"
    nb["cells"][cell_idx]["source"] = new_src.splitlines(keepends=True)
    nb["cells"][cell_idx]["outputs"] = []
    nb["cells"][cell_idx]["execution_count"] = None


def _set_md(cell_idx: int, new_src: str) -> None:
    assert nb["cells"][cell_idx]["cell_type"] == "markdown", \
        f"cell {cell_idx} is not markdown"
    nb["cells"][cell_idx]["source"] = new_src.splitlines(keepends=True)


# ---------------------------------------------------------------------------
# Cell 0 (title) — reflect V1 focus
# ---------------------------------------------------------------------------
_set_md(0, """# FireST-GraphNet  •  **V1** AF post-processing visualization + evaluation

This notebook is a **post-processing notebook only** — it does *not* rerun training or inference. It reads the saved **V1 AF (Active Fire) one-step inference outputs** produced by the legacy V1 exporter (under `.../output/inference_exports_af/`) and generates, for every sample discovered under `PATH1`:

1. one publication-quality PNG per sample (Ground Truth / Prediction / Confusion Map) → `PATH1/visualizations/`
2. one per-sample metrics CSV (F1, IoU, Accuracy, precision/recall, TP/TN/FP/FN, …) → `PATH1/sample_metrics.csv`
3. a small summary CSV with mean / median of F1, IoU, Accuracy → `PATH1/sample_metrics_summary.csv`

### V1 vs V2 layout — what's different?

| aspect | V2 (postprocess_af_inference_exports.ipynb) | **V1 (this notebook)** |
| --- | --- | --- |
| filename | `run27__best_model__train__sample_000000__pred.npy` | `sample00_idx00000_pred.npy` |
| path depth from root | `{run}/{ckpt}/{split}/predictions/file` (5 parts) | `{run}/predictions/file` (3 parts) |
| `run` / `ckpt` / `split` source | parsed from filename | parsed from path + `inference_summary.json` |
| threshold | usually present in `inference_summary.json` | absent → default `0.5` |
| sample_idx | `sample_000000` | `sample00_idx00000` (idx wins) |

Every other cell is identical to the V2 post-processor so the CSV schema, metrics, and PNG layout are unchanged.

Notebook cell organisation:

1. environment / imports
2. Google Drive mount
3. path configuration
4. helper utilities
5. recursive file scanning  *(V1-aware)*
6. data loading logic
7. metric computation functions
8. confusion map generation
9. per-sample visualization export
10. CSV export
11. summary statistics
12. optional quick preview""")


# ---------------------------------------------------------------------------
# Cell 6 — Path configuration (V1 roots)
# ---------------------------------------------------------------------------
_set_code(6, r'''# ---- candidate roots (first existing one wins) ----
# V1 AF inference exports land under `<PROJECT>/output/inference_exports_af/`
# (or `outputs/` — the user's Drive folder is plural).  Either is accepted.
_CANDIDATE_ROOTS = [
    # Colab / Google Drive copies (plural `outputs/` first, per user setup)
    "/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean/outputs/inference_exports_af",
    "/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean/output/inference_exports_af",
    "/content/drive/MyDrive/FireST-GraphNet-Clean/output/inference_exports_af",
    "/content/drive/MyDrive/FireST-GraphNet-Clean/outputs/inference_exports_af",
    # Local OneDrive copy (Windows workstation)
    r"C:\Users\22125\OneDrive - UW-Madison\FireST-GraphNet-Clean\output\inference_exports_af",
    # Linux / generic fallback
    "./output/inference_exports_af",
    "./inference_exports_af",
]

def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

PATH1 = _first_existing(_CANDIDATE_ROOTS)
if PATH1 is None:
    PATH1 = Path(_CANDIDATE_ROOTS[0])
    print(f"[warn] none of the candidate roots exist -- defaulting PATH1 to {PATH1}")

# ---- derived output locations ----
VIS_DIR     = PATH1 / "visualizations_postprocess"   # keep V1 exporter's own viz/ untouched
METRICS_CSV = PATH1 / "sample_metrics.csv"
SUMMARY_CSV = PATH1 / "sample_metrics_summary.csv"

# ---- threshold ----
# V1 `inference_summary.json` does NOT record a threshold.  Override here if
# you re-binarise at a different cutoff.
DEFAULT_THRESHOLD = 0.5

# ---- visualisation parameters ----
FIG_DPI       = 220
FIG_SIZE      = (6.0, 14.0)
OVERWRITE_VIZ = False

# ---- preview limits ----
PREVIEW_N = 3

VIS_DIR.mkdir(parents=True, exist_ok=True)

print("PATH1         :", PATH1)
print("VIS_DIR       :", VIS_DIR)
print("METRICS_CSV   :", METRICS_CSV)
print("SUMMARY_CSV   :", SUMMARY_CSV)
print("threshold dflt:", DEFAULT_THRESHOLD)
''')


# ---------------------------------------------------------------------------
# Cell 10 — Scanning logic (V1-aware parse + sample_idx + summary lookup)
# ---------------------------------------------------------------------------
_set_code(10, r'''# Supported prediction / ground-truth file extensions.
_PRED_EXTS = (".npy", ".npz", ".pt", ".pth", ".png")
_GT_EXTS   = (".npy", ".npz", ".pt", ".pth", ".png")

# Filename suffixes that strongly indicate the file type.
# NB: `_hist_last` is *not* in either list, so those files are ignored.
_PRED_SUFFIXES = ("__pred", "_pred", ".pred", "-pred", "__prediction", "_prediction")
_GT_SUFFIXES   = ("__gt", "_gt", ".gt", "-gt", "__target", "_target",
                  "__label", "_label", "__mask", "_mask")


def _strip_pred_suffix(stem: str) -> Optional[str]:
    for s in _PRED_SUFFIXES:
        if stem.endswith(s):
            return stem[: -len(s)]
    return None

def _strip_gt_suffix(stem: str) -> Optional[str]:
    for s in _GT_SUFFIXES:
        if stem.endswith(s):
            return stem[: -len(s)]
    return None


def _candidate_gt_paths(pred_path: Path) -> List[Path]:
    """Given a prediction file path, list plausible matching GT file paths."""
    stem = pred_path.stem
    base = _strip_pred_suffix(stem)
    if base is None:
        base = stem
    out: List[Path] = []
    pred_parent = pred_path.parent
    alt_parents = [pred_parent]
    if pred_parent.name.lower() in {"predictions", "preds", "pred"}:
        alt_parents.append(pred_parent.parent / "ground_truth")
        alt_parents.append(pred_parent.parent / "groundtruth")
        alt_parents.append(pred_parent.parent / "gt")
        alt_parents.append(pred_parent.parent / "targets")
        alt_parents.append(pred_parent.parent / "labels")
    for parent in alt_parents:
        for suf in _GT_SUFFIXES:
            for ext in _GT_EXTS:
                cand = parent / f"{base}{suf}{ext}"
                out.append(cand)
        for ext in _GT_EXTS:
            out.append(parent / f"{base}{ext}")
    seen = set(); uniq: List[Path] = []
    for c in out:
        k = str(c).lower()
        if k in seen:
            continue
        seen.add(k); uniq.append(c)
    return uniq


def _iter_candidate_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for ext in _PRED_EXTS:
        out.extend(root.rglob(f"*{ext}"))
    return out


# ── V1-specific metadata cache (one inference_summary.json per run folder) ──
_SUMMARY_CACHE: Dict[Path, Dict[str, Any]] = {}

def _find_run_summary(pred_path: Path, root: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Walk up from the prediction file looking for an inference_summary.json.
    Caches the parsed dict per containing directory."""
    for up in (pred_path.parent, pred_path.parent.parent, pred_path.parent.parent.parent):
        if up == root.parent or up == up.parent:
            break
        cand = up / "inference_summary.json"
        if cand.exists():
            if cand not in _SUMMARY_CACHE:
                try:
                    _SUMMARY_CACHE[cand] = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    _SUMMARY_CACHE[cand] = {}
            return up, _SUMMARY_CACHE[cand]
    return None, None


def _parse_run_ckpt_split(pred_path: Path, root: Path) -> Dict[str, Optional[str]]:
    """V1-aware parser.

    Priority order:
      1. V2-style filename `run__ckpt__split__sample_IDX` (kept for safety so this
         notebook also works if a V2-style export sneaks in).
      2. V1 layout: path parts `{run}/predictions/{file}`; `ckpt` and `split`
         come from the sibling `inference_summary.json`.
      3. Fallback: path components if there are enough of them.
    """
    info: Dict[str, Optional[str]] = {
        "run": None, "ckpt": None, "split": None, "sample_idx": None,
    }
    stem = pred_path.stem
    base = _strip_pred_suffix(stem) or stem

    # V2-style filename -------------------------------------------------------
    m = re.match(
        r"^(?P<run>[^_]+)__(?P<ckpt>[^_]+)__(?P<split>[^_]+)__sample_(?P<idx>\d+)$",
        base,
    )
    if m:
        info["run"]        = m.group("run")
        info["ckpt"]       = m.group("ckpt")
        info["split"]      = m.group("split")
        info["sample_idx"] = int(m.group("idx"))
        return info

    # V1 layout ---------------------------------------------------------------
    try:
        rel = pred_path.relative_to(root).parts
    except ValueError:
        rel = pred_path.parts

    # `.../{run}/predictions/{file}` → run = rel[-3]
    if len(rel) >= 3 and rel[-2].lower() in {"predictions", "preds", "pred"}:
        info["run"] = rel[-3]
    elif len(rel) >= 5:
        info["run"]   = rel[-5]
        info["ckpt"]  = rel[-4]
        info["split"] = rel[-3]

    # sample_idx: V1 uses `sampleNN_idxNNNNN_pred` → prefer the `idxNNNNN` token.
    m_idx = re.search(r"idx[_-]?(\d+)", base)
    if m_idx:
        info["sample_idx"] = int(m_idx.group(1))
    else:
        m_idx = re.search(r"sample[_-]?(\d+)", base)
        if m_idx:
            info["sample_idx"] = int(m_idx.group(1))

    # ckpt / split via sibling inference_summary.json -------------------------
    run_dir, summary = _find_run_summary(pred_path, root)
    if summary:
        if not info["split"] and summary.get("split"):
            info["split"] = str(summary["split"])
        if not info["ckpt"]:
            ckpt_val = summary.get("checkpoint") or summary.get("checkpoint_path")
            if ckpt_val:
                info["ckpt"] = Path(str(ckpt_val)).stem    # e.g. "best_model"
        if not info["run"] and summary.get("run_name"):
            info["run"] = str(summary["run_name"])

    # Conservative defaults when summary is missing
    if info["ckpt"] is None:
        info["ckpt"] = "best_model"
    return info


def _guess_threshold_for(pred_path: Path) -> float:
    """V1 inference_summary.json does NOT store a threshold; return default.
    Kept API-compatible with the V2 notebook so downstream code is unchanged."""
    for up in (pred_path.parent, pred_path.parent.parent, pred_path.parent.parent.parent):
        cand = up / "inference_summary.json"
        if cand.exists():
            try:
                meta = json.loads(cand.read_text(encoding="utf-8"))
                for k in ("threshold", "thr", "thr_best", "best_threshold"):
                    if k in meta and meta[k] is not None:
                        return float(meta[k])
            except Exception:
                pass
    return DEFAULT_THRESHOLD


def scan_samples(root: Path) -> List[Dict[str, Any]]:
    """Build a list of {pred, gt, meta} records discovered under `root`."""
    cand = _iter_candidate_files(root)
    cand = [p for p in cand
            if VIS_DIR not in p.parents and p != METRICS_CSV and p != SUMMARY_CSV]
    pred_files = [p for p in cand if _strip_pred_suffix(p.stem) is not None]
    pred_files.sort()

    recs: List[Dict[str, Any]] = []
    for p in pred_files:
        gt_path: Optional[Path] = None
        for g in _candidate_gt_paths(p):
            if g.exists():
                gt_path = g
                break
        info = _parse_run_ckpt_split(p, root)
        thr  = _guess_threshold_for(p)
        recs.append({
            "pred_path": p,
            "gt_path": gt_path,
            "threshold": thr,
            **info,
        })
    return recs


samples = scan_samples(PATH1)
n_with_gt = sum(1 for r in samples if r["gt_path"] is not None)
log(f"Discovered {len(samples)} prediction file(s); {n_with_gt} have a matching GT.")
if samples:
    log("First 3 records:")
    for r in samples[:3]:
        log(f"  pred={r['pred_path'].name}  gt={r['gt_path'].name if r['gt_path'] else '-'}  "
            f"run={r['run']} ckpt={r['ckpt']} split={r['split']} idx={r['sample_idx']} thr={r['threshold']}")
''')


# ---------------------------------------------------------------------------
# Serialise
# ---------------------------------------------------------------------------
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

# Round-trip validation
nb2 = json.loads(OUT.read_text(encoding="utf-8"))
assert len(nb2["cells"]) == len(nb["cells"]), "cell count mismatch"
print(f"WROTE {OUT}")
print(f"cells: {len(nb['cells'])} (markdown {sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}, "
      f"code {sum(1 for c in nb['cells'] if c['cell_type']=='code')})")
