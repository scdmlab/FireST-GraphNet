# FireST-GraphNet

**Wildfire Progression Prediction via a Temporal Graph Neural Network with Sequence Learning**

---

## Overview

FireST-GraphNet predicts next-day active fire masks from multi-spectral satellite imagery by combining:

1. **CNN Encoder + CBAM** — extracts spatial features with channel & spatial attention
2. **Graph Neural Network** — constructs a patch-level graph and propagates information via kNN message passing
3. **LSTM Temporal Model** — captures temporal dynamics across input frames
4. **FiLM-conditioned Decoder** — fuses temporal context with spatial features for pixel-level prediction

## Project Structure

```
firest_graphnet/
├── configs/
│   └── base.yaml                  # Default hyper-parameters
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # TSSatFireSequenceDataset
│   ├── inspect_dataset.py         # Mandatory dataset inspector
│   ├── transforms.py              # Augmentation & normalisation
│   ├── weather.py                 # HRRR weather provider (placeholder)
│   └── download_hrrr.md           # HRRR download instructions
├── models/
│   ├── __init__.py
│   ├── cbam.py                    # CBAM attention module
│   ├── encoder.py                 # Multi-source CNN encoder
│   ├── graph_modules.py           # Graph construction + GNN layers
│   └── firest_graphnet.py         # Full FireSTGraphNet model
├── training/
│   ├── __init__.py
│   ├── losses.py                  # BCEWithLogitsLoss + pos_weight
│   ├── metrics.py                 # IoU, F1, Precision, Recall
│   └── trainer.py                 # Training loop
├── scripts/
│   ├── __init__.py
│   ├── train.py                   # Training entry-point
│   └── eval.py                    # Evaluation entry-point
├── utils/
│   ├── __init__.py
│   ├── seed.py                    # Reproducibility
│   ├── io.py                      # Config & file I/O
│   └── logging.py                 # CSV metric logger
└── README.md
```

## Dataset (TS-SatFire)

After running the inspector, the dataset has the following structure:

| File | Shape | Description |
|------|-------|-------------|
| `af_train_img_seqtoseq_alll_1i_1.npy` | (2254, 8, 1, 256, 256) | 2254 samples × 8 bands × 256² |
| `af_train_label_seqtoseq_alll_1i_1.npy` | (2254, 3, 1, 256, 256) | Ch0: mask, Ch1: **binary fire** {0,1}, Ch2: FRP |
| `af_val_img_seqtoseq_alll_1i_1.npy` | (278, 8, 1, 256, 256) | Validation images |
| `af_val_label_seqtoseq_alll_1i_1.npy` | (278, 3, 1, 256, 256) | Validation labels |

**8 bands** include reflectance-range bands (0–130) and thermal bands (270–380 K).

**Class imbalance:** ~2.4% fire pixels on average → handled via `pos_weight` in the loss.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pandas pyyaml tqdm matplotlib
# Optional (recommended) for graph acceleration:
pip install torch-geometric
```

### 2. Inspect Dataset

```bash
python -m firest_graphnet.data.inspect_dataset --data_root "C:\Users\22125\Desktop\FireST-GraphNet"
```

### 3. Train

```bash
python -m firest_graphnet.scripts.train --data_root "C:\Users\22125\Desktop\FireST-GraphNet"
```

With custom parameters:

```bash
python -m firest_graphnet.scripts.train \
    --data_root "C:\Users\22125\Desktop\FireST-GraphNet" \
    --batch_size 2 \
    --epochs 30 \
    --lr 0.0005 \
    --patch_size 16 \
    --k 8 \
    --T_in 9 \
    --device cuda
```

### 4. Evaluate

```bash
python -m firest_graphnet.scripts.eval \
    --data_root "C:\Users\22125\Desktop\FireST-GraphNet" \
    --checkpoint outputs/best_model.pt \
    --save_vis
```

## Outputs

After training, the `outputs/` directory contains:

| File | Content |
|------|---------|
| `config.yaml` | Saved configuration |
| `metrics.csv` | Per-epoch train/val metrics |
| `best_model.pt` | Best checkpoint (by val IoU) |
| `final_model.pt` | Final epoch weights |
| `checkpoint_epoch*.pt` | Periodic checkpoints |
| `vis/` | Prediction vs ground-truth images |

## Architecture Details

### (A) CNN Encoder + CBAM

- 3 Conv blocks: 8→32→64→128 channels
- Each block: Conv2d → BatchNorm → GELU → CBAM (optional)
- CBAM: channel attention (squeeze-excite) + spatial attention (7×7 conv)

### (B) Graph Module

- Feature map divided into 8×8 patches → 1024 graph nodes
- Node features: mean-pooled patch embeddings
- kNN graph (k=8) constructed per sample
- 2 GNN message-passing layers with residual connections
- Outputs upsampled back to full resolution
- **Fallback:** pure-PyTorch message passing if `torch_geometric` is unavailable

### (C) LSTM Temporal Model

- Global average-pooled node embeddings per timestep → (T_in, 128)
- 2-layer LSTM with hidden size 256
- Last hidden state used as temporal context

### (D) FiLM Decoder

- 3 Conv blocks: 128→128→64→32→1
- FiLM conditioning: LSTM output generates per-channel γ and β
- Final 1×1 conv → raw logits

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch_geometric` | Install via `pip install torch-geometric` or the model falls back to pure PyTorch automatically |
| `FileNotFoundError: dataset_train` | Check `--data_root` points to the directory containing `dataset_train/` |
| Out of memory (CUDA OOM) | Reduce `--batch_size` to 1–2, or use `--patch_size 16` for fewer graph nodes |
| Training loss not decreasing | Try lower `--lr` (e.g., 0.0001), or increase `pos_weight` in config |
| All predictions are zero | Increase `pos_weight` (default 10), check dataset has fire pixels |
| NaN in loss | Check for NaN in input data; some tmp files contain NaN — only `af_*` files are used |
| Windows path issues | Use raw strings: `--data_root "C:\Users\..."` |
| `metrics.csv` not updating | Ensure `outputs/` directory is writable |

## Citation

If you use this code, please cite:

```
@article{firest_graphnet2026,
    title={FireST-GraphNet: Wildfire Progression Prediction via a Temporal
           Graph Neural Network with Sequence Learning},
    year={2026}
}
```

## License

Research use only.
