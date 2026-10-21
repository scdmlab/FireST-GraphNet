# FireST-GraphNet

Code accompanying **FireST-GraphNet: Physics-informed Spatiotemporal Graph Neural Network for Active Fire Detection from Multi-Source Remote Sensing**.

Authors: Tang Sui, Songxi Yang, Song Gao, Paul C. Stoy, and Qunying Huang.

FireST-GraphNet combines satellite image time series, environmental information, convolutional feature extraction, patch-level graph reasoning, temporal aggregation, and FiLM-conditioned decoding. The project includes data preparation, training, evaluation, and visualization utilities. Shared development code also retains multi-horizon forecasting and weather-provider interfaces; use the configuration and notebook associated with the experiment being run.

## Repository contents

| Path | Contents |
| --- | --- |
| `firest_graphnet/models/` | CNN/CBAM encoder, graph modules, temporal model, and decoders |
| `firest_graphnet/data/` | TS-SatFire loaders, preprocessing, and environmental-data providers |
| `firest_graphnet/training/` | Training loops, losses, normalization, and metrics |
| `firest_graphnet/physics/` | Physics-informed constraints and schedules |
| `firest_graphnet/configs/base.yaml` | Shared starting configuration |
| `firest_graphnet/scripts/` | Training, evaluation, and dataset-statistics entry points |
| `data_tools/` | Data preparation instructions, alignment checks, and Colab workflows |
| `scripts/` | Figure-processing utilities |
| `postprocess_af_inference_exports*.ipynb` | Metrics and visualization from previously exported AF predictions |

The source files retain their original experiment layout. Notebook outputs and execution metadata have been cleared for publication. Local data, checkpoints, credentials, environments, caches, manuscript drafts, and experiment-output directories are not included.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
git clone https://github.com/scdmlab/FireST-GraphNet.git
cd FireST-GraphNet
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CPU/GPU. Optional geospatial preprocessing and weather-download dependencies are listed in `requirements-geospatial.txt`. Some notebook-specific baseline comparisons additionally require MONAI; graph acceleration optionally uses PyTorch Geometric. These dependencies are not needed for every workflow.

## Data

The TS-SatFire dataset is distributed by its original authors:

- Dataset: <https://www.kaggle.com/datasets/z789456sx/ts-satfire>
- Dataset description: <https://doi.org/10.1038/s41597-025-06271-3>
- Upstream preprocessing and benchmark code: <https://github.com/zhaoyutim/TS-SatFire>

Follow `data_tools/README.md` and the guides in `data_tools/docs/` to prepare aligned inputs. Configure paths for your own machine before running the scripts or notebooks. Kaggle authentication must use your own credentials; never commit `kaggle.json`.

The package expects a data root containing `dataset_train/` and `dataset_val/`. Typical array names include:

```text
dataset_train/
  af_train_img_seqtoseq_alll_1i_1.npy
  af_train_label_seqtoseq_alll_1i_1.npy
  firepred_train.npy
  lulc_train.npy
dataset_val/
  af_val_img_seqtoseq_alll_1i_1.npy
  af_val_label_seqtoseq_alll_1i_1.npy
  firepred_val.npy
  lulc_val.npy
```

Use the split layout and label channel required by the chosen experiment. Dataset images, auxiliary arrays, and labels must refer to the same samples. Data and third-party materials retain their original licenses.

## Package entry points

Inspect prepared arrays:

```bash
python -m firest_graphnet.data.inspect_dataset --data_root /path/to/data
```

Run the package's single-step training entry point:

```bash
python -m firest_graphnet.scripts.train --data_root /path/to/data --mode legacy --output_dir outputs/af_run
```

Evaluate a compatible checkpoint:

```bash
python -m firest_graphnet.scripts.eval --data_root /path/to/data --checkpoint /path/to/best_model.pt --mode legacy --output_dir eval_outputs
```

Pass `--config /path/to/config.yaml` to use a specific run configuration. Training supports command-line overrides; inspect `--help` for the available arguments. The single-step and forecast modes use different model settings and checkpoint structures and must be paired with the matching saved configuration.

For the original component smoke check:

```bash
python smoke_test.py
```

The Colab notebooks mount Google Drive and use configurable project/data paths. The AF postprocessing notebooks consume already-exported predictions and ground truth; they do not perform training or inference.

## Experiment reproducibility

`base.yaml` and the command examples above are shared development defaults, not a claim that the default invocation reproduces every table in the manuscript. Match the original experiment's model definition, input/label convention, history length, graph settings, normalization, loss, checkpoint-selection procedure, evaluation split, and calibration settings. In particular, retain the original training configuration when loading a checkpoint and fit any decision-rule parameters on validation data.

Pretrained weights and full experiment outputs are not bundled in this source release. Reproducing numerical results requires the corresponding prepared data, run configuration, and checkpoint or a new training run. No fresh training or reproduction of the reported manuscript scores was performed as part of this repository upload.

## Citation and contact

Please cite the associated manuscript by its title above. Publication details will be added when available. Please also cite TS-SatFire when using the dataset.

Corresponding author: Qunying Huang, <qhuang46@wisc.edu>.

## License

The source code and associated software documentation are distributed under the [MIT License](LICENSE). Copyright (c) 2026 Tang Sui. Third-party data and dependencies retain their original licenses.
