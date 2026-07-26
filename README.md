# Motion Capture CAE Anomaly Detection

Research prototype for one-class anomaly detection in gait-cycle motion-capture
data. A residual 1D convolutional autoencoder (CAE) is trained on gait cycles
from female participants and evaluated on held-out female and male
participants.

## Pipeline

The end-to-end pipeline:

1. Finds post-processed recording CSV files for overground walking.
2. Extracts selected lower-limb kinematic and kinetic channels.
3. Segments recordings into gait cycles, fills missing values, smooths each
   signal and resamples every cycle to 101 time points.
4. Splits data by participant to avoid subject leakage:
   - training: female participants only;
   - validation: different female participants only;
   - test: remaining female participants and all male participants.
5. Computes per-channel Z-score statistics on the training split only.
6. Trains a residual 1D CAE with GroupNorm, GELU and a temporal-derivative
   reconstruction penalty.
7. Calculates anomaly scores and reports ROC-AUC, average precision, accuracy,
   precision, recall, F1 and the confusion matrix.

The anomaly score is:

```text
score = MSE(x, reconstruction) + 0.5 * MSE(Δx, Δreconstruction)
```

Higher scores are treated as more anomalous.

## Data source

The project targets the dataset described in:

- [3D motion analysis dataset of healthy young adult volunteers walking and running on overground and treadmill](https://doi.org/10.1038/s41597-024-03420-y)
  (Riglet et al., *Scientific Data*, 2024).
- [Dataset archive on Figshare](https://doi.org/10.6084/m9.figshare.c.7056797.v1),
  including `metadata.xlsx` and `Data_Run_Walk.zip`.

Download the dataset by following the article's **Data Records** section. The
main experiment consumes post-processed CSV files, not the raw C3D files.

By default, the loader uses both sessions and the `Walk_Comfortable`
overground condition. It expects this structure:

```text
Data_Run_Walk/
└── AJ026/
    ├── Session1/
    │   └── Overground_Walk/
    │       └── Walk_Comfortable/
    │           └── Post_Process/
    │               ├── Walk_Comfortable1.csv
    │               ├── Walk_Comfortable2.csv
    │               └── ...
    └── Session2/
        └── Overground_Walk/
            └── Walk_Comfortable/
                └── Post_Process/
                    └── Walk_Comfortable1.csv
```

Only files matching `<condition><recording-number>.csv` are loaded. Participant
sex is resolved with the fixed 30-participant map in
[`data_reader.py`](data_reader.py); unknown participant IDs are skipped.

## Requirements

- Python 3.12
- c3d
- NumPy
- pandas
- SciPy
- scikit-learn
- PyTorch

A CUDA-capable PyTorch installation is optional. The runner selects CUDA when
available and otherwise uses the CPU.

## Installation

From the repository root:

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux, macOS or WSL

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file pins the versions used by the verified environment.

## Quick start

### Smoke test

```bash
python run_experiment.py \
  --smoke-test \
  --epochs 1 \
  --output-dir artifacts_smoke
```

### Real-data experiment

```powershell
python run_experiment.py --dataset-root "E:\path\to\Data_Run_Walk" --cache-file "cache\walk_comfortable_101_seed42.npz" --output-dir "artifacts" --epochs 30
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Threshold selection

`--val-percentile` selects a percentile of reconstruction scores from the
female-only validation split. A test sample is classified as anomalous when its
score is greater than or equal to that threshold.

## Command-line options

Run `python run_experiment.py --help` for the complete CLI. The main options
are:

| Option              |                            Default | Meaning                                        |
|---------------------|-----------------------------------:|------------------------------------------------|
| `--dataset-root`    | `/mnt/e/ML_datasets/Data_Run_Walk` | Dataset directory                              |
| `--output-dir`      |                        `artifacts` | Checkpoint and report directory                |
| `--epochs`          |                               `30` | Training epochs                                |
| `--batch-size`      |                               `32` | DataLoader batch size                          |
| `--lr`              |                            `0.001` | Adam learning rate                             |
| `--latent-channels` |                              `128` | Latent channel count                           |
| `--seed`            |                               `42` | NumPy and PyTorch seed                         |
| `--val-percentile`  |                             `95.0` | Validation-score threshold percentile          |
| `--n-samples`       |                              `101` | Time points per normalized gait cycle          |
| `--num-workers`     |                                `4` | Threads used while loading CSV files           |
| `--rebuild-cache`   |                                off | Explicitly replace an existing processed cache |
| `--conditions`      |                 `Walk_Comfortable` | Conditions to include                          |
| `--sessions`        |                `Session1 Session2` | Sessions to include                            |
| `--device`          |   CUDA if available, otherwise CPU | PyTorch device                                 |
| `--smoke-test`      |                                off | Use synthetic data                             |

## Method reference

The autoencoder design is informed by general reconstruction-based anomaly
detection concepts discussed in:

- [Deep Autoencoder Neural Networks: A Comprehensive Review and New Perspectives](https://doi.org/10.1007/s11831-025-10260-5)
  (Mienye and Swart, *Archives of Computational Methods in Engineering*, 2025).

This review is background material; it does not define the sex-based experiment
implemented in this repository.
