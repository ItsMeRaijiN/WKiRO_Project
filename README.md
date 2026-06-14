# Convolutional autoencoder for female vs male gait recognition

Anomaly detection in motion-capture data using a convolutional autoencoder (CAE).
The autoencoder learns one sex's gait (the "normal" class) and detects the other sex
as an anomaly based on a higher reconstruction error.

## Sources

- Dataset: https://www.nature.com/articles/s41597-024-03420-y (figshare DOI `10.6084/m9.figshare.c.7056797`)
- Method: https://link.springer.com/article/10.1007/s11831-025-10260-5

## Files

- [data_reader.py](data_reader.py) - CSV loading, event parsing, **per-cycle segmentation**, subject-wise split, normalization
- [dataset_torch.py](dataset_torch.py) - PyTorch `Dataset` wrapper
- [model.py](model.py) - 1D convolutional autoencoder with a vector bottleneck
- [train.py](train.py) - training loop (loss = MSE + derivative MSE)
- [evaluate.py](evaluate.py) - anomaly scoring and metrics (ROC-AUC, AP, threshold)
- [run_experiment.py](run_experiment.py) - full end-to-end experiment
- [mnist_experiment.py](mnist_experiment.py) - preliminary MNIST anomaly experiment (digit 9)
- [visualize.py](visualize.py) - generate figures for the gait experiment
- [evaluate_seeds.py](evaluate_seeds.py) - multi-seed evaluation (both directions, mean +/- std)
- [supervised_baseline.py](supervised_baseline.py) - supervised ceiling (subject-wise CV)
- [REPORT.md](REPORT.md) - results and analysis

## Data

The dataset is a single `Data_Run_Walk.zip` file (~28 GB) from figshare. Only the
`Walk_Comfortable` (overground) condition is needed for the experiment - it is enough to
extract the files `*/Session*/Overground_Walk/Walk_Comfortable/Post_Process/Walk_Comfortable*.csv`
and `metadata.xlsx`, preserving the directory structure:

```text
Data_Run_Walk/
  AJ026/
    Session1/Overground_Walk/Walk_Comfortable/Post_Process/Walk_Comfortable1.csv
    Session2/...
  metadata.xlsx
```

## Running

Full experiment (default: train on males, latent 16):

```powershell
python run_experiment.py --dataset-root "E:\ML_datasets\Data_Run_Walk" --normal-class M
```

Useful options:

```powershell
python run_experiment.py --epochs 60 --latent-dim 16 --normal-class M --val-percentile 95
```

Smoke test on synthetic data (no dataset needed):

```powershell
python run_experiment.py --smoke-test
```

Generate figures and run the MNIST preliminary experiment:

```powershell
python visualize.py
python mnist_experiment.py --latent-dim 8 --epochs 25
```

Reproduce the numbers cited in the report:

```powershell
python evaluate_seeds.py --seeds 10        # multi-seed mean +/- std, both directions
python supervised_baseline.py              # supervised ceiling (subject-wise CV)
```

Outputs go to `artifacts/`: `best_model.pt`, `metrics.json`, `report.md`.
Processed data is cached in `cache/processed_walk_comfortable.npz`

## What the pipeline does

1. Loads gait trials and **segments them into individual cycles** (foot strike -> next), normalizing each to 101 samples.
2. Selects 24 joint-angle channels (Hip/Knee/Ankle/Pelvis x L/R x XYZ).
3. Splits the data subject-wise: train/val = only the normal class; test = the rest of the normal class + the entire anomaly class.
4. Z-score normalizes using training statistics.
5. Trains a bottlenecked CAE on the normal class.
6. Computes a threshold from the validation errors (percentile) and test metrics:
   ROC-AUC and AP (threshold-free), accuracy/precision/recall/F1, plus
   subject-level ROC-AUC and a Mann-Whitney test of the error difference.

## Results (summary)

Trained on males, latent 16 - single run (seed 42): **ROC-AUC 0.76 (cycle) / 0.79 (subject), AP 0.91**;
over 10 seeds: **0.73 ± 0.08 (cycle), 0.78 ± 0.11 (subject)**. The MNIST preliminary experiment
(digit 9 as anomaly) reaches ROC-AUC 0.71. Full analysis and figures in [REPORT.md](REPORT.md).
