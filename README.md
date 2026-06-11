# Motion Capture CAE Anomaly Detection

Convolutional autoencoder pipeline for detecting gait anomalies in motion capture data. The project follows the idea from the cited paper: train the reconstruction model on female gait cycles and treat male gait cycles as anomalies when their reconstruction error is higher.

## Based On

- Dataset article: https://www.nature.com/articles/s41597-024-03420-y
- Method paper: https://link.springer.com/article/10.1007/s11831-025-10260-5

## Repository Layout

- [data_reader.py](data_reader.py) - CSV discovery, parsing, interpolation, and dataset splitting
- [dataset_torch.py](dataset_torch.py) - PyTorch dataset wrapper
- [model.py](model.py) - 1D convolutional autoencoder
- [train.py](train.py) - train/validation utilities
- [evaluate.py](evaluate.py) - anomaly scoring and metrics
- [run_experiment.py](run_experiment.py) - end-to-end experiment runner
- [requirements.txt](requirements.txt) - Python dependencies

## Requirements

- Python 3.12 was used in the workspace environment
- PyTorch, NumPy, Pandas, SciPy, and scikit-learn
- Access to the dataset directory downloaded from the article

## Expected Dataset Structure

The loader expects a directory layout like this:

```text
Data_Run_Walk/
	SUBJECT_ID/
		Session1/
			Overground_Walk/
				Walk_Comfortable/
					Post_Process/
						Walk_Comfortable1.csv
				Walk_Fast/
				Walk_Slow/
		Session2/
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the full experiment on the real dataset:

```bash
python3 run_experiment.py --dataset-root /path/to/Data_Run_Walk
```

Useful options:

```bash
python3 run_experiment.py --epochs 50 --batch-size 32 --val-percentile 1
```

Run a synthetic smoke test without the real dataset:

```bash
python3 run_experiment.py --smoke-test
```

Outputs are written to `artifacts/`:

- `best_model.pt`
- `metrics.json`
- `report.md`

The processed dataset cache is stored in `cache/processed_walk_comfortable.npz` and reused on later runs.

## What The Pipeline Does

1. Loads gait cycles from female and male subjects.
2. Normalizes each cycle to a fixed length.
3. Splits the data subject-wise so that training and validation use only female cycles, while test includes the remaining female cycles plus all male cycles.
4. Trains a convolutional autoencoder on female cycles.
5. Uses the validation reconstruction error to set an anomaly threshold.
6. Reports ROC-AUC, average precision, and threshold-based classification metrics on the test set.
7. Adds a derivative-based reconstruction penalty and score to better separate gait dynamics.

## Notes

- If the dataset path is wrong, the loader now fails with a clear error.
- If you want, I can also add a small results notebook or a table-ready CSV export of the metrics.
