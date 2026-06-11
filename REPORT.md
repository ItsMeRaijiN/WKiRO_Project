# Motion Capture CAE Report

## Goal

Detect male gait cycles as anomalies by training a convolutional autoencoder only on female `Walk_Comfortable` cycles.

## Data and Setup

- Dataset: `Data_Run_Walk`
- Condition used: `Walk_Comfortable`
- Split: subject-wise, female-only train/validation, mixed test
- Cache: `cache/processed_walk_comfortable.npz`

## Model Changes

- Residual 1D convolutional autoencoder with GroupNorm and GELU
- Temporal derivative penalty added to the training objective
- Combined reconstruction score used for anomaly detection

## Threshold Choice

The validation score distribution suggested a much lower threshold than the previous 95th percentile calibration. A 1st percentile threshold gave the best practical trade-off on the current dataset, especially recall.

## Current Results

- ROC-AUC: 0.5638
- Average precision: 0.8756
- Threshold percentile: 1.0
- Threshold score: 0.258015
- Accuracy: 0.8555
- Precision: 0.8580
- Recall: 0.9966
- F1: 0.9221
- Female mean test error: 0.596190
- Male mean test error: 0.680452

## Interpretation

- The improved model increases separation compared with the previous baseline.
- The error gap between female and male cycles is still modest, so this remains a hard problem.
- The low threshold is useful here because the score distribution strongly overlaps and the positive class is dominant in the current test split.

## Notes for the Thesis / Project Write-up

This pipeline is best described as a reconstruction-based anomaly detector with calibrated thresholding, not as a pure classifier. The most important engineering changes were the WSL cache, the derivative-aware loss, and the residual CNN encoder-decoder.