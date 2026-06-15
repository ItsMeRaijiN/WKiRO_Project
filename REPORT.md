# Report: Convolutional autoencoder for female vs male gait recognition

## Goal

Anomaly detection in motion-capture data: train a convolutional autoencoder (CAE)
on one sex's gait and treat the other sex as an anomaly (higher reconstruction error).
Data: *Riglet et al. 2024* (Nature Scientific Data, `s41597-024-03420-y`),
condition `Walk_Comfortable` (overground).

## Key fixes over the baseline version

Several issues were identified and fixed; together they explained the near-random
baseline result (ROC-AUC ≈ 0.53):

1. **Incorrect sex labels.** Two subjects (SA017, TK029) were labeled as male, but per
   `metadata.xlsx` they are female. After the fix: 14 females / 16 males (matching the article).
2. **No cycle segmentation.** A single `Walk_Comfortable{N}.csv` file contains a whole pass
   with **multiple gait cycles** (1-3). The previous loader interpolated the entire
   ~4-second pass to 101 samples, destroying the structure. Now each cycle is segmented by
   foot-strike events (`Right_Foot_Strike`) and normalized to 0-100% (101 samples).
3. **Mixing data blocks.** The CSV file has three blocks (model outputs 100 Hz,
   ground reaction wrench, analog 1000 Hz). The previous loader concatenated them.
   Now only block 1 (joint angles) is read.
4. **Channel selection.** Kept 24 reliable joint-angle channels (Hip/Knee/Ankle/Pelvis
   x L/R x XYZ). Dropped zero / force-plate-dependent channels (KneePower - always 0,
   GroundReactionForce - often 0 in overground).
5. **Bottleneck too wide.** The previous latent (128x26 ≈ 3328) was larger than the input
   (101x24 = 2424), so the AE reconstructed everything. Introduced a vector bottleneck (latent_dim=16).
6. **Training direction.** Training on **males** (there are more of them -> a better-estimated
   "normal" manifold; male gait is more homogeneous) gives much better separation than training
   on females. The task description allows either direction.

## Data after preprocessing

- 474 trials -> **812 individual gait cycles** (379 F / 433 M), 24 channels, 101 samples.
- Subject-wise split (no subject leakage): train/val use only the normal class (M),
  test = remaining males + all females.

## Ceiling test (does the signal exist?)

A supervised classifier (subject-wise 5-fold GroupKFold) on the same cycles:

- **Logistic regression: ROC-AUC = 0.945**
- Random Forest: 0.755
- Range of motion (ROM) alone: 0.754

Reproduce: `python supervised_baseline.py`.

Conclusion: the sex difference is strongly present in the data (nearly linearly separable).
A reconstruction autoencoder, having no labels, learns the dominant variability
(individual gait style) rather than the "sex" direction - hence its score is lower than
the supervised ceiling.

## Final results (AE, trained on males, latent_dim=16)

Evaluation over 10 seeds (different subject-wise splits), mean ± std:

| Metric | Trained on M (anomaly = F) | Trained on F (anomaly = M) |
|---|---|---|
| ROC-AUC (cycle)   | **0.731 ± 0.083** | 0.347 ± 0.098 |
| Average precision | **0.931 ± 0.031** | 0.779 ± 0.026 |
| ROC-AUC (subject) | **0.783 ± 0.110** | 0.285 ± 0.113 |

Reproduce: `python evaluate_seeds.py --seeds 10`.

A single representative run (seed=42, `artifacts/`):

- ROC-AUC (cycle): 0.7611, ROC-AUC (subject): 0.7857, AP: 0.9093
- Accuracy 0.694, Precision 0.859, Recall 0.704, F1 0.774 (threshold = 95th validation percentile)
- Confusion matrix: TN=86, FP=44, FN=112, TP=267 (positive = female / anomaly)
- Mean reconstruction error - females (anomaly): 0.517, males (normal): 0.337
- **Mann-Whitney U (anomaly > normal): p = 3.1x10^-19** (highly significant difference)

## Interpretation

- Fixing the data pipeline and the training direction raised ROC-AUC from ≈ 0.53 to ≈ 0.73-0.78.
- The reconstruction error of females is significantly higher than that of males - the
  autoencoder correctly treats female gait as an anomaly relative to the learned male pattern.
- The large variance across seeds is due to the small number of subjects (30 people, ~3 normal-class
  subjects in the test set per split). This is a dataset limitation, not a method limitation.
- The directional asymmetry (M->0.73 vs F->0.35) suggests female gait deviates from the male
  pattern consistently, while male gait falls within female variability.

## Figures (directory `figures/`)

- `gait_error_dist.png` - reconstruction-error distribution: males (normal) vs females (anomaly)
- `gait_roc.png` - ROC curve (cycle level)
- `gait_training_curve.png` - training curve (train/val)
- `gait_reconstructions.png` - original vs reconstruction of example cycles (angles in degrees)

Generate with: `python visualize.py`.

## Preliminary experiment: MNIST

Method validation on a clean benchmark (per the task description): a 2D convolutional AE
trained on digits 0-8, with digit **9 as the anomaly** (`mnist_experiment.py`).

- ROC-AUC = **0.709**, average precision = 0.177 (latent_dim=8, 25 epochs)
- Mean reconstruction error - normal (0-8): 0.0191, anomaly (9): 0.0270 (+41%)
- Digit 9 is a hard anomaly (visually similar to 4/7/8), hence the moderate AUC - the method
  still correctly assigns a higher error to the anomaly.
- Figures: `figures/mnist_error_hist.png`, `mnist_roc.png`, `mnist_reconstructions.png`.