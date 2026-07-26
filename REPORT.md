# Motion Capture CAE Report
## Required command

```bash
python run_experiment.py \
  --dataset-root "/path/to/Data_Run_Walk" \
  --cache-file "cache/walk_comfortable_101_seed42.npz" \
  --output-dir "artifacts" \
  --epochs 30 \
  --rebuild-cache
```

The generated report includes cycle-level and subject-level metrics.
