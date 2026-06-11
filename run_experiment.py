import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_reader import (
    WALK_CONDITIONS,
    SESSIONS,
    compute_normalization_stats,
    load_dataset,
    normalize_zscore,
    split_dataset,
)
from dataset_torch import MocapDataset
from evaluate import compute_scores, evaluate, select_threshold
from model import Conv1dCAE
from train import train_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder for motion-capture anomaly detection."
    )
    parser.add_argument("--dataset-root", default="/mnt/e/ML_datasets/Data_Run_Walk")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-percentile", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=101)
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for CSV loading.")
    parser.add_argument("--cache-file", default="cache/processed_walk_comfortable.npz", help="Path to a cached processed dataset on WSL.")
    parser.add_argument("--conditions", nargs="*", default=WALK_CONDITIONS)
    parser.add_argument("--sessions", nargs="*", default=SESSIONS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Use a synthetic dataset for validation.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_synthetic_dataset(n_female=48, n_male=24, n_samples=101):
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 1.0, n_samples)
    channels = [
        "LHipAngles_X",
        "RHipAngles_X",
        "LKneeAngles_X",
        "RKneeAngles_X",
        "LAnkleAngles_X",
        "RAnkleAngles_X",
    ]

    def make_wave(is_male: bool):
        base_phase = rng.uniform(0, 2 * math.pi)
        amplitude = 1.0 if not is_male else 1.25
        shift = 0.0 if not is_male else 0.25
        signal = []
        for idx, _ in enumerate(channels):
            freq = 1.0 + 0.15 * idx
            harmonic = amplitude * np.sin(2 * math.pi * freq * time + base_phase + shift * idx)
            harmonic += 0.25 * np.sin(4 * math.pi * time + 0.3 * idx)
            if is_male:
                harmonic += 0.08 * np.cos(8 * math.pi * time + 0.2 * idx)
            else:
                harmonic += 0.03 * rng.normal(size=n_samples)
            signal.append(harmonic.astype(np.float32))
        return np.stack(signal, axis=-1)

    female = np.stack([make_wave(False) for _ in range(n_female)], axis=0)
    male = np.stack([make_wave(True) for _ in range(n_male)], axis=0)
    X = np.concatenate([female, male], axis=0)
    y = np.concatenate([np.zeros(n_female, dtype=np.int8), np.ones(n_male, dtype=np.int8)])
    gender = ["F"] * n_female + ["M"] * n_male
    meta = [
        {"subject_id": f"F{i:03d}", "session": "synthetic", "condition": "Walk_Comfortable", "cycle_num": i + 1}
        for i in range(n_female)
    ] + [
        {"subject_id": f"M{i:03d}", "session": "synthetic", "condition": "Walk_Comfortable", "cycle_num": i + 1}
        for i in range(n_male)
    ]
    stats = {"F": n_female, "M": n_male, "unknown": 0, "errors": 0}
    return {"X": X, "y": y, "gender": gender, "meta": meta, "channels": channels, "stats": stats}


def prepare_dataset(args):
    cache_path = Path(args.cache_file)

    if args.smoke_test:
        print("Using synthetic smoke-test dataset.")
        dataset = build_synthetic_dataset(n_samples=args.n_samples)
        splits = {
            "train": {"X": dataset["X"][:32], "y": dataset["y"][:32], "gender": dataset["gender"][:32], "meta": dataset["meta"][:32]},
            "val": {"X": dataset["X"][32:40], "y": dataset["y"][32:40], "gender": dataset["gender"][32:40], "meta": dataset["meta"][32:40]},
            "test": {"X": dataset["X"][40:], "y": dataset["y"][40:], "gender": dataset["gender"][40:], "meta": dataset["meta"][40:]},
            "channels": dataset["channels"],
        }
        return dataset, splits

    if cache_path.exists():
        print(f"Loading processed dataset cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)

        splits = {
            "train": {"X": cache["X_train"], "y": cache["y_train"]},
            "val": {"X": cache["X_val"], "y": cache["y_val"]},
            "test": {"X": cache["X_test"], "y": cache["y_test"]},
            "channels": cache["channels"].tolist(),
        }
        dataset = {
            "X": np.concatenate([splits["train"]["X"], splits["val"]["X"], splits["test"]["X"]], axis=0),
            "y": np.concatenate([splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]], axis=0),
            "gender": None,
            "meta": None,
            "channels": splits["channels"],
            "stats": {"cached": True},
        }
        return dataset, splits

    dataset = load_dataset(
        dataset_root=args.dataset_root,
        conditions=args.conditions,
        sessions=args.sessions,
        n_samples=args.n_samples,
        verbose=True,
        num_workers=args.num_workers,
    )
    splits = split_dataset(dataset)

    norm = compute_normalization_stats(splits["train"]["X"])
    for split_name in ["train", "val", "test"]:
        splits[split_name]["X"] = normalize_zscore(splits[split_name]["X"], norm)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=splits["train"]["X"],
        y_train=splits["train"]["y"],
        X_val=splits["val"]["X"],
        y_val=splits["val"]["y"],
        X_test=splits["test"]["X"],
        y_test=splits["test"]["y"],
        norm_mean=norm["mean"],
        norm_std=norm["std"],
        channels=np.array(splits["channels"], dtype=object),
    )
    print(f"Saved processed dataset cache to {cache_path}")
    return dataset, splits


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, splits = prepare_dataset(args)

    cache_path = Path(args.cache_file)
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        norm = {"mean": cache["norm_mean"], "std": cache["norm_std"]}
    else:
        norm = compute_normalization_stats(splits["train"]["X"])
        for split_name in ["train", "val", "test"]:
            splits[split_name]["X"] = normalize_zscore(splits[split_name]["X"], norm)

    train_ds = MocapDataset(splits["train"]["X"])
    val_ds = MocapDataset(splits["val"]["X"], splits["val"]["y"])
    test_ds = MocapDataset(splits["test"]["X"], splits["test"]["y"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    n_channels = splits["train"]["X"].shape[-1]
    model = Conv1dCAE(n_channels=n_channels, latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=1.0)
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_mean": norm["mean"],
            "norm_std": norm["std"],
            "channels": splits["channels"],
            "config": vars(args),
        },
        output_dir / "best_model.pt",
    )

    val_scores, _ = compute_scores(model, val_loader, device)
    threshold = select_threshold(val_scores, percentile=args.val_percentile)

    test_scores, test_labels = compute_scores(model, test_loader, device)
    metrics = evaluate(test_scores, test_labels, threshold=threshold)

    female_scores = test_scores[test_labels == 0]
    male_scores = test_scores[test_labels == 1]
    if len(female_scores):
        print(f"Female mean error: {female_scores.mean():.6f}")
    if len(male_scores):
        print(f"Male mean error:   {male_scores.mean():.6f}")

    metrics.update(
        {
            "val_threshold_percentile": args.val_percentile,
            "val_threshold_score": float(threshold),
            "val_loss": float(best_val_loss),
            "score_quantiles_val": [float(x) for x in np.quantile(val_scores, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])],
            "score_quantiles_test": [float(x) for x in np.quantile(test_scores, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])],
            "train_samples": int(len(splits["train"]["X"])),
            "val_samples": int(len(splits["val"]["X"])),
            "test_samples": int(len(splits["test"]["X"])),
            "female_mean_test_error": float(female_scores.mean()) if len(female_scores) else None,
            "male_mean_test_error": float(male_scores.mean()) if len(male_scores) else None,
        }
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved checkpoint to: {output_dir / 'best_model.pt'}")
    print(f"Saved metrics to:    {output_dir / 'metrics.json'}")

    report_path = output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Motion Capture CAE Report\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- Dataset: Walk_Comfortable only\n")
        handle.write(f"- Train samples: {metrics['train_samples']}\n")
        handle.write(f"- Val samples: {metrics['val_samples']}\n")
        handle.write(f"- Test samples: {metrics['test_samples']}\n")
        handle.write(f"- ROC-AUC: {metrics['roc_auc']:.4f}\n")
        handle.write(f"- AP: {metrics['average_precision']:.4f}\n")
        handle.write(f"- Threshold percentile: {metrics['val_threshold_percentile']:.1f}\n")
        handle.write(f"- Threshold score: {metrics['val_threshold_score']:.6f}\n\n")
        handle.write("## Score Distribution\n\n")
        handle.write(f"- Validation quantiles: {metrics['score_quantiles_val']}\n")
        handle.write(f"- Test quantiles: {metrics['score_quantiles_test']}\n")
        handle.write(f"- Female mean test error: {metrics['female_mean_test_error']:.6f}\n")
        handle.write(f"- Male mean test error: {metrics['male_mean_test_error']:.6f}\n\n")
        handle.write("## Notes\n\n")
        handle.write("- Reconstruction score uses both signal MSE and derivative MSE.\n")
        handle.write("- Threshold is calibrated from the validation error distribution using a low percentile selected from score analysis.\n")
    print(f"Saved report to:        {report_path}")


if __name__ == "__main__":
    main()