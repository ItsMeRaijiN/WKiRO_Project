"""
run_experiment.py

Train a convolutional autoencoder for gait anomaly detection and sex
recognition based on reconstruction error.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

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

NORMAL_MAP = {"F": 0, "M": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CAE for gait anomaly detection (sex recognition)."
    )
    parser.add_argument("--dataset-root", default=r"E:\ML_datasets\Data_Run_Walk")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--normal-class", choices=["F", "M"], default="M",
                        help="The 'normal' class to train the AE on; the other sex = anomaly. Default M.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent vector dimension (AE bottleneck).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-percentile", type=float, default=95.0,
                        help="Percentile of validation errors used as the anomaly threshold.")
    parser.add_argument("--n-samples", type=int, default=101)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of threads for CSV parsing.")
    parser.add_argument("--cache-file", default="cache/processed_walk_comfortable.npz",
                        help="Path to the processed-data cache (.npz).")
    parser.add_argument("--conditions", nargs="*", default=WALK_CONDITIONS)
    parser.add_argument("--sessions", nargs="*", default=SESSIONS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic data (pipeline test).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_synthetic_dataset(n_female=48, n_male=48, n_samples=101):
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 1.0, n_samples)
    channels = ["LHipAngles_X", "RHipAngles_X", "LKneeAngles_X",
                "RKneeAngles_X", "LAnkleAngles_X", "RAnkleAngles_X"]

    def make_wave(is_male: bool):
        base_phase = rng.uniform(0, 2 * math.pi)
        amplitude = 1.0 if not is_male else 1.25
        shift = 0.0 if not is_male else 0.25
        signal = []
        for idx, _ in enumerate(channels):
            freq = 1.0 + 0.15 * idx
            harmonic = amplitude * np.sin(2 * math.pi * freq * time + base_phase + shift * idx)
            harmonic += 0.25 * np.sin(4 * math.pi * time + 0.3 * idx)
            harmonic += 0.03 * rng.normal(size=n_samples)
            signal.append(harmonic.astype(np.float32))
        return np.stack(signal, axis=-1)

    female = np.stack([make_wave(False) for _ in range(n_female)], axis=0)
    male = np.stack([make_wave(True) for _ in range(n_male)], axis=0)
    X = np.concatenate([female, male], axis=0)
    y = np.concatenate([np.zeros(n_female, dtype=np.int8), np.ones(n_male, dtype=np.int8)])
    gender = ["F"] * n_female + ["M"] * n_male
    meta = [{"subject_id": f"F{i // 4:03d}"} for i in range(n_female)] + \
           [{"subject_id": f"M{i // 4:03d}"} for i in range(n_male)]
    stats = {"F": n_female, "M": n_male, "unknown": 0, "errors": 0}
    return {"X": X, "y": y, "gender": gender, "meta": meta, "channels": channels, "stats": stats}


def _attach_subjects(splits):
    for s in ["train", "val", "test"]:
        splits[s]["subj"] = [m["subject_id"] for m in splits[s]["meta"]]
    return splits


def _process(dataset, normal_label):
    splits = split_dataset(dataset, normal_label=normal_label)
    splits = _attach_subjects(splits)
    norm = compute_normalization_stats(splits["train"]["X"])
    for s in ["train", "val", "test"]:
        splits[s]["X"] = normalize_zscore(splits[s]["X"], norm)
    return splits, norm


def _save_cache(cache_path, splits, norm, normal_label):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=splits["train"]["X"], y_train=splits["train"]["y"],
        X_val=splits["val"]["X"], y_val=splits["val"]["y"],
        X_test=splits["test"]["X"], y_test=splits["test"]["y"],
        gender_test=np.array(splits["test"]["gender"], dtype=object),
        subj_test=np.array(splits["test"]["subj"], dtype=object),
        norm_mean=norm["mean"], norm_std=norm["std"],
        channels=np.array(splits["channels"], dtype=object),
        normal_label=np.array(normal_label),
    )
    print(f"Saved cache: {cache_path}")


def _load_cache(cache_path):
    print(f"Loading cache: {cache_path}")
    c = np.load(cache_path, allow_pickle=True)
    splits = {
        "train": {"X": c["X_train"], "y": c["y_train"]},
        "val":   {"X": c["X_val"], "y": c["y_val"]},
        "test":  {"X": c["X_test"], "y": c["y_test"],
                  "gender": c["gender_test"].tolist(), "subj": c["subj_test"].tolist()},
        "channels": c["channels"].tolist(),
    }
    norm = {"mean": c["norm_mean"], "std": c["norm_std"]}
    normal_label = int(c["normal_label"])
    return splits, norm, normal_label


def get_splits(args, normal_label):
    cache_path = Path(args.cache_file)
    if args.smoke_test:
        print("Synthetic data (smoke test).")
        dataset = build_synthetic_dataset(n_samples=args.n_samples)
        splits, norm = _process(dataset, normal_label)
        return splits, norm, normal_label

    if cache_path.exists():
        splits, norm, cached_label = _load_cache(cache_path)
        if cached_label == normal_label:
            return splits, norm, cached_label
        print(f"[INFO] Cache was built for normal_label={cached_label}, but {normal_label} "
              f"was requested - rebuilding the cache from the dataset.")

    dataset = load_dataset(
        dataset_root=args.dataset_root, conditions=args.conditions, sessions=args.sessions,
        n_samples=args.n_samples, verbose=True, num_workers=args.num_workers,
    )
    splits, norm = _process(dataset, normal_label)
    _save_cache(cache_path, splits, norm, normal_label)
    return splits, norm, normal_label


def subject_level_roc(scores, anomaly, subjects):
    subjects = np.asarray(subjects)
    s_scores, s_labels = [], []
    for s in sorted(set(subjects)):
        m = subjects == s
        s_scores.append(float(scores[m].mean()))
        s_labels.append(int(anomaly[m][0]))
    if len(set(s_labels)) < 2:
        return float("nan"), s_scores, s_labels
    return float(roc_auc_score(s_labels, s_scores)), s_scores, s_labels


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_label = NORMAL_MAP[args.normal_class]
    splits, norm, normal_label = get_splits(args, normal_label)
    anomaly_name = "F" if normal_label == 1 else "M"
    print(f"\nNormal class: {args.normal_class} (label={normal_label}) | anomaly: {anomaly_name}")

    test_y = np.asarray(splits["test"]["y"]).astype(int)
    test_anom = (test_y != normal_label).astype(np.float32)
    test_gender = np.asarray(splits["test"]["gender"])
    test_subj = splits["test"]["subj"]
    train_loader = DataLoader(MocapDataset(splits["train"]["X"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MocapDataset(splits["val"]["X"]), batch_size=args.batch_size)
    test_loader = DataLoader(MocapDataset(splits["test"]["X"], test_anom), batch_size=args.batch_size)
    n_channels = splits["train"]["X"].shape[-1]
    model = Conv1dCAE(n_channels=n_channels, latent_dim=args.latent_dim, seq_len=args.n_samples).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state, best_val_loss = None, float("inf")
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
        {"model_state_dict": model.state_dict(), "norm_mean": norm["mean"], "norm_std": norm["std"],
         "channels": splits["channels"], "normal_label": normal_label, "config": vars(args)},
        output_dir / "best_model.pt",
    )

    val_scores, _ = compute_scores(model, val_loader, device)
    threshold = select_threshold(val_scores, percentile=args.val_percentile)

    test_scores, test_labels = compute_scores(model, test_loader, device)
    metrics = evaluate(test_scores, test_labels, threshold=threshold)
    female_scores = test_scores[test_gender == "F"]
    male_scores = test_scores[test_gender == "M"]
    anomaly_scores = test_scores[test_labels == 1]
    normal_scores = test_scores[test_labels == 0]
    print(f"Female mean error: {female_scores.mean():.6f}" if len(female_scores) else "no F")
    print(f"Male mean error:   {male_scores.mean():.6f}" if len(male_scores) else "no M")

    u_stat, p_value = mannwhitneyu(anomaly_scores, normal_scores, alternative="greater")
    print(f"Mann-Whitney U (anomaly > normal): U={u_stat:.1f}, p={p_value:.3e}")

    subj_roc, _, _ = subject_level_roc(test_scores, test_labels, test_subj)
    print(f"ROC-AUC (subject level): {subj_roc:.4f}")

    metrics.update({
        "normal_class": args.normal_class,
        "anomaly_class": anomaly_name,
        "latent_dim": args.latent_dim,
        "roc_auc_subject_level": subj_roc,
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_value),
        "val_threshold_percentile": args.val_percentile,
        "val_threshold_score": float(threshold),
        "val_loss": float(best_val_loss),
        "score_quantiles_test": [float(x) for x in np.quantile(test_scores, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])],
        "train_samples": int(len(splits["train"]["X"])),
        "val_samples": int(len(splits["val"]["X"])),
        "test_samples": int(len(splits["test"]["X"])),
        "female_mean_test_error": float(female_scores.mean()) if len(female_scores) else None,
        "male_mean_test_error": float(male_scores.mean()) if len(male_scores) else None,
        "n_test_subjects": int(len(set(test_subj))),
    })

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as h:
        json.dump(metrics, h, indent=2, ensure_ascii=False)

    report_path = output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as h:
        h.write("# CAE Report - gait anomaly detection (sex)\n\n")
        h.write("## Configuration\n\n")
        h.write(f"- Condition: Walk_Comfortable (overground), cycles segmented per foot strike\n")
        h.write(f"- Normal class: **{args.normal_class}**, anomaly: **{anomaly_name}**\n")
        h.write(f"- Latent dimension: {args.latent_dim}\n")
        h.write(f"- Samples - train: {metrics['train_samples']}, val: {metrics['val_samples']}, test: {metrics['test_samples']}\n\n")
        h.write("## Results\n\n")
        h.write(f"- **ROC-AUC (cycle): {metrics['roc_auc']:.4f}**\n")
        h.write(f"- **ROC-AUC (subject): {metrics['roc_auc_subject_level']:.4f}** ({metrics['n_test_subjects']} test subjects)\n")
        h.write(f"- Average precision: {metrics['average_precision']:.4f}\n")
        h.write(f"- Threshold ({metrics['val_threshold_percentile']:.0f}th percentile): {metrics['val_threshold_score']:.6f}\n")
        h.write(f"- Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}\n\n")
        h.write("## Statistical analysis of reconstruction error\n\n")
        h.write(f"- Mean error - females: {metrics['female_mean_test_error']:.6f}, "
                f"males: {metrics['male_mean_test_error']:.6f}\n")
        h.write(f"- Mann-Whitney U (anomaly > normal): U={metrics['mannwhitney_u']:.1f}, "
                f"p={metrics['mannwhitney_p']:.3e}\n")
        h.write(f"- Error quantiles: {metrics['score_quantiles_test']}\n\n")
        h.write("## Notes\n\n")
        h.write("- Anomaly score = reconstruction MSE + 0.5 * temporal-derivative MSE.\n")
        h.write("- Subject-wise split\n")

    print(f"\nSaved: {output_dir/'best_model.pt'}, {output_dir/'metrics.json'}, {report_path}")


if __name__ == "__main__":
    main()
