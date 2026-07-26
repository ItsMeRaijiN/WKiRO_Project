import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_reader import (
    EXPECTED_FEATURE_COLUMNS,
    GENDER_MAP,
    SESSIONS,
    WALK_CONDITIONS,
    compute_normalization_stats,
    load_dataset,
    normalize_zscore,
    split_dataset,
)
from dataset_torch import MocapDataset
from evaluate import (
    aggregate_scores_by_subject,
    compute_scores,
    evaluate,
    select_threshold,
)
from model import Conv1dCAE
from train import train_epoch, validate

CACHE_SCHEMA_VERSION = 2


class CacheValidationError(ValueError):
    """Raised when a processed cache cannot be safely reused."""
def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def _percentile(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 100:
        raise argparse.ArgumentTypeError("must be in the interval (0, 100]")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder for motion-capture anomaly detection."
    )
    parser.add_argument("--dataset-root", default="/mnt/e/ML_datasets/Data_Run_Walk")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--latent-channels", type=_positive_int, default=128)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--val-percentile", type=_percentile, default=95.0)
    parser.add_argument("--n-samples", type=_positive_int, default=101)
    parser.add_argument(
        "--num-workers",
        type=_positive_int,
        default=4,
        help="Number of worker threads for CSV loading.",
    )
    parser.add_argument(
        "--cache-file",
        default="cache/processed_walk_comfortable.npz",
        help="Path to a versioned processed-dataset cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Explicitly overwrite an existing cache with freshly processed data.",
    )
    parser.add_argument("--conditions", nargs="+", default=WALK_CONDITIONS)
    parser.add_argument("--sessions", nargs="+", default=SESSIONS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a synthetic dataset for integration testing.",
    )
    args = parser.parse_args(argv)
    if args.n_samples < 2:
        parser.error("--n-samples must be at least 2")
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_synthetic_dataset(
    n_female: int = 48,
    n_male: int = 24,
    n_samples: int = 101,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 1.0, n_samples)
    channels = [
        "LHipAngles_X",
        "RHipAngles_X",
        "LKneeAngles_X",
        "RKneeAngles_X",
        "LAnkleAngles_X",
        "RAnkleAngles_X",
    ]

    def make_wave(is_male: bool) -> np.ndarray:
        base_phase = rng.uniform(0, 2 * math.pi)
        amplitude = 1.0 if not is_male else 1.25
        shift = 0.0 if not is_male else 0.25
        signal = []
        for index, _ in enumerate(channels):
            frequency = 1.0 + 0.15 * index
            harmonic = amplitude * np.sin(
                2 * math.pi * frequency * time_axis + base_phase + shift * index
            )
            harmonic += 0.25 * np.sin(4 * math.pi * time_axis + 0.3 * index)
            if is_male:
                harmonic += 0.08 * np.cos(8 * math.pi * time_axis + 0.2 * index)
            else:
                harmonic += 0.03 * rng.normal(size=n_samples)
            signal.append(harmonic.astype(np.float32))
        return np.stack(signal, axis=-1)

    female = np.stack([make_wave(False) for _ in range(n_female)], axis=0)
    male = np.stack([make_wave(True) for _ in range(n_male)], axis=0)
    X = np.concatenate([female, male], axis=0)
    y = np.concatenate(
        [np.zeros(n_female, dtype=np.int8), np.ones(n_male, dtype=np.int8)]
    )
    gender = ["F"] * n_female + ["M"] * n_male
    meta = [
        {
            "subject_id": f"F{index:03d}",
            "session": "synthetic",
            "condition": "Walk_Comfortable",
            "cycle_num": 1,
        }
        for index in range(n_female)
    ] + [
        {
            "subject_id": f"M{index:03d}",
            "session": "synthetic",
            "condition": "Walk_Comfortable",
            "cycle_num": 1,
        }
        for index in range(n_male)
    ]
    stats = {"F": n_female, "M": n_male, "unknown": 0, "errors": 0}
    return {
        "X": X,
        "y": y,
        "gender": gender,
        "meta": meta,
        "channels": channels,
        "stats": stats,
    }


def _synthetic_splits(dataset: dict) -> dict:
    return {
        "train": {
            "X": dataset["X"][:32],
            "y": dataset["y"][:32],
            "gender": dataset["gender"][:32],
            "meta": dataset["meta"][:32],
        },
        "val": {
            "X": dataset["X"][32:40],
            "y": dataset["y"][32:40],
            "gender": dataset["gender"][32:40],
            "meta": dataset["meta"][32:40],
        },
        "test": {
            "X": dataset["X"][40:],
            "y": dataset["y"][40:],
            "gender": dataset["gender"][40:],
            "meta": dataset["meta"][40:],
        },
        "channels": dataset["channels"],
    }


def _cache_metadata(args) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve(strict=False)),
        "conditions": list(args.conditions),
        "sessions": list(args.sessions),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "selected_channels": EXPECTED_FEATURE_COLUMNS,
        "gender_map": GENDER_MAP,
    }


def _subject_ids(split: dict) -> np.ndarray:
    return np.asarray([item["subject_id"] for item in split["meta"]], dtype=np.str_)


def _validate_processed_splits(splits: dict, norm: dict, n_samples: int) -> None:
    channels = list(splits["channels"])
    if not channels:
        raise CacheValidationError("Cache does not contain any channel names.")

    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        X = np.asarray(split["X"])
        y = np.asarray(split["y"])
        if X.ndim != 3 or X.shape[0] == 0:
            raise CacheValidationError(f"Cache split '{split_name}' is empty or malformed.")
        if X.shape[1:] != (n_samples, len(channels)):
            raise CacheValidationError(
                f"Cache split '{split_name}' has shape {X.shape}; expected "
                f"(N, {n_samples}, {len(channels)})."
            )
        if len(y) != len(X) or len(split["meta"]) != len(X):
            raise CacheValidationError(
                f"Cache split '{split_name}' has inconsistent sample metadata."
            )
        if not np.isfinite(X).all():
            raise CacheValidationError(f"Cache split '{split_name}' is not finite.")

    if not np.all(splits["train"]["y"] == 0) or not np.all(
        splits["val"]["y"] == 0
    ):
        raise CacheValidationError("Training and validation caches must be female-only.")
    if set(np.unique(splits["test"]["y"])) != {0, 1}:
        raise CacheValidationError("Test cache must contain both labels 0 and 1.")

    expected_norm_shape = (1, 1, len(channels))
    if (
        np.asarray(norm["mean"]).shape != expected_norm_shape
        or np.asarray(norm["std"]).shape != expected_norm_shape
    ):
        raise CacheValidationError("Cache normalization statistics are incompatible.")
    if not np.isfinite(norm["mean"]).all() or not np.isfinite(norm["std"]).all():
        raise CacheValidationError("Cache normalization statistics are not finite.")
    if np.any(np.asarray(norm["std"]) <= 0):
        raise CacheValidationError("Cache normalization standard deviations must be positive.")


def save_processed_cache(
    cache_path: Path,
    splits: dict,
    norm: dict,
    metadata: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=splits["train"]["X"],
        y_train=splits["train"]["y"],
        subject_ids_train=_subject_ids(splits["train"]),
        X_val=splits["val"]["X"],
        y_val=splits["val"]["y"],
        subject_ids_val=_subject_ids(splits["val"]),
        X_test=splits["test"]["X"],
        y_test=splits["test"]["y"],
        subject_ids_test=_subject_ids(splits["test"]),
        norm_mean=norm["mean"],
        norm_std=norm["std"],
        channels=np.asarray(splits["channels"], dtype=np.str_),
        cache_metadata=np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            dtype=np.str_,
        ),
    )


def load_processed_cache(
    cache_path: Path,
    expected_metadata: dict,
) -> tuple[dict, dict]:
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            required_keys = {
                "X_train",
                "y_train",
                "subject_ids_train",
                "X_val",
                "y_val",
                "subject_ids_val",
                "X_test",
                "y_test",
                "subject_ids_test",
                "norm_mean",
                "norm_std",
                "channels",
                "cache_metadata",
            }
            missing = required_keys - set(cache.files)
            if missing:
                raise CacheValidationError(
                    "Cache uses an obsolete or incomplete schema; missing: "
                    + ", ".join(sorted(missing))
                )

            actual_metadata = json.loads(str(cache["cache_metadata"].item()))
            if actual_metadata != expected_metadata:
                raise CacheValidationError(
                    "Cache configuration does not match this run. "
                    "Use a different --cache-file or pass --rebuild-cache."
                )

            channels = cache["channels"].astype(str).tolist()
            splits = {"channels": channels}
            for split_name in ("train", "val", "test"):
                X = cache[f"X_{split_name}"].astype(np.float32, copy=True)
                raw_y = cache[f"y_{split_name}"].copy()
                if not np.isin(raw_y, [0, 1]).all():
                    raise CacheValidationError(
                        f"Cache split '{split_name}' contains invalid labels."
                    )
                y = raw_y.astype(np.int8, copy=False)
                subject_ids = cache[f"subject_ids_{split_name}"].astype(str).tolist()
                splits[split_name] = {
                    "X": X,
                    "y": y,
                    "gender": ["F" if label == 0 else "M" for label in y],
                    "meta": [{"subject_id": subject_id} for subject_id in subject_ids],
                }
            norm = {
                "mean": cache["norm_mean"].astype(np.float32, copy=True),
                "std": cache["norm_std"].astype(np.float32, copy=True),
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, CacheValidationError):
            raise
        raise CacheValidationError(f"Cannot read processed cache: {exc}") from exc

    _validate_processed_splits(splits, norm, expected_metadata["n_samples"])
    return splits, norm


def _normalize_splits(splits: dict) -> dict:
    norm = compute_normalization_stats(splits["train"]["X"])
    for split_name in ("train", "val", "test"):
        splits[split_name]["X"] = normalize_zscore(splits[split_name]["X"], norm)
    return norm


def prepare_dataset(args) -> tuple[dict, dict, dict]:
    if args.smoke_test:
        print("Using synthetic smoke-test dataset.")
        dataset = build_synthetic_dataset(
            n_samples=args.n_samples,
            seed=args.seed,
        )
        splits = _synthetic_splits(dataset)
        norm = _normalize_splits(splits)
        return dataset, splits, norm

    cache_path = Path(args.cache_file)
    metadata = _cache_metadata(args)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading validated processed dataset cache from {cache_path}")
        splits, norm = load_processed_cache(cache_path, metadata)
        dataset = {
            "X": None,
            "y": None,
            "gender": None,
            "meta": None,
            "channels": splits["channels"],
            "stats": {"cached": True},
        }
        return dataset, splits, norm

    dataset = load_dataset(
        dataset_root=args.dataset_root,
        conditions=args.conditions,
        sessions=args.sessions,
        n_samples=args.n_samples,
        verbose=True,
        num_workers=args.num_workers,
    )
    splits = split_dataset(dataset, seed=args.seed)
    norm = _normalize_splits(splits)
    _validate_processed_splits(splits, norm, args.n_samples)

    if cache_path.exists():
        print(f"Overwriting processed cache after explicit --rebuild-cache: {cache_path}")
    save_processed_cache(cache_path, splits, norm, metadata)
    print(f"Saved processed dataset cache to {cache_path}")
    return dataset, splits, norm


def _prefixed_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{args.device}' was requested, but CUDA is unavailable."
        )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, splits, norm = prepare_dataset(args)

    train_ds = MocapDataset(splits["train"]["X"])
    val_ds = MocapDataset(splits["val"]["X"], splits["val"]["y"])
    test_ds = MocapDataset(splits["test"]["X"], splits["test"]["y"])

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    n_channels = splits["train"]["X"].shape[-1]
    model = Conv1dCAE(
        n_channels=n_channels,
        latent_channels=args.latent_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=1.0,
        )
        val_loss = validate(model, val_loader, device)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError("Training produced a non-finite loss.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    if best_state is None or best_epoch is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_mean": torch.from_numpy(np.asarray(norm["mean"])),
            "norm_std": torch.from_numpy(np.asarray(norm["std"])),
            "channels": list(splits["channels"]),
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        output_dir / "best_model.pt",
    )

    val_scores, _ = compute_scores(model, val_loader, device)
    threshold = select_threshold(val_scores, percentile=args.val_percentile)

    test_scores, test_labels = compute_scores(model, test_loader, device)
    cycle_metrics = evaluate(
        test_scores,
        test_labels,
        threshold=threshold,
        title="CYCLE-LEVEL EVALUATION",
    )

    val_subject_scores, _ = aggregate_scores_by_subject(
        val_scores,
        splits["val"]["y"],
        splits["val"]["meta"],
    )
    subject_threshold = select_threshold(
        val_subject_scores,
        percentile=args.val_percentile,
    )
    test_subject_scores, test_subject_labels = aggregate_scores_by_subject(
        test_scores,
        test_labels,
        splits["test"]["meta"],
    )
    subject_metrics = evaluate(
        test_subject_scores,
        test_subject_labels,
        threshold=subject_threshold,
        title="SUBJECT-LEVEL EVALUATION",
    )

    female_scores = test_scores[test_labels == 0]
    male_scores = test_scores[test_labels == 1]
    print(f"Female mean cycle error: {female_scores.mean():.6f}")
    print(f"Male mean cycle error:   {male_scores.mean():.6f}")

    metrics = {
        **_prefixed_metrics(cycle_metrics, "cycle"),
        **_prefixed_metrics(subject_metrics, "subject"),
        "val_threshold_percentile": args.val_percentile,
        "cycle_val_threshold_score": float(threshold),
        "subject_val_threshold_score": float(subject_threshold),
        "best_epoch": int(best_epoch),
        "val_loss": float(best_val_loss),
        "score_quantiles_val": [
            float(value)
            for value in np.quantile(
                val_scores,
                [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            )
        ],
        "score_quantiles_test": [
            float(value)
            for value in np.quantile(
                test_scores,
                [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            )
        ],
        "train_samples": len(splits["train"]["X"]),
        "val_samples": len(splits["val"]["X"]),
        "test_samples": len(splits["test"]["X"]),
        "test_subjects": len(test_subject_scores),
        "female_mean_test_error": float(female_scores.mean()),
        "male_mean_test_error": float(male_scores.mean()),
        "conditions": list(args.conditions),
        "sessions": list(args.sessions),
        "seed": args.seed,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\nSaved checkpoint to: {output_dir / 'best_model.pt'}")
    print(f"Saved metrics to:    {output_dir / 'metrics.json'}")

    report_path = output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Motion Capture CAE Report\n\n")
        handle.write("## Configuration\n\n")
        handle.write(f"- Conditions: {', '.join(args.conditions)}\n")
        handle.write(f"- Sessions: {', '.join(args.sessions)}\n")
        handle.write(f"- Seed: {args.seed}\n")
        handle.write(f"- Best epoch: {best_epoch}\n")
        handle.write(f"- Train cycles: {metrics['train_samples']}\n")
        handle.write(f"- Validation cycles: {metrics['val_samples']}\n")
        handle.write(f"- Test cycles: {metrics['test_samples']}\n")
        handle.write(f"- Test subjects: {metrics['test_subjects']}\n\n")
        handle.write("## Cycle-level metrics\n\n")
        handle.write(f"- ROC-AUC: {metrics['cycle_roc_auc']:.4f}\n")
        handle.write(f"- Average precision: {metrics['cycle_average_precision']:.4f}\n")
        handle.write(f"- F1: {metrics['cycle_f1']:.4f}\n")
        handle.write(f"- Threshold: {metrics['cycle_val_threshold_score']:.6f}\n\n")
        handle.write("## Subject-level metrics\n\n")
        handle.write(f"- ROC-AUC: {metrics['subject_roc_auc']:.4f}\n")
        handle.write(
            f"- Average precision: {metrics['subject_average_precision']:.4f}\n"
        )
        handle.write(f"- F1: {metrics['subject_f1']:.4f}\n")
        handle.write(f"- Threshold: {metrics['subject_val_threshold_score']:.6f}\n\n")
        handle.write("## Notes\n\n")
        handle.write(
            "- Reconstruction score combines signal MSE and derivative MSE.\n"
        )
        handle.write(
            "- Thresholds are calibrated independently on female-only validation "
            "cycle and subject scores.\n"
        )
    print(f"Saved report to:      {report_path}")


if __name__ == "__main__":
    main()
