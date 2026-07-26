"""
CSV file structure:
  Row 0: metadata (FrameNumber, FirstFrame, PointFrequency, AnalogFrequency).
  Following rows: foot-strike/off event times in seconds.
  Next: channel names, units, X/Y/Z axes, and the numeric point-data table.
  Subsequent force-platform and analog tables are separate from point data.

Selected feature groups:
  - Hip, knee, and ankle joint angles.
  - Pelvis angles.
  - Ground reaction forces.
  - Knee moments and powers.
"""

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


SELECTED_CHANNELS = [
    "LHipAngles",       "RHipAngles",
    "LKneeAngles",      "RKneeAngles",
    "LAnkleAngles",     "RAnkleAngles",
    "LPelvisAngles",    "RPelvisAngles",
    "LGroundReactionForce", "RGroundReactionForce",
    "LKneeMoment",      "RKneeMoment",
    "LKneePower",       "RKneePower",
]

N_SAMPLES = 101
WALK_CONDITIONS = ["Walk_Comfortable"]
EXPECTED_FEATURE_COLUMNS = [
    f"{channel}_{axis}"
    for channel in SELECTED_CHANNELS
    for axis in ("X", "Y", "Z")
]
GENDER_MAP = {
    "AJ026": "F",
    "BD004": "F",
    "BL025": "M",
    "CF027": "M",
    "CL007": "F",
    "DA013": "M",
    "DC005": "M",
    "GF022": "F",
    "GM001": "M",
    "GQ016": "M",
    "HN021": "F",
    "HS018": "F",
    "JS009": "F",
    "LA014": "F",
    "LD002": "M",
    "LF003": "M",
    "LL011": "M",
    "LM012": "M",
    "MR015": "M",
    "OB010": "M",
    "PC008": "F",
    "PQ006": "M",
    "RC020": "F",
    "RC023": "F",
    "RV028": "M",
    "SA017": "F",
    "SM019": "F",
    "TB030": "M",
    "TK029": "F",
    "YX024": "M",
}
SESSIONS = ["Session1", "Session2"]

def get_gender(subject_id: str) -> str | None:
    return GENDER_MAP.get(subject_id)

def _normalize_header_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

def _detect_delimiter(path: Path) -> str:
    candidates = [",", ";", "\t", "|"]
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            sample_lines = [handle.readline() for _ in range(20)]
    except OSError:
        return ","

    sample_lines = [line for line in sample_lines if line.strip()]
    if not sample_lines:
        return ","

    return max(candidates, key=lambda candidate: sum(line.count(candidate) for line in sample_lines))

def _read_csv_rows(filepath: Path) -> list[list[str]]:
    delimiter = _detect_delimiter(filepath)
    with filepath.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.reader(handle, delimiter=delimiter))

def _find_signal_header(rows: list[list[str]]) -> int | None:
    selected_tokens = {_normalize_header_token(channel) for channel in SELECTED_CHANNELS}
    best_index = None
    best_score = 0

    for index, row in enumerate(rows):
        tokens = {_normalize_header_token(value) for value in row if str(value).strip()}
        if not tokens:
            continue

        channel_hits = len(tokens & selected_tokens)
        time_hit = "time" in tokens
        score = channel_hits * 10 + int(time_hit) * 25
        if score > best_score:
            best_index = index
            best_score = score

    return best_index


def _read_frame_count(rows: list[list[str]], header_index: int) -> int | None:
    for row in rows[:header_index]:
        if len(row) < 2 or _normalize_header_token(row[0]) != "framenumber":
            continue
        try:
            frame_count = int(float(row[1]))
        except ValueError:
            return None
        return frame_count if frame_count > 0 else None
    return None


def _read_events(rows: list[list[str]], header_index: int) -> dict[str, list[float]]:
    events: dict[str, list[float]] = {}
    canonical_names = {
        "leftfootstrike": "Left_Foot_Strike",
        "rightfootstrike": "Right_Foot_Strike",
        "leftfootoff": "Left_Foot_Off",
        "rightfootoff": "Right_Foot_Off",
    }

    for row in rows[:header_index]:
        if not row:
            continue
        canonical = canonical_names.get(_normalize_header_token(row[0]))
        if canonical is None:
            continue

        times = []
        for value in row[1:]:
            try:
                times.append(float(value))
            except (TypeError, ValueError):
                continue
        if times:
            events[canonical] = sorted(set(times))

    return events


def _build_channel_map(names_row: list[str]) -> dict[str, list[int]]:
    selected_by_token = {
        _normalize_header_token(channel): channel for channel in SELECTED_CHANNELS
    }
    channel_map: dict[str, list[int]] = {}
    current_channel = None

    for column_index, raw_name in enumerate(names_row):
        token = _normalize_header_token(raw_name)
        if token:
            current_channel = selected_by_token.get(token)
            if current_channel is not None:
                channel_map[current_channel] = []
        if current_channel is not None:
            channel_map[current_channel].append(column_index)

    return channel_map


def parse_mocap_recording(filepath: str | Path) -> dict | None:
    filepath = Path(filepath)

    try:
        rows = _read_csv_rows(filepath)
    except (OSError, csv.Error) as exc:
        print(f"  [ERROR] Cannot read {filepath.name}: {exc}")
        return None

    if not rows:
        return None

    header_index = _find_signal_header(rows)
    if header_index is None or header_index + 3 >= len(rows):
        print(f"  [WARNING] Could not detect a signal header in {filepath.name}")
        return None

    names_row = rows[header_index]
    axes_row = rows[header_index + 2]
    channel_map = _build_channel_map(names_row)

    normalized_names = [_normalize_header_token(name) for name in names_row]
    time_column = next(
        (index for index, token in enumerate(normalized_names) if token == "time"),
        None,
    )
    if time_column is None:
        print(f"  [WARNING] Missing Time column in {filepath.name}")
        return None

    frame_count = _read_frame_count(rows, header_index)
    section_rows: list[list[str]] = []
    for row in rows[header_index + 3 :]:
        if not row or not any(str(value).strip() for value in row):
            if section_rows:
                break
            continue
        if time_column >= len(row):
            if section_rows:
                break
            continue
        try:
            float(row[time_column])
        except (TypeError, ValueError):
            if section_rows:
                break
            continue

        padded = row[: len(names_row)] + [""] * max(0, len(names_row) - len(row))
        section_rows.append(padded)
        if frame_count is not None and len(section_rows) == frame_count:
            break

    if len(section_rows) < 10:
        print(f"  [WARNING] Too few kinematic frames in {filepath.name}")
        return None
    if frame_count is not None and len(section_rows) != frame_count:
        print(
            f"  [WARNING] Frame-count mismatch in {filepath.name}: "
            f"expected {frame_count}, read {len(section_rows)}"
        )
        return None

    data_raw = pd.DataFrame(section_rows)
    selected_columns: dict[str, pd.Series] = {
        "Time": pd.to_numeric(data_raw.iloc[:, time_column], errors="coerce")
    }

    for channel in SELECTED_CHANNELS:
        column_indices = channel_map.get(channel, [])
        for position, column_index in enumerate(column_indices[:3]):
            axis = (
                str(axes_row[column_index]).strip().upper()
                if column_index < len(axes_row)
                else ""
            )
            if axis not in {"X", "Y", "Z"}:
                axis = ("X", "Y", "Z")[position]
            selected_columns[f"{channel}_{axis}"] = pd.to_numeric(
                data_raw.iloc[:, column_index],
                errors="coerce",
            )

    missing_columns = [
        column for column in EXPECTED_FEATURE_COLUMNS if column not in selected_columns
    ]
    if missing_columns:
        print(
            f"  [WARNING] Missing required channels in {filepath.name}: "
            f"{', '.join(missing_columns)}"
        )
        return None

    dataframe = pd.DataFrame(selected_columns)[["Time", *EXPECTED_FEATURE_COLUMNS]]
    dataframe = dataframe.dropna(subset=["Time"]).reset_index(drop=True)
    all_nan_columns = [
        column
        for column in EXPECTED_FEATURE_COLUMNS
        if dataframe[column].isna().all()
    ]
    if all_nan_columns:
        print(
            f"  [WARNING] Empty channels in {filepath.name}: "
            f"{', '.join(all_nan_columns)}"
        )
        return None

    return {
        "data": dataframe,
        "events": _read_events(rows, header_index),
        "frame_count": frame_count,
    }


def parse_mocap_csv(filepath: str | Path) -> pd.DataFrame | None:
    recording = parse_mocap_recording(filepath)
    return None if recording is None else recording["data"]


def segment_gait_cycles(
    recording: dict,
    *,
    min_samples: int = 20,
) -> list[dict]:
    dataframe = recording["data"]
    events = recording["events"]
    cycles = []

    for side in ("Left", "Right"):
        strikes = events.get(f"{side}_Foot_Strike", [])
        for cycle_index, (start_time, end_time) in enumerate(pairwise(strikes), start=1):
            if end_time <= start_time:
                continue
            mask = (dataframe["Time"] >= start_time) & (dataframe["Time"] <= end_time)
            cycle_data = dataframe.loc[mask].reset_index(drop=True)
            if len(cycle_data) < min_samples:
                continue
            cycles.append(
                {
                    "data": cycle_data,
                    "side": side,
                    "cycle_num": cycle_index,
                    "cycle_start_sec": float(start_time),
                    "cycle_end_sec": float(end_time),
                }
            )

    return cycles

def normalize_cycle_length(df: pd.DataFrame, n_samples: int = N_SAMPLES) -> np.ndarray:
    feature_cols = [c for c in df.columns if c != "Time"]
    n_orig = len(df)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_orig < 2:
        raise ValueError("A gait cycle must contain at least 2 samples.")
    if not feature_cols:
        raise ValueError("A gait cycle does not contain any feature channels.")

    x_orig = np.linspace(0, 1, n_orig)
    x_new = np.linspace(0, 1, n_samples)

    result = np.zeros((n_samples, len(feature_cols)), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        y = df[col].values.astype(np.float64)
        invalid = ~np.isfinite(y)
        if invalid.all():
            raise ValueError(f"Channel {col} has no finite samples in this gait cycle.")
        if invalid.any():
            y[invalid] = np.interp(
                np.where(invalid)[0],
                np.where(~invalid)[0],
                y[~invalid],
            )

        if n_orig >= 11:
            window_length = min(11, n_orig if n_orig % 2 == 1 else n_orig - 1)
            y = savgol_filter(y, window_length=window_length, polyorder=3)

        f = interp1d(x_orig, y, kind="linear", fill_value="extrapolate")
        result[:, i] = f(x_new).astype(np.float32)

    return result

def find_cycle_files(
    dataset_root: str | Path,
    conditions: list[str] = WALK_CONDITIONS,
    sessions: list[str] = SESSIONS,
) -> list[dict]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    records = []

    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        gender = get_gender(subject_id)

        for session in sessions:
            session_dir = subject_dir / session
            if not session_dir.exists():
                continue

            ow_dir = session_dir / "Overground_Walk"
            if not ow_dir.exists():
                continue

            for condition in conditions:
                cond_dir = ow_dir / condition / "Post_Process"
                if not cond_dir.exists():
                    continue
                pattern = re.compile(
                    rf"^{re.escape(condition)}(\d+)\.csv$", re.IGNORECASE
                )
                for f in sorted(cond_dir.iterdir()):
                    m = pattern.match(f.name)
                    if m:
                        records.append({
                            "path":       f,
                            "subject_id": subject_id,
                            "gender":     gender,
                            "session":    session,
                            "condition":  condition,
                            "recording_num": int(m.group(1)),
                        })

    return records

def load_dataset(
    dataset_root: str | Path,
    conditions: list[str] = WALK_CONDITIONS,
    sessions: list[str] = SESSIONS,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
    num_workers: int = 1,
) -> dict:
    records = find_cycle_files(dataset_root, conditions, sessions)

    if verbose:
        print(f"Found {len(records)} CSV recordings across all participants.")

    X_list, y_list, gender_list, meta_list = [], [], [], []
    channels = EXPECTED_FEATURE_COLUMNS.copy()
    stats = {"F": 0, "M": 0, "unknown": 0, "errors": 0}

    def handle_record(rec: dict):
        gender = rec["gender"]
        if gender is None:
            return "unknown", rec, None, 0.0

        start = time.perf_counter()
        recording = parse_mocap_recording(rec["path"])
        elapsed = time.perf_counter() - start
        if recording is None:
            return "error", rec, None, elapsed

        cycles = segment_gait_cycles(recording)
        if not cycles:
            print(f"  [WARNING] No complete gait cycles in {rec['path'].name}")
            return "error", rec, None, elapsed

        return "ok", rec, cycles, elapsed

    def consume_result(index: int, result: tuple) -> None:
        status, rec, cycles, elapsed = result

        if verbose:
            print(f"  [FILE] {rec['path'].name} | {elapsed:.2f}s")

        if status == "unknown":
            stats["unknown"] += 1
            return
        if status == "error":
            stats["errors"] += 1
            return

        gender = rec["gender"]
        label = 0 if gender == "F" else 1
        accepted_cycles = 0
        for cycle_info in cycles:
            try:
                cycle = normalize_cycle_length(cycle_info["data"], n_samples)
            except ValueError as exc:
                if verbose:
                    print(
                        f"  [WARNING] Skipped a cycle from {rec['path'].name}: {exc}"
                    )
                continue

            if cycle.shape != (n_samples, len(channels)) or not np.isfinite(cycle).all():
                continue

            sample_meta = {
                **rec,
                "side": cycle_info["side"],
                "cycle_num": cycle_info["cycle_num"],
                "cycle_start_sec": cycle_info["cycle_start_sec"],
                "cycle_end_sec": cycle_info["cycle_end_sec"],
            }
            X_list.append(cycle)
            y_list.append(label)
            gender_list.append(gender)
            meta_list.append(sample_meta)
            stats[gender] += 1
            accepted_cycles += 1

        if accepted_cycles == 0:
            stats["errors"] += 1

        if verbose and (index == 1 or index % 50 == 0 or index == len(records)):
            print(
                f"  [PROGRESS] {index:4d}/{len(records)} | "
                f"F={stats['F']} M={stats['M']} "
                f"unknown={stats['unknown']} errors={stats['errors']}"
            )

    worker_count = max(1, int(num_workers))
    use_threads = worker_count > 1 and len(records) > 1
    if use_threads:
        worker_count = min(worker_count, len(records))
        if verbose:
            print(f"Processing CSV files with {worker_count} threads")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(handle_record, records)
            for index, result in enumerate(results, start=1):
                consume_result(index, result)
    else:
        for index, result in enumerate(map(handle_record, records), start=1):
            consume_result(index, result)

    if not X_list:
        raise ValueError("No data was loaded. Check the dataset path and file format.")

    X = np.stack(X_list, axis=0)   # (N, n_samples, n_channels)
    y = np.array(y_list, dtype=np.int8)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Loaded samples:      {len(X)}")
        print(f"  Female (F=0):        {stats['F']}")
        print(f"  Male (M=1):          {stats['M']}")
        print(f"  Unknown sex records: {stats['unknown']}")
        print(f"  Loading errors:      {stats['errors']}")
        print(f"  X shape:             {X.shape}")
        print(f"  Channels ({len(channels)}):")
        for i, ch in enumerate(channels):
            print(f"    [{i:2d}] {ch}")
        print(f"{'='*50}\n")

    return {
        "X":        X,
        "y":        y,
        "gender":   gender_list,
        "meta":     meta_list,
        "channels": channels,
        "stats":    stats,
    }

def compute_normalization_stats(X_train: np.ndarray) -> dict:
    if X_train.ndim != 3 or X_train.shape[0] == 0 or X_train.shape[-1] == 0:
        raise ValueError("X_train must be a non-empty 3D array with feature channels.")
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values.")

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def normalize_zscore(X: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])
    if X.ndim != 3 or mean.shape != std.shape or mean.shape != (1, 1, X.shape[-1]):
        raise ValueError(
            "Normalization statistics are incompatible with the dataset channels."
        )
    normalized = (X - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced NaN or infinite values.")
    return normalized.astype(np.float32, copy=False)


def denormalize_zscore(X_norm: np.ndarray, stats: dict) -> np.ndarray:
    return X_norm * stats["std"] + stats["mean"]

def split_dataset(
    dataset: dict,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = 42,
) -> dict:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    rng = np.random.default_rng(seed)

    meta = dataset["meta"]
    y = np.asarray(dataset["y"])
    X = np.asarray(dataset["X"])
    gender = dataset["gender"]
    if X.ndim != 3:
        raise ValueError("dataset['X'] must be a 3D array.")
    if not (len(X) == len(y) == len(meta) == len(gender)):
        raise ValueError("Dataset arrays and metadata have inconsistent lengths.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Dataset labels must contain only 0 (female) and 1 (male).")

    female_subjects = sorted(
        {m["subject_id"] for m, label in zip(meta, y, strict=True) if label == 0}
    )
    if len(female_subjects) < 3:
        raise ValueError(
            "At least three female participants are required for subject-wise "
            "train, validation and test splits."
        )
    if not np.any(y == 1):
        raise ValueError("At least one male sample is required for test evaluation.")

    rng.shuffle(female_subjects)

    n_f = len(female_subjects)
    n_tr = max(1, int(n_f * train_ratio))
    n_va = max(1, int(n_f * val_ratio))
    if n_tr + n_va >= n_f:
        n_tr = n_f - n_va - 1
    if n_tr < 1 or n_va < 1 or n_f - n_tr - n_va < 1:
        raise ValueError("Split ratios do not leave female participants in every split.")

    train_subjects = set(female_subjects[:n_tr])
    val_subjects = set(female_subjects[n_tr:n_tr + n_va])

    idx_train, idx_val, idx_test = [], [], []

    for i, (m, label) in enumerate(zip(meta, y, strict=True)):
        sid = m["subject_id"]
        if label == 1:
            idx_test.append(i)
        elif sid in train_subjects:
            idx_train.append(i)
        elif sid in val_subjects:
            idx_val.append(i)
        else:
            idx_test.append(i)

    def subset(indices):
        indices = np.asarray(indices, dtype=np.int64)
        return {
            "X": X[indices],
            "y": y[indices],
            "gender": [gender[i] for i in indices],
            "meta": [meta[i] for i in indices],
        }

    splits = {
        "train":    subset(idx_train),
        "val":      subset(idx_val),
        "test":     subset(idx_test),
        "channels": dataset["channels"],
    }

    print("Subject-wise dataset split:")
    print(
        f"  TRAIN: {len(idx_train):4d} samples | "
        f"F={sum(y[idx_train] == 0)} M={sum(y[idx_train] == 1)} | "
        f"female subjects: {len(train_subjects)}"
    )
    print(
        f"  VAL:   {len(idx_val):4d} samples | "
        f"F={sum(y[idx_val] == 0)} M={sum(y[idx_val] == 1)} | "
        f"female subjects: {len(val_subjects)}"
    )
    print(
        f"  TEST:  {len(idx_test):4d} samples | "
        f"F={sum(y[idx_test] == 0)} M={sum(y[idx_test] == 1)}"
    )

    return splits

def describe_dataset(dataset: dict) -> None:
    X, y, channels = dataset["X"], dataset["y"], dataset["channels"]
    print(f"\nShape:      {X.shape}  (samples x time x channels)")
    print(f"Labels:     0=F ({(y==0).sum()}), 1=M ({(y==1).sum()})")
    print("\nChannel statistics (min / mean / max), first five:")
    for i, ch in enumerate(channels[:5]):
        vals = X[:, :, i]
        print(f"  {ch:<35s}  min={vals.min():8.2f}  mean={vals.mean():8.2f}  max={vals.max():8.2f}")
    if len(channels) > 5:
        print(f"  ... ({len(channels) - 5} more channels)")

def _main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load, segment and describe the motion-capture CSV dataset."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")

    dataset = load_dataset(
        dataset_root=args.dataset_root,
        conditions=WALK_CONDITIONS,
        sessions=SESSIONS,
        n_samples=N_SAMPLES,
        verbose=True,
        num_workers=args.num_workers,
    )
    describe_dataset(dataset)
    split_dataset(dataset)


if __name__ == "__main__":
    _main()
