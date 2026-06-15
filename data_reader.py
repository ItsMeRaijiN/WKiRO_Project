"""
data_reader.py
==============
Reader and preprocessor for the motion-capture dataset by Riglet et al. (2024).
Goal: prepare data for training a 1D convolutional autoencoder (CAE) for anomaly
detection (female vs male gait).

CSV file layout (Vicon "Post_Process" export):
  Rows 0-3: metadata (FrameNumber, FirstFrame, PointFrequency, AnalogFrequency)
  Next rows: events (Right/Left_Foot_Strike/Off) - times in seconds
  Then THE FILE CONTAINS SEVERAL DATA BLOCKS, each with its own "Time" header:
    BLOCK 1 (model outputs, 100 Hz): joint angles/moments/powers  <-- ONLY this one is read
    BLOCK 2 (ground reaction wrench, 100 Hz)
    BLOCK 3 (analog, 1000 Hz): raw forces Fx/Fy/Fz
  Each block header is 3 rows: names / units / axes (X/Y/Z), followed by data.

IMPORTANT: a single Walk_Comfortable{N}.csv file contains a WHOLE pass with SEVERAL
gait cycles (usually 1-3). We therefore segment it into individual gait cycles using
foot-strike events, and normalize each cycle to 0-100% (101 samples).

Sex labels:
  Each subject's sex comes from the dataset's metadata.xlsx sheet (column "Sex").
  GENDER_MAP was verified 1:1 against metadata.xlsx (14 females / 16 males).
  Female (F) -> label 0 (normal), Male (M) -> label 1 (anomaly).
  Subjects outside the map (e.g. the "Python Code" folder) are skipped.

Selected channels:
  By default 24 joint-angle channels (joint rotations) - fully reliable and aligned
  with the task description. Kinetics (GRF, moments, powers) in overground walking
  depend on force-plate contact and are often zero, so they are not the default.
"""

import csv
import re
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Channels to extract - names from the block-1 header row (without X/Y/Z axes).
# Default: lower-limb and pelvis joint angles (8 groups x 3 axes = 24 channels).
SELECTED_CHANNELS = [
    "LHipAngles",    "RHipAngles",
    "LKneeAngles",   "RKneeAngles",
    "LAnkleAngles",  "RAnkleAngles",
    "LPelvisAngles", "RPelvisAngles",
]

# Optional channels (kinetics) - can be added, but are often zero in overground:
#   "LGroundReactionForce", "RGroundReactionForce",
#   "LKneeMoment", "RKneeMoment",     # usually reliable
#   "LKneePower",  "RKneePower",      # in this dataset effectively always zero

# Number of samples after cycle normalization (0-100% of the gait cycle)
N_SAMPLES = 101

# Leg whose foot strikes define cycles (cycle = strike -> next strike of the same leg)
CYCLE_LEG = "Right"

# Sanity check for a single cycle (at 100 Hz a gait cycle is ~0.8-1.4 s)
MIN_CYCLE_FRAMES = 40
MIN_CYCLE_SEC = 0.5
MAX_CYCLE_SEC = 2.5

# Walking speeds to include
WALK_CONDITIONS = ["Walk_Comfortable"]

# Sex per subject ID - verified against metadata.xlsx (14 F / 16 M)
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
    "SA017": "F",   # verified against metadata.xlsx (was incorrectly "M")
    "SM019": "F",
    "TB030": "M",
    "TK029": "F",   # verified against metadata.xlsx (was incorrectly "M")
    "YX024": "M",
}
# Total: 14 females / 16 males - consistent with the article and metadata.xlsx.

# Sessions (the dataset has Session1 and Session2 for each subject)
SESSIONS = ["Session1", "Session2"]


def get_gender(subject_id: str) -> Optional[str]:
    """Return 'F'/'M' for a known ID, otherwise None."""
    return GENDER_MAP.get(subject_id)


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


# ---------------------------------------------------------------------------
# Parsing a single CSV file (block 1 + events only)
# ---------------------------------------------------------------------------

_EVENT_RE = re.compile(r"^(Left|Right)_Foot_(Strike|Off)$", re.IGNORECASE)


def _detect_delimiter(path: Path) -> str:
    candidates = [",", ";", "\t", "|"]
    try:
        sample_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ","
    sample_lines = [line for line in sample_lines[:20] if line.strip()]
    if not sample_lines:
        return ","
    best, best_score = ",", -1
    for cand in candidates:
        score = sum(line.count(cand) for line in sample_lines)
        if score > best_score:
            best_score, best = score, cand
    return best


def parse_mocap_trial(filepath: str | Path) -> Optional[dict]:
    """
    Read a single trial file and return a dict:
        {
            "df":     pd.DataFrame with columns [Time, <channel_axis>, ...] (block 1 only),
            "events": {"Right_Foot_Strike": [t, ...], "Left_Foot_Strike": [...], ...},
            "meta":   {"first_frame": int, "point_freq": float},
        }
    or None if the file is corrupt / required data is missing.
    Reads only the first data block (model outputs, 100 Hz).
    """
    filepath = Path(filepath)
    try:
        delimiter = _detect_delimiter(filepath)
        with filepath.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
    except Exception as e:
        print(f"  [ERROR] Could not read {filepath.name}: {e}")
        return None

    if not rows:
        return None

    def as_num(x: str) -> Optional[float]:
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    # --- Metadata and events (top of file, before the first "Time" header) ---
    meta = {"first_frame": None, "point_freq": 100.0}
    events: dict[str, list[float]] = {}
    header_idx = None
    for i, row in enumerate(rows):
        if not row:
            continue
        key = str(row[0]).strip()
        if _normalize_token(key) == "time":
            header_idx = i  # first header = block 1
            break
        if key == "FirstFrame":
            meta["first_frame"] = as_num(row[1]) if len(row) > 1 else None
        elif key == "PointFrequency":
            pf = as_num(row[1]) if len(row) > 1 else None
            if pf:
                meta["point_freq"] = pf
        elif _EVENT_RE.match(key):
            vals = [as_num(c) for c in row[1:] if str(c).strip()]
            events[key] = [v for v in vals if v is not None]

    if header_idx is None:
        print(f"  [WARNING] No 'Time' header in {filepath.name}")
        return None

    # Block-1 header: names / units / axes ; data starts at header_idx + 3
    names_row = rows[header_idx]
    axes_row = rows[header_idx + 2] if header_idx + 2 < len(rows) else []
    data_start = header_idx + 3

    # --- Map: channel_name -> column indices (X, Y, Z) ---
    channel_cols: dict[str, list[int]] = {}
    current = None
    for col_idx, name in enumerate(names_row):
        name = str(name).strip()
        if name and _normalize_token(name) != "time":
            current = name
            channel_cols.setdefault(current, [])
        if current and _normalize_token(current) != "time":
            channel_cols[current].append(col_idx)

    # --- Block-1 data: read until empty row / non-numeric Time / next block ---
    block_rows: list[list[str]] = []
    for row in rows[data_start:]:
        if not row or not str(row[0]).strip():
            break
        if as_num(row[0]) is None:
            break
        block_rows.append(row)

    if len(block_rows) < MIN_CYCLE_FRAMES:
        return None

    n = len(block_rows)
    axes_labels = ["X", "Y", "Z"]

    time_vals = np.array([as_num(r[0]) for r in block_rows], dtype=np.float64)
    out_cols = {"Time": time_vals}

    for ch in SELECTED_CHANNELS:
        if ch not in channel_cols:
            continue
        for k, cidx in enumerate(channel_cols[ch][:3]):
            axis = axes_row[cidx].strip() if cidx < len(axes_row) and str(axes_row[cidx]).strip() else axes_labels[k]
            col = np.empty(n, dtype=np.float64)
            for r_i, row in enumerate(block_rows):
                col[r_i] = as_num(row[cidx]) if cidx < len(row) and str(row[cidx]).strip() else np.nan
            out_cols[f"{ch}_{axis}"] = col

    if len(out_cols) <= 1:
        print(f"  [WARNING] No selected channels in {filepath.name}")
        return None

    df = pd.DataFrame(out_cols)
    df = df[~df["Time"].isna()].reset_index(drop=True)
    return {"df": df, "events": events, "meta": meta}


# ---------------------------------------------------------------------------
# Segmentation into individual gait cycles + normalization to 101 samples
# ---------------------------------------------------------------------------

def _resample_to(segment: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Interpolate a cycle (frames, n_channels) to (n_samples, n_channels) - normalization
    to 0-100% of the cycle. Fills NaNs and lightly smooths (Savitzky-Golay).
    """
    n_orig, n_ch = segment.shape
    x_orig = np.linspace(0.0, 1.0, n_orig)
    x_new = np.linspace(0.0, 1.0, n_samples)
    out = np.zeros((n_samples, n_ch), dtype=np.float32)

    for c in range(n_ch):
        y = segment[:, c].astype(np.float64)
        nans = np.isnan(y)
        if nans.all():
            continue
        if nans.any():
            y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])
        if n_orig >= 11:
            win = min(11, n_orig if n_orig % 2 == 1 else n_orig - 1)
            if win >= 5:
                y = savgol_filter(y, window_length=win, polyorder=3)
        out[:, c] = np.interp(x_new, x_orig, y).astype(np.float32)

    return out


def segment_cycles(
    parsed: dict,
    n_samples: int = N_SAMPLES,
    leg: str = CYCLE_LEG,
) -> list[np.ndarray]:
    """
    Split a trial into individual gait cycles by foot strikes of the chosen leg.
    Cycle = [strike_i, strike_{i+1}]; each normalized to n_samples.
    Returns a list of arrays (n_samples, n_channels).
    """
    df = parsed["df"]
    events = parsed["events"]
    feature_cols = [c for c in df.columns if c != "Time"]
    if not feature_cols:
        return []

    time_vals = df["Time"].to_numpy(dtype=np.float64)
    strikes = sorted(t for t in events.get(f"{leg}_Foot_Strike", []) if t is not None)

    feats = df[feature_cols].to_numpy(dtype=np.float64)
    cycles: list[np.ndarray] = []

    for i in range(len(strikes) - 1):
        t0, t1 = strikes[i], strikes[i + 1]
        duration = t1 - t0
        if not (MIN_CYCLE_SEC <= duration <= MAX_CYCLE_SEC):
            continue
        idx = np.flatnonzero((time_vals >= t0) & (time_vals <= t1))
        if len(idx) < MIN_CYCLE_FRAMES:
            continue
        cycles.append(_resample_to(feats[idx], n_samples))

    return cycles


# ---------------------------------------------------------------------------
# Dataset directory scan
# ---------------------------------------------------------------------------

def find_cycle_files(
    dataset_root: str | Path,
    conditions: list[str] = WALK_CONDITIONS,
    sessions: list[str] = SESSIONS,
) -> list[dict]:
    """
    Walk the dataset tree and return a list of trial files:
        {"path", "subject_id", "gender", "session", "condition", "trial_num"}
    Pattern: {root}/{ID}/{Session}/Overground_Walk/{condition}/Post_Process/{condition}{N}.csv
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records = []
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        gender = get_gender(subject_id)

        for session in sessions:
            ow_dir = subject_dir / session / "Overground_Walk"
            if not ow_dir.exists():
                continue
            for condition in conditions:
                cond_dir = ow_dir / condition / "Post_Process"
                if not cond_dir.exists():
                    continue
                pattern = re.compile(rf"^{re.escape(condition)}(\d+)\.csv$", re.IGNORECASE)
                for f in sorted(cond_dir.iterdir()):
                    m = pattern.match(f.name)
                    if m:
                        records.append({
                            "path":       f,
                            "subject_id": subject_id,
                            "gender":     gender,
                            "session":    session,
                            "condition":  condition,
                            "trial_num":  int(m.group(1)),
                        })
    return records


# ---------------------------------------------------------------------------
# Main loader - builds the ready dataset (per gait cycle)
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_root: str | Path,
    conditions: list[str] = WALK_CONDITIONS,
    sessions: list[str] = SESSIONS,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
    num_workers: int = 1,
    leg: str = CYCLE_LEG,
) -> dict:
    """
    Load the whole dataset as individual gait cycles and return a dict:
        {
            "X":        np.ndarray (N, n_samples, n_channels),
            "y":        np.ndarray (N,)  0=female, 1=male,
            "gender":   list[str],
            "meta":     list[dict]  (trial metadata + cycle index),
            "channels": list[str],
            "stats":    dict,
        }
    Samples with unknown sex are skipped.
    """
    records = find_cycle_files(dataset_root, conditions, sessions)
    if verbose:
        print(f"Found {len(records)} trial files (all sexes).")

    X_list, y_list, gender_list, meta_list = [], [], [], []
    channels: Optional[list[str]] = None
    stats = {"F": 0, "M": 0, "unknown": 0, "errors": 0, "trials": 0}

    def handle_record(rec: dict):
        if rec["gender"] is None:
            return "unknown", rec, None, None, 0.0
        start = time.perf_counter()
        parsed = parse_mocap_trial(rec["path"])
        elapsed = time.perf_counter() - start
        if parsed is None:
            return "error", rec, None, None, elapsed
        cols = [c for c in parsed["df"].columns if c != "Time"]
        cycles = segment_cycles(parsed, n_samples=n_samples, leg=leg)
        return "ok", rec, cycles, cols, elapsed

    def consume(status, rec, cycles, cols):
        nonlocal channels
        if status == "unknown":
            stats["unknown"] += 1
            return
        if status == "error" or not cycles:
            stats["errors"] += 1
            return
        if channels is None:
            channels = cols
        stats["trials"] += 1
        gender = rec["gender"]
        label = 0 if gender == "F" else 1
        for c_idx, cyc in enumerate(cycles):
            if cyc.shape[1] != len(channels):
                stats["errors"] += 1
                continue
            X_list.append(cyc)
            y_list.append(label)
            gender_list.append(gender)
            m = dict(rec)
            m["cycle_index"] = c_idx
            meta_list.append(m)
            stats[gender] += 1

    worker_count = max(1, int(num_workers))
    use_threads = worker_count > 1 and len(records) > 1

    if use_threads:
        worker_count = min(worker_count, len(records))
        if verbose:
            print(f"Parsing CSV in parallel: {worker_count} threads")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(handle_record, rec) for rec in records]
            for idx, future in enumerate(as_completed(futures), start=1):
                status, rec, cycles, cols, elapsed = future.result()
                consume(status, rec, cycles, cols)
                if verbose and (idx == 1 or idx % 50 == 0 or idx == len(records)):
                    print(f"  [PROGRESS] {idx:4d}/{len(records)} | "
                          f"cycles F={stats['F']} M={stats['M']} "
                          f"unknown={stats['unknown']} errors={stats['errors']}")
    else:
        for idx, rec in enumerate(records, start=1):
            status, rec, cycles, cols, elapsed = handle_record(rec)
            consume(status, rec, cycles, cols)
            if verbose and (idx == 1 or idx % 50 == 0 or idx == len(records)):
                print(f"  [PROGRESS] {idx:4d}/{len(records)} | "
                      f"cycles F={stats['F']} M={stats['M']} "
                      f"unknown={stats['unknown']} errors={stats['errors']}")

    if not X_list:
        raise ValueError("No cycles loaded. Check the dataset path.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int8)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Trials OK:         {stats['trials']}")
        print(f"  Cycles total:      {len(X)}")
        print(f"  Females (F=0):     {stats['F']}")
        print(f"  Males (M=1):       {stats['M']}")
        print(f"  Unknown sex:       {stats['unknown']}")
        print(f"  Errors/empty:      {stats['errors']}")
        print(f"  Shape X:           {X.shape}")
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


# ---------------------------------------------------------------------------
# Amplitude normalization (per-channel z-score)
# ---------------------------------------------------------------------------

def compute_normalization_stats(X_train: np.ndarray) -> dict:
    """Compute per-channel mean and std on the training set (normal class)."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def normalize_zscore(X: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score normalize X using the training statistics."""
    return (X - stats["mean"]) / stats["std"]


def denormalize_zscore(X_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Invert the z-score normalization."""
    return X_norm * stats["std"] + stats["mean"]


# ---------------------------------------------------------------------------
# Train/val/test split (subject-wise)
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: dict,
    normal_label: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split strategy (anomaly detection):
      - "normal" class = normal_label (0=F, 1=M; default 1=males)
      - TRAIN: 70% of normal-class subjects
      - VAL:   15% of normal-class subjects -> for tuning and threshold
      - TEST:  remaining normal-class subjects + ALL anomaly-class subjects
    Subject-wise split (no subject leakage between train/val/test).
    """
    rng = np.random.default_rng(seed)
    meta, y, X = dataset["meta"], dataset["y"], dataset["X"]

    normal_subjects = sorted(set(
        m["subject_id"] for m, label in zip(meta, y) if label == normal_label
    ))
    rng.shuffle(normal_subjects)

    n_s = len(normal_subjects)
    n_tr = int(n_s * train_ratio)
    n_va = int(n_s * val_ratio)
    train_subjects = set(normal_subjects[:n_tr])
    val_subjects = set(normal_subjects[n_tr:n_tr + n_va])

    idx_train, idx_val, idx_test = [], [], []
    for i, (m, label) in enumerate(zip(meta, y)):
        sid = m["subject_id"]
        if label != normal_label:
            idx_test.append(i)
        elif sid in train_subjects:
            idx_train.append(i)
        elif sid in val_subjects:
            idx_val.append(i)
        else:
            idx_test.append(i)

    def subset(indices):
        indices = np.array(indices)
        return {
            "X":      X[indices],
            "y":      y[indices],
            "gender": [dataset["gender"][i] for i in indices],
            "meta":   [meta[i] for i in indices],
        }

    splits = {
        "train":    subset(idx_train),
        "val":      subset(idx_val),
        "test":     subset(idx_test),
        "channels": dataset["channels"],
    }

    print("Dataset split (subject-wise):")
    print(f"  TRAIN: {len(idx_train):4d} cycles | F={int(sum(y[idx_train]==0))} M={int(sum(y[idx_train]==1))} | subjects: {len(train_subjects)}")
    print(f"  VAL:   {len(idx_val):4d} cycles | F={int(sum(y[idx_val]==0))} M={int(sum(y[idx_val]==1))} | subjects: {len(val_subjects)}")
    print(f"  TEST:  {len(idx_test):4d} cycles | F={int(sum(y[idx_test]==0))} M={int(sum(y[idx_test]==1))}")
    return splits


# ---------------------------------------------------------------------------
# Data preview (diagnostics)
# ---------------------------------------------------------------------------

def describe_dataset(dataset: dict) -> None:
    """Print basic dataset statistics."""
    X, y, channels = dataset["X"], dataset["y"], dataset["channels"]
    print(f"\nShape:      {X.shape}  (cycles x time x channels)")
    print(f"Labels:     0=F ({int((y==0).sum())}), 1=M ({int((y==1).sum())})")
    print("\nChannel statistics (min / mean / max) - first 5:")
    for i, ch in enumerate(channels[:5]):
        vals = X[:, :, i]
        print(f"  {ch:<24s}  min={vals.min():8.2f}  mean={vals.mean():8.2f}  max={vals.max():8.2f}")
    if len(channels) > 5:
        print(f"  ... ({len(channels) - 5} more channels)")


# ---------------------------------------------------------------------------
# Entry point - usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = sys.argv[1] if len(sys.argv) > 1 else r"E:\ML_datasets\Data_Run_Walk"
    print(f"Dataset root: {ROOT}\n")

    dataset = load_dataset(dataset_root=ROOT, verbose=True, num_workers=8)
    describe_dataset(dataset)
    splits = split_dataset(dataset)
    norm_stats = compute_normalization_stats(splits["train"]["X"])
    print("\nZ-score normalization ready.")
