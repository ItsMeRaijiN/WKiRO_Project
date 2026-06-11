"""
mocap_reader.py
===============
Czytnik i preprocessor danych motion capture z datasetu Gueugnon et al. (2024).
Cel: przygotowanie danych do treningu konwolucyjnego autoenkodera (CAE)
     do detekcji anomalii (chód kobiet vs mężczyzn).

Struktura pliku CSV:
  Wiersz 0: metadane (FrameNumber, FirstFrame, PointFrequency, AnalogFrequency)
  Wiersze 1-3: zdarzenia (foot strikes/offs) - czasy w sekundach
  Wiersz 4: nazwy kanałów (Time, PELVISO, ..., LKneeAngles, ...)
  Wiersz 5: jednostki (s, mm, deg, N, Nmm, W)
  Wiersz 6: osie X/Y/Z
  Wiersze 7+: dane numeryczne

Konwencja ID uczestników:
  GF*** = kobieta (Female)  -> etykieta 0 (normalne)
  GM*** = mężczyzna (Male)  -> etykieta 1 (anomalia)
  Inne prefiksy: sprawdzane przez plik metadata lub domyślnie pomijane

Wybrane kanały (biomechanika kolana wg ScienceDirect 2020):
  - Kąty stawowe: LKneeAngles (X/Y/Z), RKneeAngles (X/Y/Z)
  - Kąty bioder: LHipAngles, RHipAngles
  - Kąty stawów skokowych: LAnkleAngles, RAnkleAngles
  - Kąty miednicy: LPelvisAngles, RPelvisAngles
  - Siły reakcji podłoża: LGroundReactionForce, RGroundReactionForce
  - Momenty kolana: LKneeMoment, RKneeMoment
  - Moce kolana: LKneePower, RKneePower
"""

import csv
import os
import re
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

# Kanały do wyodrębnienia — nazwy z wiersza nagłówkowego (bez osi X/Y/Z)
SELECTED_CHANNELS = [
    # Kąty stawowe kończyn dolnych (stopnie)
    "LHipAngles",       "RHipAngles",
    "LKneeAngles",      "RKneeAngles",
    "LAnkleAngles",     "RAnkleAngles",
    # Kąty miednicy i kręgosłupa
    "LPelvisAngles",    "RPelvisAngles",
    # Siły reakcji podłoża (N) — kluczowe dla różnic płciowych w kolanie
    "LGroundReactionForce", "RGroundReactionForce",
    # Momenty kolana (Nmm) — bezpośrednio z artykułu ScienceDirect
    "LKneeMoment",      "RKneeMoment",
    # Moce kolana (W)
    "LKneePower",       "RKneePower",
]

# Liczba próbek po normalizacji cyklu (0-100% cyklu chodu)
N_SAMPLES = 101

# Prędkości chodu do uwzględnienia
WALK_CONDITIONS = ["Walk_Comfortable"]
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
    "SA017": "M",
    "SM019": "F",
    "TB030": "M",
    "TK029": "M",
    "YX024": "M",
}
# Sesje (dataset ma Session1 i Session2 dla każdego uczestnika)
SESSIONS = ["Session1", "Session2"]

# Prefiks ID uczestnika -> płeć
def get_gender(subject_id: str) -> Optional[str]:
    return GENDER_MAP.get(subject_id)

# ---------------------------------------------------------------------------
# Parsowanie pojedynczego pliku CSV
# ---------------------------------------------------------------------------

def parse_mocap_csv(filepath: str | Path) -> Optional[pd.DataFrame]:
    """
    Wczytuje jeden plik CSV cyklu chodu.

    Zwraca DataFrame z kolumnami: Time + wybrane kanały (X/Y/Z jako osobne kolumny),
    lub None jeśli plik jest uszkodzony / brakuje wymaganych kolumn.

    Schemat kolumn wynikowych (przykład):
        Time, LHipAngles_X, LHipAngles_Y, LHipAngles_Z,
              RHipAngles_X, ..., LKneePower_X, ...
    """
    filepath = Path(filepath)

    def normalize_header_token(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def detect_delimiter(path: Path) -> str:
        candidates = [",", ";", "\t", "|"]
        try:
            sample_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return ","

        sample_lines = [line for line in sample_lines[:20] if line.strip()]
        if not sample_lines:
            return ","

        best_delimiter = ","
        best_score = -1
        for candidate in candidates:
            score = sum(line.count(candidate) for line in sample_lines)
            if score > best_score:
                best_score = score
                best_delimiter = candidate

        return best_delimiter

    try:
        # --- Wczytaj surowy plik ---
        delimiter = detect_delimiter(filepath)
        with filepath.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            rows = [row for row in csv.reader(handle, delimiter=delimiter) if any(cell.strip() for cell in row)]

        if not rows:
            return None

        max_width = max(len(row) for row in rows)
        padded_rows = [row + [""] * (max_width - len(row)) for row in rows]
        raw = pd.DataFrame(padded_rows, dtype=str)

    except Exception as e:
        print(f"  [BŁĄD] Nie można wczytać {filepath.name}: {e}")
        return None

    # --- Znajdź wiersz z nazwami kanałów ---
    selected_channel_tokens = {normalize_header_token(ch) for ch in SELECTED_CHANNELS}
    header_row_idx = None
    best_score = -1
    for i, row in raw.iterrows():
        tokens = {
            normalize_header_token(x)
            for x in row.values
            if str(x).strip()
        }
        if not tokens:
            continue

        channel_hits = len(tokens & selected_channel_tokens)
        time_hit = 1 if "time" in tokens else 0
        score = channel_hits * 10 + time_hit * 25

        if score > best_score:
            best_score = score
            header_row_idx = i

    if header_row_idx is None or best_score <= 0:
        if len(raw) > 4:
            header_row_idx = 4
        else:
            print(f"  [OSTRZEŻENIE] Nie udało się wykryć nagłówka w {filepath.name}")
            return None

    # Wiersze nagłówkowe: nazwy, jednostki, osie
    names_row  = raw.iloc[header_row_idx].fillna("").tolist()
    units_row  = raw.iloc[header_row_idx + 1].fillna("").tolist() if header_row_idx + 1 < len(raw) else []
    axes_row   = raw.iloc[header_row_idx + 2].fillna("").tolist() if header_row_idx + 2 < len(raw) else []

    # Dane numeryczne zaczynają się dwa wiersze po osiach
    data_start = header_row_idx + 3
    data_raw = raw.iloc[data_start:].reset_index(drop=True)

    # --- Zbuduj mapę: nazwa_kanału -> lista indeksów kolumn (X, Y, Z) ---
    channel_col_map = {}  # {"LKneeAngles": [col_x, col_y, col_z], ...}
    current_channel = None

    for col_idx, name in enumerate(names_row):
        name = str(name).strip()
        if name and normalize_header_token(name) != "time":
            current_channel = name
            channel_col_map[current_channel] = []
        if current_channel and normalize_header_token(current_channel) != "time":
            channel_col_map[current_channel].append(col_idx)

    # --- Wyodrębnij dane numeryczne dla wybranych kanałów ---
    # Preferuj jawny nagłówek Time, a jeśli go nie ma, użyj pierwszej kolumny.
    normalized_names = [normalize_header_token(name) for name in names_row]
    time_col_idx = next((idx for idx, token in enumerate(normalized_names) if token == "time"), None)
    if time_col_idx is None:
        time_col_idx = next((idx for idx, token in enumerate(normalized_names) if token.startswith("time")), None)
    if time_col_idx is None:
        time_col_idx = 0

    try:
        time_col = pd.to_numeric(data_raw.iloc[:, time_col_idx], errors="coerce")
    except Exception:
        return None

    selected_cols = {"Time": time_col}

    for ch in SELECTED_CHANNELS:
        if ch not in channel_col_map:
            continue
        col_indices = channel_col_map[ch]
        axes_labels = ["X", "Y", "Z"]
        for i, cidx in enumerate(col_indices[:3]):  # max 3 osie
            axis = axes_labels[i] if i < len(axes_labels) else str(i)
            col_name = f"{ch}_{axis}"
            try:
                selected_cols[col_name] = pd.to_numeric(
                    data_raw.iloc[:, cidx], errors="coerce"
                )
            except Exception:
                pass

    df = pd.DataFrame(selected_cols)

    # Usuń wiersze gdzie Time jest NaN (stopki C3D artefakty)
    df = df.dropna(subset=["Time"]).reset_index(drop=True)

    if len(df) < 10:
        return None

    return df


# ---------------------------------------------------------------------------
# Normalizacja cyklu do stałej długości
# ---------------------------------------------------------------------------

def normalize_cycle_length(df: pd.DataFrame, n_samples: int = N_SAMPLES) -> np.ndarray:
    """
    Interpoluje wszystkie sygnały do stałej liczby próbek (normalizacja cyklu).

    Zwraca array shape (n_samples, n_channels).
    """
    feature_cols = [c for c in df.columns if c != "Time"]
    n_orig = len(df)

    x_orig = np.linspace(0, 1, n_orig)
    x_new  = np.linspace(0, 1, n_samples)

    result = np.zeros((n_samples, len(feature_cols)), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        y = df[col].values.astype(np.float64)
        # Zastąp NaN interpolacją liniową
        nans = np.isnan(y)
        if nans.all():
            result[:, i] = 0.0
            continue
        if nans.any():
            y[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], y[~nans])

        # Lekkie wygładzenie Savitzky-Golay przed interpolacją
        if n_orig >= 11:
            y = savgol_filter(y, window_length=min(11, n_orig if n_orig % 2 == 1 else n_orig - 1), polyorder=3)

        f = interp1d(x_orig, y, kind="linear", fill_value="extrapolate")
        result[:, i] = f(x_new).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Skan katalogu datasetu
# ---------------------------------------------------------------------------

def find_cycle_files(
    dataset_root: str | Path,
    conditions: list[str] = WALK_CONDITIONS,
    sessions: list[str] = SESSIONS,
) -> list[dict]:
    """
    Przeszukuje drzewo katalogów datasetu i zwraca listę słowników:
        {
            "path": Path(...),
            "subject_id": "GF022",
            "gender": "F",          # "F", "M" lub None
            "session": "Session1",
            "condition": "Walk_Comfortable",
            "cycle_num": 3
        }
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records = []

    # Wzorzec: {root}/{SubjectID}/{Session}/{Overground_Walk}/{condition}/Post_Process/{condition}{N}.csv
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

                # Pliki cykli: Walk_Comfortable1.csv, Walk_Comfortable2.csv, ...
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
                            "cycle_num":  int(m.group(1)),
                        })

    return records


# ---------------------------------------------------------------------------
# Główny loader — buduje gotowy dataset
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_root: str | Path,
    conditions: list[str] = WALK_CONDITIONS,
    sessions: list[str] = SESSIONS,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
    num_workers: int = 1,
) -> dict:
    """
    Wczytuje cały dataset i zwraca słownik:
        {
            "X":        np.ndarray  shape (N, n_samples, n_channels),
            "y":        np.ndarray  shape (N,)  0=female, 1=male
            "gender":   list[str]   "F" lub "M" dla każdej próbki
            "meta":     list[dict]  metadane każdej próbki
            "channels": list[str]   nazwy kanałów (kolumny)
            "stats":    dict        liczba próbek per płeć/warunek
        }
    Próbki o nieznanej płci są POMIJANE.
    """
    records = find_cycle_files(dataset_root, conditions, sessions)

    if verbose:
        print(f"Znaleziono {len(records)} plików CSV (wszystkie płcie).")

    X_list, y_list, gender_list, meta_list = [], [], [], []
    channels = None
    stats = {"F": 0, "M": 0, "unknown": 0, "errors": 0}

    def handle_record(rec: dict):
        gender = rec["gender"]
        if gender is None:
            return "unknown", rec, None, 0.0

        start = time.perf_counter()
        df = parse_mocap_csv(rec["path"])
        elapsed = time.perf_counter() - start
        if df is None:
            return "error", rec, None, elapsed

        return "ok", rec, df, elapsed

    worker_count = max(1, int(num_workers))
    use_threads = worker_count > 1 and len(records) > 1
    if use_threads:
        worker_count = min(worker_count, len(records))
        if verbose:
            print(f"Przetwarzanie CSV równolegle: {worker_count} wątków")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(handle_record, rec) for rec in records]
            for idx, future in enumerate(as_completed(futures), start=1):
                status, rec, df, elapsed = future.result()

                if verbose:
                    print(f"  [PLIK] {rec['path'].name} | {elapsed:.2f}s")

                if status == "unknown":
                    stats["unknown"] += 1
                    continue
                if status == "error":
                    stats["errors"] += 1
                    continue

                # Zapamiętaj nazwy kanałów z pierwszego poprawnego pliku
                if channels is None:
                    channels = [c for c in df.columns if c != "Time"]

                cycle = normalize_cycle_length(df, n_samples)

                # Upewnij się że wymiary są spójne
                if cycle.shape != (n_samples, len(channels)):
                    stats["errors"] += 1
                    continue

                gender = rec["gender"]
                label = 0 if gender == "F" else 1
                X_list.append(cycle)
                y_list.append(label)
                gender_list.append(gender)
                meta_list.append(rec)
                stats[gender] += 1

                if verbose and (idx == 1 or idx % 50 == 0 or idx == len(records)):
                    print(
                        f"  [POSTĘP] {idx:4d}/{len(records)} | "
                        f"F={stats['F']} M={stats['M']} unknown={stats['unknown']} errors={stats['errors']}"
                    )
    else:
        for idx, result in enumerate(map(handle_record, records), start=1):
            status, rec, df, elapsed = result

            if verbose:
                print(f"  [PLIK] {rec['path'].name} | {elapsed:.2f}s")

            if status == "unknown":
                stats["unknown"] += 1
                continue
            if status == "error":
                stats["errors"] += 1
                continue

            # Zapamiętaj nazwy kanałów z pierwszego poprawnego pliku
            if channels is None:
                channels = [c for c in df.columns if c != "Time"]

            cycle = normalize_cycle_length(df, n_samples)

            # Upewnij się że wymiary są spójne
            if cycle.shape != (n_samples, len(channels)):
                stats["errors"] += 1
                continue

            gender = rec["gender"]
            label = 0 if gender == "F" else 1
            X_list.append(cycle)
            y_list.append(label)
            gender_list.append(gender)
            meta_list.append(rec)
            stats[gender] += 1

            if verbose and (idx == 1 or idx % 50 == 0 or idx == len(records)):
                print(
                    f"  [POSTĘP] {idx:4d}/{len(records)} | "
                    f"F={stats['F']} M={stats['M']} unknown={stats['unknown']} errors={stats['errors']}"
                )

    if not X_list:
        raise ValueError("Nie wczytano żadnych danych. Sprawdź ścieżkę do datasetu.")

    X = np.stack(X_list, axis=0)   # (N, n_samples, n_channels)
    y = np.array(y_list, dtype=np.int8)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Wczytano próbek:   {len(X)}")
        print(f"  Kobiety (F=0):     {stats['F']}")
        print(f"  Mężczyźni (M=1):   {stats['M']}")
        print(f"  Nieznana płeć:     {stats['unknown']}")
        print(f"  Błędy wczytywania: {stats['errors']}")
        print(f"  Shape X:           {X.shape}")
        print(f"  Kanały ({len(channels)}):")
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
# Normalizacja amplitudy (Z-score per kanał)
# ---------------------------------------------------------------------------

def compute_normalization_stats(X_train: np.ndarray) -> dict:
    """
    Oblicza mean i std per kanał na zbiorze treningowym (kobiety).
    X_train shape: (N, n_samples, n_channels)
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, n_channels)
    std  = X_train.std(axis=(0, 1), keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)             # unikaj dzielenia przez 0
    return {"mean": mean, "std": std}


def normalize_zscore(X: np.ndarray, stats: dict) -> np.ndarray:
    """Normalizuje X z-score używając statystyk z treningu."""
    return (X - stats["mean"]) / stats["std"]


def denormalize_zscore(X_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Odwraca normalizację z-score."""
    return X_norm * stats["std"] + stats["mean"]


# ---------------------------------------------------------------------------
# Podział na train/val/test
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: dict,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = 42,
) -> dict:
    """
    Strategia podziału:
      - TRAIN: tylko kobiety (70% cykli kobiet)
      - VAL:   tylko kobiety (15% cykli kobiet)  -> do strojenia i progu
      - TEST:  pozostałe kobiety (15%) + WSZYSCY mężczyźni

    Podział per-uczestnik (subject-wise), żeby uniknąć data leakage.
    """
    rng = np.random.default_rng(seed)

    meta = dataset["meta"]
    y    = dataset["y"]
    X    = dataset["X"]

    # Zbierz unikalne ID uczestniczek
    female_subjects = sorted(set(
        m["subject_id"] for m, label in zip(meta, y) if label == 0
    ))
    rng.shuffle(female_subjects)

    n_f  = len(female_subjects)
    n_tr = int(n_f * train_ratio)
    n_va = int(n_f * val_ratio)

    train_subjects = set(female_subjects[:n_tr])
    val_subjects   = set(female_subjects[n_tr:n_tr + n_va])
    # reszta kobiet -> test

    idx_train, idx_val, idx_test = [], [], []

    for i, (m, label) in enumerate(zip(meta, y)):
        sid = m["subject_id"]
        if label == 1:
            # Mężczyźni zawsze w test
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

    print("Podział datasetu (subject-wise):")
    print(f"  TRAIN: {len(idx_train):4d} próbek | "
          f"F={sum(y[idx_train]==0)} M={sum(y[idx_train]==1)} | "
          f"uczestniczki: {len(train_subjects)}")
    print(f"  VAL:   {len(idx_val):4d} próbek | "
          f"F={sum(y[idx_val]==0)} M={sum(y[idx_val]==1)} | "
          f"uczestniczki: {len(val_subjects)}")
    print(f"  TEST:  {len(idx_test):4d} próbek | "
          f"F={sum(y[idx_test]==0)} M={sum(y[idx_test]==1)}")

    return splits


# ---------------------------------------------------------------------------
# Podgląd danych (diagnostyka)
# ---------------------------------------------------------------------------

def describe_dataset(dataset: dict) -> None:
    """Wyświetla podstawowe statystyki datasetu."""
    X, y, channels = dataset["X"], dataset["y"], dataset["channels"]
    print(f"\nShape:      {X.shape}  (próbki × czas × kanały)")
    print(f"Etykiety:   0=F ({(y==0).sum()}), 1=M ({(y==1).sum()})")
    print(f"\nStatystyki kanałów (min / mean / max) — pierwsze 5:")
    for i, ch in enumerate(channels[:5]):
        vals = X[:, :, i]
        print(f"  {ch:<35s}  min={vals.min():8.2f}  mean={vals.mean():8.2f}  max={vals.max():8.2f}")
    if len(channels) > 5:
        print(f"  ... ({len(channels) - 5} więcej kanałów)")


# ---------------------------------------------------------------------------
# Punkt wejścia — przykład użycia
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Ścieżka do datasetu — podaj jako argument lub ustaw tu
    if len(sys.argv) > 1:
        ROOT = sys.argv[1]
    else:
        ROOT = "/mnt/e/ML_datasets/Data_Run_Walk"  # domyślna ścieżka z Twojego systemu

    print(f"Dataset root: {ROOT}\n")

    # 1. Wczytaj wszystkie dane
    dataset = load_dataset(
        dataset_root=ROOT,
        conditions=WALK_CONDITIONS,   # ["Walk_Comfortable", "Walk_Fast", "Walk_Slow"]
        sessions=SESSIONS,
        n_samples=N_SAMPLES,
        verbose=True,
    )

    # 2. Opisz dataset
    describe_dataset(dataset)

    # 3. Podziel na train/val/test
    splits = split_dataset(dataset, train_ratio=0.70, val_ratio=0.15)

    # 4. Oblicz statystyki normalizacji na zbiorze treningowym
    norm_stats = compute_normalization_stats(splits["train"]["X"])

    # 5. Zastosuj normalizację
    for split_name in ["train", "val", "test"]:
        splits[split_name]["X_norm"] = normalize_zscore(
            splits[split_name]["X"], norm_stats
        )

    print("\nNormalizacja Z-score zastosowana.")
    print(f"Mean shape: {norm_stats['mean'].shape}")
    print(f"Std shape:  {norm_stats['std'].shape}")

    # 6. Zapis do pliku .npz (opcjonalnie)
    save_path = Path(ROOT) / "processed_dataset.npz"
    np.savez_compressed(
        save_path,
        X_train=splits["train"]["X_norm"],
        y_train=splits["train"]["y"],
        X_val=splits["val"]["X_norm"],
        y_val=splits["val"]["y"],
        X_test=splits["test"]["X_norm"],
        y_test=splits["test"]["y"],
        norm_mean=norm_stats["mean"],
        norm_std=norm_stats["std"],
        channels=np.array(splits["channels"]),
    )
    print(f"\nZapisano przetworzone dane: {save_path}")
