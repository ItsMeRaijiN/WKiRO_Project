import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data_reader import (
    EXPECTED_FEATURE_COLUMNS,
    GENDER_MAP,
    load_dataset,
    normalize_cycle_length,
    parse_mocap_recording,
    segment_gait_cycles,
    split_dataset,
)
from evaluate import aggregate_scores_by_subject, evaluate, select_threshold
from model import Conv1dCAE
from run_experiment import (
    CacheValidationError,
    _cache_metadata,
    _normalize_splits,
    _synthetic_splits,
    build_synthetic_dataset,
    load_processed_cache,
    parse_args,
    prepare_dataset,
    save_processed_cache,
)


def write_recording_fixture(path: Path, frame_count: int = 61) -> None:
    channel_names = [column.rsplit("_", 1)[0] for column in EXPECTED_FEATURE_COLUMNS]
    unique_channels = list(dict.fromkeys(channel_names))

    names = ["Time"]
    units = ["s"]
    axes = [""]
    for channel in unique_channels:
        names.extend([channel, "", ""])
        units.extend(["unit", "unit", "unit"])
        axes.extend(["X", "Y", "Z"])

    rows = [
        ["FrameNumber", str(frame_count)],
        ["FirstFrame", "0"],
        ["PointFrequency", "100"],
        ["AnalogFrequency", "1000"],
        [],
        ["Left_Foot_Strike", "0.00", "0.30", "0.60"],
        ["Right_Foot_Strike", "0.00", "0.30", "0.60"],
        [],
        names,
        units,
        axes,
    ]

    for frame in range(frame_count):
        time_value = frame / 100
        values = [f"{time_value:.2f}"]
        for channel_index, _ in enumerate(EXPECTED_FEATURE_COLUMNS):
            values.append(str(time_value + channel_index))
        rows.append(values)

    rows.extend(
        [
            [],
            ["Time", "Force_Fx1"],
            ["s", "N"],
            ["", "X"],
            *[[str(index / 1000), str(index)] for index in range(200)],
        ]
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


class ParserTests(unittest.TestCase):
    def test_parser_stops_before_later_csv_sections_and_segments_cycles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "recording.csv"
            write_recording_fixture(path)

            recording = parse_mocap_recording(path)
            self.assertIsNotNone(recording)
            self.assertEqual(recording["data"].shape, (61, 43))
            self.assertEqual(recording["frame_count"], 61)

            cycles = segment_gait_cycles(recording)
            self.assertEqual(len(cycles), 4)
            self.assertEqual({cycle["side"] for cycle in cycles}, {"Left", "Right"})

            normalized = [
                normalize_cycle_length(cycle["data"], n_samples=101)
                for cycle in cycles
            ]
            self.assertTrue(all(array.shape == (101, 42) for array in normalized))
            self.assertTrue(all(np.isfinite(array).all() for array in normalized))

    def test_dataset_loader_emits_individual_cycles_with_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording_directory = (
                root
                / "AJ026"
                / "Session1"
                / "Overground_Walk"
                / "Walk_Comfortable"
                / "Post_Process"
            )
            recording_directory.mkdir(parents=True)
            write_recording_fixture(
                recording_directory / "Walk_Comfortable1.csv"
            )

            dataset = load_dataset(
                root,
                conditions=["Walk_Comfortable"],
                sessions=["Session1"],
                n_samples=101,
                verbose=False,
            )
            self.assertEqual(dataset["X"].shape, (4, 101, 42))
            self.assertTrue(np.all(dataset["y"] == 0))
            self.assertEqual(
                {item["side"] for item in dataset["meta"]},
                {"Left", "Right"},
            )
            self.assertTrue(
                all(item["recording_num"] == 1 for item in dataset["meta"])
            )

    def test_gender_map_matches_official_counts_and_corrected_subjects(self):
        values, counts = np.unique(list(GENDER_MAP.values()), return_counts=True)
        self.assertEqual(dict(zip(values, counts, strict=True)), {"F": 14, "M": 16})
        self.assertEqual(GENDER_MAP["SA017"], "F")
        self.assertEqual(GENDER_MAP["TK029"], "F")


class SplitAndMetricTests(unittest.TestCase):
    def test_small_subject_split_is_non_empty_and_leak_free(self):
        subjects = ["F001", "F002", "F003", "M001"]
        dataset = {
            "X": np.zeros((4, 10, 2), dtype=np.float32),
            "y": np.asarray([0, 0, 0, 1], dtype=np.int8),
            "gender": ["F", "F", "F", "M"],
            "meta": [{"subject_id": subject} for subject in subjects],
            "channels": ["a", "b"],
        }

        splits = split_dataset(dataset, seed=42)
        train_subjects = {item["subject_id"] for item in splits["train"]["meta"]}
        val_subjects = {item["subject_id"] for item in splits["val"]["meta"]}
        test_subjects = {item["subject_id"] for item in splits["test"]["meta"]}

        self.assertTrue(train_subjects)
        self.assertTrue(val_subjects)
        self.assertIn("M001", test_subjects)
        self.assertFalse(train_subjects & val_subjects)
        self.assertFalse(train_subjects & test_subjects)
        self.assertFalse(val_subjects & test_subjects)

    def test_upper_tail_threshold_and_evaluation_validation(self):
        scores = np.arange(100, dtype=np.float64)
        threshold = select_threshold(scores, percentile=95)
        self.assertEqual(int((scores >= threshold).sum()), 5)

        with self.assertRaises(ValueError):
            evaluate(np.asarray([0.1, 0.2]), np.asarray([0, 0]))

    def test_subject_aggregation_averages_cycles(self):
        scores, labels = aggregate_scores_by_subject(
            np.asarray([1.0, 3.0, 5.0]),
            np.asarray([0, 0, 1]),
            [
                {"subject_id": "F001"},
                {"subject_id": "F001"},
                {"subject_id": "M001"},
            ],
        )
        np.testing.assert_allclose(scores, [2.0, 5.0])
        np.testing.assert_array_equal(labels, [0, 1])


class CacheAndCliTests(unittest.TestCase):
    def test_smoke_test_never_loads_an_existing_real_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "stale.npz"
            np.savez(
                cache_path,
                norm_mean=np.zeros((1, 1, 7)),
                norm_std=np.ones((1, 1, 7)),
            )
            args = parse_args(
                [
                    "--smoke-test",
                    "--cache-file",
                    str(cache_path),
                    "--epochs",
                    "1",
                ]
            )
            _, splits, norm = prepare_dataset(args)
            self.assertEqual(splits["train"]["X"].shape[-1], 6)
            self.assertEqual(norm["mean"].shape[-1], 6)

    def test_cache_roundtrip_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "processed.npz"
            args = parse_args(
                [
                    "--dataset-root",
                    str(Path(temporary_directory) / "dataset"),
                    "--cache-file",
                    str(cache_path),
                ]
            )
            dataset = build_synthetic_dataset(seed=args.seed)
            splits = _synthetic_splits(dataset)
            norm = _normalize_splits(splits)
            metadata = _cache_metadata(args)
            save_processed_cache(cache_path, splits, norm, metadata)

            loaded_splits, loaded_norm = load_processed_cache(cache_path, metadata)
            self.assertEqual(loaded_splits["channels"], splits["channels"])
            np.testing.assert_allclose(loaded_norm["mean"], norm["mean"])

            mismatched_metadata = {**metadata, "seed": metadata["seed"] + 1}
            with self.assertRaises(CacheValidationError):
                load_processed_cache(cache_path, mismatched_metadata)

    def test_cli_rejects_zero_epochs_and_defaults_to_upper_percentile(self):
        self.assertEqual(parse_args([]).val_percentile, 95.0)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--epochs", "0"])


class ModelTests(unittest.TestCase):
    def test_model_preserves_time_and_channel_dimensions(self):
        model = Conv1dCAE(n_channels=6, latent_channels=16)
        input_tensor = torch.randn(2, 101, 6)
        self.assertEqual(model(input_tensor).shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()
