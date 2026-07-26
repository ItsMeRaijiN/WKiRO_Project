import argparse
from pathlib import Path

import c3d
import numpy as np


def _clean_label(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()

def _parameter_array(reader: c3d.Reader, key: str, attribute: str):
    parameter = reader.get(key)
    return None if parameter is None else getattr(parameter, attribute)

def read_c3d(filepath: str | Path) -> dict:
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"C3D file does not exist: {filepath}")

    with filepath.open("rb") as handle:
        reader = c3d.Reader(handle)
        first_frame = reader.first_frame
        last_frame = reader.last_frame
        point_rate = float(reader.point_rate)
        analog_rate = float(reader.analog_rate)

        point_labels = [_clean_label(label) for label in reader.point_labels]
        analog_labels = [_clean_label(label) for label in reader.analog_labels]
        all_points = {label: [] for label in point_labels if label}
        all_analogs = {label: [] for label in analog_labels if label}

        frame_numbers = []
        for frame_number, points, analog in reader.read_frames():
            frame_numbers.append(int(frame_number))
            for index, label in enumerate(point_labels):
                if label:
                    all_points[label].append(points[index, :3])
            for index, label in enumerate(analog_labels):
                if label and analog.size:
                    all_analogs[label].extend(analog[index, :])

        all_points = {
            label: np.asarray(values, dtype=np.float64)
            for label, values in all_points.items()
        }
        all_analogs = {
            label: np.asarray(values, dtype=np.float64)
            for label, values in all_analogs.items()
        }

        events = {}
        contexts = _parameter_array(reader, "EVENT:CONTEXTS", "string_array")
        labels = _parameter_array(reader, "EVENT:LABELS", "string_array")
        times = _parameter_array(reader, "EVENT:TIMES", "float_array")
        if contexts is not None and labels is not None and times is not None:
            contexts = np.asarray(contexts).reshape(-1)
            labels = np.asarray(labels).reshape(-1)
            times = np.asarray(times).reshape(-1, 2)
            event_count = min(len(contexts), len(labels), len(times))
            for index in range(event_count):
                side = _clean_label(contexts[index])
                event_name = _clean_label(labels[index]).replace(" ", "_")
                time_sec = float(times[index, 1])
                frame = round(time_sec * point_rate)
                events.setdefault(f"{side}_{event_name}", []).append(frame)

        height = _parameter_array(reader, "PROCESSING:HEIGHT", "float_array")
        bodymass = _parameter_array(reader, "PROCESSING:BODYMASS", "float_array")

    expected_frame_count = last_frame - first_frame + 1
    if len(frame_numbers) != expected_frame_count:
        raise ValueError(
            f"C3D frame count mismatch: expected {expected_frame_count}, "
            f"read {len(frame_numbers)}."
        )

    return {
        "path": filepath,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frame_count": expected_frame_count,
        "point_rate": point_rate,
        "analog_rate": analog_rate,
        "height": None if height is None else float(np.asarray(height).reshape(-1)[0]),
        "bodymass": (
            None if bodymass is None else float(np.asarray(bodymass).reshape(-1)[0])
        ),
        "points": all_points,
        "analogs": all_analogs,
        "events": events,
    }

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c3d_file", help="Path to a C3D recording.")
    return parser.parse_args(argv)

def main(argv=None) -> None:
    args = parse_args(argv)
    result = read_c3d(args.c3d_file)

    print(f"File:         {result['path']}")
    print(f"Frames:       {result['frame_count']}")
    print(f"Points rate:  {result['point_rate']:.2f} Hz")
    print(f"Analogs rate: {result['analog_rate']:.2f} Hz")
    print(f"Point labels: {len(result['points'])}")
    print(f"Analog labels:{len(result['analogs']):2d}")
    print(f"Event groups: {len(result['events'])}")
    if result["height"] is not None:
        print(f"Height:       {result['height']:.1f} mm")
    if result["bodymass"] is not None:
        print(f"Body mass:    {result['bodymass']:.1f} kg")

    for event_name, frames in sorted(result["events"].items()):
        print(f"  {event_name}: {frames}")


if __name__ == "__main__":
    main()
