import c3d
import numpy as np

# =============================
# 0) Read C3D file
# =============================
name_c3d_file = "/mnt/e/ML_datasets/Data_Run_Walk/AJ026/Session1/Treadmill_Run/Treadmill_Run_Comfortable/Post_Process/Treadmill_Run_Comfortable.c3d"

with open(name_c3d_file, 'rb') as handle:
    reader = c3d.Reader(handle)

    # =============================
    # 1) Metadata
    # =============================
    first_frame = reader.header.first_frame
    last_frame = reader.header.last_frame
    number_points_frame = last_frame - first_frame
    frequency_points = reader.header.frame_rate

    # Pobranie pierwszej ramki
    points0, analogs0 = next(reader.read_frames())
    analog_subframes = analogs0.shape[0] if analogs0 is not None else 1
    frequency_analogs = frequency_points * analog_subframes

    print("Points rate:", frequency_points, "Hz")
    print("Analogs rate:", frequency_analogs, "Hz")

    # Anthropometry (jeśli istnieją)
    height_subject = None
    bodymass_subject = None
    try:
        processing = reader.groups['PROCESSING']
        height_subject = processing.get('HEIGHT').float_array[0]
        bodymass_subject = processing.get('BODYMASS').float_array[0]
    except Exception:
        pass

    # =============================
    # 2) Points (markers)
    # =============================
    all_points = {}
    point_labels = reader.point_labels
    for label in point_labels:
        if label.strip():
            all_points[label] = []

    # =============================
    # 3) Analogs
    # =============================
    all_analogs = {}
    analog_labels = reader.analog_labels
    for label in analog_labels:
        if label.strip():
            all_analogs[label] = []

    # =============================
    # 4) Iteracja po wszystkich ramkach
    # =============================
    # Pierwsza ramka już pobrana
    frames = [(points0, analogs0)] + list(reader.read_frames())
    for points, analogs in frames:
        # --- Points ---
        for idx, label in enumerate(point_labels):
            if not label.strip():
                continue
            all_points[label].append(points[idx, :3])  # tylko XYZ

        # --- Analogs ---
        if analogs is not None:
            for ch_idx, label in enumerate(analog_labels):
                if not label.strip():
                    continue
                all_analogs[label].extend(analogs[:, ch_idx])

    # Konwersja do numpy
    for k in all_points:
        all_points[k] = np.array(all_points[k])
    for k in all_analogs:
        all_analogs[k] = np.array(all_analogs[k])

    # =============================
    # 5) Events
    # =============================
    all_events = {}
    try:
        events_group = reader.groups['EVENT']
        contexts = events_group.get('CONTEXTS').string_array
        labels = events_group.get('LABELS').string_array
        times = events_group.get('TIMES').float_array.reshape(-1, 2)

        for i in range(len(labels)):
            side_event = contexts[i].strip()
            name_event = labels[i].strip().replace(" ", "_")
            key = f"{side_event}_{name_event}"
            if key not in all_events:
                all_events[key] = []

            # konwersja czasu (s) → frame index
            time_sec = times[i][1]
            frame = int(round(time_sec * frequency_points))
            all_events[key].append(frame)

    except Exception:
        pass

    # =============================
    # 6) Gait cycles example (hip/knee/ankle sagittal angles)
    # =============================
    gait_data = {}
    for side in ['Left', 'Right']:
        key = f"{side}_Foot_Strike"
        if key not in all_events:
            continue

        number_cycles = len(all_events[key]) - 1
        for c in range(number_cycles):
            begin_cycle = all_events[key][c] - first_frame
            end_cycle = all_events[key][c+1] - first_frame
            for joint in ["Hip", "Knee", "Ankle"]:
                label = f"{side[0]}{joint}Angles"
                if label not in all_points:
                    continue
                kinematic_data = all_points[label][begin_cycle:end_cycle]
                kinematic_sagittal_plane = kinematic_data[:, 0]

                # zapis do dict
                gait_data.setdefault(side, {}).setdefault(joint, []).append(kinematic_sagittal_plane)

print("Pipeline loaded successfully!")
print("Markers:", list(all_points.keys())[:5])
print("Analogs:", list(all_analogs.keys())[:5])
print("Events:", list(all_events.keys()))