import os
import sys
import io
import pandas as pd
import pyxdf
from datetime import datetime
from zoneinfo import ZoneInfo
from glob import glob
import traceback
from lib.config import (
    UNSYNCED_DIR, SHARED_DIR,
    PSYCHOPY_START_TIMES,
    QUESTIONNAIRE_MARKERS, RELAX_MARKERS, RELAX_MARKERS_LABELS,
    TASK_MARKERS, TASK_MARKERS_LABELS,
)

RECORDING_TZ = ZoneInfo("Europe/Zurich")

# Path constants
unsynced_dir = str(UNSYNCED_DIR)
shared_dir = str(SHARED_DIR)

# Note: In sync.py, s_026 is actually subject 1 (s_001 is mapped to 26),
# "SYNC_SUBJECT_FOLDER = os.path.join(SYNC_DATA_FOLDER, f's_%03d' % (subject + 25))"
# So the index in the psychopy_start_time list is int(subject_id[2:]) - 26
def get_lsl_intervals(subject_num):
    # Subject index for psychopy times
    subject_index = subject_num - 26

    # LSL file index mapping (1 = s_026, 2 = s_027, etc.)
    # In the original data, lsl folders are named sub-xx...
    # We must find the folder containing the subject number
    lsl_idx = subject_num - 25

    lsl_folder = os.path.join(unsynced_dir, 'lsl')
    lsl_subject_folders = [f for f in os.listdir(lsl_folder) if f"{lsl_idx:02d}" in f]

    if not lsl_subject_folders:
        print(
            f"    Missing LSL folder for subject {subject_num} (searched for {lsl_idx:02d})"
        )
        return None, None

    lsl_subject_folder = lsl_subject_folders[0]
    eeg_dir = os.path.join(lsl_folder, lsl_subject_folder, "ses-S001", "eeg")

    if not os.path.exists(eeg_dir):
        print(f"    Missing EEG dir: {eeg_dir}")
        return None, None

    xdf_files = [
        f for f in os.listdir(eeg_dir) if f.endswith(".xdf") and "old" not in f
    ]
    if not xdf_files:
        print(f"    Missing XDF file in {eeg_dir}")
        return None, None

    xdf_path = os.path.join(eeg_dir, xdf_files[0])

    # Load XDF
    text_trap = io.StringIO()
    sys.stdout = text_trap
    try:
        data, header = pyxdf.load_xdf(xdf_path)
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"    Failed to load XDF: {e}")
        return None, None
    sys.stdout = sys.__stdout__

    lsl_start_timestamp = datetime.strptime(
        PSYCHOPY_START_TIMES[subject_index], "%Y-%m-%d %Hh%M.%S.%f%z"
    ).timestamp()

    try:
        stream_index = [
            i
            for i, stream in enumerate(data)
            if stream["info"]["name"][0].lower() == "lindapsychopy"
        ][0]
    except IndexError:
        print(f"    Could not find 'lindapsychopy' stream in {xdf_path}")
        return None, None

    ts = pd.DataFrame(data[stream_index]["time_series"], columns=["timeseries"])
    ts["timeseries"] = ts["timeseries"].str.lower()
    ts["local_timestamps"] = data[stream_index]["time_stamps"]
    ts["local_timestamps"] = ts["local_timestamps"] - ts["local_timestamps"][0]
    ts["timestamps"] = ts["local_timestamps"] + lsl_start_timestamp
    ts["timestamps"] = ts["timestamps"].astype(float)

    # Parsing logic straight from sync.py
    questionnaire_markers = QUESTIONNAIRE_MARKERS
    relax_markers = list(RELAX_MARKERS)
    relax_markers_labels = list(RELAX_MARKERS_LABELS)
    task_markers = list(TASK_MARKERS)
    task_markers_labels = list(TASK_MARKERS_LABELS)

    current_task = None
    current_label = None
    labels = []
    to_be_amended = []

    if lsl_idx == 28:
        task_markers = task_markers[2:]
        task_markers_labels = task_markers_labels[2:]
        start_timestamp = ts.loc[2094, "timestamps"]
        ts = ts[ts["timestamps"] > start_timestamp]

    for index, row in ts.iterrows():
        if (
            any(
                row["timeseries"].startswith(marker) for marker in questionnaire_markers
            )
            or row["timeseries"].startswith("hello")
            or row["timeseries"].startswith("ready")
            or row["timeseries"].startswith("thanks for participating")
        ):
            if (
                index > 0
                and "ready" in ts.loc[index - 1, "timeseries"]
                and row["timestamps"] - ts.loc[index - 1, "timestamps"] > 2
                and current_label in task_markers_labels
                and task_markers_labels.index(current_label) > 7
            ):
                to_be_amended.append((index - 1, index))
            labels.append("questionnaire")
        elif any(marker in row["timeseries"] for marker in relax_markers):
            current_task = next(
                marker for marker in relax_markers if marker in row["timeseries"]
            )
            current_label = relax_markers_labels[relax_markers.index(current_task)]
            labels.append(current_label)
        elif any(marker in row["timeseries"] for marker in task_markers):
            current_task = next(
                marker for marker in task_markers if marker in row["timeseries"]
            )
            current_label = task_markers_labels[task_markers.index(current_task)]
            labels.append("questionnaire")
        else:
            labels.append(current_label)

    ts["label"] = labels

    labels = task_markers_labels[-4:]
    for index, (start, end) in enumerate(to_be_amended):
        # We need to use valid indices that exist in the original index.
        # But wait, original code used .loc on the original index.
        # Since we concat and do not reset_index inside the loop, the index becomes non-unique or non-monotonic.
        # Let's just use .iloc because we know it corresponds to the position in the original dataframe.
        # However, to be safe, we just capture the values BEFORE we start concatenating.
        start_lt = ts["local_timestamps"].values[start]
        end_lt = ts["local_timestamps"].values[end]
        start_t = ts["timestamps"].values[start]
        end_t = ts["timestamps"].values[end]

        ts = pd.concat(
            [
                ts,
                pd.DataFrame.from_dict(
                    {
                        "timeseries": ["task_marker"] * 2,
                        "local_timestamps": [start_lt + 0.1, end_lt - 0.1],
                        "timestamps": [start_t + 0.1, end_t - 0.1],
                        "label": [labels[index]] * 2,
                    }
                ),
            ],
            ignore_index=True,
        )
    ts = ts.sort_values(by="timestamps").reset_index(drop=True)

    intervals = ts.loc[
        ts["label"].ne(ts["label"].shift()) | ts["label"].ne(ts["label"].shift(-1))
    ]

    streams_reference = {}
    questionnaire_count = 1
    for index, row in intervals.iterrows():
        if (
            row["label"] == "questionnaire"
            and f"questionnaire_{questionnaire_count}" not in streams_reference
        ):
            streams_reference[f"questionnaire_{questionnaire_count}"] = row[
                "timestamps"
            ]
        elif row["label"] == "questionnaire":
            streams_reference[f"questionnaire_{questionnaire_count}"] = (
                streams_reference[f"questionnaire_{questionnaire_count}"],
                row["timestamps"],
            )
            questionnaire_count += 1
        else:
            if row["label"] not in streams_reference:
                streams_reference[row["label"]] = row["timestamps"]
            else:
                streams_reference[row["label"]] = (
                    streams_reference[row["label"]],
                    row["timestamps"],
                )

    if "questionnaire_16" not in streams_reference:
        last_questionnaire = max(
            [
                int(key.split("_")[1])
                for key in streams_reference.keys()
                if "questionnaire" in key
            ]
        )
        streams_reference["passive_viewing"] = (
            streams_reference["passive_viewing"][0],
            streams_reference[f"questionnaire_{last_questionnaire}"][0],
        )
    else:
        streams_reference["passive_viewing"] = (
            streams_reference["passive_viewing"][0],
            streams_reference["questionnaire_16"][0],
        )

    for (phase1, times1), (phase2, times2) in zip(
        streams_reference.items(), list(streams_reference.items())[1:]
    ):
        end1 = times1[1] if isinstance(times1, tuple) else times1
        start2 = times2[0] if isinstance(times2, tuple) else times2
        if start2 - end1 > 1:
            if "questionnaire" in phase1 and any(task in phase2 for task in ["images"]):
                new_start2 = start2 - int(start2 - end1)
                streams_reference[phase2] = (
                    new_start2,
                    times2[1] if isinstance(times2, tuple) else times2,
                )

    return streams_reference, lsl_start_timestamp


def assign_label(abs_time, streams_reference):
    for phase, times in streams_reference.items():
        if isinstance(times, tuple):
            if times[0] <= abs_time <= times[1]:
                return "questionnaire" if "questionnaire" in phase else phase
        else:
            # Catch edge cases where start/end are exact same float
            if abs_time == times:
                return "questionnaire" if "questionnaire" in phase else phase
    return "central_position"


def parse_rgb_timestamp(ts_str):
    # The camera logs local wall-clock time at USI (Lugano); pin the zone so the
    # result does not depend on the timezone of the machine running this script.
    dt = datetime.strptime(ts_str.strip(), "%Y-%m-%d$%H-%M-%S-%f").replace(tzinfo=RECORDING_TZ)
    return dt.timestamp()


def process_subject(subject_folder):
    subject_id = os.path.basename(subject_folder.rstrip("/"))
    subject_num = int(subject_id.split("_")[1])
    print(f"Processing {subject_id}...")

    rgb_path = os.path.join(subject_folder, "rgb", f"{subject_id}.txt")
    eyetracker_path = os.path.join(subject_folder, "eyetracker", f"{subject_id}.tsv")

    if not os.path.exists(rgb_path):
        print(f"  -> Missing RGB frame-clock file at {rgb_path}. Skipping.")
        return
    if not os.path.exists(eyetracker_path):
        print(f"  -> Missing eye-tracker file at {eyetracker_path}. Skipping.")
        return

    # 1. Get exact intervals from LSL
    streams_reference, lsl_start_time = get_lsl_intervals(subject_num)
    if streams_reference is None:
        return

    # 2. Process the RGB frame clock
    rgb_data = []
    with open(rgb_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    abs_time = parse_rgb_timestamp(parts[1])
                    rgb_data.append({"abs_time": abs_time})
                except ValueError:
                    continue

    rgb_df = pd.DataFrame(rgb_data).sort_values("abs_time")

    # Apply labels
    rgb_df["label"] = rgb_df["abs_time"].apply(
        lambda t: assign_label(t, streams_reference)
    )

    # Calculate relative timestamp
    rgb_df["timestamp"] = (rgb_df["abs_time"] - lsl_start_time).round(3)
    rgb_out = rgb_df[["timestamp", "label"]]

    rgb_out_path = os.path.join(subject_folder, "rgb", f"{subject_id}.csv")
    rgb_out.to_csv(rgb_out_path, index=False)

    # 3. Process eye-tracker (Tobii) data
    # Skip the first 7 lines (pygaze header)
    eyetracker_df = pd.read_csv(eyetracker_path, sep="\t", skiprows=7)

    # Rename TimeStamp to timestamp
    if "TimeStamp" in eyetracker_df.columns:
        eyetracker_df = eyetracker_df.rename(columns={"TimeStamp": "timestamp"})

    # Drop distance columns
    cols_to_drop = ["DistanceLeft", "DistanceRight", "AverageDistance"]
    eyetracker_df = eyetracker_df.drop(columns=[c for c in cols_to_drop if c in eyetracker_df.columns])

    # Relative timestamp in seconds (original is in ms)
    eyetracker_start_ms = eyetracker_df["timestamp"].iloc[0]

    # Create an absolute time column matching the lsl start
    # Since Tobii starts at the exact same time as the RGB camera/LSL, we add the relative seconds to the true start
    eyetracker_df["abs_time"] = lsl_start_time + (
        (eyetracker_df["timestamp"] - eyetracker_start_ms) / 1000.0
    )

    # Assign labels
    eyetracker_df["label"] = eyetracker_df["abs_time"].apply(
        lambda t: assign_label(t, streams_reference)
    )

    # Create final relative timestamp (starting at 0.0)
    eyetracker_df["timestamp"] = (eyetracker_df["abs_time"] - lsl_start_time).round(3)

    if "abs_time" in eyetracker_df.columns:
        eyetracker_df = eyetracker_df.drop(columns=["abs_time"])

    eyetracker_out_path = os.path.join(subject_folder, "eyetracker", f"{subject_id}.csv")
    eyetracker_df.to_csv(eyetracker_out_path, index=False)

    print(f"  -> Saved {rgb_out_path}")
    print(f"  -> Saved {eyetracker_out_path}")


if __name__ == "__main__":
    # Process all s_026 to s_030... or simply process whatever is in data/shared
    folders = glob(os.path.join(shared_dir, 's_*', ''))
    for folder in sorted(folders):
        try:
            process_subject(folder)
        except Exception as e:
            print(f"Failed processing {folder}: {e}")
            traceback.print_exc()
