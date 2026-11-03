import os
import sys
import io
import pandas as pd
import pyxdf
from datetime import datetime
from rich.console import Console
console = Console()
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

import argparse
from lib.config import (
    DATA_DIR,
    PSYCHOPY_START_TIMES, PSYCHOPY_END_TIMES,
    QUESTIONNAIRE_MARKERS, RELAX_MARKERS, RELAX_MARKERS_LABELS,
    TASK_MARKERS, TASK_MARKERS_LABELS,
)

parser = argparse.ArgumentParser(description='Synchronize raw USI data (XDF + OCO)')
parser.add_argument('--base-path', type=str, default=None,
                    help='Override base data directory (default: <repo>/data)')
parser.add_argument('--output-suffix', type=str, default='_oco',
                    help='Suffix for output filenames (default: _oco, use empty string for no suffix)')
parser.add_argument('--zero-timestamps', action='store_true',
                    help='Zero-base timestamps and rename experimental_phase to label')
args = parser.parse_args()

output_suffix = args.output_suffix

if args.base_path:
    base_data = args.base_path
else:
    base_data = str(DATA_DIR)
SYNC_DATA_FOLDER = os.path.join(base_data, 'usi-synced')
UNSYNC_DATA_FOLDER = os.path.join(base_data, 'usi-unsynced')

subjects = range(1, 31)

def rename_column(col: str):
    if "Raw" in col:
        col = col.replace("/Raw", "")
    if "Software" in col:
        col = col.replace("Software", "Software_")
    col = col.replace("/", "_")
    col = col.replace(" ", "_")
    col = col.replace(".", "_")
    col = col.replace("[", "_")
    col = col.replace("]", "")
    if "Prox" in col or "Nav" in col or "Expression_Intensity" in col:
        col = "_".join(col.split("_")[:-1]).lower() + "_" + col.split("_")[-1]
    else:
        col = col.lower()
    return col

def oco_dataloader(path: str):
    with open(path) as f:
        lines = f.readlines()
        
        metadata_lines = lines[:40]
        metadata = {}
        for line in metadata_lines:
            if '[]' in line:
                key, value = line[1:].strip().split('[]')
                value = value.split(',')[1:]
            else:
                key, value = line[1:].strip().split(',')
            key_parts = key.split('/')
            key_parts[-1] = key_parts[-1].replace('.', ' ')
            key = '_'.join(key_parts)
            metadata[key] = value
        
        data_lines = lines[41:]
        cols = data_lines[0].strip().split(',')
        if 'LSLinletMarkers' in cols:
            cols.remove('LSLinletMarkers')
        data_lines = [line.strip().split(',') if len(line.strip().split(',')) < 57 else line.strip().split(',')[0:-1] for line in data_lines[1:]]
        data = pd.DataFrame.from_records(data_lines, columns=cols)
        data.columns = [rename_column(col) for col in data.columns]
    return metadata, data

for subject in subjects:
    # if subject == 25:
    #     continue
    with console.status(f'[bold green]Processing') as status:
        console.log(f"Subject {subject}")
        SYNC_SUBJECT_FOLDER = os.path.join(SYNC_DATA_FOLDER, f's_%03d' % (subject + 25))
        LSL_FOLDER = os.path.join(UNSYNC_DATA_FOLDER, 'lsl')
        LSL_SUBJECT_FOLDER = [f for f in os.listdir(LSL_FOLDER) if (f'%02d' % subject) in f][0]
        xdf_filename = [f for f in os.listdir(os.path.join(LSL_FOLDER, LSL_SUBJECT_FOLDER, 'ses-S001', 'eeg')) if f.endswith('.xdf') and 'old' not in f][0]
        xdf_path = os.path.join(LSL_FOLDER, LSL_SUBJECT_FOLDER, 'ses-S001', 'eeg', xdf_filename)

        console.log('\tLoading LSL file')
        text_trap = io.StringIO()
        sys.stdout = text_trap
        data, header = pyxdf.load_xdf(xdf_path)
        sys.stdout = sys.__stdout__

        lsl_start_timestamp = datetime.strptime(PSYCHOPY_START_TIMES[subject-1], "%Y-%m-%d %Hh%M.%S.%f%z").timestamp()
        lsl_end_timestamp = datetime.strptime(PSYCHOPY_END_TIMES[subject-1], "%Y-%m-%d %H:%M:%S").timestamp()
        
        stream_index = [i for i, stream in enumerate(data) if stream['info']['name'][0].lower() == 'lindapsychopy'][0]
        ts = pd.DataFrame(data[stream_index]['time_series'], columns=['timeseries'])
        ts['timeseries'] = ts['timeseries'].str.lower()
        ts['local_timestamps'] = data[stream_index]['time_stamps']
        ts['local_timestamps'] = ts['local_timestamps'] - ts['local_timestamps'][0]
        ts['timestamps'] = ts['local_timestamps'] + lsl_start_timestamp
        ts['timestamps'] = ts['timestamps'].astype(float)

        os.makedirs(os.path.join(SYNC_SUBJECT_FOLDER, 'oco'), exist_ok=True)
        OCO_SUBJECT_FOLDER = os.path.join(UNSYNC_DATA_FOLDER, 'oco', f'%02d' % subject)
        OCO_FILENAME = [f for f in os.listdir(OCO_SUBJECT_FOLDER) if f.endswith('.txt') and not 'missing' in f and not 'corrupted' in f]
        if len(OCO_FILENAME) == 0:
            log.warning('\tNo OCO data found')
            with open(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', 'oco_missing.txt'), 'w') as f:
                f.write('No OCO data found')
            continue
        OCO_FILENAME = OCO_FILENAME[0]

        console.log('\tSync OCO start to LSL file creation')
        oco_metadata, oco_data = oco_dataloader(os.path.join(OCO_SUBJECT_FOLDER, OCO_FILENAME))

        if 'software_timestamp' not in oco_data.columns:
            oco_data['software_timestamp'] = oco_data['timestamp']
        oco_data['software_timestamp'] = oco_data['software_timestamp'].astype(float)
        oco_data['timestamp'] = oco_data['timestamp'].astype(float)
        oco_data_synced = oco_data.copy()

        with open(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', f's_%03d{output_suffix}_metadata.txt' % (subject + 25)), 'w') as f:
            for key, value in oco_metadata.items():
                f.write(f'{key}: {value}\n')

        # oco_data.to_csv(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', 'data.csv'), index=False)

        if oco_data_synced.size == 0:
            log.warning('\tNo OCO data to sync')
            with open(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', 'oco_data_corrupted.txt'), 'w') as f:
                f.write('No OCO data to sync')
            continue

        console.log('\tMerging OCO cloud data')
        OCO_CSV_FILENAMES = [f for f in os.listdir(OCO_SUBJECT_FOLDER) if f.endswith('.csv')]
        for filename in OCO_CSV_FILENAMES:
            if "processed" not in filename:
                oco_data = pd.read_csv(os.path.join(OCO_SUBJECT_FOLDER, filename))
                oco_data = oco_data.rename(columns=lambda x: "ml_" + rename_column(x))
                oco_data = oco_data.rename(columns={'ml_timestamp': 'timestamp'})
                oco_data_synced = pd.merge(oco_data_synced, oco_data, on='timestamp', how='left')
                if len(oco_data_synced) == 0:
                    log.warning('\tOCO data empty after merge')
                    with open(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', 'oco_data_corrupted.txt'), 'w') as f:
                        f.write('OCO data empty after merge')
                    break

        if len(oco_data_synced) == 0:
            log.warning('\tNo OCO data after filtering')
            continue

        # Apply timestamp adjustments AFTER all merges
        oco_data_synced['timestamp'] = oco_data_synced['timestamp'] - oco_data_synced['timestamp'].iloc[0] + oco_data_synced['software_timestamp'].iloc[0]
        timestamp_diffs = oco_data_synced['timestamp'].diff()
        if any(timestamp_diffs.abs() > 0.1):
            for i, diff in enumerate(timestamp_diffs):
                if abs(diff) > 0.1:
                    oco_data_synced.loc[i:, 'timestamp'] -= diff + 0.02
        oco_data_synced.drop_duplicates('timestamp', inplace=True)
        oco_data_synced = oco_data_synced[(oco_data_synced['timestamp'] > lsl_start_timestamp) & (oco_data_synced['timestamp'] < lsl_end_timestamp)]

        if len(oco_data_synced) == 0:
            log.warning('\tNo OCO data after timestamp filtering')
            continue

        console.log('\tExtracting timesamps from LSL streams')
        questionnaire_markers = QUESTIONNAIRE_MARKERS
        relax_markers = list(RELAX_MARKERS)
        relax_markers_labels = list(RELAX_MARKERS_LABELS)
        task_markers = list(TASK_MARKERS)
        task_markers_labels = list(TASK_MARKERS_LABELS)

        current_task = None
        current_label = None
        questionnaire_count = 1
        labels = []
        to_be_amended = []

        if subject == 28:
            task_markers = task_markers[2:]
            task_markers_labels = task_markers_labels[2:]
            start_timestamp = ts.loc[2094, 'timestamps']
            ts = ts[ts['timestamps'] > start_timestamp]
            oco_data_synced = oco_data_synced[(oco_data_synced['timestamp'] > lsl_start_timestamp) & (oco_data_synced['timestamp'] < lsl_end_timestamp)]

        for index, row in ts.iterrows():
            if any(row["timeseries"].startswith(marker) for marker in questionnaire_markers) or row["timeseries"].startswith("hello") or row["timeseries"].startswith("ready") or row["timeseries"].startswith("thanks for participating"):
                if index > 0 and "ready" in ts.loc[index-1, "timeseries"] and row['timestamps'] - ts.loc[index-1, 'timestamps'] > 2 and current_label in task_markers_labels and task_markers_labels.index(current_label) > 7:
                    to_be_amended.append((index - 1, index))
                labels.append("questionnaire")
            elif any(marker in row["timeseries"] for marker in relax_markers):
                current_task = next(marker for marker in relax_markers if marker in row["timeseries"])
                current_label = relax_markers_labels[relax_markers.index(current_task)]
                labels.append(current_label)
            elif any(marker in row["timeseries"] for marker in task_markers):
                current_task = next(marker for marker in task_markers if marker in row["timeseries"])
                current_label = task_markers_labels[task_markers.index(current_task)]
                labels.append("questionnaire")
            else:
                labels.append(current_label)
                
        ts['label'] = labels

        labels = task_markers_labels[-4:]
        # console.print(labels)
        for index, (start, end) in enumerate(to_be_amended):
            ts = pd.concat(
                [
                    ts,
                    pd.DataFrame.from_dict({
                            'timeseries': ['task_marker']*2,
                            'local_timestamps': [ts.loc[start, 'local_timestamps']+0.1, ts.loc[end, 'local_timestamps']-0.1],
                            'timestamps': [ts.loc[start, 'timestamps']+0.1, ts.loc[end, 'timestamps']-0.1],
                            'label': [labels[index]]*2
                        })
                ]
            )
        ts = ts.sort_values(by='timestamps').reset_index(drop=True)

        intervals = ts.loc[ts['label'].ne(ts['label'].shift()) | ts['label'].ne(ts['label'].shift(-1))]

        streams_reference = {}
        questionnaire_count = 1
        for index, row in intervals.iterrows():
            if row['label'] == "questionnaire" and f"questionnaire_{questionnaire_count}" not in streams_reference:
                streams_reference[f"questionnaire_{questionnaire_count}"] = row['timestamps']
            elif row['label'] == "questionnaire":
                streams_reference[f"questionnaire_{questionnaire_count}"] = (streams_reference[f"questionnaire_{questionnaire_count}"], row['timestamps'])
                questionnaire_count += 1
            else:
                if row['label'] not in streams_reference:
                    streams_reference[row['label']] = row['timestamps']
                else:
                    streams_reference[row['label']] = (streams_reference[row['label']], row['timestamps'])
        if 'questionnaire_16' not in streams_reference:
            last_questionnaire = max([int(key.split('_')[1]) for key in streams_reference.keys() if 'questionnaire' in key])
            streams_reference['passive_viewing'] = (streams_reference['passive_viewing'][0], streams_reference[f'questionnaire_{last_questionnaire}'][0])
        else:
            streams_reference['passive_viewing'] = (streams_reference['passive_viewing'][0], streams_reference['questionnaire_16'][0])

        # Check that every two consecutive phases have a gap of at most 1 second
        for (phase1, times1), (phase2, times2) in zip(streams_reference.items(), list(streams_reference.items())[1:]):
            end1 = times1[1] if isinstance(times1, tuple) else times1
            start2 = times2[0] if isinstance(times2, tuple) else times2
            if start2 - end1 > 1:
                log.warning(f'\tGap of {start2 - end1} seconds between {phase1} and {phase2}')
                # If the gap is between questionnaire and following task starting with "images", subtract the integer part of the gap from the start of the task
                if 'questionnaire' in phase1 and any(task in phase2 for task in ['images']):
                    new_start2 = start2 - int(start2 - end1)
                    streams_reference[phase2] = (new_start2, times2[1] if isinstance(times2, tuple) else times2)
                    console.log(f'\tAdjusted start of {phase2} to {new_start2}')
        console.log('\tExperimental phases:')
        console.log(streams_reference)
        console.log('\tTrimming OCO data')
        end = streams_reference['passive_viewing'][1]
        oco_data_synced = oco_data_synced[oco_data_synced['software_timestamp'] <= end]

        console.log('\tApplying labels to OCO data. This may take a while...')
        for index, row in oco_data_synced.iterrows():
            for stream_name, (start, end) in streams_reference.items():
                if start <= row['software_timestamp'] <= end:
                    oco_data_synced.loc[index, 'experimental_phase'] = stream_name if 'questionnaire' not in stream_name else 'questionnaire'
                    break
        oco_data_synced = oco_data_synced[['experimental_phase', 'timestamp'] + [col for col in oco_data_synced.columns if col not in ['experimental_phase', 'timestamp']]]
        oco_data_synced['experimental_phase'] = oco_data_synced['experimental_phase'].fillna('questionnaire')
        if args.zero_timestamps:
            oco_data_synced['timestamp'] = oco_data_synced['timestamp'] - oco_data_synced['timestamp'].iloc[0]
            oco_data_synced = oco_data_synced.rename(columns={'experimental_phase': 'label'})
        oco_data_synced.to_csv(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', f's_%03d{output_suffix}.csv' % (subject + 25)), index=False)
        if not args.zero_timestamps:
            oco_data_synced.groupby('experimental_phase').sample(1).sort_values(by='timestamp').to_csv(os.path.join(SYNC_SUBJECT_FOLDER, 'oco', f's_%03d{output_suffix}_sample.csv' % (subject + 25)), index=False)