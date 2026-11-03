"""
Export LightGBM predictions for OCOSense smartglasses data.
Generates predictions for LOSO binary, LOSO three-class, and LOGO three-class experiments.
Output goes to data/results/ocosense/{experiment}/lgbm_predictions.csv.
"""

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from rich.console import Console
from lib.config import (
    OCO_DIR, RESULTS_DIR, EXCLUDED_SUBJECTS, NAN_DROP_THRESHOLD,
    NO_COG_LOAD, LOW_COG_LOAD, HIGH_COG_LOAD, LOGO_GROUPS,
    SENSOR_AXES, SENSOR_AXES_ALL, ML_COLS, PROBA_COLS,
    EXPR_COL, EXPR_VALUES, FULL_FEATURE_CONFIG,
)

console = Console()

# ---- PATHS ----
oco_dir = str(OCO_DIR)
output_base = RESULTS_DIR / "ocosense"

# ---- DATA LOADING WITH TIMESTAMP PRESERVATION ----
console.log("Loading segmented OCO data...")
segmented_files = [
    os.path.join(oco_dir, folder, f"{folder}_oco_segmented.csv")
    for folder in os.listdir(oco_dir)
    if os.path.exists(os.path.join(oco_dir, folder, f"{folder}_oco_segmented.csv"))
]

frames = []
window_timestamps_list = []

for file_path in tqdm(segmented_files, desc="Processing subjects"):
    df_segmented = pd.read_csv(file_path)
    if df_segmented.empty:
        console.log(f"Skipping {os.path.basename(os.path.dirname(file_path))} because the segmented file is empty")
        continue

    folder = os.path.basename(os.path.dirname(file_path))
    df_segmented["subject"] = folder

    # Extract window timestamps BEFORE feature extraction
    window_ts = df_segmented.groupby(['subject', 'window_index']).agg({
        'timestamp': ['min', 'max'],
        'experimental_phase': 'first'
    }).reset_index()
    window_ts.columns = ['subject', 'window_index', 'start_timestamp', 'end_timestamp', 'experimental_phase']
    window_timestamps_list.append(window_ts)

    ml_columns = [col for col in df_segmented.columns if col.startswith("ml")]
    sensor_columns = [col for col in df_segmented.columns if col.startswith("acc") or col.startswith("gyro") or col.startswith("mag")]
    df_segmented = df_segmented[["subject", "window_index", "experimental_phase"] + sensor_columns + ml_columns]
    frames.append(df_segmented)

df = pd.concat(frames, ignore_index=True)
window_timestamps = pd.concat(window_timestamps_list, ignore_index=True)

console.log(f"Total rows after concatenation: {len(df)}")
console.log(f"Total subjects: {df['subject'].nunique()}")
console.log(f"Total columns: {len(df.columns)}")

# Drop excluded subjects (matches modeling.py)
console.log(f"Rows before dropping excluded subjects: {len(df)}")
for subj in EXCLUDED_SUBJECTS:
    df = df[df['subject'] != subj]
console.log(f"Rows after dropping excluded subjects: {len(df)}")
console.log(f"Subjects remaining: {df['subject'].nunique()}")

# Drop rows with >50% NaN values (keep rows with at least 50% non-NaN) - matches modeling.py
min_non_nan = int(len(df.columns) * (1 - NAN_DROP_THRESHOLD))
console.log(f"Dropping rows with >{100-50}% NaN (keeping rows with at least {min_non_nan}/{len(df.columns)} non-NaN columns)")
rows_before = len(df)
df = df.dropna(axis=0, thresh=min_non_nan)
console.log(f"Rows after NaN filtering: {len(df)} (dropped {rows_before - len(df)} rows)")

# Filter windows with single experimental_phase and exclude questionnaire
phase_counts = df.groupby(['subject', 'window_index'])['experimental_phase'].nunique().reset_index()
valid_windows = phase_counts[phase_counts['experimental_phase'] == 1][['subject', 'window_index']]
df = df.merge(valid_windows, on=['subject', 'window_index'], how='inner')
df = df[df['experimental_phase'] != 'questionnaire']

# Filter timestamps to exclude excluded subjects and match valid windows
for subj in EXCLUDED_SUBJECTS:
    window_timestamps = window_timestamps[window_timestamps['subject'] != subj]
window_timestamps = window_timestamps.merge(valid_windows, on=['subject', 'window_index'], how='inner')
window_timestamps = window_timestamps[window_timestamps['experimental_phase'] != 'questionnaire']

console.log(f"Loaded {len(df['subject'].unique())} subjects, {len(df)} samples")

# ---- FEATURE EXTRACTION FUNCTIONS ----
from clues.features import (
    compute_basic_stats, compute_quartiles, compute_expression_counts,
    compute_signal_magnitude_area, compute_energy,
    compute_skew_kurtosis, compute_axis_correlations,
    compute_ml_proba_stats, compute_trend,
    extract_features,
)

# ---- FEATURE CONFIG (same as modeling.py) ----
FEATURE_CONFIG = FULL_FEATURE_CONFIG

sensor_axes = dict(SENSOR_AXES)
sensor_axes["all"] = SENSOR_AXES_ALL
ml_cols = [col for col in ML_COLS if col in df.columns]
proba_cols = [col for col in PROBA_COLS if col in df.columns]
expr_col = EXPR_COL
expr_values = list(EXPR_VALUES)
cont_ml_cols = ml_cols

# Check if expression column exists
if expr_col not in df.columns:
    console.log(f"[yellow]Warning: {expr_col} column not found, disabling expression counts[/yellow]")
    FEATURE_CONFIG["expression_counts"] = False

console.log(f"Available ML columns: {ml_cols}")
console.log(f"Available probability columns: {len(proba_cols)}")

# ---- EXTRACT FEATURES ----
console.log("Extracting features...")
features_df = extract_features(df, sensor_axes, ml_cols, proba_cols, expr_col, expr_values, cont_ml_cols, FEATURE_CONFIG)
console.log(f"Extracted features shape: {features_df.shape}")

# ---- LABEL DEFINITIONS ----
features_df['target_binary'] = features_df['experimental_phase'].apply(
    lambda x: 0 if x in NO_COG_LOAD else 1
)

def map_threeclass(phase):
    if phase in NO_COG_LOAD:
        return 0
    elif phase in LOW_COG_LOAD:
        return 1
    elif phase in HIGH_COG_LOAD:
        return 2
    else:
        return np.nan

features_df['target_threeclass'] = features_df['experimental_phase'].apply(map_threeclass)

# Get feature column names
exclude_cols = {"subject", "window_index", "experimental_phase", "target_binary", "target_threeclass"}
feature_cols = [col for col in features_df.columns if col not in exclude_cols]

console.log(f"Number of features: {len(feature_cols)}")

# ---- EXPERIMENT 1: LOSO BINARY CLASSIFICATION ----
console.log("\n=== Running LOSO Binary Classification ===")
predictions_binary = []
subjects = features_df['subject'].unique()

for test_subject in tqdm(subjects, desc="LOSO Binary"):
    train_mask = features_df['subject'] != test_subject
    test_mask = features_df['subject'] == test_subject

    X_train = features_df.loc[train_mask, feature_cols]  # Keep as DataFrame to preserve feature names
    y_train = features_df.loc[train_mask, 'target_binary'].values
    X_test = features_df.loc[test_mask, feature_cols]  # Keep as DataFrame to preserve feature names
    y_test = features_df.loc[test_mask, 'target_binary'].values

    model = LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    test_indices = features_df.loc[test_mask].index
    for i, idx in enumerate(test_indices):
        row = features_df.loc[idx]
        ts_row = window_timestamps[
            (window_timestamps['subject'] == row['subject']) &
            (window_timestamps['window_index'] == row['window_index'])
        ].iloc[0]

        predictions_binary.append({
            'subject_id': row['subject'],
            'y_true': int(y_test[i]),
            'y_pred': int(y_pred[i]),
            'prob_class_0': y_proba[i, 0],
            'prob_class_1': y_proba[i, 1],
            'start_timestamp': ts_row['start_timestamp'],
            'end_timestamp': ts_row['end_timestamp']
        })

loso_binary_df = pd.DataFrame(predictions_binary)
console.log(f"LOSO Binary predictions: {len(loso_binary_df)} samples")

# ---- EXPERIMENT 2: LOSO THREE-CLASS CLASSIFICATION ----
console.log("\n=== Running LOSO Three-Class Classification ===")
predictions_threeclass = []

for test_subject in tqdm(subjects, desc="LOSO Three-Class"):
    train_mask = features_df['subject'] != test_subject
    test_mask = features_df['subject'] == test_subject

    X_train = features_df.loc[train_mask, feature_cols]  # Keep as DataFrame to preserve feature names
    y_train = features_df.loc[train_mask, 'target_threeclass'].values
    X_test = features_df.loc[test_mask, feature_cols]  # Keep as DataFrame to preserve feature names
    y_test = features_df.loc[test_mask, 'target_threeclass'].values

    model = LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    test_indices = features_df.loc[test_mask].index
    for i, idx in enumerate(test_indices):
        row = features_df.loc[idx]
        ts_row = window_timestamps[
            (window_timestamps['subject'] == row['subject']) &
            (window_timestamps['window_index'] == row['window_index'])
        ].iloc[0]

        predictions_threeclass.append({
            'subject_id': row['subject'],
            'y_true': int(y_test[i]),
            'y_pred': int(y_pred[i]),
            'prob_class_0': y_proba[i, 0],
            'prob_class_1': y_proba[i, 1],
            'prob_class_2': y_proba[i, 2],
            'start_timestamp': ts_row['start_timestamp'],
            'end_timestamp': ts_row['end_timestamp']
        })

loso_threeclass_df = pd.DataFrame(predictions_threeclass)
console.log(f"LOSO Three-Class predictions: {len(loso_threeclass_df)} samples")

# ---- EXPERIMENT 3: LOGO THREE-CLASS CLASSIFICATION ----
console.log("\n=== Running LOGO Three-Class Classification ===")
predictions_logo = []

for fold_id, test_tasks in enumerate(LOGO_GROUPS, start=1):
    console.log(f"Fold {fold_id}: Testing on {test_tasks}")
    test_mask = features_df['experimental_phase'].isin(test_tasks)
    train_mask = ~test_mask

    X_train = features_df.loc[train_mask, feature_cols]  # Keep as DataFrame to preserve feature names
    y_train = features_df.loc[train_mask, 'target_threeclass'].values
    X_test = features_df.loc[test_mask, feature_cols]  # Keep as DataFrame to preserve feature names
    y_test = features_df.loc[test_mask, 'target_threeclass'].values

    model = LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    test_indices = features_df.loc[test_mask].index
    for i, idx in enumerate(test_indices):
        row = features_df.loc[idx]
        ts_row = window_timestamps[
            (window_timestamps['subject'] == row['subject']) &
            (window_timestamps['window_index'] == row['window_index'])
        ].iloc[0]

        predictions_logo.append({
            'fold_id': fold_id,
            'task': row['experimental_phase'],
            'y_true': int(y_test[i]),
            'y_pred': int(y_pred[i]),
            'prob_class_0': y_proba[i, 0],
            'prob_class_1': y_proba[i, 1],
            'prob_class_2': y_proba[i, 2],
            'subject_id': row['subject'],
            'start_timestamp': ts_row['start_timestamp'],
            'end_timestamp': ts_row['end_timestamp']
        })

logo_threeclass_df = pd.DataFrame(predictions_logo)
console.log(f"LOGO Three-Class predictions: {len(logo_threeclass_df)} samples")

# ---- SAVE RESULTS ----
console.log("\n=== Saving results ===")
for d in ["loso_2class", "loso_3class", "logo_3class"]:
    (output_base / d).mkdir(parents=True, exist_ok=True)

loso_binary_path = output_base / "loso_2class" / "lgbm_predictions.csv"
loso_threeclass_path = output_base / "loso_3class" / "lgbm_predictions.csv"
logo_threeclass_path = output_base / "logo_3class" / "lgbm_predictions.csv"

loso_binary_df.to_csv(loso_binary_path, index=False)
loso_threeclass_df.to_csv(loso_threeclass_path, index=False)
logo_threeclass_df.to_csv(logo_threeclass_path, index=False)

console.log(f"✓ Saved {loso_binary_path}")
console.log(f"✓ Saved {loso_threeclass_path}")
console.log(f"✓ Saved {logo_threeclass_path}")

console.log("\n=== Done! ===")
