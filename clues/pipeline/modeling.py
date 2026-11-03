# %%
# ---- IMPORTS ----
import pandas as pd
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb
from lightgbm import LGBMClassifier, plot_importance
from sklearn.dummy import DummyClassifier
from lib.config import (
    OCO_DIR, EXCLUDED_SUBJECTS, NAN_DROP_THRESHOLD,
    NO_COG_LOAD, LOW_COG_LOAD, HIGH_COG_LOAD, LOGO_GROUPS,
    SENSOR_AXES, SENSOR_AXES_ALL, ML_COLS, PROBA_COLS,
    EXPR_COL, EXPR_VALUES, FULL_FEATURE_CONFIG, RESULTS_DIR,
)

# ---- ARGUMENT PARSING ----
parser = argparse.ArgumentParser(description='Run cognitive load classification with LightGBM, XGBoost, and RandomForest')
parser.add_argument('--xai', action='store_true', help='Enable XAI analysis (feature importance and SHAP plots)')
parser.add_argument('--oco-dir', default=None, help='override OCO_DIR input root')
parser.add_argument('--out-dir', default=None, help='override results output dir')
args = parser.parse_args()

console = Console()

if args.xai:
    console.log("[bold green]XAI analysis enabled - will generate feature importance and SHAP plots[/bold green]")
else:
    console.log("[bold yellow]XAI analysis disabled - skipping feature importance and SHAP plots[/bold yellow]")

script_dir = os.path.dirname(os.path.abspath(__file__))
oco_dir = args.oco_dir or str(OCO_DIR)

# ---- OUTPUT DIRECTORIES ----
results_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "ocosense"
artifacts_dir = results_dir / "artifacts"
for exp_dir in ["loso_2class", "loso_3class", "logo_3class"]:
    (results_dir / exp_dir).mkdir(parents=True, exist_ok=True)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Target → experiment directory mapping
TARGET_EXP_MAP = {
    "target_binary": "loso_2class",
    "target_threeclass": "loso_3class",
    "target_logo_threeclass": "logo_3class",
}

# Model name → filename mapping
MODEL_FILE_MAP = {
    "Baseline": "baseline_predictions.csv",
    "RandomForest": "rf_predictions.csv",
    "XGBoost": "xgb_predictions.csv",
    "LightGBM": "lgbm_predictions.csv",
}


def save_predictions(predictions_list, target, model_name):
    """Save collected predictions to the appropriate CSV file."""
    df_pred = pd.DataFrame(predictions_list)
    exp_dir = TARGET_EXP_MAP[target]
    filename = MODEL_FILE_MAP[model_name]
    path = results_dir / exp_dir / filename
    df_pred.to_csv(path, index=False)
    console.log(f"  Saved {len(df_pred)} predictions to {path}")


# %%
# ---- DATA LOADING ----
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

    # Extract window timestamps before column subsetting
    window_ts = df_segmented.groupby(['subject', 'window_index']).agg({
        'timestamp': ['min', 'max'],
        'experimental_phase': 'first'
    }).reset_index()
    window_ts.columns = ['subject', 'window_index', 'start_timestamp', 'end_timestamp', 'experimental_phase']
    window_timestamps_list.append(window_ts)

    ml_columns = [col for col in df_segmented.columns if col.startswith("ml")]
    sensor_columns = [col for col in df_segmented.columns if col.startswith("acc") or col.startswith("gyro") or col.startswith("mag")]
    df_segmented = df_segmented[["subject", "window_index", "experimental_phase", "timestamp", "software_timestamp"] + sensor_columns + ml_columns]
    frames.append(df_segmented)

df = pd.concat(frames, ignore_index=True)
window_timestamps = pd.concat(window_timestamps_list, ignore_index=True)
console.log(f"Total rows after concatenation: {len(df)}")
console.log(f"Total subjects: {df['subject'].nunique()}")
console.log(f"Total columns: {len(df.columns)}")

# Drop excluded subjects
console.log(f"Rows before dropping excluded subjects: {len(df)}")
for subj in EXCLUDED_SUBJECTS:
    df = df[df['subject'] != subj]
    window_timestamps = window_timestamps[window_timestamps['subject'] != subj]
console.log(f"Rows after dropping excluded subjects: {len(df)}")
console.log(f"Subjects remaining: {df['subject'].nunique()}")

# Drop rows with >50% NaN values (keep rows with at least 50% non-NaN)
min_non_nan = int(len(df.columns) * (1 - NAN_DROP_THRESHOLD))
console.log(f"Dropping rows with >{100-50}% NaN (keeping rows with at least {min_non_nan}/{len(df.columns)} non-NaN columns)")
rows_before = len(df)
df = df.dropna(axis=0, thresh=min_non_nan)
console.log(f"Rows after NaN filtering: {len(df)} (dropped {rows_before - len(df)} rows)")

phase_counts = df.groupby(['subject', 'window_index'])['experimental_phase'].nunique().reset_index()
valid_windows = phase_counts[phase_counts['experimental_phase'] == 1][['subject', 'window_index']]
df = df.merge(valid_windows, on=['subject', 'window_index'], how='inner')
console.log(f"Windows after removing multi-phase windows: {df.groupby(['subject', 'window_index']).ngroups}")
df = df[df['experimental_phase'] != 'questionnaire']
console.log(f"Windows after removing questionnaire phase: {df.groupby(['subject', 'window_index']).ngroups}")

# Filter window timestamps to match
window_timestamps = window_timestamps.merge(valid_windows, on=['subject', 'window_index'], how='inner')
window_timestamps = window_timestamps[window_timestamps['experimental_phase'] != 'questionnaire']

# %%
# ---- FEATURE EXTRACTION PIPELINE ----
from clues.features import (
    compute_basic_stats, compute_quartiles, compute_expression_counts,
    compute_signal_magnitude_area, compute_energy, compute_range,
    compute_skew_kurtosis, compute_zero_crossing, compute_axis_correlations,
    compute_ml_proba_stats, compute_trend, compute_first_last_diff,
    extract_features,
)

# ---- FEATURE EXTRACTION CONFIGURATIONS ----
FEATURE_CONFIGS = [FULL_FEATURE_CONFIG]

sensor_axes = dict(SENSOR_AXES)
sensor_axes["all"] = SENSOR_AXES_ALL
ml_cols = list(ML_COLS)
proba_cols = list(PROBA_COLS)
expr_col = EXPR_COL
expr_values = list(EXPR_VALUES)
cont_ml_cols = ml_cols

# %% ---- MAIN FEATURE EXTRACTION + ML LOOP ----
for config_idx, FEATURE_CONFIG in enumerate(FEATURE_CONFIGS):
    console.print(f"\n\n===== FEATURE CONFIG {config_idx+1}/{len(FEATURE_CONFIGS)} =====")
    console.print(FEATURE_CONFIG)
    features_df = extract_features(df, sensor_axes, ml_cols, proba_cols, expr_col, expr_values, cont_ml_cols, FEATURE_CONFIG)
    console.log(f"Extracted features shape: {features_df.shape}")

    # ---- DISTRIBUTION OF experimental_phase ----
    phase_dist = features_df['experimental_phase'].value_counts().reset_index()
    phase_dist.columns = ['experimental_phase', 'count']
    phase_dist.to_csv(str(artifacts_dir / "experimental_phase_distribution.csv"), index=False)
    console.log(f"Saved experimental_phase distribution to {artifacts_dir / 'experimental_phase_distribution.csv'}")

    # ---- LABELS CREATION ----
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
    features_df["target_logo_threeclass"] = features_df["target_threeclass"]

    features_df = features_df[["subject", "window_index", "experimental_phase", "target_binary", "target_threeclass", "target_logo_threeclass"] + list(features_df.columns[3:-3])]

    # Merge window timestamps into features_df
    features_df = features_df.merge(
        window_timestamps[['subject', 'window_index', 'start_timestamp', 'end_timestamp']],
        on=['subject', 'window_index'],
        how='left'
    )

    console.log(f"Features DataFrame shape after adding targets: {features_df.shape}")

    # Save features to artifacts (metadata + features, no target columns)
    features_to_save = features_df.drop(
        columns=["target_binary", "target_threeclass", "target_logo_threeclass"]
    )
    features_to_save.to_csv(str(artifacts_dir / "features.csv"), index=False)
    console.log(f"Saved features ({features_to_save.shape}) to {artifacts_dir / 'features.csv'}")

    # Log class distributions
    console.log(f"\n[bold]Class Distribution Summary:[/bold]")
    console.log(f"  Binary (0=No Load, 1=Cog Load): {dict(features_df['target_binary'].value_counts().sort_index())}")
    console.log(f"  Three-class (0=No, 1=Low, 2=High): {dict(features_df['target_threeclass'].value_counts().sort_index())}")

    # Log subject-level class distribution
    console.log(f"\n[bold]Subject-level class distribution:[/bold]")
    for subj in sorted(features_df['subject'].unique()):
        subj_data = features_df[features_df['subject'] == subj]
        binary_dist = dict(subj_data['target_binary'].value_counts().sort_index())
        three_dist = dict(subj_data['target_threeclass'].value_counts().sort_index())
        console.log(f"  {subj}: n={len(subj_data)}, binary={binary_dist}, three={three_dist}")

    # Results dict for this config
    results = {
        "target_binary": [],
        "target_threeclass": [],
        "target_logo_threeclass": []
    }

    # ---- Baseline (majority predictor) with LOSO and LOGO ----
    from collections import Counter
    import warnings

    def mean_std(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arr = np.array(arr)
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            return mean, std

    def compute_baseline_metrics(y_true, y_pred, y_proba, is_multiclass):
        acc = accuracy_score(y_true, y_pred)
        if is_multiclass:
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro")
            try:
                auc = roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
            except Exception:
                auc = 0.50
        else:
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                auc = 0.50
        return acc, precision, recall, f1, auc

    def baseline_loso(features_df, target):
        subjects = features_df['subject'].unique()
        console.log(f"  Running Baseline LOSO for {target} with {len(subjects)} subjects")
        accs, precisions, recalls, f1s, aucs = [], [], [], [], []
        all_predictions = []
        is_multiclass = features_df[target].nunique() > 2

        for idx, subj in enumerate(subjects, 1):
            test_mask = features_df['subject'] == subj
            train_mask = ~test_mask

            X_train = features_df.loc[train_mask, target]
            X_test = features_df.loc[test_mask, target]
            y_train = features_df.loc[train_mask, target]
            y_test = features_df.loc[test_mask, target]

            if len(y_test) == 0:
                console.log(f"  [{idx}/{len(subjects)}] {subj}: Skipped (no test samples)")
                continue

            # Diagnostic: Check if only one class present in test set
            unique_classes = y_test.unique()
            if len(unique_classes) == 1:
                console.log(f"  [{idx}/{len(subjects)}] {subj}: [yellow]Only class {unique_classes[0]} in test set (n={len(y_test)})[/yellow]")
            else:
                console.log(f"  [{idx}/{len(subjects)}] {subj}: {len(y_test)} test samples, classes: {sorted(unique_classes.tolist())}")

            dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_clf.fit(X_train, y_train)

            y_pred = dummy_clf.predict(X_test)
            y_proba = dummy_clf.predict_proba(X_test)

            acc, precision, recall, f1, auc = compute_baseline_metrics(y_test, y_pred, y_proba, is_multiclass)
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)

            # Collect predictions
            test_rows = features_df.loc[test_mask]
            for i, (_, row) in enumerate(test_rows.iterrows()):
                pred = {
                    'subject_id': row['subject'],
                    'y_true': int(y_test.iloc[i]),
                    'y_pred': int(y_pred[i]),
                    'start_timestamp': row['start_timestamp'],
                    'end_timestamp': row['end_timestamp'],
                }
                for c in range(y_proba.shape[1]):
                    pred[f'prob_class_{c}'] = y_proba[i, c]
                all_predictions.append(pred)

        metrics = mean_std(accs) + mean_std(precisions) + mean_std(recalls) + mean_std(f1s) + mean_std(aucs)
        return metrics, all_predictions


    def baseline_logo(features_df, target, logo_groups):
        encoder = LabelEncoder().fit(features_df['experimental_phase'])
        exp_phases = features_df['experimental_phase']
        y = features_df[target]
        X = features_df[target]
        accs, precisions, recalls, f1s, aucs = [], [], [], [], []
        all_predictions = []
        is_multiclass = features_df[target].nunique() > 2

        for fold_id, group in enumerate(logo_groups, start=1):
            group_enc = encoder.transform(group)
            test_mask = np.isin(encoder.transform(exp_phases), group_enc)
            train_mask = ~test_mask

            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            if len(y_test) == 0:
                continue

            dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_clf.fit(X_train, y_train)

            y_pred = dummy_clf.predict(X_test)
            y_proba = dummy_clf.predict_proba(X_test)

            acc, precision, recall, f1, auc = compute_baseline_metrics(y_test, y_pred, y_proba, is_multiclass)
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)

            # Collect predictions
            test_rows = features_df.iloc[np.where(test_mask)[0]]
            for i, (_, row) in enumerate(test_rows.iterrows()):
                pred = {
                    'fold_id': fold_id,
                    'task': row['experimental_phase'],
                    'y_true': int(y_test.iloc[i]),
                    'y_pred': int(y_pred[i]),
                    'subject_id': row['subject'],
                    'start_timestamp': row['start_timestamp'],
                    'end_timestamp': row['end_timestamp'],
                }
                for c in range(y_proba.shape[1]):
                    pred[f'prob_class_{c}'] = y_proba[i, c]
                all_predictions.append(pred)

        metrics = mean_std(accs) + mean_std(precisions) + mean_std(recalls) + mean_std(f1s) + mean_std(aucs)
        return metrics, all_predictions


    for target in tqdm(["target_binary", "target_threeclass", "target_logo_threeclass"], desc="Running Baseline"):
        console.log(f"\n[bold cyan]Running Baseline for {target}[/bold cyan]")
        y_true = features_df[target]
        is_multiclass = y_true.nunique() > 2
        # LOGO (only for target_logo_threeclass)
        if target == "target_logo_threeclass":
            (acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std), predictions = baseline_logo(features_df, target, LOGO_GROUPS)
            results[target].append(
                f"BL  + LOGO + high c.l. VS. low c.l. VS. no c.l. | "
                f"Macro F1: {f1_mean:.2f} ± {f1_std:.2f} | "
                f"AUC: {auc_mean:.2f} ± {auc_std:.2f} | "
                f"Acc: {acc_mean:.2f} ± {acc_std:.2f} | "
                f"Prec: {prec_mean:.2f} ± {prec_std:.2f} | "
                f"Rec: {rec_mean:.2f} ± {rec_std:.2f}"
            )
        # LOSO
        else:
            # LOSO
            (acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std), predictions = baseline_loso(features_df, target)
            results[target].append(
                f"BL  + LOSO + {'c.l. VS. no c.l.' if target == 'target_binary' else 'high c.l. VS. low c.l. VS. no c.l.'} | "
                f"Macro F1: {f1_mean:.2f} ± {f1_std:.2f} | "
                f"AUC: {auc_mean:.2f} ± {auc_std:.2f} | "
                f"Acc: {acc_mean:.2f} ± {acc_std:.2f} | "
                f"Prec: {prec_mean:.2f} ± {prec_std:.2f} | "
                f"Rec: {rec_mean:.2f} ± {rec_std:.2f}"
            )
        save_predictions(predictions, target, "Baseline")

    # ---- ML PIPELINE ----
    class CustomLOGO:
        def __init__(self):
            self.encoder = LabelEncoder().fit(features_df['experimental_phase'])
            self.groups = [list(g) for g in LOGO_GROUPS]
            self.experimental_phase = self.encoder.transform(features_df['experimental_phase'])
            for i, group in enumerate(self.groups):
                self.groups[i] = self.encoder.transform(group)

        def split(self, X, y, groups=None):
            for group in self.groups:
                test_mask = np.isin(self.experimental_phase, group)
                train_mask = ~test_mask
                yield np.where(train_mask)[0], np.where(test_mask)[0]

    def evaluate_model_cv(X, y, groups, model, is_multiclass, model_name="Model"):
        if groups is not None:
            logo = LeaveOneGroupOut()
            cv_type = "LOSO"
            # Count number of folds
            n_folds = len(np.unique(groups))
        else:
            logo = CustomLOGO()
            cv_type = "LOGO"
            n_folds = len(logo.groups)

        console.log(f"  Running {model_name} with {cv_type} ({n_folds} folds)")
        accs, precisions, recalls, f1s, aucs = [], [], [], [], []
        all_predictions = []

        fold_idx = 0
        for train_idx, test_idx in logo.split(X, y, groups):
            fold_idx += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Identify which subject/group is being tested
            if groups is not None:
                test_subject = groups[test_idx[0]] if len(test_idx) > 0 else "unknown"
                console.log(f"    Fold {fold_idx}/{n_folds}: Testing on {test_subject} ({len(test_idx)} samples)")
            else:
                console.log(f"    Fold {fold_idx}/{n_folds}: Testing on group ({len(test_idx)} samples)")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)

            # Use macro averaging for both binary and multiclass
            precisions.append(precision_score(y_test, y_pred, average="macro", zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

            if is_multiclass:
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test), average="macro", multi_class="ovr")
                except Exception:
                    auc = float('nan')
            else:
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                except Exception:
                    auc = float('nan')
            aucs.append(auc)

            auc_str = f"{aucs[-1]:.3f}" if not np.isnan(aucs[-1]) else "N/A"
            f1_str = f"{f1s[-1]:.3f}"
            console.log(f"      Macro F1: {f1_str}, AUC: {auc_str}")

            # Collect per-window predictions
            for i, idx in enumerate(test_idx):
                row = features_df.iloc[idx]
                if groups is None:  # LOGO
                    pred = {
                        'fold_id': fold_idx,
                        'task': row['experimental_phase'],
                        'y_true': int(y_test[i]),
                        'y_pred': int(y_pred[i]),
                        'subject_id': row['subject'],
                        'start_timestamp': row['start_timestamp'],
                        'end_timestamp': row['end_timestamp'],
                    }
                else:  # LOSO
                    pred = {
                        'subject_id': row['subject'],
                        'y_true': int(y_test[i]),
                        'y_pred': int(y_pred[i]),
                        'start_timestamp': row['start_timestamp'],
                        'end_timestamp': row['end_timestamp'],
                    }
                for c in range(y_proba.shape[1]):
                    pred[f'prob_class_{c}'] = y_proba[i, c]
                all_predictions.append(pred)

        def mean_std(arr):
            arr = np.array(arr)
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            return mean, std
        acc_mean, acc_std = mean_std(accs)
        prec_mean, prec_std = mean_std(precisions)
        rec_mean, rec_std = mean_std(recalls)
        f1_mean, f1_std = mean_std(f1s)
        auc_mean, auc_std = mean_std(aucs)
        return acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std, all_predictions

    exclude_cols = {"subject", "window_index", "experimental_phase", "target_binary", "target_threeclass", "target_logo_threeclass", "start_timestamp", "end_timestamp"}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    def run_and_log_cv(X, y, groups, model, is_multiclass, label, model_name):
        acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std, predictions = evaluate_model_cv(X, y, groups, model, is_multiclass, model_name)
        results[target].append(
            f"{label} | "
            f"Macro F1: {f1_mean:.2f} ± {f1_std:.2f} | "
            f"AUC: {auc_mean:.2f} ± {auc_std:.2f} | "
            f"Acc: {acc_mean:.2f} ± {acc_std:.2f} | "
            f"Prec: {prec_mean:.2f} ± {prec_std:.2f} | "
            f"Rec: {rec_mean:.2f} ± {rec_std:.2f}"
        )
        save_predictions(predictions, target, model_name)

    X = features_df[feature_cols]  # Keep as DataFrame to preserve feature names

    # ---- RANDOM FOREST CLASSIFIER ----
    for target in tqdm(["target_binary", "target_threeclass", "target_logo_threeclass"], desc="Running RF"):
        console.log(f"\n[bold cyan]Running RandomForest for {target}[/bold cyan]")
        y = features_df[target].values
        if target == "target_logo_threeclass":
            groups = None
            label = "RF  + LOGO + high c.l. VS. low c.l. VS. no c.l."
        else:
            groups = features_df["subject"].values
            label = f"RF  + LOSO + {'c.l. VS. no c.l.' if target == 'target_binary' else 'high c.l. VS. low c.l. VS. no c.l.'}"
        is_multiclass = len(np.unique(y)) > 2
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        run_and_log_cv(X, y, groups, rf, is_multiclass, label, "RandomForest")

    # ---- XGBOOST CLASSIFIER ----
    for target in tqdm(["target_binary", "target_threeclass", "target_logo_threeclass"], desc="Running XGB"):
        console.log(f"\n[bold cyan]Running XGBoost for {target}[/bold cyan]")
        y = features_df[target].values
        if target == "target_logo_threeclass":
            groups = None
            label = "XGB + LOGO + high c.l. VS. low c.l. VS. no c.l."
        else:
            groups = features_df["subject"].values
            label = f"XGB + LOSO + {'c.l. VS. no c.l.' if target == 'target_binary' else 'high c.l. VS. low c.l. VS. no c.l.'}"
        is_multiclass = len(np.unique(y)) > 2
        xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
        run_and_log_cv(X, y, groups, xgb_model, is_multiclass, label, "XGBoost")

    # ---- LIGHTGBM CLASSIFIER ----
    for target in tqdm(["target_binary", "target_threeclass", "target_logo_threeclass"], desc="Running LGB"):
        console.log(f"\n[bold cyan]Running LightGBM for {target}[/bold cyan]")
        y = features_df[target].values
        if target == "target_logo_threeclass":
            groups = None
            label = "LGB + LOGO + high c.l. VS. low c.l. VS. no c.l."
        else:
            groups = features_df["subject"].values
            label = f"LGB + LOSO + {'c.l. VS. no c.l.' if target == 'target_binary' else 'high c.l. VS. low c.l. VS. no c.l.'}"
        is_multiclass = len(np.unique(y)) > 2
        lgb_model = LGBMClassifier(random_state=42, verbosity=-1)
        run_and_log_cv(X, y, groups, lgb_model, is_multiclass, label, "LightGBM")

    # ---- FINAL LOGGING FOR THIS CONFIG ----
    console.print(f"\n---- RESULTS FOR FEATURE CONFIG {config_idx+1} ----")
    for target, logs in results.items():
        for log in logs:
            console.print(log)
        console.print("\n")

# %% --- FEATURE IMPORTANCE PLOTTING ---
FEATURE_CONFIG = FULL_FEATURE_CONFIG
console.print(f"\n\n===== FEATURE CONFIG =====")
console.print(FEATURE_CONFIG)
features_df = extract_features(df, sensor_axes, ml_cols, proba_cols, expr_col, expr_values, cont_ml_cols, FEATURE_CONFIG)
console.log(f"Extracted features shape: {features_df.shape}")

# ---- DISTRIBUTION OF experimental_phase ----
phase_dist = features_df['experimental_phase'].value_counts().reset_index()
phase_dist.columns = ['experimental_phase', 'count']
phase_dist.to_csv(str(artifacts_dir / "experimental_phase_distribution.csv"), index=False)
console.log(f"Saved experimental_phase distribution to {artifacts_dir / 'experimental_phase_distribution.csv'}")

# ---- LABELS CREATION ----
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

features_df = features_df[["subject", "window_index", "experimental_phase", "target_binary", "target_threeclass"] + list(features_df.columns[3:-3])]

results = {
    "target_binary": [],
    "target_threeclass": [],
    "target_logo_threeclass": []
}

# %%
# Remove higly correlated features > 0.9
corr_matrix = features_df.drop(columns=["subject", "window_index", "experimental_phase", "target_binary", "target_threeclass"]).corr().abs()
# Find pairs of features with correlation > 0.9
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find features to drop (keep first occurrence, drop second)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

console.log(f"Features to drop due to high correlation (>0.9): {len(to_drop)}")
console.log(f"Features: {to_drop}")

# Drop highly correlated features
features_df = features_df.drop(columns=to_drop)
console.log(f"Features DataFrame shape after removing highly correlated features: {features_df.shape}")

# %%
# ---- XAI: FEATURE IMPORTANCE AND SHAP ANALYSIS ----
if args.xai:
    console.log("\n[bold green]Starting XAI analysis (feature importance and SHAP)[/bold green]")
    from lightgbm import LGBMClassifier, plot_importance
    import shap
    import matplotlib.pyplot as plt

    def plot_feature_importance(model: LGBMClassifier, target: str):

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(model, max_num_features=20, importance_type='gain', ax=ax)
        plt.title(f'Feature Importance (Gain) for {"Binary" if target == "target_binary" else "Three-class"} Target')
        plt.tight_layout()
        plt.savefig(str(artifacts_dir / f'feature_importance_gain_{target}.pdf'), bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(model, max_num_features=20, importance_type='split', ax=ax)
        plt.title(f'Feature Importance (Split) for {"Binary" if target == "target_binary" else "Three-class"} Target')
        plt.tight_layout()
        plt.savefig(str(artifacts_dir / f'feature_importance_split_{target}.pdf'), bbox_inches='tight')
        plt.show()

    for target in ["target_binary", "target_threeclass"]:
        X = features_df.drop(columns=["subject", "window_index", "experimental_phase", "target_binary", "target_threeclass"])
        y = features_df[target]
        lgb_model = LGBMClassifier(random_state=42, verbosity=-1)
        lgb_model.fit(X, y, eval_set=[(X, y)])
        plot_feature_importance(lgb_model, target)

    # %%
    import matplotlib.pyplot as plt
    X = features_df.drop(columns=["subject", "window_index", "experimental_phase", "target_binary", "target_threeclass"])
    y = features_df["target_binary"]
    lgb_model = LGBMClassifier(random_state=42, verbosity=-1)
    lgb_model.fit(X, y, eval_set=[(X, y)])
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=15, show=False)
    plt.savefig(str(artifacts_dir / 'shap_summary_plot_binary.pdf'), bbox_inches='tight')
    plt.show()

    # %%
    def map_labels_to_class(label):
        if label == 0:
            return "no_cogload"
        elif label == 1:
            return "low_cogload"
        elif label == 2:
            return "high_cogload"
    for label in features_df["target_threeclass"].unique():
        print(f"Processing label: {map_labels_to_class(label)}")
        X = features_df.drop(columns=["subject", "window_index", "experimental_phase", "target_binary", "target_threeclass"])
        y = features_df["target_threeclass"].apply(lambda x: 1 if x == label else 0)
        lgb_model = LGBMClassifier(random_state=42, verbosity=-1)
        lgb_model.fit(X, y, eval_set=[(X, y)])
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, max_display=15, show=False)
        plt.savefig(str(artifacts_dir / f'shap_summary_plot_{map_labels_to_class(label)}.pdf'), bbox_inches='tight')
        plt.show()


else:
    console.log("[bold yellow]XAI analysis skipped (use --xai flag to enable)[/bold yellow]")
