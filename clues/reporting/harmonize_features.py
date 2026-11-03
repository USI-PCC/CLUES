"""
Harmonize feature files across all modalities.

1. Extracts features from empatica prediction CSVs → empatica/artifacts/features.csv
2. Standardizes metadata columns across all modality feature files to:
   subject, task, start_time, end_time, <feature_columns>

Prerequisites:
  - Empatica predictions must exist (from external pipeline)
  - OCOSense features.csv must exist (run modeling.py first)
  - Eye-tracker eyetracker_features.csv and RGB rgb_features.csv must exist (from external pipelines)

Usage: python -m clues.reporting.harmonize_features
"""

import pandas as pd
from rich.console import Console
from lib.config import RESULTS_DIR

console = Console()

META_COLS = ["subject", "task", "start_time", "end_time"]


def reorder_cols(df):
    """Put metadata columns first, then feature columns."""
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df[META_COLS + feature_cols]


def harmonize_empatica():
    """Combine empatica raw features with prediction metadata → artifacts/features.csv.

    The new empatica pipeline stores raw features in artifacts/empatica_features.csv
    (no metadata columns) and per-row metadata (label, subject, time_start, time_end)
    in the prediction CSVs. Rows in both files are aligned by position.
    """
    dst_dir = RESULTS_DIR / "empatica" / "artifacts"
    dst_dir.mkdir(parents=True, exist_ok=True)

    feats_src = dst_dir / "empatica_features.csv"
    meta_src = RESULTS_DIR / "empatica" / "loso_3class" / "lgbm_predictions.csv"

    console.log(f"Reading {feats_src}")
    feats = pd.read_csv(feats_src)
    console.log(f"Reading {meta_src}")
    meta = pd.read_csv(meta_src)

    if len(feats) != len(meta):
        raise ValueError(
            f"Empatica row mismatch: features={len(feats)} vs predictions={len(meta)}"
        )

    # Pull metadata from prediction file
    meta = meta[["subject", "label", "time_start", "time_end"]].rename(columns={
        "label": "task",
        "time_start": "start_time",
        "time_end": "end_time",
    })

    df = pd.concat([meta.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    df = reorder_cols(df)

    dst = dst_dir / "features.csv"
    df.to_csv(dst, index=False)
    console.log(f"[green]Empatica:[/green] {df.shape} → {dst}")


def harmonize_eyetracker():
    """Standardize eyetracker feature file metadata."""
    src = RESULTS_DIR / "eyetracker" / "artifacts" / "eyetracker_features.csv"
    dst = RESULTS_DIR / "eyetracker" / "artifacts" / "features.csv"

    console.log(f"Reading {src}")
    df = pd.read_csv(src)

    # Drop unnecessary metadata columns + unnamed index
    drop_cols = ["Filename", "Index_start", "Index_end", "Win len", "Overlap Percent"]
    drop_cols += [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Rename metadata
    df = df.rename(columns={
        "Subject": "subject",
        "Label": "task",
        "Time_start": "start_time",
        "Time_end": "end_time",
    })

    # Clean subject IDs: "s_001.csv" → "s_001"
    df["subject"] = df["subject"].str.replace(r"\.csv$", "", regex=True)

    df = reorder_cols(df)
    df.to_csv(dst, index=False)
    console.log(f"[green]Eyetracker:[/green] {df.shape} → {dst}")


def harmonize_rgb():
    """Standardize RGB feature file metadata."""
    src = RESULTS_DIR / "rgb" / "artifacts" / "rgb_features.csv"
    dst = RESULTS_DIR / "rgb" / "artifacts" / "features.csv"

    console.log(f"Reading {src}")
    df = pd.read_csv(src)

    # Drop unnecessary metadata columns (both legacy and new pipeline variants)
    drop_cols = ["Path", "RelativePath", "Filename", "Modality", "Fs", "Assumed fs",
                 "Start_i", "End_i", "Frame_i", "Frame_t", "Timing mode",
                 "Index_in_sequence", "Detection",
                 "Crop mode", "Face detector backend",
                 "DetectorSuccess", "TemporalReuse", "MediapipeSuccess"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Rename metadata
    df = df.rename(columns={
        "Subject": "subject",
        "Label": "task",
        "Start_t": "start_time",
        "End_t": "end_time",
    })

    df = reorder_cols(df)
    df.to_csv(dst, index=False)
    console.log(f"[green]RGB:[/green] {df.shape} → {dst}")


def harmonize_ocosense():
    """Standardize OCOSense feature file metadata."""
    src = RESULTS_DIR / "ocosense" / "artifacts" / "features.csv"

    console.log(f"Reading {src}")
    df = pd.read_csv(src)

    # Drop window_index (internal tracking, not needed in feature file)
    drop_cols = ["window_index"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Rename metadata
    df = df.rename(columns={
        "experimental_phase": "task",
        "start_timestamp": "start_time",
        "end_timestamp": "end_time",
    })

    df = reorder_cols(df)
    df.to_csv(src, index=False)
    console.log(f"[green]OCOSense:[/green] {df.shape} → {src}")


def print_summary():
    """Print a summary of all feature files."""
    console.print("\n[bold]Feature files summary:[/bold]")
    for modality in ["empatica", "eyetracker", "rgb", "ocosense"]:
        path = RESULTS_DIR / modality / "artifacts" / "features.csv"
        if path.exists():
            df = pd.read_csv(path, nrows=0)
            n_features = len([c for c in df.columns if c not in META_COLS])
            console.print(f"  {modality:12s}: {n_features:4d} features, metadata={list(df.columns[:4])}")
        else:
            console.print(f"  {modality:12s}: [red]not found[/red]")


if __name__ == "__main__":
    harmonize_empatica()
    harmonize_eyetracker()
    harmonize_rgb()
    harmonize_ocosense()
    print_summary()
    console.print("\n[bold green]All feature files harmonized.[/bold green]")
