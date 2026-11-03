"""
Compute dataset statistics for the OCOsense modality and emit a JSON blob
suitable for a reviewer-facing summary table.

Reports per subject:
  - recording_duration_seconds (max - min timestamp in raw preprocessed CSV)
  - n_raw_samples (row count in raw preprocessed CSV)
  - pct_missing_sensor_cells (fraction of *sentinel* NaN in acc/gyro/mag/nav
    columns; OCO fills missing readings with 0 rather than NaN, so this is
    typically 0% and must be read together with the window-level metric)
  - n_segmented_windows, n_usable_windows, pct_windows_dropped
  - n_windows_dropped_questionnaire, n_windows_dropped_multi_phase
    (count of the two filter-level dropout reasons)
  - usable (bool: True if subject contributes any usable window)
  - data_status ("ok", "no_data_recorded", "excluded_config")

Hardware-level reasons for data loss (device calibration, participant
dropouts, etc.) require author input and are left as a placeholder in
``data_loss_reasons.hardware_recording_notes``.

Usage:
    python -m clues.analysis.oco_dataset_stats
    # -> data/results/ocosense/artifacts/dataset_stats.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from lib.config import OCO_DIR, RESULTS_DIR, EXCLUDED_SUBJECTS

console = Console()

SENSOR_PREFIXES = ("accelerometer_", "gyroscope_", "magnetometer_", "nav_", "prox_")
OUTPUT_PATH = RESULTS_DIR / "ocosense" / "artifacts" / "dataset_stats.json"


def _sensor_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith(SENSOR_PREFIXES)]


def _subject_from_path(p: Path) -> str:
    return p.name


def _compute_subject_stats(subject_dir: Path) -> dict:
    """Return the per-subject stats dict (handles missing-data cases)."""
    subject = _subject_from_path(subject_dir)
    stats = {
        "subject": subject,
        "data_status": None,
        "recording_duration_seconds": None,
        "n_raw_samples": None,
        "pct_missing_sensor_cells": None,
        "n_segmented_windows": None,
        "n_usable_windows": None,
        "n_windows_dropped_questionnaire": None,
        "n_windows_dropped_multi_phase": None,
        "pct_windows_dropped": None,
        "usable": False,
    }

    # Explicit "no data" marker file written by the sync/prep pipeline
    if (subject_dir / "oco_missing.txt").exists():
        stats["data_status"] = "no_data_recorded"
        return stats

    # Config-level exclusion (known data quality issue)
    if subject in EXCLUDED_SUBJECTS:
        stats["data_status"] = "excluded_config"

    raw_path = subject_dir / f"{subject}_oco.csv"
    segmented_path = subject_dir / f"{subject}_oco_segmented.csv"

    if not raw_path.exists():
        # Metadata-only directory (recording aborted before any usable capture)
        if stats["data_status"] is None:
            stats["data_status"] = "no_data_recorded"
        return stats

    # ---- Raw file: duration, sample count, NaN rate on sensor cells ----
    df = pd.read_csv(raw_path, low_memory=False)
    sensor_cols = _sensor_columns(df)
    numeric = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    n_cells = numeric.size
    n_nan = int(numeric.isna().sum().sum())

    stats["n_raw_samples"] = int(len(df))
    stats["recording_duration_seconds"] = float(
        df["timestamp"].max() - df["timestamp"].min()
    ) if "timestamp" in df.columns and len(df) else 0.0
    stats["pct_missing_sensor_cells"] = (100.0 * n_nan / n_cells) if n_cells else None

    # ---- Segmented file: windows pre and post modeling.py filters ----
    if segmented_path.exists():
        seg = pd.read_csv(segmented_path, low_memory=False)
        windows = seg.groupby("window_index")
        stats["n_segmented_windows"] = int(windows.ngroups)

        # Replicate modeling.py filters:
        # 1. drop windows spanning >1 experimental_phase (multi-phase)
        # 2. drop windows whose (sole) phase == "questionnaire"
        phase_counts = windows["experimental_phase"].nunique()
        multi_phase_wins = set(phase_counts[phase_counts > 1].index)

        single_phase_wins = phase_counts[phase_counts == 1].index
        seg_single = seg[seg["window_index"].isin(single_phase_wins)]
        # First (== only) phase per single-phase window
        window_phase = seg_single.groupby("window_index")["experimental_phase"].first()
        questionnaire_wins = set(window_phase[window_phase == "questionnaire"].index)
        usable_wins = set(window_phase[window_phase != "questionnaire"].index)

        total_wins = int(windows.ngroups)
        stats["n_usable_windows"] = len(usable_wins)
        stats["n_windows_dropped_questionnaire"] = len(questionnaire_wins)
        stats["n_windows_dropped_multi_phase"] = len(multi_phase_wins)
        stats["pct_windows_dropped"] = (
            100.0 * (total_wins - len(usable_wins)) / total_wins if total_wins else None
        )
    else:
        stats["n_segmented_windows"] = 0
        stats["n_usable_windows"] = 0
        stats["n_windows_dropped_questionnaire"] = 0
        stats["n_windows_dropped_multi_phase"] = 0
        stats["pct_windows_dropped"] = None

    # Status & usability (subjects in EXCLUDED_SUBJECTS stay un-usable even
    # if they have windows, because modeling.py drops them upstream)
    if stats["data_status"] == "excluded_config":
        stats["usable"] = False
    else:
        stats["data_status"] = "ok"
        stats["usable"] = (stats["n_usable_windows"] or 0) > 0

    return stats


def _aggregate(subjects: list) -> dict:
    usable = [s for s in subjects if s["usable"]]

    def _vals(key):
        return [s[key] for s in usable if s.get(key) is not None]

    durations = _vals("recording_duration_seconds")
    missing = _vals("pct_missing_sensor_cells")
    raw_samples = _vals("n_raw_samples")
    seg_windows = _vals("n_segmented_windows")
    usable_windows = _vals("n_usable_windows")
    win_drop = _vals("pct_windows_dropped")
    dropped_quest = _vals("n_windows_dropped_questionnaire")
    dropped_multi = _vals("n_windows_dropped_multi_phase")

    return {
        "n_subjects_total": len(subjects),
        "n_subjects_usable": len(usable),
        "n_subjects_excluded": len(subjects) - len(usable),
        "recording_duration_seconds": {
            "mean": float(np.mean(durations)) if durations else None,
            "std":  float(np.std(durations))  if durations else None,
            "min":  float(np.min(durations))  if durations else None,
            "max":  float(np.max(durations))  if durations else None,
            "total": float(np.sum(durations)) if durations else None,
        },
        "pct_missing_sensor_cells_per_subject": {
            "mean": float(np.mean(missing)) if missing else None,
            "std":  float(np.std(missing))  if missing else None,
            "min":  float(np.min(missing))  if missing else None,
            "max":  float(np.max(missing))  if missing else None,
        },
        "n_raw_samples_total": int(np.sum(raw_samples)) if raw_samples else 0,
        "n_segmented_windows_total": int(np.sum(seg_windows)) if seg_windows else 0,
        "n_usable_windows_total":   int(np.sum(usable_windows)) if usable_windows else 0,
        "pct_windows_dropped_per_subject": {
            "mean": float(np.mean(win_drop)) if win_drop else None,
            "std":  float(np.std(win_drop))  if win_drop else None,
            "min":  float(np.min(win_drop))  if win_drop else None,
            "max":  float(np.max(win_drop))  if win_drop else None,
        },
        "n_windows_dropped_questionnaire_total": int(np.sum(dropped_quest)) if dropped_quest else 0,
        "n_windows_dropped_multi_phase_total":  int(np.sum(dropped_multi))  if dropped_multi  else 0,
    }


def main():
    subject_dirs = sorted(p for p in Path(OCO_DIR).iterdir() if p.is_dir() and p.name.startswith("s_"))
    console.log(f"Found {len(subject_dirs)} subject directories under {OCO_DIR}")

    per_subject = []
    for d in subject_dirs:
        console.log(f"  {d.name} ...")
        per_subject.append(_compute_subject_stats(d))

    reasons_for_loss = {s["subject"]: s["data_status"] for s in per_subject if not s["usable"]}

    out = {
        "modality": "ocosense",
        "notes": {
            "sampling_rate_hz_imu": 50,
            "sampling_rate_hz_nav_prox": 25,
            "window_size_seconds": 30.0,
            "window_overlap_seconds": 7.5,
        },
        "overall":  _aggregate(per_subject),
        "per_subject": per_subject,
        "data_loss_reasons": {
            "pipeline_filters": [
                "Subject s_046 is excluded upstream due to known data quality issues (config.EXCLUDED_SUBJECTS).",
                "Windows that span more than one experimental phase are dropped (multi-phase windows).",
                "Windows falling inside a questionnaire phase are dropped (not task-related).",
                "During feature extraction, rows with more than 50% NaN sensor values are dropped (config.NAN_DROP_THRESHOLD).",
                "Subjects whose preprocessing produced zero usable windows are dropped implicitly.",
            ],
            "per_subject_status": reasons_for_loss,
            "hardware_recording_notes": (
                "REQUIRES AUTHOR INPUT — e.g. recording aborted, glasses calibration "
                "failure, battery depletion, or consent withdrawn. Subjects currently "
                "flagged 'no_data_recorded' had empty OCO captures; 'excluded_config' "
                "is the hand-curated quality exclusion."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    console.log(f"[bold green]Wrote {OUTPUT_PATH}[/bold green]")

    # Short console summary
    ov = out["overall"]
    console.print(
        f"\n[bold]OCOsense dataset stats[/bold]\n"
        f"  Usable subjects: {ov['n_subjects_usable']} / {ov['n_subjects_total']}\n"
        f"  Mean recording duration: {ov['recording_duration_seconds']['mean']:.1f} s\n"
        f"  Mean NaN sensor cells: {ov['pct_missing_sensor_cells_per_subject']['mean']:.2f}%"
        " (OCO writes 0 rather than NaN — window-level metric is more meaningful)\n"
        f"  Mean windows dropped (questionnaire + multi-phase): "
        f"{ov['pct_windows_dropped_per_subject']['mean']:.1f}%\n"
        f"  Total raw samples: {ov['n_raw_samples_total']:,}\n"
        f"  Total segmented windows: {ov['n_segmented_windows_total']:,}\n"
        f"  Total usable windows: {ov['n_usable_windows_total']:,}"
    )


if __name__ == "__main__":
    main()
