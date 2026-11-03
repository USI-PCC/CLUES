#!/usr/bin/env python3
"""
Script to check for timestamp gaps in OCO data files.

This script checks if consecutive samples have a timestamp difference
of at most 0.04 seconds. If not, it prints the sample rows and the difference.
"""

import pandas as pd
from pathlib import Path


def check_timestamp_gaps():
    """Check timestamp gaps for all subjects' OCO files."""

    # Define base directories
    base_dir = Path(__file__).parent.parent  # Go up to project root

    # Get all subject folders in the project root
    subject_folders = sorted([d for d in base_dir.iterdir()
                             if d.is_dir() and d.name.startswith("s_")])

    print(f"Found {len(subject_folders)} subjects to check")

    for subject_folder in subject_folders:
        subject_id = subject_folder.name

        # Look for OCO CSV file in the oco subfolder
        csv_file = subject_folder / "oco" / f"{subject_id}.csv"

        if not csv_file.exists():
            print(f"\nSkipping {subject_id}: OCO CSV file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Checking {subject_id}...")
        print(f"{'='*60}")

        # Read the CSV file
        df = pd.read_csv(csv_file)

        if 'timestamp' not in df.columns:
            print(f"  Warning: 'timestamp' column not found in {subject_id}")
            continue

        # Calculate timestamp differences between consecutive samples
        timestamp_diffs = df['timestamp'].diff()

        # Find gaps larger than 0.04 seconds
        large_gaps = timestamp_diffs[timestamp_diffs > 1]

        if large_gaps.empty:
            print(f"  OK: All consecutive samples have timestamp difference <= 0.04s")
        else:
            print(f"  WARNING: Found {len(large_gaps)} timestamp gaps > 0.04s:")
            print()
            for idx in large_gaps.index:
                diff = timestamp_diffs.iloc[idx]
                prev_row = idx - 1
                prev_timestamp = df.iloc[prev_row]['timestamp']
                curr_timestamp = df.iloc[idx]['timestamp']
                print(f"  Row {prev_row} (t={prev_timestamp:.6f}) -> Row {idx} (t={curr_timestamp:.6f}): diff={diff:.6f}s")

    print(f"\n{'='*60}")
    print("All subjects checked!")
    print(f"{'='*60}")


if __name__ == "__main__":
    check_timestamp_gaps()
