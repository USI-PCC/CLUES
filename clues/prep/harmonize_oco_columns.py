"""
Harmonize OCO columns across all subjects by dropping columns that don't exist in newer subjects.

Removes: ml_arousal, ml_expression_Intensity, ml_valence from subjects s_001-s_025
This ensures all subjects have consistent column structure for modeling.
"""

import pandas as pd
import os
from rich.console import Console

from lib.config import OCO_DIR, COLUMNS_TO_DROP_OLD_SUBJECTS, OLD_SUBJECTS

console = Console()

# Path to OCO data
oco_dir = str(OCO_DIR)

# Columns to drop from older subjects
columns_to_drop = COLUMNS_TO_DROP_OLD_SUBJECTS

# Subjects to process (s_001 through s_025)
subjects_to_process = OLD_SUBJECTS

console.log(f"Harmonizing OCO columns for {len(subjects_to_process)} subjects")
console.log(f"Columns to drop: {', '.join(columns_to_drop)}")

processed_count = 0
skipped_count = 0

for subject in subjects_to_process:
    subject_folder = os.path.join(oco_dir, subject)
    oco_file = os.path.join(subject_folder, f"{subject}_oco.csv")
    segmented_file = os.path.join(subject_folder, f"{subject}_oco_segmented.csv")

    if not os.path.exists(oco_file):
        console.log(f"[yellow]Skipping {subject} - file not found[/yellow]")
        skipped_count += 1
        continue

    console.log(f"[cyan]Processing {subject}...[/cyan]")

    try:
        # Process main OCO file
        df = pd.read_csv(oco_file)
        original_cols = df.shape[1]

        # Drop columns that exist
        cols_dropped = []
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                cols_dropped.append(col)

        if cols_dropped:
            df.to_csv(oco_file, index=False)
            console.log(f"  {oco_file}: Dropped {len(cols_dropped)} columns: {', '.join(cols_dropped)}")
        else:
            console.log(f"  {oco_file}: No columns to drop (already harmonized)")

        # Process segmented file if it exists
        if os.path.exists(segmented_file):
            df_seg = pd.read_csv(segmented_file)
            cols_dropped_seg = []
            for col in columns_to_drop:
                if col in df_seg.columns:
                    df_seg = df_seg.drop(columns=[col])
                    cols_dropped_seg.append(col)

            if cols_dropped_seg:
                df_seg.to_csv(segmented_file, index=False)
                console.log(f"  {segmented_file}: Dropped {len(cols_dropped_seg)} columns")

        processed_count += 1

    except Exception as e:
        console.log(f"[red]Error processing {subject}: {e}[/red]")
        skipped_count += 1
        continue

console.print(f"\n[bold green]Column harmonization complete![/bold green]")
console.print(f"  Processed: {processed_count} subjects")
console.print(f"  Skipped: {skipped_count} subjects")

# Verify harmonization by checking a few subjects
console.print(f"\n[bold cyan]Verification:[/bold cyan]")
verify_subjects = ['s_001', 's_010', 's_025', 's_026', 's_040']
for subject in verify_subjects:
    oco_file = os.path.join(oco_dir, subject, f"{subject}_oco.csv")
    if os.path.exists(oco_file):
        df = pd.read_csv(oco_file, nrows=0)
        ml_cols = [col for col in df.columns if col.startswith('ml_')]
        console.print(f"  {subject}: {len(ml_cols)} ML columns")
