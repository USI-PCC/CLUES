"""Sliding window segmentation for OCO data.

Creates 30s windows with 7.5s overlap. Can process all subjects or only
subjects from a specified starting subject ID onwards.

Usage:
    python -m clues.pipeline.segmentation                       # all subjects
    python -m clues.pipeline.segmentation --from-subject s_026  # only s_026+
    python -m clues.pipeline.segmentation --skip-existing       # skip if segmented file exists
"""

import argparse
import pandas as pd
import os
from rich.console import Console

from lib.config import WINDOW_SIZE, WINDOW_OVERLAP, OCO_DIR

console = Console()


def segment_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Segment a per-subject DataFrame into sliding 30s windows with 7.5s overlap.

    Parameters
    ----------
    df:
        Must contain a ``software_timestamp`` column (numeric seconds).

    Returns
    -------
    pd.DataFrame or None
        Concatenated windowed rows with a ``window_index`` column prepended,
        or ``None`` when no full window fits in the data span.
    """
    start_time = df["software_timestamp"].min()
    end_time = df["software_timestamp"].max()

    segmented = None
    current_start = start_time
    window_index = 0

    while current_start + WINDOW_SIZE <= end_time:
        current_end = current_start + WINDOW_SIZE
        segment = df[
            (df["software_timestamp"] >= current_start)
            & (df["software_timestamp"] < current_end)
        ].copy()
        segment["window_index"] = window_index

        if segmented is None:
            segmented = segment
        else:
            segmented = pd.concat([segmented, segment], ignore_index=True)

        current_start += WINDOW_SIZE - WINDOW_OVERLAP
        window_index += 1

    if segmented is None or len(segmented) == 0:
        return None

    # Move window_index to front
    cols = list(segmented.columns)
    cols.insert(0, cols.pop(cols.index("window_index")))
    segmented = segmented[cols]

    return segmented


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment OCO data into sliding windows")
    parser.add_argument("--from-subject", type=str, default=None,
                        help="Only process subjects >= this ID (e.g., s_026)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip subjects that already have a segmented file")
    args = parser.parse_args()

    oco_dir = str(OCO_DIR)
    all_folders = sorted([f for f in os.listdir(oco_dir) if f.startswith("s_")])

    if args.from_subject:
        all_folders = [f for f in all_folders if f >= args.from_subject]
        console.log(f"Filtering to subjects >= {args.from_subject}: {len(all_folders)} subjects")

    processed_count = 0
    skipped_count = 0

    for folder in all_folders:
        input_file = os.path.join(oco_dir, folder, f"{folder}_oco.csv")
        output_file = os.path.join(oco_dir, folder, f"{folder}_oco_segmented.csv")

        if args.skip_existing and os.path.exists(output_file):
            console.log(f"[yellow]Skipping {folder} - segmented file already exists[/yellow]")
            skipped_count += 1
            continue

        if not os.path.exists(input_file):
            console.log(f"Skipping {folder} as the CSV file does not exist.")
            skipped_count += 1
            continue

        console.log(f"Processing {folder}...")

        try:
            subject_df = pd.read_csv(input_file)

            if "software_timestamp" not in subject_df.columns:
                console.log(f"[red]Error: {folder} missing 'software_timestamp' column[/red]")
                skipped_count += 1
                continue

            segmented = segment_frame(subject_df)

            if segmented is None or len(segmented) == 0:
                console.log(f"[yellow]Warning: {folder} produced no segments[/yellow]")
                skipped_count += 1
                continue

            window_index = segmented["window_index"].max() + 1
            console.log(f"  Segmented {folder} into {window_index} windows.")

            segmented.to_csv(output_file, index=False)
            console.log(f"  Saved segmented data for {folder}")
            processed_count += 1

        except Exception as e:
            console.log(f"[red]Error processing {folder}: {e}[/red]")
            skipped_count += 1

    console.print(f"\n[bold green]Segmentation complete![/bold green]")
    console.print(f"  Processed: {processed_count} subjects")
    console.print(f"  Skipped: {skipped_count} subjects")
