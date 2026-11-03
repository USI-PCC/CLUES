"""
Measure alignment between RGB and eyetracker 3-second windows.

This script compares temporal alignment of RGB and eyetracker modalities
independent of empatica data (which has known flaws). It normalizes timestamps
per subject and measures coverage and offset statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def load_and_normalize_rgb(path):
    """
    Load RGB data and normalize timestamps per subject.

    Args:
        path: Path to RGB CSV file

    Returns:
        DataFrame with normalized timestamps in seconds
    """
    console.print(f"[cyan]Loading RGB data from {path}...[/cyan]")
    df = pd.read_csv(path)

    # Convert deciseconds to seconds
    df['start_sec'] = df['Start_i'] / 10.0
    df['end_sec'] = df['End_i'] / 10.0

    # Remove duplicates
    console.print(f"[yellow]Rows before deduplication: {len(df)}[/yellow]")
    df = df.drop_duplicates(subset=['Subject', 'Start_i', 'End_i'])
    console.print(f"[yellow]Rows after deduplication: {len(df)}[/yellow]")

    # Normalize per subject (subtract min time so each starts at 0)
    normalized_data = []
    for subject in df['Subject'].unique():
        subject_df = df[df['Subject'] == subject].copy()
        min_time = subject_df['start_sec'].min()
        subject_df['start_norm'] = subject_df['start_sec'] - min_time
        subject_df['end_norm'] = subject_df['end_sec'] - min_time
        normalized_data.append(subject_df)

    df_normalized = pd.concat(normalized_data, ignore_index=True)
    console.print(f"[green]RGB data loaded: {len(df_normalized)} windows from {df_normalized['Subject'].nunique()} subjects[/green]")

    return df_normalized


def load_and_normalize_eyetracker(path):
    """
    Load eyetracker data and normalize timestamps per subject.

    Args:
        path: Path to eyetracker CSV file

    Returns:
        DataFrame with normalized timestamps in seconds
    """
    console.print(f"[cyan]Loading eyetracker data from {path}...[/cyan]")
    df = pd.read_csv(path)

    # Convert milliseconds to seconds
    df['start_sec'] = df['time_start'] / 1000.0
    df['end_sec'] = df['time_end'] / 1000.0

    # Normalize per subject (subtract min time so each starts at 0)
    normalized_data = []
    for subject in df['subject'].unique():
        subject_df = df[df['subject'] == subject].copy()
        min_time = subject_df['start_sec'].min()
        subject_df['start_norm'] = subject_df['start_sec'] - min_time
        subject_df['end_norm'] = subject_df['end_sec'] - min_time
        normalized_data.append(subject_df)

    df_normalized = pd.concat(normalized_data, ignore_index=True)
    console.print(f"[green]Eyetracker data loaded: {len(df_normalized)} windows from {df_normalized['subject'].nunique()} subjects[/green]")

    return df_normalized


def find_closest_match(window_start, window_end, candidate_windows, tolerance_sec=1.0):
    """
    Find the closest matching window within tolerance.

    Args:
        window_start: Start time of reference window
        window_end: End time of reference window
        candidate_windows: DataFrame of candidate windows
        tolerance_sec: Maximum allowed offset in seconds

    Returns:
        Tuple of (matched_index, offset) or (None, None) if no match
    """
    window_center = (window_start + window_end) / 2.0

    # Calculate center of each candidate window
    candidate_centers = (candidate_windows['start_norm'] + candidate_windows['end_norm']) / 2.0

    # Find offsets
    offsets = np.abs(candidate_centers - window_center)

    # Find minimum offset within tolerance
    min_offset_idx = offsets.idxmin()
    min_offset = offsets[min_offset_idx]

    if min_offset <= tolerance_sec:
        return min_offset_idx, min_offset
    else:
        return None, None


def measure_alignment(df_rgb, df_et, tolerance_sec=1.0):
    """
    Measure alignment between RGB and eyetracker 3s windows.

    Args:
        df_rgb: RGB DataFrame with normalized timestamps
        df_et: Eyetracker DataFrame with normalized timestamps
        tolerance_sec: Maximum allowed offset for matching (default: 1.0 second)

    Returns:
        Tuple of (per_subject_stats, overall_summary)
    """
    console.print(f"\n[cyan]Measuring alignment with tolerance = {tolerance_sec}s...[/cyan]")

    per_subject_stats = []
    all_offsets = []

    # Get common subjects
    rgb_subjects = set(df_rgb['Subject'].unique())
    et_subjects = set(df_et['subject'].unique())
    common_subjects = sorted(rgb_subjects & et_subjects)

    console.print(f"[yellow]Common subjects: {len(common_subjects)}[/yellow]")
    console.print(f"[yellow]RGB-only subjects: {len(rgb_subjects - et_subjects)}[/yellow]")
    console.print(f"[yellow]ET-only subjects: {len(et_subjects - rgb_subjects)}[/yellow]")

    for subject in common_subjects:
        rgb_sub = df_rgb[df_rgb['Subject'] == subject].copy()
        et_sub = df_et[df_et['subject'] == subject].copy()

        # Match RGB windows to eyetracker
        rgb_matched = 0
        rgb_offsets = []

        for idx, row in rgb_sub.iterrows():
            match_idx, offset = find_closest_match(
                row['start_norm'],
                row['end_norm'],
                et_sub,
                tolerance_sec
            )
            if match_idx is not None:
                rgb_matched += 1
                rgb_offsets.append(offset)

        # Match eyetracker windows to RGB
        et_matched = 0
        et_offsets = []

        for idx, row in et_sub.iterrows():
            match_idx, offset = find_closest_match(
                row['start_norm'],
                row['end_norm'],
                rgb_sub,
                tolerance_sec
            )
            if match_idx is not None:
                et_matched += 1
                et_offsets.append(offset)

        # Calculate coverage percentages
        rgb_coverage = (rgb_matched / len(rgb_sub)) * 100 if len(rgb_sub) > 0 else 0
        et_coverage = (et_matched / len(et_sub)) * 100 if len(et_sub) > 0 else 0

        # Store stats
        per_subject_stats.append({
            'subject': subject,
            'rgb_windows': len(rgb_sub),
            'et_windows': len(et_sub),
            'rgb_matched': rgb_matched,
            'et_matched': et_matched,
            'rgb_coverage_pct': rgb_coverage,
            'et_coverage_pct': et_coverage,
            'mean_offset_sec': np.mean(rgb_offsets + et_offsets) if (rgb_offsets or et_offsets) else 0,
            'median_offset_sec': np.median(rgb_offsets + et_offsets) if (rgb_offsets or et_offsets) else 0
        })

        all_offsets.extend(rgb_offsets + et_offsets)

    # Calculate overall summary
    total_rgb_windows = sum(s['rgb_windows'] for s in per_subject_stats)
    total_et_windows = sum(s['et_windows'] for s in per_subject_stats)
    total_rgb_matched = sum(s['rgb_matched'] for s in per_subject_stats)
    total_et_matched = sum(s['et_matched'] for s in per_subject_stats)

    overall_summary = {
        'total_subjects': len(common_subjects),
        'total_rgb_windows': total_rgb_windows,
        'total_et_windows': total_et_windows,
        'total_rgb_matched': total_rgb_matched,
        'total_et_matched': total_et_matched,
        'overall_rgb_coverage_pct': (total_rgb_matched / total_rgb_windows * 100) if total_rgb_windows > 0 else 0,
        'overall_et_coverage_pct': (total_et_matched / total_et_windows * 100) if total_et_windows > 0 else 0,
        'mean_offset_sec': np.mean(all_offsets) if all_offsets else 0,
        'median_offset_sec': np.median(all_offsets) if all_offsets else 0,
        'std_offset_sec': np.std(all_offsets) if all_offsets else 0
    }

    return per_subject_stats, overall_summary


def print_results(per_subject_stats, overall_summary):
    """
    Print alignment results in a formatted table.

    Args:
        per_subject_stats: List of per-subject statistics dictionaries
        overall_summary: Overall summary statistics dictionary
    """
    # Per-subject table
    console.print("\n[bold cyan]═══ Per-Subject Alignment Statistics ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Subject", style="cyan")
    table.add_column("RGB Windows", justify="right")
    table.add_column("ET Windows", justify="right")
    table.add_column("RGB Matched", justify="right")
    table.add_column("ET Matched", justify="right")
    table.add_column("RGB Coverage %", justify="right")
    table.add_column("ET Coverage %", justify="right")
    table.add_column("Mean Offset (s)", justify="right")

    for stat in per_subject_stats:
        table.add_row(
            stat['subject'],
            str(stat['rgb_windows']),
            str(stat['et_windows']),
            str(stat['rgb_matched']),
            str(stat['et_matched']),
            f"{stat['rgb_coverage_pct']:.1f}",
            f"{stat['et_coverage_pct']:.1f}",
            f"{stat['mean_offset_sec']:.3f}"
        )

    console.print(table)

    # Overall summary
    console.print("\n[bold cyan]═══ Overall Alignment Summary ═══[/bold cyan]\n")

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total Subjects", str(overall_summary['total_subjects']))
    summary_table.add_row("Total RGB Windows", str(overall_summary['total_rgb_windows']))
    summary_table.add_row("Total ET Windows", str(overall_summary['total_et_windows']))
    summary_table.add_row("RGB Windows Matched", str(overall_summary['total_rgb_matched']))
    summary_table.add_row("ET Windows Matched", str(overall_summary['total_et_matched']))
    summary_table.add_row("Overall RGB Coverage %", f"{overall_summary['overall_rgb_coverage_pct']:.2f}")
    summary_table.add_row("Overall ET Coverage %", f"{overall_summary['overall_et_coverage_pct']:.2f}")
    summary_table.add_row("Mean Offset (s)", f"{overall_summary['mean_offset_sec']:.4f}")
    summary_table.add_row("Median Offset (s)", f"{overall_summary['median_offset_sec']:.4f}")
    summary_table.add_row("Std Offset (s)", f"{overall_summary['std_offset_sec']:.4f}")

    console.print(summary_table)


def save_results(per_subject_stats, overall_summary, output_path):
    """
    Save alignment results to CSV file.

    Args:
        per_subject_stats: List of per-subject statistics dictionaries
        overall_summary: Overall summary statistics dictionary
        output_path: Path to output CSV file
    """
    # Save per-subject stats
    df_stats = pd.DataFrame(per_subject_stats)
    df_stats.to_csv(output_path, index=False)
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    # Also save summary as a separate file
    summary_path = str(output_path).replace('.csv', '_summary.csv')
    df_summary = pd.DataFrame([overall_summary])
    df_summary.to_csv(summary_path, index=False)
    console.print(f"[green]Summary saved to {summary_path}[/green]")


def main():
    """Main execution function."""
    # Input paths
    script_dir = Path(__file__).parent
    rebuttal_dir = script_dir / '..' / 'rebuttal'
    rgb_path = rebuttal_dir / 'RGB' / '01_RESULTS_JANUARY' / 'loso_low_vs_high_vs_no.csv'
    et_path = rebuttal_dir / 'eyetracker' / 'loso_df3_LGBM_predictions.csv'

    # Output path
    output_path = rebuttal_dir / 'rgb_eyetracker_alignment_stats.csv'

    console.print("[bold green]═══ RGB-Eyetracker Alignment Measurement ═══[/bold green]\n")

    # Load and normalize data
    df_rgb = load_and_normalize_rgb(rgb_path)
    df_et = load_and_normalize_eyetracker(et_path)

    # Measure alignment with 1.0 second tolerance
    per_subject_stats, overall_summary = measure_alignment(df_rgb, df_et, tolerance_sec=1.0)

    # Print results
    print_results(per_subject_stats, overall_summary)

    # Save results
    save_results(per_subject_stats, overall_summary, output_path)

    console.print("\n[bold green]✓ Alignment measurement complete![/bold green]")


if __name__ == "__main__":
    main()
