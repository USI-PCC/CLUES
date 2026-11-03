"""Dataset path resolution."""

from pathlib import Path


def clues_subject_csv(subject_dir):
    """Path to a subject's raw OCO CSV under either the fresh CLUES layout
    ({subject}/ocosense/{subject}.csv) or the legacy layout ({subject}/{subject}_oco.csv)."""
    subject_dir = Path(subject_dir)
    fresh = subject_dir / "ocosense" / f"{subject_dir.name}.csv"
    if fresh.exists():
        return fresh
    return subject_dir / f"{subject_dir.name}_oco.csv"
