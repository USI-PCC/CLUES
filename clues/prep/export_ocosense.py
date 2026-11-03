"""Export synced OCOSense recordings to the shared CLUES dataset layout.

Reads <synced-dir>/<subject>/oco/<subject>_oco.csv (the output of
clues.pipeline.sync) and writes <out-dir>/<subject>/ocosense/<subject>.csv,
the file distributed in the shared dataset:
  - 'experimental_phase' is renamed to 'label'
  - 'timestamp' is made relative to the first sample (seconds)
  - 'software_timestamp' is dropped

Usage: python -m clues.prep.export_ocosense --out-dir /path/to/CLUES
"""

import argparse
from pathlib import Path

import pandas as pd

from lib.config import SYNCED_DIR


def export_subject(src_csv, dest_csv):
    df = pd.read_csv(src_csv)
    df = df.rename(columns={"experimental_phase": "label"})
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]
    df = df.drop(columns=["software_timestamp"], errors="ignore")
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description="Export synced OCOSense recordings to the shared dataset layout.")
    ap.add_argument("--synced-dir", type=Path, default=SYNCED_DIR, help="sync output root (default: data/usi-synced)")
    ap.add_argument("--out-dir", type=Path, required=True, help="dataset root to write <subject>/ocosense/ into")
    args = ap.parse_args()
    for subject_dir in sorted(args.synced_dir.glob("s_*")):
        s = subject_dir.name
        src = subject_dir / "oco" / f"{s}_oco.csv"
        if not src.exists():
            print(f"{s}: no synced OCOSense file, skipped")
            continue
        export_subject(src, args.out_dir / s / "ocosense" / f"{s}.csv")
        print(f"{s}: exported")


if __name__ == "__main__":
    main()
