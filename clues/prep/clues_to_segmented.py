"""Build classic-pipeline segmented CSVs from the fresh CLUES ocosense dump.

Bridges the schema gap between /data/datasets/CLUES ({subject}/ocosense/{subject}.csv,
`label` phase column, no software_timestamp) and what modeling.py expects
({subject}/{subject}_oco_segmented.csv).

Path resolution is delegated to lib.paths.clues_subject_csv, which is the
canonical single source of truth for the CLUES nested layout.
"""
import os
from pathlib import Path

import pandas as pd

from lib.config import EXCLUDED_SUBJECTS
from lib.paths import clues_subject_csv
from clues.pipeline.segmentation import segment_frame


def _load_clues_frame(csv_path):
    df = pd.read_csv(csv_path)
    if "label" in df.columns and "experimental_phase" not in df.columns:
        df = df.rename(columns={"label": "experimental_phase"})
    if "software_timestamp" not in df.columns:
        df["software_timestamp"] = df["timestamp"]
    return df


def build_segmented(clues_root, out_root):
    """For each {subject}/ocosense/{subject}.csv under clues_root, segment and write
    {out_root}/{subject}/{subject}_oco_segmented.csv.

    Skips subjects in EXCLUDED_SUBJECTS and subjects whose CSV is missing or
    whose segmentation produces no windows.
    """
    clues_root, out_root = Path(clues_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in clues_root.iterdir() if p.is_dir()):
        if d.name in EXCLUDED_SUBJECTS:
            continue
        csv = clues_subject_csv(d)
        if not csv.exists():
            continue
        seg = segment_frame(_load_clues_frame(csv))
        if seg is None or len(seg) == 0:
            continue
        subj_out = out_root / d.name
        subj_out.mkdir(parents=True, exist_ok=True)
        seg.to_csv(subj_out / f"{d.name}_oco_segmented.csv", index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clues-dir", default=os.environ.get("CLUES_DIR", "/data/datasets/CLUES"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    build_segmented(a.clues_dir, a.out_dir)
    print(f"segmented -> {a.out_dir}")
