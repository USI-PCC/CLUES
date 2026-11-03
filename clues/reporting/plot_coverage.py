"""
Compact coverage visualization for email/paper use.

Two horizontal bar panels (LOSO and LOGO) showing each modality's yield
against the empatica 30 s reference, with subject coverage and matched-window
counts annotated directly on the bars.

Output: data/results/plots/fusion_coverage.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.config import RESULTS_DIR, REBUTTAL_DIR

MODALITIES = ["empatica", "ocosense", "eyetracker", "rgb"]
COLORS = {
    "empatica":   "#4C72B0",
    "ocosense":   "#55A868",
    "eyetracker": "#C44E52",
    "rgb":        "#8172B3",
}
EXPERIMENTS = [
    ("LOSO (2- & 3-class, identical coverage)", "loso_two_class"),
    ("LOGO 3-class", "logo_three_class"),
]


def coverage_row(df, mod):
    col = f"y_pred_{mod}"
    total_rows = len(df)
    total_subj = df["subject_id"].nunique()
    present = df[col] != -1
    matched = int(present.sum())
    subj_with_data = int(df.loc[present, "subject_id"].nunique())
    if subj_with_data:
        sub_df = df[df.subject_id.isin(df.loc[present, "subject_id"].unique())]
        within = 100.0 * (sub_df[col] != -1).sum() / len(sub_df)
    else:
        within = 0.0
    total_yield = 100.0 * matched / total_rows
    return total_rows, total_subj, matched, total_yield, within, subj_with_data


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    y_pos = np.arange(len(MODALITIES))

    for ax, (title, exp) in zip(axes, EXPERIMENTS):
        path = REBUTTAL_DIR / "fusion" / f"majority_voting_{exp}.csv"
        df = pd.read_csv(path)
        ref_total, ref_subj, *_ = coverage_row(df, "empatica")

        total_yields = []
        within_yields = []
        annotations = []
        for mod in MODALITIES:
            total, _, matched, t_yield, w_yield, subj = coverage_row(df, mod)
            total_yields.append(t_yield)
            within_yields.append(w_yield)
            annotations.append(
                f"{matched:,}/{ref_total:,} windows  ·  {subj}/{ref_subj} subj"
            )

        bars = ax.barh(
            y_pos, total_yields,
            color=[COLORS[m] for m in MODALITIES],
            edgecolor="black", linewidth=0.4, height=0.55,
            label="Total yield (of empatica reference)",
        )

        # Overlay within-subject yield as a thin darker bar
        ax.barh(
            y_pos, within_yields,
            color=[COLORS[m] for m in MODALITIES],
            alpha=0.35, edgecolor="none", height=0.85, zorder=0,
        )

        # Annotations inside bars (total yield value) + right-side context
        for i, (tv, wv, ann) in enumerate(zip(total_yields, within_yields, annotations)):
            ax.text(min(tv - 2, 98), i, f"{tv:.0f}%",
                    va="center", ha="right", fontsize=9,
                    color="white", fontweight="bold")
            ax.text(101, i, f"within-subj {wv:.0f}%   {ann}",
                    va="center", ha="left", fontsize=8, color="#333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(MODALITIES, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Total yield (% of empatica reference windows matched)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.invert_yaxis()
        # Extra right margin so annotations fit
        ax.set_xlim(0, 100)

    # Shared overall title
    fig.suptitle(
        "Fusion data coverage per modality  ·  empatica as 30 s reference",
        fontsize=12, fontweight="bold", y=1.02,
    )

    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fusion_coverage.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
