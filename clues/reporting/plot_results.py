"""
Visualise the merged results in ``data/results/all_results.csv``.

Produces two figures under ``data/results/plots/``:

1. ``all_results_bars.png`` — one panel per experiment (three experiments,
   three metric rows). Each bar is a method; per-modality bars are coloured
   by modality, fusion bars share a single accent colour. Sorted by macro F1
   within each panel. Error bars show the 95 % confidence interval across
   LOSO subjects (or LOGO folds).

2. ``all_results_heatmap.png`` — methods × experiments heatmap of macro F1
   with ±CI annotations. Makes cross-experiment consistency easy to read at
   a glance.

Usage: python -m clues.reporting.plot_results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from lib.config import RESULTS_DIR

console = Console()

EXPERIMENT_ORDER = ["loso_two_class", "loso_three_class", "logo_three_class"]
METRICS = ["accuracy", "macro_f1", "auc"]
CI_COL = {
    "accuracy": "accuracy_ci95",
    "macro_f1": "macro_f1_ci95",
    "auc": "auc_ci95",
}

# Colour palette: one colour per modality plus a distinct fusion colour.
MODALITY_COLORS = {
    "empatica":   "#4C72B0",
    "ocosense":   "#55A868",
    "eyetracker": "#C44E52",
    "rgb":        "#8172B3",
}
FUSION_COLOR = "#DD8452"


def _label_for_row(row) -> str:
    """Human-readable label: per-modality → '{modality}/{method}', fusion → method."""
    if row["category"] == "fusion":
        return row["method"]
    return f"{row['modality']}/{row['method']}"


def _color_for_row(row) -> str:
    if row["category"] == "fusion":
        return FUSION_COLOR
    return MODALITY_COLORS.get(row["modality"], "#7f7f7f")


def plot_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar charts: rows = metrics (acc, F1, AUC), cols = experiments."""
    fig, axes = plt.subplots(
        len(METRICS), len(EXPERIMENT_ORDER),
        figsize=(6 * len(EXPERIMENT_ORDER), 4 * len(METRICS)),
        sharey=False,
    )

    for col, exp in enumerate(EXPERIMENT_ORDER):
        sub = df[df["experiment"] == exp].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("macro_f1", ascending=True)
        labels = sub.apply(_label_for_row, axis=1).tolist()
        colors = sub.apply(_color_for_row, axis=1).tolist()

        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            y_pos = np.arange(len(sub))
            values = sub[metric].values
            ci_col = CI_COL[metric]
            errs = sub[ci_col].values if ci_col in sub.columns else np.zeros_like(values)
            errs = np.nan_to_num(errs, nan=0.0)

            ax.barh(
                y_pos, values, color=colors,
                edgecolor="black", linewidth=0.3,
                xerr=errs, error_kw=dict(ecolor="#333", capsize=2, linewidth=0.6),
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.5)
            ax.grid(axis="x", linestyle="--", alpha=0.4)

            if row == 0:
                ax.set_title(exp, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric, fontsize=10, fontweight="bold")

            # Annotate each bar with "value ± ci95" just past the end of the bar + error
            for i, (v, e) in enumerate(zip(values, errs)):
                if np.isnan(v):
                    continue
                ax.text(v + e + 0.005, i, f"{v:.3f}±{e:.3f}", va="center", fontsize=5.5)

    # Legend (one entry per modality + fusion) at the top
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=m)
        for m, c in MODALITY_COLORS.items()
    ]
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, color=FUSION_COLOR, label="fusion")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"[green]Bar chart →[/green] {out_path}")


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Method × experiment heatmap of macro F1."""
    df = df.copy()
    df["label"] = df.apply(_label_for_row, axis=1)

    # Sort rows: fusion methods first (grouped), then per-modality grouped by modality.
    method_order = (
        df[["category", "modality", "label"]]
        .drop_duplicates()
        .assign(_ord=lambda d: (d["category"] != "fusion").astype(int))
        .sort_values(["_ord", "modality", "label"])["label"]
        .tolist()
    )

    pivot = df.pivot_table(
        index="label", columns="experiment", values="macro_f1", aggfunc="first"
    ).reindex(index=method_order, columns=EXPERIMENT_ORDER)
    pivot_ci = df.pivot_table(
        index="label", columns="experiment", values="macro_f1_ci95", aggfunc="first"
    ).reindex(index=method_order, columns=EXPERIMENT_ORDER)

    fig, ax = plt.subplots(figsize=(7, max(6, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Annotate each cell with its mean and ±CI
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            ci = pivot_ci.values[i, j] if not np.isnan(pivot_ci.values[i, j]) else 0.0
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n±{ci:.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    # Horizontal separator between fusion and per-modality methods
    n_fusion = (df.loc[df["label"].isin(method_order), "category"] == "fusion").sum()
    # n_fusion counts rows across experiments; derive unique fusion method count instead
    n_fusion_unique = df[df["category"] == "fusion"]["label"].nunique()
    if 0 < n_fusion_unique < len(method_order):
        ax.axhline(n_fusion_unique - 0.5, color="white", linewidth=1.5)

    ax.set_title("Macro F1 ± 95 % CI — methods × experiments",
                 fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("macro F1", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"[green]Heatmap →[/green] {out_path}")


def main():
    results_path = RESULTS_DIR / "all_results.csv"
    if not results_path.exists():
        console.log(f"[red]Not found:[/red] {results_path}. Run `python -m clues.reporting.merge_results` first.")
        return

    df = pd.read_csv(results_path)
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_bars(df, plots_dir / "all_results_bars.png")
    plot_heatmap(df, plots_dir / "all_results_heatmap.png")

    console.log(f"[bold green]Saved plots to {plots_dir}[/bold green]")


if __name__ == "__main__":
    main()
