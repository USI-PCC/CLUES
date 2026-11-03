"""Fusion runner: loads predictions, dispatches to methods, writes results."""

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from rich.console import Console

from lib.config import MODALITIES, EXPERIMENTS, REBUTTAL_DIR
from clues.data_loaders import load_modality_predictions
from clues.fusion.utils import compute_metrics
from clues.fusion.majority_voting import MajorityVotingFusion
from clues.fusion.weighted_probability import WeightedProbabilityFusion

try:
    from clues.fusion.confidence_based import ConfidenceBasedFusion
    _HAS_CONFIDENCE = True
except ImportError:
    _HAS_CONFIDENCE = False

try:
    from clues.fusion.dst_voting import DSTVotingFusion
    _HAS_DST = True
except ImportError:
    _HAS_DST = False

try:
    from clues.fusion.stacking import StackingFusion
    _HAS_STACKING = True
except ImportError:
    _HAS_STACKING = False

try:
    from clues.fusion.early_fusion import (
        EarlyFusion,
        EarlyFusionPCA,
        EarlyFusionICA,
        EarlyFusionSelectK,
    )
    _HAS_EARLY = True
except ImportError:
    _HAS_EARLY = False

console = Console()

# Build the list of available methods, skipping those not yet implemented.
AVAILABLE_METHODS = [
    MajorityVotingFusion(),
    WeightedProbabilityFusion(),
]
if _HAS_CONFIDENCE:
    AVAILABLE_METHODS.append(ConfidenceBasedFusion(alpha=1.5))
if _HAS_DST:
    AVAILABLE_METHODS.append(DSTVotingFusion())
if _HAS_STACKING:
    AVAILABLE_METHODS.append(StackingFusion())
if _HAS_EARLY:
    AVAILABLE_METHODS.append(EarlyFusion())
    AVAILABLE_METHODS.append(EarlyFusionPCA())
    AVAILABLE_METHODS.append(EarlyFusionICA())
    AVAILABLE_METHODS.append(EarlyFusionSelectK())

METHOD_MAP = {m.name: m for m in AVAILABLE_METHODS}

# Methods to use in the subset sweep — top-2 per family (U1 design).
SUBSET_SWEEP_METHOD_NAMES = [
    "stacking",
    "weighted_probability",
    "early_fusion",
    "early_fusion_selectk",
]


def _subset_tag(modalities):
    """Underscore-joined, alphabetically-sorted modality names."""
    return "__".join(sorted(modalities))


def enumerate_subsets(modalities=None, min_size=2):
    """Return all subsets of the given modalities of size >= min_size.

    Default: all 11 non-trivial subsets of the 4 global modalities.
    """
    pool = list(modalities) if modalities is not None else list(MODALITIES)
    subsets = []
    for k in range(min_size, len(pool) + 1):
        for combo in combinations(pool, k):
            subsets.append(list(combo))
    return subsets


def run_fusion_subsets(methods=None, experiments=None, subsets=None):
    """Run the subset ablation sweep.

    For each (subset, method, experiment), compute fusion and save to
    ``rebuttal/fusion/subsets/<subset_tag>/<method>_<experiment>.csv``.
    Every subset is required to contain empatica (the reference-window
    modality); subsets that don't are silently extended with empatica,
    then duplicates produced by that injection are removed.
    """
    if methods is None:
        methods = [METHOD_MAP[n] for n in SUBSET_SWEEP_METHOD_NAMES if n in METHOD_MAP]
    if experiments is None:
        experiments = EXPERIMENTS
    if subsets is None:
        subsets = enumerate_subsets()

    # Deduplicate after injecting empatica.
    seen = set()
    unique_subsets = []
    for s in subsets:
        canonical = tuple(sorted(set(s) | {"empatica"}))
        if canonical not in seen:
            seen.add(canonical)
            unique_subsets.append(list(canonical))
    subsets = unique_subsets

    base_out = REBUTTAL_DIR / "fusion" / "subsets"
    base_out.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:
        console.rule(f"[bold cyan]Experiment: {experiment}[/bold cyan]")

        predictions = {}
        for modality in MODALITIES:
            try:
                df = load_modality_predictions(modality, experiment)
                if df is not None and not df.empty:
                    predictions[modality] = df
            except Exception as exc:
                console.print(f"  [red]Skipped[/red] {modality}: {exc}")

        if "empatica" not in predictions:
            console.print("[red]Empatica predictions missing — cannot run subsets.[/red]")
            continue

        for subset in subsets:
            tag = _subset_tag(subset)
            out_dir = base_out / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for method in methods:
                console.print(f"\n[bold]{method.name}[/bold] on subset {tag}")
                try:
                    fused_df = method.fuse(
                        predictions, experiment,
                        modality_subset=subset,
                    )
                except Exception as exc:
                    console.print(
                        f"  [red]Error during {method.name} on {tag} / {experiment}:[/red] {exc}"
                    )
                    console.print_exception()
                    continue

                y_true = fused_df["y_true"].values
                y_pred = fused_df["fused_prediction"].values
                num_cls = _num_classes(experiment)
                prob_cols = [f"fused_prob_{i}" for i in range(num_cls)]
                y_proba = fused_df[prob_cols].values if all(
                    c in fused_df.columns for c in prob_cols
                ) else None

                acc, f1, auc, n_valid = compute_metrics(y_true, y_pred, y_proba)
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                console.print(
                    f"  Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}  |  "
                    f"AUC: {auc_str}  |  n={n_valid}"
                )

                out_path = out_dir / f"{method.name}_{experiment}.csv"
                fused_df.to_csv(out_path, index=False)
                console.print(f"  Saved -> {out_path}")

    console.print("\n[bold green]Subset ablation complete.[/bold green]")


def _num_classes(experiment: str) -> int:
    return 2 if experiment == "loso_two_class" else 3


def run_fusion(methods=None, experiments=None):
    """Load predictions, run all fusion methods, and save results.

    For each experiment, loads available modality predictions into a dict,
    then calls each method's fuse() to produce fused predictions.  Metrics
    are printed to the console and results are saved as CSVs under
    REBUTTAL_DIR / "fusion".

    Args:
        methods: List of FusionMethod instances to run.  Defaults to
            AVAILABLE_METHODS.
        experiments: List of experiment names to run.  Defaults to
            EXPERIMENTS from config.
    """
    if methods is None:
        methods = AVAILABLE_METHODS
    if experiments is None:
        experiments = EXPERIMENTS

    output_dir = REBUTTAL_DIR / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:
        console.rule(f"[bold cyan]Experiment: {experiment}[/bold cyan]")

        # ------------------------------------------------------------------
        # Load predictions for each modality
        # ------------------------------------------------------------------
        predictions = {}
        for modality in MODALITIES:
            try:
                df = load_modality_predictions(modality, experiment)
                if df is not None and not df.empty:
                    predictions[modality] = df
                    console.print(
                        f"  [green]Loaded[/green] {modality}: "
                        f"{len(df)} rows"
                    )
                else:
                    console.print(
                        f"  [yellow]Empty[/yellow] {modality}: skipping"
                    )
            except Exception as exc:
                console.print(
                    f"  [red]Skipped[/red] {modality}: {exc}"
                )

        if not predictions:
            console.print("[red]No modality data available — skipping experiment.[/red]")
            continue

        # ------------------------------------------------------------------
        # Run each fusion method
        # ------------------------------------------------------------------
        for method in methods:
            console.print(f"\n[bold]{method.name}[/bold]")
            try:
                fused_df = method.fuse(predictions, experiment)
            except Exception as exc:
                console.print(f"  [red]Error during fusion:[/red] {exc}")
                continue

            # Compute metrics
            y_true = fused_df["y_true"].values
            y_pred = fused_df["fused_prediction"].values

            num_cls = _num_classes(experiment)
            prob_cols = [f"fused_prob_{i}" for i in range(num_cls)]
            if all(c in fused_df.columns for c in prob_cols):
                y_proba = fused_df[prob_cols].values
            else:
                y_proba = None

            acc, f1, auc, n_valid = compute_metrics(y_true, y_pred, y_proba)

            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            console.print(
                f"  Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}  |  "
                f"AUC: {auc_str}  |  n={n_valid}"
            )

            # Save results
            out_path = output_dir / f"{method.name}_{experiment}.csv"
            fused_df.to_csv(out_path, index=False)
            console.print(f"  Saved -> {out_path}")

    console.print("\n[bold green]Fusion complete.[/bold green]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run multimodal late-fusion experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m clues.fusion.runner\n"
            "  python -m clues.fusion.runner --methods majority_voting,weighted_probability\n"
            "  python -m clues.fusion.runner --experiments loso_two_class\n"
            "  python -m clues.fusion.runner --methods majority_voting --experiments loso_three_class\n"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated list of fusion method names to run.  "
            f"Available: {', '.join(METHOD_MAP.keys())}.  "
            "Defaults to all available methods."
        ),
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help=(
            "Comma-separated list of experiments to run.  "
            f"Available: {', '.join(EXPERIMENTS)}.  "
            "Defaults to all experiments."
        ),
    )
    parser.add_argument(
        "--subsets",
        action="store_true",
        help=(
            "Run the subset ablation sweep: for each of the 7 unique modality "
            "subsets containing empatica, run the top-2 methods per fusion "
            "family. Output goes under rebuttal/fusion/subsets/<tag>/. "
            "Takes precedence over --modalities."
        ),
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help=(
            "Comma-separated list of modality names (e.g. ocosense,empatica) "
            "for a single-subset run. Output goes under "
            "rebuttal/fusion/subsets/<tag>/ just like --subsets. Ignored if "
            "--subsets is set."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve method names
    if args.methods is not None:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in requested if m not in METHOD_MAP]
        if unknown:
            console.print(
                f"[red]Unknown method(s): {unknown}.  "
                f"Available: {list(METHOD_MAP.keys())}[/red]"
            )
            sys.exit(1)
        selected_methods = [METHOD_MAP[m] for m in requested]
    else:
        selected_methods = None  # defaults to AVAILABLE_METHODS inside run_fusion

    # Resolve experiment names
    if args.experiments is not None:
        requested_exps = [e.strip() for e in args.experiments.split(",") if e.strip()]
        unknown_exps = [e for e in requested_exps if e not in EXPERIMENTS]
        if unknown_exps:
            console.print(
                f"[red]Unknown experiment(s): {unknown_exps}.  "
                f"Available: {EXPERIMENTS}[/red]"
            )
            sys.exit(1)
        selected_experiments = requested_exps
    else:
        selected_experiments = None  # defaults to EXPERIMENTS inside run_fusion

    if args.subsets:
        run_fusion_subsets(
            methods=selected_methods,
            experiments=selected_experiments,
        )
    elif args.modalities is not None:
        subset = [m.strip() for m in args.modalities.split(",") if m.strip()]
        run_fusion_subsets(
            methods=selected_methods,
            experiments=selected_experiments,
            subsets=[subset],
        )
    else:
        run_fusion(methods=selected_methods, experiments=selected_experiments)
