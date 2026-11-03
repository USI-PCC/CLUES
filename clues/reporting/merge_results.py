"""
Merge all per-modality and fusion results into a single summary CSV.

Scans:
  - data/results/{modality}/{exp_dir}/{model}_predictions.csv
    for modality in {empatica, ocosense, eyetracker, rgb}
    for exp_dir in {loso_2class, loso_3class, logo_3class}
    for model   in {baseline, lgbm, rf, xgb}
  - rebuttal/fusion/{method}_{experiment}.csv
    for fusion method outputs

For each file, metrics are computed **per fold** — per subject for LOSO
experiments, per task group for LOGO — and then aggregated as
mean ± 95% CI (t-distribution). This gives honest uncertainty estimates
driven by cross-subject / cross-fold variation, rather than reporting a
single pooled-prediction number.

AUC is only averaged over folds that contain at least two classes in the
held-out set; folds with a single class receive NaN and are excluded.

Per-modality files are loaded via ``data_loaders.load_modality_predictions``
which normalises labels to the LABEL_MAP convention. Fusion files are read
directly — they already use LABEL_MAP encoding.

Usage: python -m clues.reporting.merge_results
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from lib.config import RESULTS_DIR, REBUTTAL_DIR
from clues.data_loaders import load_modality_predictions, EXP_DIR_MAP

console = Console()

MODALITIES = ["empatica", "ocosense", "eyetracker", "rgb"]
MODELS = ["baseline", "lgbm", "rf", "xgb"]
EXPERIMENTS = list(EXP_DIR_MAP.keys())  # loso_two_class, loso_three_class, logo_three_class

# Thermal results are stored in a legacy layout that differs from the other
# modalities (per-subject / per-task folders, all models in one CSV, no
# explicit baseline). The helpers below aggregate thermal into the same
# per-fold schema expected by ``compute_metrics`` so it can be merged with
# the other modalities in ``all_results.csv``.
THERMAL_EXP_DIR = {
    "loso_two_class":   "loso_2class",
    "loso_three_class": "loso_3class",
    "logo_three_class": "logo_3class",
}
# our_name -> thermal column suffix
THERMAL_MODEL_MAP = {"rf": "rf", "xgb": "xgb", "lgbm": "lgb"}
# probability-column order matches LABEL_MAP_*CLASS in config.py
THERMAL_PROB_SUFFIXES_2 = ["load", "rest"]                    # class 0, 1
THERMAL_PROB_SUFFIXES_3 = ["high_load", "low_load", "rest"]   # class 0, 1, 2


def _num_classes(experiment: str) -> int:
    return 3 if "three" in experiment else 2


def _ci95_halfwidth(values: np.ndarray) -> float:
    """95 % confidence interval half-width from a 1-D sample using Student-t."""
    clean = values[~np.isnan(values)]
    n = len(clean)
    if n < 2:
        return float("nan")
    se = float(clean.std(ddof=1) / np.sqrt(n))
    t_crit = float(student_t.ppf(0.975, n - 1))
    return t_crit * se


def _fold_metrics(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  proba: np.ndarray | None,
                  num_classes: int):
    """Per-fold (acc, macro_f1, auc). AUC is NaN if fold has <2 classes."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    auc = float("nan")
    if proba is not None and len(np.unique(y_true)) >= 2:
        try:
            if num_classes == 2:
                auc = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                # OvR multi-class AUC needs y_true to contain all training
                # classes; scikit-learn complains otherwise. Fall back to
                # per-class AUCs, averaged over the classes that are present.
                present = sorted(np.unique(y_true).astype(int).tolist())
                per_class = []
                for cls in present:
                    y_bin = (y_true == cls).astype(int)
                    if len(np.unique(y_bin)) < 2:
                        continue
                    per_class.append(
                        roc_auc_score(y_bin, proba[:, cls])
                    )
                auc = float(np.mean(per_class)) if per_class else float("nan")
        except (ValueError, IndexError):
            auc = float("nan")

    return acc, f1, auc


def compute_metrics(df: pd.DataFrame,
                    num_classes: int,
                    prob_cols: list,
                    group_col: str):
    """Compute per-fold metrics then aggregate as mean and 95 % CI.

    Returns a dict with:
        n_samples, n_folds,
        accuracy, accuracy_ci95,
        macro_f1, macro_f1_ci95,
        auc, auc_ci95, auc_n_folds (folds contributing to AUC mean).

    Expects ``df`` to contain y_true (int), y_pred (int), optional prob
    columns in LABEL_MAP class order, and ``group_col`` identifying the
    fold each row belongs to.
    """
    valid = df["y_pred"] != -1
    work = df.loc[valid].copy()
    n_samples = int(valid.sum())

    empty_result = dict(
        n_samples=0, n_folds=0,
        accuracy=float("nan"), accuracy_ci95=float("nan"),
        macro_f1=float("nan"), macro_f1_ci95=float("nan"),
        auc=float("nan"), auc_ci95=float("nan"), auc_n_folds=0,
    )
    if n_samples == 0 or group_col not in work.columns:
        return empty_result

    has_probs = bool(prob_cols) and all(c in work.columns for c in prob_cols)
    accs, f1s, aucs = [], [], []

    for _, fold_df in work.groupby(group_col, sort=False):
        y_true = fold_df["y_true"].astype(int).values
        y_pred = fold_df["y_pred"].astype(int).values
        proba = fold_df[prob_cols].values if has_probs else None
        acc, f1, auc = _fold_metrics(y_true, y_pred, proba, num_classes)
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

    accs_arr = np.array(accs, dtype=float)
    f1s_arr = np.array(f1s, dtype=float)
    aucs_arr = np.array(aucs, dtype=float)
    auc_valid = aucs_arr[~np.isnan(aucs_arr)]

    return dict(
        n_samples=n_samples,
        n_folds=int(len(accs)),
        accuracy=float(accs_arr.mean()),
        accuracy_ci95=_ci95_halfwidth(accs_arr),
        macro_f1=float(f1s_arr.mean()),
        macro_f1_ci95=_ci95_halfwidth(f1s_arr),
        auc=float(auc_valid.mean()) if len(auc_valid) else float("nan"),
        auc_ci95=_ci95_halfwidth(aucs_arr),
        auc_n_folds=int(len(auc_valid)),
    )


def _group_col_for_experiment(experiment: str, df: pd.DataFrame) -> str:
    """LOGO → fold_id if available, else subject_id. LOSO → subject_id."""
    if "logo" in experiment and "fold_id" in df.columns:
        return "fold_id"
    return "subject_id"


def scan_modality_results() -> pd.DataFrame:
    """Compute metrics for every (modality, experiment, model) combination."""
    rows = []
    for experiment in EXPERIMENTS:
        n_cls = _num_classes(experiment)
        prob_cols = [f"prob_class_{i}" for i in range(n_cls)]

        for modality in MODALITIES:
            for model in MODELS:
                path = RESULTS_DIR / modality / EXP_DIR_MAP[experiment] / f"{model}_predictions.csv"
                if not path.exists():
                    continue
                try:
                    df = load_modality_predictions(modality, experiment, model=model)
                except Exception as exc:
                    console.log(f"[red]Failed[/red] {modality}/{experiment}/{model}: {exc}")
                    continue
                if df.empty:
                    continue
                group_col = _group_col_for_experiment(experiment, df)
                metrics = compute_metrics(df, n_cls, prob_cols, group_col)
                rows.append({
                    "category": "per_modality",
                    "method": model,
                    "modality": modality,
                    "experiment": experiment,
                    **metrics,
                })
                console.log(
                    f"  {modality:11s} / {experiment:17s} / {model:8s}  "
                    f"acc={metrics['accuracy']:.4f}±{metrics['accuracy_ci95']:.4f}  "
                    f"f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_ci95']:.4f}  "
                    f"auc={metrics['auc']:.4f}±{metrics['auc_ci95']:.4f}  "
                    f"n={metrics['n_samples']}  folds={metrics['n_folds']}"
                )
    return pd.DataFrame(rows)


def _load_thermal_experiment(experiment: str) -> pd.DataFrame:
    """Concatenate all thermal test-instance CSVs for an experiment.

    Adds ``subject_id`` (from the per-subject folder name) and, for LOGO,
    ``fold_id`` and ``task`` (from the task folder name).
    """
    thermal_base = RESULTS_DIR / "thermal" / THERMAL_EXP_DIR[experiment]
    if not thermal_base.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if "logo" in experiment:
        # LOGO layout: {task}/test_instances_{task}.csv — one pooled file per
        # task; no per-subject subfolders and no subject_id column. Group by
        # fold_id = task name, which matches how LOGO metrics are aggregated
        # for the other modalities.
        for task_dir in sorted(p for p in thermal_base.iterdir() if p.is_dir()):
            csvs = list(task_dir.glob("test_instances_*.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            df["fold_id"] = task_dir.name
            df["task"] = task_dir.name
            frames.append(df)
    else:
        by_subj = thermal_base / "predictions_by_subject"
        if not by_subj.exists():
            return pd.DataFrame()
        for subj_dir in sorted(p for p in by_subj.iterdir() if p.is_dir()):
            csvs = list(subj_dir.glob("test_instances_*.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            df["subject_id"] = subj_dir.name
            frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scan_thermal_results() -> pd.DataFrame:
    """Compute metrics for thermal across all (experiment, model) combinations.

    Thermal stores every model's predictions in the same CSV so we load the
    experiment data once and project into the standard schema per model.
    The baseline row is synthesised as a global-mode predictor.
    """
    rows = []
    for experiment in EXPERIMENTS:
        combined = _load_thermal_experiment(experiment)
        if combined.empty:
            continue

        n_cls = _num_classes(experiment)
        prob_suffixes = THERMAL_PROB_SUFFIXES_2 if n_cls == 2 else THERMAL_PROB_SUFFIXES_3
        group_col = "fold_id" if "logo" in experiment else "subject_id"

        # ML models: rf, xgb, lgbm (mapped from `lgb` in thermal files)
        for our_name, suffix in THERMAL_MODEL_MAP.items():
            y_pred_col = f"y_pred_{suffix}"
            if y_pred_col not in combined.columns:
                continue
            work = combined[[group_col, "y_true", y_pred_col]].copy()
            work = work.rename(columns={y_pred_col: "y_pred"})
            for i, class_suffix in enumerate(prob_suffixes):
                proba_col = f"proba_{suffix}_{class_suffix}"
                if proba_col in combined.columns:
                    work[f"prob_class_{i}"] = combined[proba_col].values
            prob_cols = [f"prob_class_{i}" for i in range(n_cls)]
            metrics = compute_metrics(work, n_cls, prob_cols, group_col)
            rows.append({
                "category": "per_modality", "method": our_name,
                "modality": "thermal", "experiment": experiment, **metrics,
            })
            console.log(
                f"  {'thermal':11s} / {experiment:17s} / {our_name:8s}  "
                f"acc={metrics['accuracy']:.4f}±{metrics['accuracy_ci95']:.4f}  "
                f"f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_ci95']:.4f}  "
                f"auc={metrics['auc']:.4f}±{metrics['auc_ci95']:.4f}  "
                f"n={metrics['n_samples']}  folds={metrics['n_folds']}"
            )

        # Baseline: global-mode majority-class predictor (same convention as the
        # other modalities' baseline_predictions.csv after averaging per fold).
        majority = int(combined["y_true"].mode().iloc[0])
        work = combined[[group_col, "y_true"]].copy()
        work["y_pred"] = majority
        for i in range(n_cls):
            work[f"prob_class_{i}"] = 1.0 if i == majority else 0.0
        prob_cols = [f"prob_class_{i}" for i in range(n_cls)]
        metrics = compute_metrics(work, n_cls, prob_cols, group_col)
        rows.append({
            "category": "per_modality", "method": "baseline",
            "modality": "thermal", "experiment": experiment, **metrics,
        })
        console.log(
            f"  {'thermal':11s} / {experiment:17s} / {'baseline':8s}  "
            f"acc={metrics['accuracy']:.4f}±{metrics['accuracy_ci95']:.4f}  "
            f"f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_ci95']:.4f}  "
            f"auc={metrics['auc']:.4f}±{metrics['auc_ci95']:.4f}  "
            f"n={metrics['n_samples']}  folds={metrics['n_folds']}"
        )

    return pd.DataFrame(rows)


def scan_fusion_results() -> pd.DataFrame:
    """Compute metrics for every fusion CSV in rebuttal/fusion/."""
    fusion_dir = REBUTTAL_DIR / "fusion"
    if not fusion_dir.exists():
        console.log(f"[yellow]No fusion directory at {fusion_dir} — skipping.[/yellow]")
        return pd.DataFrame()

    rows = []
    for path in sorted(fusion_dir.glob("*.csv")):
        name = path.stem
        matched_exp = None
        for exp in EXPERIMENTS:
            if name.endswith(f"_{exp}"):
                matched_exp = exp
                method = name[: -len(f"_{exp}")]
                break
        if matched_exp is None:
            console.log(f"[yellow]Skipping unparseable fusion file:[/yellow] {path.name}")
            continue

        df = pd.read_csv(path)
        # Normalize column names to match compute_metrics expectations
        df = df.rename(columns={"fused_prediction": "y_pred"})
        n_cls = _num_classes(matched_exp)
        prob_cols = [f"fused_prob_{i}" for i in range(n_cls)]

        if "y_true" not in df.columns or "y_pred" not in df.columns:
            console.log(f"[yellow]Missing y_true/y_pred in[/yellow] {path.name}")
            continue

        group_col = _group_col_for_experiment(matched_exp, df)
        metrics = compute_metrics(df, n_cls, prob_cols, group_col)
        rows.append({
            "category": "fusion",
            "method": method,
            "modality": "fusion",
            "experiment": matched_exp,
            **metrics,
        })
        console.log(
            f"  [fusion] {method:22s} / {matched_exp:17s}  "
            f"acc={metrics['accuracy']:.4f}±{metrics['accuracy_ci95']:.4f}  "
            f"f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_ci95']:.4f}  "
            f"auc={metrics['auc']:.4f}±{metrics['auc_ci95']:.4f}  "
            f"n={metrics['n_samples']}  folds={metrics['n_folds']}"
        )
    return pd.DataFrame(rows)


def scan_fusion_subsets_results() -> pd.DataFrame:
    """Compute metrics for every fusion CSV under rebuttal/fusion/subsets/.

    Expected layout: rebuttal/fusion/subsets/<subset_tag>/<method>_<experiment>.csv
    where <subset_tag> is underscore-joined sorted modality names.
    """
    subsets_dir = REBUTTAL_DIR / "fusion" / "subsets"
    if not subsets_dir.exists():
        console.log(f"[yellow]No fusion subsets directory at {subsets_dir} — skipping.[/yellow]")
        return pd.DataFrame()

    rows = []
    for subset_dir in sorted(p for p in subsets_dir.iterdir() if p.is_dir()):
        subset_tag = subset_dir.name
        modalities = subset_tag.split("__")
        for path in sorted(subset_dir.glob("*.csv")):
            name = path.stem
            matched_exp = None
            for exp in EXPERIMENTS:
                if name.endswith(f"_{exp}"):
                    matched_exp = exp
                    method = name[: -len(f"_{exp}")]
                    break
            if matched_exp is None:
                console.log(f"[yellow]Skipping unparseable subset file:[/yellow] {path}")
                continue

            df = pd.read_csv(path).rename(columns={"fused_prediction": "y_pred"})
            n_cls = _num_classes(matched_exp)
            prob_cols = [f"fused_prob_{i}" for i in range(n_cls)]
            if "y_true" not in df.columns or "y_pred" not in df.columns:
                console.log(f"[yellow]Missing y_true/y_pred in[/yellow] {path}")
                continue

            group_col = _group_col_for_experiment(matched_exp, df)
            metrics = compute_metrics(df, n_cls, prob_cols, group_col)
            rows.append({
                "category": "fusion_subset",
                "method": method,
                "modality": "fusion",
                "experiment": matched_exp,
                "modalities": ",".join(modalities),
                "n_modalities": len(modalities),
                "subset_tag": subset_tag,
                **metrics,
            })
            console.log(
                f"  [subset] {method:22s} / {matched_exp:17s} / {subset_tag:40s}  "
                f"f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_ci95']:.4f}  "
                f"auc={metrics['auc']:.4f}±{metrics['auc_ci95']:.4f}  "
                f"n={metrics['n_samples']}"
            )
    return pd.DataFrame(rows)


def main():
    console.rule("[bold cyan]Per-modality results[/bold cyan]")
    mod_df = scan_modality_results()

    console.rule("[bold cyan]Thermal (legacy layout)[/bold cyan]")
    thermal_df = scan_thermal_results()

    console.rule("[bold cyan]Fusion results[/bold cyan]")
    fus_df = scan_fusion_results()

    console.rule("[bold cyan]Fusion subset ablation[/bold cyan]")
    subset_df = scan_fusion_subsets_results()

    merged = pd.concat([mod_df, thermal_df, fus_df], ignore_index=True)

    # Order for readability: per_modality first, then fusion; by experiment, method, modality
    category_order = {"per_modality": 0, "fusion": 1}
    merged["_cat_ord"] = merged["category"].map(category_order).fillna(9)
    merged = merged.sort_values(
        ["_cat_ord", "experiment", "modality", "method"]
    ).drop(columns="_cat_ord").reset_index(drop=True)

    # Round floats for readability (CSV stores full precision — this is for display only)
    display = merged.copy()
    float_cols = ["accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95", "auc", "auc_ci95"]
    for col in float_cols:
        if col in display.columns:
            display[col] = display[col].round(4)

    out_path = RESULTS_DIR / "all_results.csv"
    merged.to_csv(out_path, index=False)

    console.rule("[bold green]Summary[/bold green]")
    console.print(display.to_string(index=False))
    console.log(f"\nWrote {len(merged)} rows to {out_path}")

    if not subset_df.empty:
        subset_out = RESULTS_DIR / "subsets_results.csv"
        subset_df_sorted = subset_df.sort_values(
            ["experiment", "method", "n_modalities", "modalities"]
        ).reset_index(drop=True)
        subset_df_sorted.to_csv(subset_out, index=False)
        console.log(f"Wrote {len(subset_df_sorted)} rows to {subset_out}")


if __name__ == "__main__":
    main()
