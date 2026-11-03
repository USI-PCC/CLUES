"""
Generic modality prediction loader for multimodal cognitive load fusion.

Provides a single public entry point `load_modality_predictions` that dispatches
to per-modality private loaders. Each loader reads from
``RESULTS_DIR/{modality}/{exp_dir}/lgbm_predictions.csv`` and normalises the
output to a common schema:

    subject_id, start_time, end_time, y_true, y_pred,
    prob_class_0, prob_class_1, [prob_class_2],
    [fold_id, task]   (LOGO experiments only)

Integer labels follow LABEL_MAP (from config.py):
    3-class: 0=high_load, 1=low_load, 2=rest/no_load
    2-class: 0=load, 1=rest/no_load
"""

import os

import numpy as np
import pandas as pd

from lib.config import RESULTS_DIR, LABEL_MAP_2CLASS, LABEL_MAP_3CLASS

# ---------------------------------------------------------------------------
# Experiment → on-disk directory
# ---------------------------------------------------------------------------

EXP_DIR_MAP = {
    "loso_two_class":   "loso_2class",
    "loso_three_class": "loso_3class",
    "logo_three_class": "logo_3class",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_classes(experiment: str) -> int:
    return 3 if "three" in experiment else 2


def _label_map(experiment: str) -> dict:
    return LABEL_MAP_3CLASS if "three" in experiment else LABEL_MAP_2CLASS


def _is_logo(experiment: str) -> bool:
    return "logo" in experiment


def _map_labels(series: pd.Series, label_map: dict) -> pd.Series:
    """Convert string labels to integers via label_map (case-insensitive)."""
    return series.astype(str).str.lower().map(label_map).fillna(-1).astype(int)


def _prediction_path(modality: str, experiment: str, model: str = "lgbm") -> str:
    return os.path.join(
        str(RESULTS_DIR), modality, EXP_DIR_MAP[experiment], f"{model}_predictions.csv"
    )


# Cache of per-subject empatica first-window times, used to align ocosense
# timestamps (which are Unix and get zero-based by the loader) into empatica's
# recording-relative frame. Loaded once on first access.
_EMPATICA_FIRST_WINDOW: dict | None = None


def _empatica_first_window_offsets() -> dict:
    """Return {subject_id: first empatica window start_time (recording-relative)}.

    Used by ``_load_ocosense`` to shift ocosense predictions onto empatica's
    time axis. Empatica and ocosense share the first task event (``2_back``
    for 50+/52 subjects), so adding empatica's first-window offset to
    ocosense's zero-aligned times puts both modalities on the same origin.
    """
    global _EMPATICA_FIRST_WINDOW
    if _EMPATICA_FIRST_WINDOW is not None:
        return _EMPATICA_FIRST_WINDOW

    path = _prediction_path("empatica", "loso_three_class")
    df = pd.read_csv(path)
    _EMPATICA_FIRST_WINDOW = (
        df.groupby("subject")["time_start"].min().to_dict()
    )
    return _EMPATICA_FIRST_WINDOW


# ---------------------------------------------------------------------------
# Private per-modality loaders
# ---------------------------------------------------------------------------

def _load_ocosense(experiment: str, model: str = "lgbm") -> pd.DataFrame:
    """OCOSense predictions (30 s windows).

    Source columns: subject_id, y_true (int), y_pred (int),
        start_timestamp, end_timestamp, prob_class_0, prob_class_1, [prob_class_2]
        (+ fold_id, task for LOGO).

    modeling.py label convention is the REVERSE of LABEL_MAP:
        3-class: 0=no_load, 1=low_load, 2=high_load  →  swap 0↔2
        2-class: 0=no_load, 1=cog_load              →  swap 0↔1
    Absolute Unix timestamps are normalised per subject to start at 0.
    """
    df = pd.read_csv(_prediction_path("ocosense", experiment, model))
    n_cls = _num_classes(experiment)

    # Remap labels: modeling.py convention → LABEL_MAP convention
    if n_cls == 3:
        swap = {0: 2, 1: 1, 2: 0}
        prob_reorder = ["prob_class_2", "prob_class_1", "prob_class_0"]
    else:
        swap = {0: 1, 1: 0}
        prob_reorder = ["prob_class_1", "prob_class_0"]

    df["y_true"] = df["y_true"].map(swap).astype(int)
    df["y_pred"] = df["y_pred"].map(swap).astype(int)

    out = pd.DataFrame({
        "subject_id": df["subject_id"],
        "start_time": df["start_timestamp"],
        "end_time":   df["end_timestamp"],
        "y_true":     df["y_true"],
        "y_pred":     df["y_pred"],
    })
    for i, src in enumerate(prob_reorder):
        out[f"prob_class_{i}"] = df[src]

    # Absolute Unix timestamps → recording-relative per subject, aligned
    # to empatica's time origin (both modalities share the first task event).
    empatica_offset = _empatica_first_window_offsets()
    for subj in out["subject_id"].unique():
        mask = out["subject_id"] == subj
        t0 = out.loc[mask, "start_time"].min()
        shift = empatica_offset.get(subj, 0.0)
        out.loc[mask, "start_time"] = out.loc[mask, "start_time"] - t0 + shift
        out.loc[mask, "end_time"]   = out.loc[mask, "end_time"]   - t0 + shift

    if _is_logo(experiment):
        out["fold_id"] = df["fold_id"]
        out["task"]    = df["task"]

    return out


def _load_empatica(experiment: str, model: str = "lgbm") -> pd.DataFrame:
    """Empatica predictions (30 s windows).

    Source columns (LOSO): label, subject, y_true (str), y_pred (str),
        time_start, time_end, prob_no_load, prob_low_load, prob_high_load
        (2-class: prob_no_load, prob_load)
    Source columns (LOGO): label, time_start, time_end, fold_id, test_set,
        y_true, y_pred, original_index, prob_no_load, prob_low_load, prob_high_load
        — no subject column. Subject is recovered by joining `original_index`
        against the LOSO predictions (same feature-set ordering).
    """
    df = pd.read_csv(_prediction_path("empatica", experiment, model))
    n_cls = _num_classes(experiment)
    lmap  = _label_map(experiment)

    # Probabilities: map to class_0, class_1, [class_2] per LABEL_MAP convention
    if n_cls == 3:
        prob_map = {
            "prob_class_0": "prob_high_load",
            "prob_class_1": "prob_low_load",
            "prob_class_2": "prob_no_load",
        }
    else:
        prob_map = {
            "prob_class_0": "prob_load",
            "prob_class_1": "prob_no_load",
        }

    # Recover subject for LOGO from LOSO via original_index
    if _is_logo(experiment):
        loso_3class = pd.read_csv(_prediction_path("empatica", "loso_three_class", model))
        subject_lookup = loso_3class["subject"].to_dict()
        df["subject"] = df["original_index"].map(subject_lookup)

    out = pd.DataFrame({
        "subject_id": df["subject"],
        "start_time": df["time_start"],
        "end_time":   df["time_end"],
        "y_true":     _map_labels(df["y_true"], lmap),
        "y_pred":     _map_labels(df["y_pred"], lmap),
    })
    for dst, src in prob_map.items():
        out[dst] = df[src]

    if _is_logo(experiment):
        out["fold_id"] = df["fold_id"]
        out["task"]    = df["label"]

    return out


def _load_rgb(experiment: str, model: str = "lgbm") -> pd.DataFrame:
    """RGB predictions (3 s windows).

    Source columns: subject, time_start, time_end, y_true (str), y_pred (str),
        fold_id, and per-class prob columns.
        3-class: prob_high_load, prob_low_load, prob_no_load (already class 0/1/2 order).
        2-class: prob_load, prob_no_load (already class 0/1 order).
    """
    df = pd.read_csv(_prediction_path("rgb", experiment, model))
    n_cls = _num_classes(experiment)
    lmap  = _label_map(experiment)

    if n_cls == 3:
        prob_cols = ["prob_high_load", "prob_low_load", "prob_no_load"]
    else:
        prob_cols = ["prob_load", "prob_no_load"]

    out = pd.DataFrame({
        "subject_id": df["subject"],
        "start_time": df["time_start"],
        "end_time":   df["time_end"],
        "y_true":     _map_labels(df["y_true"], lmap),
        "y_pred":     _map_labels(df["y_pred"], lmap),
    })
    for i, src in enumerate(prob_cols):
        out[f"prob_class_{i}"] = df[src]

    if _is_logo(experiment):
        out["fold_id"] = df["fold_id"].values

    # RGB predictions can contain duplicates across overlapping windows. For LOGO
    # the same (subject, window) pair legitimately appears in multiple folds (a
    # subject's no-load window often serves as the matched control across
    # several held-out task pairs), so the dedupe key MUST include fold_id to
    # avoid collapsing those distinct per-fold predictions into one.
    dedupe_keys = ["subject_id", "start_time", "end_time"]
    if _is_logo(experiment):
        dedupe_keys.append("fold_id")
    out = out.drop_duplicates(subset=dedupe_keys)

    return out.reset_index(drop=True)


def _load_eyetracker(experiment: str, model: str = "lgbm") -> pd.DataFrame:
    """Eye-tracker predictions (3 s windows).

    Source columns: subject (may carry a ``.csv`` suffix), time_start, time_end,
        y_true (str), y_pred (str), fold_id, and per-class prob columns.
        3-class: prob_high_load, prob_low_load, prob_rest.
        2-class: prob_load, prob_rest.
    """
    df = pd.read_csv(_prediction_path("eyetracker", experiment, model))
    n_cls = _num_classes(experiment)
    lmap  = _label_map(experiment)

    if n_cls == 3:
        prob_cols = ["prob_high_load", "prob_low_load", "prob_rest"]
    else:
        prob_cols = ["prob_load", "prob_rest"]

    out = pd.DataFrame({
        "subject_id": df["subject"].astype(str).str.replace(r"\.csv$", "", regex=True),
        "start_time": df["time_start"],
        "end_time":   df["time_end"],
        "y_true":     _map_labels(df["y_true"], lmap),
        "y_pred":     _map_labels(df["y_pred"], lmap),
    })
    for i, src in enumerate(prob_cols):
        out[f"prob_class_{i}"] = df[src]

    if _is_logo(experiment):
        out["fold_id"] = df["fold_id"]

    return out


def _load_thermal(experiment: str, model: str = "lgbm") -> pd.DataFrame:
    """Thermal predictions are not yet produced in the new results layout.

    Returns an empty DataFrame so fusion can still run without thermal data.
    """
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODALITY_LOADERS = {
    "empatica":    _load_empatica,
    "ocosense":    _load_ocosense,
    "rgb":         _load_rgb,
    "thermal":     _load_thermal,
    "eyetracker":  _load_eyetracker,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_modality_predictions(
    modality: str,
    experiment: str,
    model: str = "lgbm",
    base_dir: str = None,   # kept for backwards compatibility; ignored
) -> pd.DataFrame:
    """Load and normalise predictions for one modality and experiment.

    Parameters
    ----------
    modality:
        One of "empatica", "ocosense", "rgb", "thermal", "eyetracker".
    experiment:
        One of "loso_two_class", "loso_three_class", "logo_three_class".
    base_dir:
        Deprecated, accepted for backwards compatibility. Predictions are
        always read from ``RESULTS_DIR`` defined in config.py.

    Returns
    -------
    pd.DataFrame
        Normalised predictions with columns:
        subject_id, start_time, end_time, y_true, y_pred,
        prob_class_0, prob_class_1, [prob_class_2],
        [fold_id, task] for LOGO experiments.

    Raises
    ------
    ValueError
        If ``modality`` or ``experiment`` is not recognised.
    """
    if modality not in _MODALITY_LOADERS:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Valid options: {sorted(_MODALITY_LOADERS)}"
        )

    if experiment not in EXP_DIR_MAP:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid options: {sorted(EXP_DIR_MAP)}"
        )

    loader = _MODALITY_LOADERS[modality]
    return loader(experiment, model)
