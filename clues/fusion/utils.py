"""Shared utilities for fusion methods."""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from lib.config import MODALITIES


def align_3s_to_30s(df_3s, start_time, end_time, subject_id, num_classes,
                    method="mean_probability"):
    """Aggregate 3s-window predictions that fall within a 30s reference window.

    Filters rows where subject_id matches and start_time >= start_time and
    start_time < end_time (the 30s window boundaries).

    Args:
        df_3s: DataFrame with 3s-window predictions for a modality.
        start_time: Start of the 30s reference window (inclusive).
        end_time: End of the 30s reference window (exclusive).
        subject_id: Subject identifier to filter on.
        num_classes: Number of prediction classes (2 or 3).
        method: Aggregation method — "mean_probability" (default) uses mean of
            probabilities with argmax for y_pred; "majority_vote" uses mode of
            y_pred with mean of probabilities.

    Returns:
        Tuple (y_pred, prob_dict) where prob_dict maps class index to mean
        probability. Returns (None, None) if no matching rows are found.
    """
    mask = (
        (df_3s["subject_id"] == subject_id) &
        (df_3s["start_time"] >= start_time) &
        (df_3s["start_time"] < end_time)
    )
    subset = df_3s[mask]

    if subset.empty:
        return None, None

    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    mean_probs = {}
    for i in range(num_classes):
        col = f"prob_class_{i}"
        if col in subset.columns:
            mean_probs[i] = float(subset[col].mean())
        else:
            mean_probs[i] = 0.0

    if method == "majority_vote":
        counts = Counter(subset["y_pred"].tolist())
        y_pred = counts.most_common(1)[0][0]
    else:  # mean_probability (default)
        y_pred = int(np.argmax([mean_probs[i] for i in range(num_classes)]))

    return y_pred, mean_probs


def compute_confidence_margin(probs):
    """Compute margin between top 2 probabilities.

    Args:
        probs: Dict mapping class index -> probability, or None.

    Returns:
        Float margin (top_prob - second_prob). Returns 0.0 if probs is None
        or has fewer than 2 classes.
    """
    if probs is None:
        return 0.0
    values = sorted(probs.values(), reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def compute_uncertainty(probs, num_classes):
    """Normalized entropy of a probability distribution in [0, 1].

    0 = perfectly confident (one class has probability 1).
    1 = maximally uncertain (uniform distribution across classes).
    Accepts a dict {class_idx: prob} or a sequence of probabilities.
    """
    if probs is None or num_classes <= 1:
        return 0.0

    if isinstance(probs, dict):
        values = np.array([probs.get(i, 0.0) for i in range(num_classes)], dtype=float)
    else:
        values = np.asarray(probs, dtype=float)

    total = values.sum()
    if total <= 0:
        return 0.0
    values = values / total
    values = np.clip(values, 1e-12, 1.0)
    entropy = -np.sum(values * np.log(values))
    max_entropy = np.log(num_classes)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_fold_modality_f1(aligned, train_mask, modalities=None):
    """Macro-F1 of each modality on the training rows of one CV fold.

    Used by weighted / confidence-based fusion to derive modality weights
    that are strictly nested-CV safe: each held-out fold sees weights
    estimated only from rows *outside* the fold.

    Args:
        aligned: DataFrame from ``align_all_modalities`` with y_true and
            per-modality y_pred_{mod} columns. Rows where y_pred_{mod} == -1
            are treated as the modality being unavailable for that row and
            excluded from the F1 computation.
        train_mask: Boolean array selecting rows to use for training
            (typically the complement of the held-out subject / fold).
        modalities: Optional iterable of modality names to score.
            Defaults to the global ``MODALITIES``.

    Returns:
        Dict mapping modality name -> macro-F1 in [0, 1]. Modalities with
        no valid rows in the training fold receive 0.0.
    """
    mods = list(modalities) if modalities is not None else list(MODALITIES)
    train = aligned.loc[train_mask]
    weights = {}
    for mod in mods:
        y_pred_col = f"y_pred_{mod}"
        if y_pred_col not in train.columns:
            weights[mod] = 0.0
            continue
        valid = train[train[y_pred_col] != -1]
        if len(valid) == 0:
            weights[mod] = 0.0
            continue
        try:
            weights[mod] = float(f1_score(
                valid["y_true"], valid[y_pred_col],
                average="macro", zero_division=0,
            ))
        except Exception:
            weights[mod] = 0.0
    return weights


def get_cv_groups(aligned, experiment):
    """Return the CV grouping array (fold_id for LOGO, subject_id otherwise)."""
    if experiment == "logo_three_class" and "fold_id" in aligned.columns:
        return aligned["fold_id"].values
    return aligned["subject_id"].values


def compute_confidence_entropy(probs, num_classes):
    """Compute confidence as 1 minus normalized entropy.

    Args:
        probs: Dict mapping class index -> probability, or None.
        num_classes: Number of classes (used for normalization).

    Returns:
        Float in [0, 1] where 1 means perfectly confident. Returns 0.0 if
        probs is None or num_classes <= 1.
    """
    if probs is None or num_classes <= 1:
        return 0.0

    values = np.array([probs.get(i, 0.0) for i in range(num_classes)],
                      dtype=float)
    # Clip to avoid log(0)
    values = np.clip(values, 1e-12, 1.0)
    entropy = -np.sum(values * np.log(values))
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(1.0 - normalized_entropy)


def compute_metrics(y_true, y_pred, y_proba):
    """Compute accuracy, macro F1, and AUC, excluding rows with y_pred == -1.

    Args:
        y_true: Array-like of ground-truth labels.
        y_pred: Array-like of predicted labels (-1 indicates missing/invalid).
        y_proba: Array-like of shape (n_samples, n_classes) with probabilities,
            or None if not available.

    Returns:
        Tuple (accuracy, macro_f1, auc, n_valid) where n_valid is the number
        of samples after filtering. AUC is None if y_proba is None or
        computation fails.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    valid_mask = y_pred != -1
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return 0.0, 0.0, None, 0

    acc = float(accuracy_score(y_true_v, y_pred_v))
    f1 = float(f1_score(y_true_v, y_pred_v, average="macro", zero_division=0))

    auc = None
    if y_proba is not None:
        y_proba_arr = np.array(y_proba)
        y_proba_v = y_proba_arr[valid_mask]
        try:
            num_classes = y_proba_v.shape[1] if y_proba_v.ndim == 2 else 2
            if num_classes == 2:
                auc = float(roc_auc_score(y_true_v, y_proba_v[:, 1]))
            else:
                auc = float(roc_auc_score(
                    y_true_v, y_proba_v, multi_class="ovr", average="macro"
                ))
        except (ValueError, IndexError):
            auc = None

    return acc, f1, auc, n_valid


def align_all_modalities(predictions, experiment, modality_subset=None):
    """Align all modalities to 30s windows using empatica as reference.

    For each empatica window:
    - empatica: used directly as the reference.
    - ocosense: matched by time proximity (|start_time diff| < 5.0 seconds).
    - rgb, thermal, eyetracker: aggregated via align_3s_to_30s.

    Missing modality predictions are filled with -1 for y_pred and 0.0 for
    probabilities.

    Args:
        predictions: Dict mapping modality name -> DataFrame with at minimum
            subject_id, start_time, end_time, y_true, y_pred, prob_class_0,
            prob_class_1 (and prob_class_2 for three-class experiments).
            LOGO experiments should also include fold_id and task columns.
        experiment: One of "loso_two_class", "loso_three_class",
            "logo_three_class".
        modality_subset: Optional list of modality names to include. Defaults
            to all of MODALITIES.

    Returns:
        DataFrame with one row per aligned 30s window. Columns include:
        subject_id, start_time, end_time, y_true, and optionally fold_id and
        task (for LOGO). Per-modality columns: y_pred_{mod} and
        prob_{mod}_{i} for each class index i.
    """
    num_classes = 2 if experiment == "loso_two_class" else 3
    active_modalities = modality_subset if modality_subset is not None else MODALITIES
    short_window_modalities = {"rgb", "thermal", "eyetracker"}
    is_logo = experiment == "logo_three_class"

    if "empatica" not in predictions or predictions["empatica"] is None:
        raise ValueError("empatica predictions are required as the reference modality")

    ref_df = predictions["empatica"]
    rows = []

    for _, ref_row in ref_df.iterrows():
        sid = ref_row["subject_id"]
        t_start = ref_row["start_time"]
        t_end = ref_row["end_time"]
        y_true = ref_row["y_true"]

        record = {
            "subject_id": sid,
            "start_time": t_start,
            "end_time": t_end,
            "y_true": y_true,
        }
        if is_logo:
            record["fold_id"] = ref_row.get("fold_id", np.nan)
            record["task"] = ref_row.get("task", np.nan)

        for mod in active_modalities:
            prob_keys = [f"prob_{mod}_{i}" for i in range(num_classes)]

            if mod not in predictions or predictions[mod] is None:
                record[f"y_pred_{mod}"] = -1
                for k in prob_keys:
                    record[k] = 0.0
                continue

            df_mod = predictions[mod]

            if mod == "empatica":
                record[f"y_pred_{mod}"] = int(ref_row["y_pred"])
                for i in range(num_classes):
                    col = f"prob_class_{i}"
                    record[f"prob_{mod}_{i}"] = float(ref_row[col]) if col in ref_row else 0.0

            elif mod == "ocosense":
                subj_mask = df_mod["subject_id"] == sid
                candidates = df_mod[subj_mask]
                if candidates.empty:
                    record[f"y_pred_{mod}"] = -1
                    for k in prob_keys:
                        record[k] = 0.0
                else:
                    diffs = (candidates["start_time"] - t_start).abs()
                    best_idx = diffs.idxmin()
                    if diffs[best_idx] < 5.0:
                        best = candidates.loc[best_idx]
                        record[f"y_pred_{mod}"] = int(best["y_pred"])
                        for i in range(num_classes):
                            col = f"prob_class_{i}"
                            record[f"prob_{mod}_{i}"] = float(best[col]) if col in best.index else 0.0
                    else:
                        record[f"y_pred_{mod}"] = -1
                        for k in prob_keys:
                            record[k] = 0.0

            elif mod in short_window_modalities:
                y_pred_agg, prob_dict = align_3s_to_30s(
                    df_mod, t_start, t_end, sid, num_classes
                )
                if y_pred_agg is None:
                    record[f"y_pred_{mod}"] = -1
                    for k in prob_keys:
                        record[k] = 0.0
                else:
                    record[f"y_pred_{mod}"] = y_pred_agg
                    for i in range(num_classes):
                        record[f"prob_{mod}_{i}"] = float(prob_dict.get(i, 0.0))

            else:
                record[f"y_pred_{mod}"] = -1
                for k in prob_keys:
                    record[k] = 0.0

        rows.append(record)

    return pd.DataFrame(rows)


def build_stacking_features(prob_dicts, num_classes, modalities=None):
    """Build feature vector for a stacking meta-classifier.

    For each modality, appends:
    - Probability for each class (num_classes values).
    - Availability indicator: 1.0 if modality is present, 0.0 otherwise.
    - Confidence score (margin between top 2 probabilities).

    Args:
        prob_dicts: Dict mapping modality name -> probability dict
            (class_index -> float), or None if modality is unavailable.
        num_classes: Number of prediction classes (2 or 3).
        modalities: Optional iterable of modality names. Defaults to the
            global ``MODALITIES`` order.

    Returns:
        1-D numpy array of floats with length
        len(modalities) * (num_classes + 2).
    """
    mods = list(modalities) if modalities is not None else list(MODALITIES)
    features = []
    for mod in mods:
        probs = prob_dicts.get(mod, None)
        if probs is not None:
            for i in range(num_classes):
                features.append(float(probs.get(i, 0.0)))
            features.append(1.0)  # availability
            features.append(compute_confidence_margin(probs))
        else:
            for _ in range(num_classes):
                features.append(0.0)
            features.append(0.0)  # unavailable
            features.append(0.0)  # no confidence
    return np.array(features, dtype=float)
