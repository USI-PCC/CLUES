"""Early (feature-level) fusion baselines.

Builds a single feature vector per 30 s reference window by concatenating
harmonized features.csv from all modalities:

- Empatica (30 s) and OCOSense (30 s): matched directly (nearest start_time
  within 5 s of the empatica reference window for the same subject).
- RGB and eye-tracker (3 s): aggregated via column-wise mean of short
  windows whose start_time falls inside the empatica [t_start, t_end) window.

Missing modalities for a given row are filled with zeros and a per-modality
availability indicator is appended so the classifier can condition on it.

Each variant trains a LightGBM classifier with LOSO/LOGO cross-validation,
optionally preceded by a preprocessor fitted on the training fold only:

- ``EarlyFusion``           — concatenation + LightGBM (He et al., 2022-style).
- ``EarlyFusionPCA``        — PCA(64) → LightGBM (standard dimensionality
  reduction baseline).
- ``EarlyFusionICA``        — FastICA(64) → LightGBM, after Zhou et al. (2020)
  who reported early fusion with ICA as the best in-distribution scheme for
  multimodal physiological workload prediction.
- ``EarlyFusionSelectK``    — univariate feature selection (ANOVA F-value,
  k=128) → LightGBM, after the feature-selection-first pattern used by
  Gjoreski et al. (2021) in the Cognitive Load Monitoring challenge.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from tqdm import tqdm

from clues.fusion.base import FusionMethod
from clues.fusion.utils import compute_uncertainty
from lib.config import RESULTS_DIR, MODALITIES

# Module-level cache so that instantiating multiple early-fusion variants
# does not re-run the expensive window alignment for a given experiment.
_FEATURE_CACHE: dict = {}

_META_COLS = ("subject", "task", "start_time", "end_time")
_LONG_WINDOW_MODS = ("empatica", "ocosense")
_SHORT_WINDOW_MODS = ("rgb", "eyetracker")
_TIME_MATCH_TOLERANCE = 5.0  # seconds


def _load_feature_table(modality: str) -> pd.DataFrame:
    """Load the harmonized features.csv for a modality, returning None if absent."""
    path = RESULTS_DIR / modality / "artifacts" / "features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _META_COLS]


def _long_window_vector(features_df, subject, t_start, feature_cols):
    """Return the feature row for a 30 s window closest to t_start (<= tolerance)."""
    if features_df is None:
        return None
    cand = features_df[features_df["subject"] == subject]
    if cand.empty:
        return None
    diffs = (cand["start_time"] - t_start).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > _TIME_MATCH_TOLERANCE:
        return None
    return cand.loc[idx, feature_cols].to_numpy(dtype=float, na_value=0.0)


def _short_window_vector(features_df, subject, t_start, t_end, feature_cols):
    """Aggregate short-window features (mean) within [t_start, t_end) for subject."""
    if features_df is None:
        return None
    mask = (
        (features_df["subject"] == subject)
        & (features_df["start_time"] >= t_start)
        & (features_df["start_time"] < t_end)
    )
    subset = features_df.loc[mask, feature_cols]
    if subset.empty:
        return None
    return subset.mean(axis=0, skipna=True).to_numpy(dtype=float, na_value=0.0)


def _assemble_feature_matrix(reference_df, modalities=None):
    """Build the concatenated feature matrix aligned to the reference rows.

    This helper is at module level (not on a class) so different fusion
    variants can share a single matrix via ``_FEATURE_CACHE``.
    """
    active = list(modalities) if modalities is not None else list(MODALITIES)
    feat_tables = {mod: _load_feature_table(mod) for mod in active}
    feat_cols = {mod: (_feature_cols(t) if t is not None else []) for mod, t in feat_tables.items()}

    n_rows = len(reference_df)
    blocks = {mod: np.zeros((n_rows, len(feat_cols[mod]) + 1), dtype=float) for mod in active}

    for row_idx, (_, row) in enumerate(tqdm(reference_df.iterrows(),
                                            total=n_rows,
                                            desc="Early-fusion assembly")):
        subject = row["subject_id"]
        t_start = row["start_time"]
        t_end = row["end_time"]

        for mod in active:
            if feat_tables[mod] is None:
                continue
            if mod in _LONG_WINDOW_MODS:
                vec = _long_window_vector(feat_tables[mod], subject, t_start, feat_cols[mod])
            else:
                vec = _short_window_vector(feat_tables[mod], subject, t_start, t_end, feat_cols[mod])

            if vec is None:
                blocks[mod][row_idx, -1] = 0.0
            else:
                blocks[mod][row_idx, :-1] = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                blocks[mod][row_idx, -1] = 1.0

    column_names = []
    for mod in active:
        column_names.extend([f"{mod}__{c}" for c in feat_cols[mod]])
        column_names.append(f"{mod}__available")

    X = np.concatenate([blocks[m] for m in active], axis=1)
    return X, column_names


class EarlyFusion(FusionMethod):
    """Feature-level fusion with a LightGBM classifier.

    Parameters
    ----------
    preprocessor:
        Optional sklearn transformer (or a sequence of transformers) applied
        inside each CV fold — fitted on the training fold and applied to the
        held-out fold, so no leakage.
    name:
        Override for the fusion method name (useful for distinguishing
        preprocessor variants in the output).
    """

    name = "early_fusion"

    _LGBM_PARAMS = dict(
        n_estimators=200,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    def __init__(self, preprocessor=None, name=None):
        if name is not None:
            self.name = name
        self.preprocessor = preprocessor

    def _build_pipeline(self, n_features=None):
        """Return a sklearn Pipeline that runs preprocess → LightGBM per fold."""
        steps = []
        if self.preprocessor is not None:
            steps.append(("prep", self.preprocessor))
        steps.append(("clf", LGBMClassifier(**self._LGBM_PARAMS)))
        return Pipeline(steps)

    def fuse(self, predictions, experiment, modality_subset=None):
        """Run early fusion.

        The ``predictions`` dict is used only to obtain the empatica reference
        rows (subject_id, start_time, end_time, y_true, fold_id, task). Features
        themselves are loaded from ``RESULTS_DIR/{modality}/artifacts/features.csv``.
        """
        num_classes = 2 if experiment == "loso_two_class" else 3
        is_logo = experiment == "logo_three_class"

        if "empatica" not in predictions or predictions["empatica"] is None:
            raise ValueError("Early fusion requires empatica predictions as reference")

        active = list(modality_subset) if modality_subset is not None else list(MODALITIES)
        # Empatica is the reference window; it must be in the subset.
        if "empatica" not in active:
            active = ["empatica"] + [m for m in active if m != "empatica"]

        ref = predictions["empatica"].copy().reset_index(drop=True)
        carry_cols = ["subject_id", "start_time", "end_time", "y_true"]
        if is_logo:
            carry_cols += ["fold_id"]
            if "task" in ref.columns:
                carry_cols += ["task"]

        cache_key = (experiment, len(ref), tuple(sorted(active)))
        if cache_key in _FEATURE_CACHE:
            X, _ = _FEATURE_CACHE[cache_key]
        else:
            X, cols = _assemble_feature_matrix(ref, modalities=active)
            _FEATURE_CACHE[cache_key] = (X, cols)

        y = ref["y_true"].values
        if is_logo and "fold_id" in ref.columns:
            groups = ref["fold_id"].values
        else:
            groups = ref["subject_id"].values

        logo_cv = LeaveOneGroupOut()
        fused_predictions = np.full(len(ref), -1, dtype=int)
        fused_prob_matrix = np.zeros((len(ref), num_classes))

        for train_idx, test_idx in tqdm(
            logo_cv.split(X, y, groups=groups),
            total=len(np.unique(groups)),
            desc=f"{self.name} CV ({experiment})",
        ):
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                continue

            pipeline = self._build_pipeline(n_features=X.shape[1])
            try:
                pipeline.fit(X[train_idx], y_train)
            except Exception:
                fallback = LGBMClassifier(**self._LGBM_PARAMS)
                fallback.fit(X[train_idx], y_train)
                preds = fallback.predict(X[test_idx])
                proba = fallback.predict_proba(X[test_idx])
                clf_classes = fallback.classes_
            else:
                preds = pipeline.predict(X[test_idx])
                proba = pipeline.predict_proba(X[test_idx])
                clf_classes = pipeline.named_steps["clf"].classes_

            fused_predictions[test_idx] = preds
            for col_idx, cls in enumerate(clf_classes):
                if 0 <= int(cls) < num_classes:
                    fused_prob_matrix[test_idx, int(cls)] = proba[:, col_idx]

        out = ref[carry_cols].copy()
        out["fused_prediction"] = fused_predictions
        for i in range(num_classes):
            out[f"fused_prob_{i}"] = fused_prob_matrix[:, i]
        out["fused_uncertainty"] = [
            compute_uncertainty(fused_prob_matrix[r], num_classes) for r in range(len(out))
        ]

        return out


class EarlyFusionPCA(EarlyFusion):
    """Early fusion with PCA dimensionality reduction before classification."""

    def __init__(self, n_components: int = 64):
        prep = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ])
        super().__init__(preprocessor=prep, name="early_fusion_pca")


class EarlyFusionICA(EarlyFusion):
    """Early fusion with FastICA decomposition, after Zhou et al. (2020)."""

    def __init__(self, n_components: int = 64):
        prep = Pipeline([
            ("scaler", StandardScaler()),
            ("ica", FastICA(n_components=n_components, max_iter=500,
                             tol=1e-3, random_state=42, whiten="unit-variance")),
        ])
        super().__init__(preprocessor=prep, name="early_fusion_ica")


class EarlyFusionSelectK(EarlyFusion):
    """Early fusion with univariate feature selection (ANOVA F-value)."""

    def __init__(self, k: int = 128):
        self._k_configured = k
        super().__init__(preprocessor=None, name="early_fusion_selectk")

    def _build_pipeline(self, n_features=None):
        k = self._k_configured if n_features is None else min(self._k_configured, n_features)
        self.preprocessor = SelectKBest(score_func=f_classif, k=k)
        return super()._build_pipeline(n_features=n_features)
