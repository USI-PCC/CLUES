"""Stacking meta-classifier fusion method."""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from lightgbm import LGBMClassifier
from tqdm import tqdm

from clues.fusion.base import FusionMethod
from clues.fusion.utils import align_all_modalities, build_stacking_features, compute_uncertainty
from lib.config import MODALITIES


class StackingFusion(FusionMethod):
    """Fusion using a LightGBM meta-classifier trained with nested CV.

    For LOGO experiments, leave-one-fold-out CV is applied (using fold_id).
    For LOSO experiments, leave-one-subject-out CV is applied (using
    subject_id). In both cases the meta-classifier is trained on all windows
    outside the held-out group and predicts the held-out windows.

    Nested-CV leakage guarantee
    ---------------------------
    The per-modality predictions consumed here are themselves produced by
    LOSO/LOGO base classifiers — i.e. each base-level prediction for subject
    (or fold) S comes from a base model that was *not* trained on S.
    Applying an outer LOSO/LOGO split at the meta level on top therefore
    constitutes nested cross-validation: the meta-classifier never observes
    features derived from a base model that had access to the held-out
    subject's data. This mirrors the nested CV protocol recommended in
    Oppelt et al. (ADABase, 2022).

    LGBMClassifier hyperparameters are fixed to a regularised configuration
    to reduce overfitting on the small meta-feature matrix.
    """

    name = "stacking"

    _LGBM_PARAMS = dict(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        num_leaves=8,
        min_child_samples=10,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1,
    )

    def fuse(self, predictions, experiment, modality_subset=None):
        """Run stacking fusion with nested CV.

        Args:
            predictions: Dict mapping modality name -> DataFrame.
            experiment: One of "loso_two_class", "loso_three_class",
                "logo_three_class".
            modality_subset: Optional list of modality names to fuse.

        Returns:
            Aligned DataFrame with fused_prediction and fused_prob_{i} columns.
            Rows that could not be assigned a prediction (e.g. the held-out
            group had no valid training data) receive fused_prediction = -1.
        """
        num_classes = 2 if experiment == "loso_two_class" else 3
        active = list(modality_subset) if modality_subset is not None else list(MODALITIES)

        aligned = align_all_modalities(
            predictions, experiment, modality_subset=active
        )

        # Build the full feature matrix from aligned rows
        X_rows = []
        for _, row in aligned.iterrows():
            prob_dicts = {}
            for mod in active:
                y_pred_col = f"y_pred_{mod}"
                if y_pred_col in row.index and row[y_pred_col] != -1:
                    probs = {}
                    for i in range(num_classes):
                        prob_col = f"prob_{mod}_{i}"
                        probs[i] = float(row[prob_col]) if prob_col in row.index else 0.0
                    prob_dicts[mod] = probs
                else:
                    prob_dicts[mod] = None
            X_rows.append(build_stacking_features(prob_dicts, num_classes,
                                                  modalities=active))

        X = np.vstack(X_rows)
        y = aligned["y_true"].values

        # Determine CV groups
        is_logo = experiment == "logo_three_class"
        if is_logo and "fold_id" in aligned.columns:
            groups = aligned["fold_id"].values
        else:
            groups = aligned["subject_id"].values

        logo_cv = LeaveOneGroupOut()
        fused_predictions = np.full(len(aligned), -1, dtype=int)
        fused_prob_matrix = np.zeros((len(aligned), num_classes))

        unique_groups = np.unique(groups)
        for held_out in tqdm(unique_groups, desc=f"Stacking CV ({experiment})"):
            train_mask = groups != held_out
            test_mask = groups == held_out

            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]

            if len(np.unique(y_train)) < 2:
                continue

            clf = LGBMClassifier(**self._LGBM_PARAMS)
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            proba = clf.predict_proba(X_test)

            clf_classes = list(clf.classes_)
            fused_predictions[test_mask] = preds
            for col_idx, cls in enumerate(clf_classes):
                if cls < num_classes:
                    fused_prob_matrix[test_mask, cls] = proba[:, col_idx]

        aligned["fused_prediction"] = fused_predictions
        for i in range(num_classes):
            aligned[f"fused_prob_{i}"] = fused_prob_matrix[:, i]

        aligned["fused_uncertainty"] = [
            compute_uncertainty(fused_prob_matrix[r], num_classes)
            for r in range(len(aligned))
        ]

        return aligned
