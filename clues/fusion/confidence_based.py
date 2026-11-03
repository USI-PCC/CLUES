"""Confidence-based fusion method."""

import numpy as np

from clues.fusion.base import FusionMethod
from clues.fusion.utils import (
    align_all_modalities,
    compute_confidence_margin,
    compute_fold_modality_f1,
    compute_uncertainty,
    get_cv_groups,
)
from lib.config import MODALITIES


class ConfidenceBasedFusion(FusionMethod):
    """Fusion weighting each modality by macro F1 amplified by confidence.

    For each modality, weight = base_F1 * confidence^alpha, where confidence
    is the margin between the top two class probabilities. Weights are
    normalised before computing the weighted average of probabilities, and
    fall back to equal weights if the total weight is zero.

    ``base_F1`` is computed **per CV fold** from the training subjects
    (LOSO groups by subject, LOGO by fold_id) so no held-out label
    information leaks into the weighting of that fold.

    Args:
        alpha: Exponent applied to the confidence margin (default 1.5).
            Higher values give more influence to high-confidence modalities.
    """

    name = "confidence_based"

    def __init__(self, alpha=1.5):
        self.alpha = alpha

    def fuse(self, predictions, experiment):
        """Run confidence-based fusion with per-fold weights.

        Args:
            predictions: Dict mapping modality name -> DataFrame.
            experiment: One of "loso_two_class", "loso_three_class",
                "logo_three_class".

        Returns:
            Aligned DataFrame with fused_prediction, fused_prob_{i}, and
            fused_uncertainty columns.
        """
        num_classes = 2 if experiment == "loso_two_class" else 3
        aligned = align_all_modalities(predictions, experiment)
        groups = get_cv_groups(aligned, experiment)

        # Precompute per-fold modality weights: {held_out_group: {mod: f1}}
        fold_weights = {}
        for held_out in np.unique(groups):
            train_mask = groups != held_out
            fold_weights[held_out] = compute_fold_modality_f1(
                aligned, train_mask, MODALITIES
            )

        fused_predictions = []
        fused_probs = {i: [] for i in range(num_classes)}

        for row_idx, (_, row) in enumerate(aligned.iterrows()):
            weights_for_fold = fold_weights[groups[row_idx]]

            available_mods = []
            mod_probs = {}
            mod_weights = {}

            for mod in MODALITIES:
                y_pred_col = f"y_pred_{mod}"
                if y_pred_col not in row.index:
                    continue
                if row[y_pred_col] == -1:
                    continue

                probs = {}
                for i in range(num_classes):
                    prob_col = f"prob_{mod}_{i}"
                    probs[i] = float(row[prob_col]) if prob_col in row.index else 0.0

                confidence = compute_confidence_margin(probs)
                base_f1 = weights_for_fold.get(mod, 0.0)
                weight = base_f1 * (confidence ** self.alpha)

                available_mods.append(mod)
                mod_probs[mod] = probs
                mod_weights[mod] = weight

            total_weight = sum(mod_weights.values())

            if total_weight == 0 and available_mods:
                # Fallback to equal weights
                equal_w = 1.0 / len(available_mods)
                for mod in available_mods:
                    mod_weights[mod] = equal_w
                total_weight = 1.0

            if total_weight > 0:
                weighted_probs = np.zeros(num_classes)
                for mod in available_mods:
                    w = mod_weights[mod] / total_weight
                    for i in range(num_classes):
                        weighted_probs[i] += w * mod_probs[mod].get(i, 0.0)
                fused_predictions.append(int(np.argmax(weighted_probs)))
                for i in range(num_classes):
                    fused_probs[i].append(float(weighted_probs[i]))
            else:
                fused_predictions.append(-1)
                for i in range(num_classes):
                    fused_probs[i].append(0.0)

        aligned["fused_prediction"] = fused_predictions
        for i in range(num_classes):
            aligned[f"fused_prob_{i}"] = fused_probs[i]

        aligned["fused_uncertainty"] = [
            compute_uncertainty([fused_probs[i][r] for i in range(num_classes)], num_classes)
            for r in range(len(fused_predictions))
        ]

        return aligned
