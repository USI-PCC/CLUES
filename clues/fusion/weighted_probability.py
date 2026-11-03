"""Weighted probability fusion method."""

import numpy as np

from clues.fusion.base import FusionMethod
from clues.fusion.utils import (
    align_all_modalities,
    compute_fold_modality_f1,
    compute_uncertainty,
    get_cv_groups,
)
from lib.config import MODALITIES


class WeightedProbabilityFusion(FusionMethod):
    """Fusion by weighting each modality's probabilities by its macro F1.

    Weights are computed **per CV fold** from the training subjects only
    (LOSO groups by subject, LOGO by fold_id), so no held-out information
    leaks into the fusion of that fold. Weights are normalised to sum to 1
    across available modalities within a row; the fused prediction is the
    argmax of the weighted probability average.
    """

    name = "weighted_probability"

    def fuse(self, predictions, experiment, modality_subset=None):
        """Run weighted probability fusion with per-fold weights.

        Args:
            predictions: Dict mapping modality name -> DataFrame.
            experiment: One of "loso_two_class", "loso_three_class",
                "logo_three_class".
            modality_subset: Optional list of modality names to fuse. Defaults
                to the global MODALITIES.

        Returns:
            Aligned DataFrame with fused_prediction, fused_prob_{i}, and
            fused_uncertainty columns.
        """
        num_classes = 2 if experiment == "loso_two_class" else 3
        active = list(modality_subset) if modality_subset is not None else list(MODALITIES)

        aligned = align_all_modalities(
            predictions, experiment, modality_subset=active
        )
        groups = get_cv_groups(aligned, experiment)

        # Precompute per-fold modality weights: {held_out_group: {mod: f1}}
        fold_weights = {}
        for held_out in np.unique(groups):
            train_mask = groups != held_out
            fold_weights[held_out] = compute_fold_modality_f1(
                aligned, train_mask, active
            )

        fused_predictions = []
        fused_probs = {i: [] for i in range(num_classes)}

        for row_idx, (_, row) in enumerate(aligned.iterrows()):
            weights_for_fold = fold_weights[groups[row_idx]]

            weighted_probs = np.zeros(num_classes)
            total_weight = 0.0

            for mod in active:
                y_pred_col = f"y_pred_{mod}"
                if y_pred_col not in row.index:
                    continue
                if row[y_pred_col] == -1:
                    continue

                base_f1 = weights_for_fold.get(mod, 0.0)
                total_weight += base_f1

                for i in range(num_classes):
                    prob_col = f"prob_{mod}_{i}"
                    prob = float(row[prob_col]) if prob_col in row.index else 0.0
                    weighted_probs[i] += base_f1 * prob

            if total_weight > 0:
                weighted_probs /= total_weight
                fused_predictions.append(int(np.argmax(weighted_probs)))
            else:
                fused_predictions.append(-1)
                weighted_probs = np.zeros(num_classes)

            for i in range(num_classes):
                fused_probs[i].append(float(weighted_probs[i]))

        aligned["fused_prediction"] = fused_predictions
        for i in range(num_classes):
            aligned[f"fused_prob_{i}"] = fused_probs[i]

        aligned["fused_uncertainty"] = [
            compute_uncertainty([fused_probs[i][r] for i in range(num_classes)], num_classes)
            for r in range(len(fused_predictions))
        ]

        return aligned
