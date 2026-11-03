"""Majority voting fusion method."""

import numpy as np
from collections import Counter

from clues.fusion.base import FusionMethod
from clues.fusion.utils import align_all_modalities, compute_uncertainty
from lib.config import MODALITIES


class MajorityVotingFusion(FusionMethod):
    """Fusion by majority vote across modalities.

    Each modality with a valid prediction (y_pred != -1) casts one vote.
    The prediction with the most votes wins. Probabilities are averaged
    across all available modalities. If no modality has a valid prediction,
    fused_prediction is set to -1.
    """

    name = "majority_voting"

    def fuse(self, predictions, experiment):
        """Run majority voting fusion.

        Args:
            predictions: Dict mapping modality name -> DataFrame.
            experiment: One of "loso_two_class", "loso_three_class",
                "logo_three_class".

        Returns:
            Aligned DataFrame with fused_prediction and fused_prob_{i} columns.
        """
        num_classes = 2 if experiment == "loso_two_class" else 3
        aligned = align_all_modalities(predictions, experiment)

        fused_predictions = []
        fused_probs = {i: [] for i in range(num_classes)}

        for _, row in aligned.iterrows():
            votes = []
            available_probs = {i: [] for i in range(num_classes)}

            for mod in MODALITIES:
                y_pred_col = f"y_pred_{mod}"
                if y_pred_col not in row.index:
                    continue
                y_pred = row[y_pred_col]
                if y_pred == -1:
                    continue
                votes.append(int(y_pred))
                for i in range(num_classes):
                    prob_col = f"prob_{mod}_{i}"
                    if prob_col in row.index:
                        available_probs[i].append(float(row[prob_col]))

            if votes:
                counter = Counter(votes)
                fused_predictions.append(counter.most_common(1)[0][0])
                for i in range(num_classes):
                    vals = available_probs[i]
                    fused_probs[i].append(float(np.mean(vals)) if vals else 0.0)
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
