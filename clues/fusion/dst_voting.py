"""Dempster-Shafer late fusion.

Treats each modality's class-probability vector as a Basic Probability
Assignment (BPA) on the singleton hypotheses {class_0, class_1, [class_2]}.
Combines modality BPAs pairwise with Dempster's rule of combination:

    m_12(A) = (1 / (1 - K)) * sum_{B ∩ C = A} m_1(B) * m_2(C)
    K       = sum_{B ∩ C = ∅} m_1(B) * m_2(C)

Under singleton-only BPAs (our case), intersection of different classes is
the empty set, so K is just the sum of products over disagreeing class
pairs and combination reduces to an element-wise product of the probability
vectors followed by normalization. If the conflict K equals 1 the combined
BPA is undefined and we fall back to the per-class mean probability.

Reference: Zhou et al., "Multimodal Physiological Signals for Workload
Prediction in Robot-assisted Surgery", ACM THRI (2020) — reports DST voting
as the most noise-robust late-fusion scheme among the schemes tested.
"""

import numpy as np

from clues.fusion.base import FusionMethod
from clues.fusion.utils import align_all_modalities, compute_uncertainty
from lib.config import MODALITIES


class DSTVotingFusion(FusionMethod):
    """Dempster-Shafer rule of combination over per-modality singleton BPAs."""

    name = "dst_voting"

    def fuse(self, predictions, experiment):
        num_classes = 2 if experiment == "loso_two_class" else 3
        aligned = align_all_modalities(predictions, experiment)

        fused_predictions = []
        fused_probs = {i: [] for i in range(num_classes)}
        fused_uncertainty = []

        for _, row in aligned.iterrows():
            mod_vectors = []

            for mod in MODALITIES:
                y_pred_col = f"y_pred_{mod}"
                if y_pred_col not in row.index or row[y_pred_col] == -1:
                    continue
                vec = np.array(
                    [float(row.get(f"prob_{mod}_{i}", 0.0)) for i in range(num_classes)],
                    dtype=float,
                )
                # Clip zeros to avoid degenerate masses
                vec = np.clip(vec, 1e-6, 1.0)
                vec = vec / vec.sum()
                mod_vectors.append(vec)

            if not mod_vectors:
                fused_predictions.append(-1)
                for i in range(num_classes):
                    fused_probs[i].append(0.0)
                fused_uncertainty.append(0.0)
                continue

            # Combine sequentially via Dempster's rule (singleton BPA case).
            combined = mod_vectors[0].copy()
            for nxt in mod_vectors[1:]:
                prod = combined * nxt  # element-wise (singleton intersections)
                total = prod.sum()      # = 1 - K (conflict mass)
                if total <= 0:
                    # Total conflict — back off to arithmetic mean of current
                    # running combination and the next vector.
                    combined = (combined + nxt) / 2.0
                else:
                    combined = prod / total

            fused_predictions.append(int(np.argmax(combined)))
            for i in range(num_classes):
                fused_probs[i].append(float(combined[i]))
            fused_uncertainty.append(compute_uncertainty(combined, num_classes))

        aligned["fused_prediction"] = fused_predictions
        for i in range(num_classes):
            aligned[f"fused_prob_{i}"] = fused_probs[i]
        aligned["fused_uncertainty"] = fused_uncertainty

        return aligned
