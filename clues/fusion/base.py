"""Base class for all fusion methods."""

import pandas as pd


class FusionMethod:
    """Base class for all fusion methods.

    Subclasses must set `name` and implement `fuse()`.
    """

    name: str = ""

    def fuse(self, predictions, experiment, modality_subset=None):
        """Run fusion on loaded modality predictions.

        Args:
            predictions: dict mapping modality name -> DataFrame with columns:
                subject_id, start_time, end_time, y_true, y_pred,
                prob_class_0, prob_class_1, [prob_class_2],
                and optionally fold_id, task.
            experiment: One of "loso_two_class", "loso_three_class", "logo_three_class"
            modality_subset: Optional list of modality names to fuse. When
                None, subclasses default to the global MODALITIES. Methods
                that do not support subsetting should raise ``NotImplementedError``
                if a non-None value is passed.

        Returns:
            DataFrame with columns including: subject_id, start_time, end_time,
            y_true, fused_prediction, fused_prob_0, fused_prob_1, [fused_prob_2],
            plus per-modality y_pred and prob columns.
        """
        raise NotImplementedError
