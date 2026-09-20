"""Shared constants and small helpers used across training and the web app."""

import numpy as np

# Framingham risk factors passed to the CISE meta-learner alongside base model scores.
CLINICAL_FEATURES_FOR_META_LEARNER = [
    "age",
    "sysBP",
    "totChol",
    "currentSmoker",
    "diabetes",
]


def positive_class_weight(labels):
    """Ratio of negative to positive samples (for imbalanced classification)."""
    labels = np.asarray(labels)
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    return n_negative / n_positive
