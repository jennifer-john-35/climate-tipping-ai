import numpy as np
import pandas as pd


def calculate_risk(model, data):
    """Calculate risk scores for the given data using the provided model.

    Args:
        model: A fitted sklearn model with a decision_function method.
        data: Input data to score.

    Returns:
        Array of risk scores (higher = more anomalous).
    """
    scores = model.decision_function(data)
    risk = 1 - scores
    return risk


def score_batch(model, feature_matrix: np.ndarray) -> np.ndarray:
    """Score a batch of feature vectors and normalise to [0, 1].

    Args:
        model: A fitted sklearn model with a decision_function method.
        feature_matrix: Array of shape (n_samples, n_features).

    Returns:
        Normalised risk scores in [0, 1], shape (n_samples,).
    """
    scores = model.decision_function(feature_matrix)
    normalised = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return normalised


def normalise_score(raw: float) -> float:
    """Clamp a single raw decision_function output to [0, 1].

    Uses a simple approximation: np.clip(1 - raw, 0, 1).

    Args:
        raw: A single raw decision_function score.

    Returns:
        A float in [0, 1].
    """
    return float(np.clip(1 - raw, 0, 1))
