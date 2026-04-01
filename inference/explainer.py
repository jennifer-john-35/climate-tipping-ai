from __future__ import annotations

import numpy as np


class SHAPExplainer:
    """SHAP KernelExplainer wrapper for an sklearn IsolationForest."""

    def __init__(self, tipping_model, background_data: np.ndarray) -> None:
        """
        Args:
            tipping_model: fitted sklearn IsolationForest.
            background_data: shape (n_samples, 5) — training data summary.
        """
        import shap

        background_summary = shap.kmeans(
            background_data, min(100, len(background_data))
        )
        self.explainer = shap.KernelExplainer(
            tipping_model.decision_function, background_summary
        )
        self.expected_value = self.explainer.expected_value

    def explain(
        self, feature_vector: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Args:
            feature_vector: shape (1, 5).

        Returns:
            (shap_values shape (5,), base_value float).
        """
        shap_vals = self.explainer.shap_values(feature_vector, silent=True)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return shap_vals.flatten()[:5], float(self.expected_value)
