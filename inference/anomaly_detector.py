from __future__ import annotations

import numpy as np
import pandas as pd


class RollingAnomalyDetector:
    """Slide a rolling window across a climate DataFrame and flag anomalies."""

    def __init__(self, tipping_model, feature_engineer) -> None:
        """
        Args:
            tipping_model: fitted sklearn IsolationForest.
            feature_engineer: the feature_engineering module (must expose
                build_rolling_feature_vectors).
        """
        self.tipping_model = tipping_model
        self.feature_engineer = feature_engineer

    def detect(
        self,
        climate_df: pd.DataFrame,
        window: int = 30,
    ) -> tuple[list[bool], np.ndarray]:
        """Slide window across climate_df, score each slice.

        Args:
            climate_df: DataFrame with columns temp, co2, sea_ice, ocean_heat, rainfall.
            window: rolling window size (default 30).

        Returns:
            (anomaly_flags list[bool], rolling_risk_scores np.ndarray)
            Both of length len(climate_df) - window.
        """
        if len(climate_df) <= window:
            return [], np.array([])

        # 1. Build rolling feature vectors
        rolling_fvs = self.feature_engineer.build_rolling_feature_vectors(
            climate_df, window
        )

        # 2. Stack into matrix: shape (N-window, 5)
        feature_matrix = np.array(rolling_fvs)

        # 3. Score with decision_function
        raw = self.tipping_model.decision_function(feature_matrix)

        # 4. Normalise to [0, 1]  (higher = more anomalous)
        risk_scores = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        # 5. Flag anomaly where risk_score > 0.7
        anomaly_flags = [bool(s > 0.7) for s in risk_scores]

        return anomaly_flags, risk_scores
