from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Exception / Warning classes
# ---------------------------------------------------------------------------

class ModelLoadError(Exception):
    """Raised when a model file cannot be found or loaded."""


class DataValidationError(Exception):
    """Raised when required columns are missing from a data file."""


class NaNThresholdWarning(UserWarning):
    """Issued when a column exceeds the 5 % NaN threshold."""


class InferenceDimensionError(Exception):
    """Raised when a feature vector has an unexpected shape."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "temp_variance",
    "ocean_variance",
    "co2_trend",
    "ice_variance",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    temp_variance: float
    ocean_variance: float
    co2_trend: float
    ice_variance: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [[self.temp_variance, self.ocean_variance,
              self.co2_trend, self.ice_variance]]
        )


@dataclass
class ScanResult:
    region_name: str
    scan_timestamp: datetime
    risk_score: float
    anomaly_flags: list[bool]
    anomaly_dates: list[str]
    shap_values: dict[str, float]
    shap_base_value: float
    forecasts: dict[str, np.ndarray]
    forecast_lower: dict[str, np.ndarray]
    forecast_upper: dict[str, np.ndarray]
    feature_vector: FeatureVector
    rolling_risk_trend: np.ndarray


@dataclass
class AlertEvent:
    timestamp: str   # ISO 8601
    region_name: str
    risk_score: float
    threshold: float
