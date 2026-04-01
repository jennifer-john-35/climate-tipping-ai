import pandas as pd
import numpy as np


def compute_variance(series):

    return np.var(series)


def compute_trend(series):

    return series.diff().mean()


def build_feature_vector(temp, ocean, co2, rain, ice):
    """Build 5-feature vector (used for forecasting)."""
    features = {
        "temp_variance": compute_variance(temp),
        "ocean_variance": compute_variance(ocean),
        "co2_trend": compute_trend(co2),
        "rain_variance": compute_variance(rain),
        "ice_variance": compute_variance(ice)
    }
    return pd.DataFrame([features])


def build_tipping_feature_vector(temp, ocean, co2, ice) -> np.ndarray:
    """Build 4-feature vector matching the trained IsolationForest (no rainfall)."""
    features = {
        "temp_variance": compute_variance(temp),
        "ocean_variance": compute_variance(ocean),
        "co2_trend": compute_trend(co2),
        "ice_variance": compute_variance(ice),
    }
    return pd.DataFrame([features]).values  # shape (1, 4)


def build_rolling_feature_vectors(
    climate_df: pd.DataFrame,
    window: int = 30,
) -> list[np.ndarray]:
    """Slide a window across climate_df and build a 4-feature tipping vector per position."""
    results = []
    n = len(climate_df)
    for i in range(window, n):
        slice_df = climate_df.iloc[i - window: i]
        fv = build_tipping_feature_vector(
            temp=slice_df["temp"],
            ocean=slice_df["ocean_heat"],
            co2=slice_df["co2"],
            ice=slice_df["sea_ice"],
        )
        results.append(fv[0])  # shape (4,)
    return results
