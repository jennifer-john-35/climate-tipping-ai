import pandas as pd
import numpy as np


def compute_variance(series):

    return np.var(series)


def compute_trend(series):

    return series.diff().mean()


def build_feature_vector(temp, ocean, co2, rain, ice):

    features = {
        "temp_variance": compute_variance(temp),
        "ocean_variance": compute_variance(ocean),
        "co2_trend": compute_trend(co2),
        "rain_variance": compute_variance(rain),
        "ice_variance": compute_variance(ice)
    }

    return pd.DataFrame([features])
