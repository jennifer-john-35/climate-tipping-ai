import os
from datetime import datetime

import pandas as pd


def _attach_freshness(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """Attach last-modified datetime as df.attrs['_freshness']."""
    mtime = os.path.getmtime(filepath)
    df.attrs["_freshness"] = datetime.fromtimestamp(mtime)
    return df


def load_temperature():
    path = "data/global_temperature.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_ocean_heat():
    path = "data/ocean_heat.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_co2():
    path = "data/co2_levels.csv"
    df = pd.read_csv(path, comment="#")
    return _attach_freshness(df, path)


def load_rainfall():
    path = "data/rainfall.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_sea_ice():
    path = "data/sea_ice.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_grid():
    path = "data/earth_grid.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_risk_scores():
    path = "data/global_risk_scores.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_climate_training():
    path = "data/climate_training_data.csv"
    df = pd.read_csv(path)
    return _attach_freshness(df, path)


def load_all_climate_data(region_name: str) -> dict[str, pd.DataFrame]:
    """Load all climate datasets and return as a dict keyed by dataset name.

    Each DataFrame has a '_freshness' attribute (last-modified datetime).
    The region_name parameter is accepted for API compatibility; all files
    are global datasets so the same data is returned regardless of region.
    """
    return {
        "temperature": load_temperature(),
        "co2": load_co2(),
        "ocean_heat": load_ocean_heat(),
        "rainfall": load_rainfall(),
        "sea_ice": load_sea_ice(),
        "grid": load_grid(),
        "risk_scores": load_risk_scores(),
        "climate_training": load_climate_training(),
    }
