import pandas as pd


def load_temperature():

    return pd.read_csv("data/global_temperature.csv")


def load_ocean_heat():

    return pd.read_csv("data/ocean_heat.csv")


def load_co2():

    return pd.read_csv("data/co2_levels.csv")


def load_rainfall():

    return pd.read_csv("data/rainfall.csv")


def load_sea_ice():

    return pd.read_csv("data/sea_ice.csv")


def load_grid():

    return pd.read_csv("data/earth_grid.csv")
