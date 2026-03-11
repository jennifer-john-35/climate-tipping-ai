import pandas as pd


def load_temperature():

    df = pd.read_csv("data/global_temperature.csv", skiprows=1)

    df.columns = df.columns.str.strip()

    df = df.rename(columns={"Year": "year"})

    df = df.melt(id_vars=["year"], var_name="month", value_name="temp_anomaly")

    df = df.dropna()

    return df


def load_co2():

    df = pd.read_csv("data/co2_levels.csv", comment="#")

    df = df[["year", "month", "average"]]

    df.rename(columns={"average": "co2"}, inplace=True)

    return df


def load_sea_ice():

    df = pd.read_csv("data/sea_ice.csv")

    df.columns = df.columns.str.strip().str.lower()

    # find the extent column automatically
    extent_col = None
    for col in df.columns:
        if "extent" in col:
            extent_col = col
            break

    if extent_col is None:
        raise ValueError("Sea ice extent column not found")

    df = df[[extent_col]]

    df.rename(columns={extent_col: "sea_ice_extent"}, inplace=True)

    return df


def load_ocean_heat():

    df = pd.read_csv("data/ocean_heat.csv")

    df.rename(columns={"Mean": "ocean_temp"}, inplace=True)

    return df


def load_rainfall():

    df = pd.read_csv("data/rainfall.csv")

    return df


def combine_datasets():

    temp = load_temperature()
    co2 = load_co2()
    ice = load_sea_ice()
    ocean = load_ocean_heat()

    temp_vals = temp["temp_anomaly"].dropna().values
    co2_vals = co2["co2"].dropna().values
    ice_vals = ice["sea_ice_extent"].dropna().values
    ocean_vals = ocean["ocean_temp"].dropna().values

    min_len = min(len(temp_vals), len(co2_vals), len(ice_vals), len(ocean_vals))

    df = pd.DataFrame({
        "temp": temp_vals[:min_len],
        "co2": co2_vals[:min_len],
        "sea_ice": ice_vals[:min_len],
        "ocean_heat": ocean_vals[:min_len]
    })

    return df
