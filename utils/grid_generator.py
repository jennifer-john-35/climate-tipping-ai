import pandas as pd
import numpy as np


def generate_global_grid(resolution=5):

    latitudes = np.arange(-90, 90 + resolution, resolution)
    longitudes = np.arange(-180, 180 + resolution, resolution)

    grid_points = []

    for lat in latitudes:
        for lon in longitudes:
            grid_points.append({
                "latitude": lat,
                "longitude": lon
            })

    grid_df = pd.DataFrame(grid_points)

    return grid_df


if __name__ == "__main__":

    grid = generate_global_grid()

    grid.to_csv("data/earth_grid.csv", index=False)

    print("Earth grid created with", len(grid), "points")
