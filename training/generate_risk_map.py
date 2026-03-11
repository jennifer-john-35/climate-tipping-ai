import sys
import os
import pandas as pd
import reverse_geocoder as rg
from utils.data_loader import load_grid
from utils.risk_calculator import calculate_risk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():

    print("Loading grid...")
    grid = load_grid()

    print("Loading climate features...")
    sample_features = pd.read_csv("data/climate_training_data.csv")

    sample_features = sample_features.apply(pd.to_numeric, errors="coerce")
    sample_features = sample_features.dropna()

    # match grid size
    sample_features = sample_features.sample(len(grid), replace=True)

    print("Calculating risk scores...")
    risk_scores = calculate_risk(sample_features)

    grid["risk_score"] = risk_scores

    print("Reverse geocoding locations...")

    coords = list(zip(grid["latitude"], grid["longitude"]))

    # single-thread mode (avoids Windows multiprocessing bug)
    results = rg.search(coords, mode=1)

    names = []

    for r in results:
        city = r["name"]
        country = r["cc"]
        names.append(f"{city}, {country}")

    grid["name"] = names

    print("Saving results...")

    grid.to_csv("data/global_risk_scores.csv", index=False)

    print("✅ Global risk map created")


if __name__ == "__main__":
    main()
