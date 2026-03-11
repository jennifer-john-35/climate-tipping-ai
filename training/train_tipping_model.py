import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


df = pd.read_csv("data/climate_training_data.csv")

# convert all values to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# remove invalid rows
df = df.dropna()

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(df)

joblib.dump(model, "models/tipping_model.pkl")

print("Tipping point model trained and saved")
