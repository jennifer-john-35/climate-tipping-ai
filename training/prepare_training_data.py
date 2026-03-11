import sys
import os
from utils.data_processor import combine_datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

df = combine_datasets()

print("Training data shape:", df.shape)

print(df.head())

df.to_csv("data/climate_training_data.csv", index=False)

print("Training dataset created")
