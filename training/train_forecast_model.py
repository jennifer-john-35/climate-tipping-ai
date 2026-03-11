import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("data/climate_training_data.csv")

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

values = df.values

scaler = MinMaxScaler()
values = scaler.fit_transform(values)

X = []
y = []

window = 10

for i in range(len(values)-window-1):
    X.append(values[i:(i+window)])
    y.append(values[i+window])

X = np.array(X)
y = np.array(y)

model = Sequential()

model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(X.shape[2]))

model.compile(optimizer="adam", loss="mse")

model.fit(X, y, epochs=20, batch_size=16)

model.save("models/forecast_model.h5")

joblib.dump(scaler, "models/forecast_scaler.pkl")

print("Forecast model trained")
