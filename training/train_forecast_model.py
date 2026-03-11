import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]
model = Sequential()

model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(X.shape[2]))

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=20, batch_size=16)
y_pred = model.predict(X_test)
print("\nDATASET SPLIT")
print("-------------------")
print("Total samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFORECAST MODEL SCORES")
print("-----------------------")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
model.save("models/forecast_model.h5")
print("Forecast model trained")
joblib.dump(scaler, "models/forecast_scaler.pkl")
