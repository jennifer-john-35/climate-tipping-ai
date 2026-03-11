import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# load dataset
df = pd.read_csv("data/climate_training_data.csv")

# convert to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# drop invalid rows
df = df.dropna()

# split dataset
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# model
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

# train
model.fit(X_train)

# predictions
y_pred = model.predict(X_test)

# convert output (-1 anomaly, 1 normal)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# fake labels for evaluation (since anomaly detection has no true labels)
y_true = [0]*len(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nMODEL SCORES")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# save model
joblib.dump(model, "models/tipping_model.pkl")

print("\nTipping point model trained and saved")
