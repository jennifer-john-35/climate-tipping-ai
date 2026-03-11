import joblib
import pandas as pd

model = joblib.load("models/tipping_model.pkl")


def calculate_risk(data):

    scores = model.decision_function(data)

    risk = 1 - scores

    return risk
