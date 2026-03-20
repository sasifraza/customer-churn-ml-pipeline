import joblib
import pandas as pd


def load_model(model_path: str = "models/model.pkl"):
    return joblib.load(model_path)


def predict_single(model, payload: dict):
    input_df = pd.DataFrame([payload])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    }