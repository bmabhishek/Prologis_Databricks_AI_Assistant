"""
SageMaker inference script for the Logistic Regression bank marketing model.
"""
import json
import os
import joblib
import pandas as pd

NUMERIC_FEATURES = ["age", "balance", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        # Fill any missing categorical with 'unknown', missing numeric with 0
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = 0
        return df[ALL_FEATURES]
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    preds = model.predict(input_data).tolist()
    probs = model.predict_proba(input_data)[:, 1].tolist()
    return [
        {
            "subscribed": bool(p),
            "label": "yes" if p == 1 else "no",
            "probability": float(prob),
        }
        for p, prob in zip(preds, probs)
    ]


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
