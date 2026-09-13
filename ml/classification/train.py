"""
Train Logistic Regression on the UCI Bank Marketing dataset.
Downloads from UCI on first run, caches locally.

Run: python ml/classification/train.py
"""
import io
import json
import zipfile
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

OUT_DIR = Path(__file__).parent
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
LOCAL_CSV = DATA_DIR / "bank.csv"

NUMERIC_FEATURES = ["age", "balance", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]
TARGET = "y"


def download_dataset():
    """Download bank.csv from UCI if not already present."""
    if LOCAL_CSV.exists():
        print(f"Using cached dataset: {LOCAL_CSV}")
        return
    print(f"Downloading from {UCI_URL}")
    r = requests.get(UCI_URL, timeout=60)
    r.raise_for_status()

    # The UCI zip contains another zip ("bank.zip") with the CSVs
    with zipfile.ZipFile(io.BytesIO(r.content)) as outer:
        with outer.open("bank.zip") as inner_zip:
            with zipfile.ZipFile(io.BytesIO(inner_zip.read())) as inner:
                with inner.open("bank-full.csv") as csv:
                    LOCAL_CSV.write_bytes(csv.read())
    print(f"Saved {LOCAL_CSV}")


def main():
    download_dataset()
    # UCI bank CSV uses ; as separator
    df = pd.read_csv(LOCAL_CSV, sep=";")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df[TARGET].value_counts()}")

    # Encode target: yes=1, no=0
    df[TARGET] = (df[TARGET] == "yes").astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Pre-processing: scale numerics, one-hot encode categoricals
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # dataset is imbalanced (~88% no, 12% yes)
            random_state=42,
        )),
    ])

    print("\nTraining Logistic Regression...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "LogisticRegression",
        "dataset": "UCI Bank Marketing",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    print("\n--- Metrics ---")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=pred):")
    print(f"    {metrics['confusion_matrix']}")

    # Save artifacts
    model_path = OUT_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    # Sanity check
    sample = X_test.iloc[0:1]
    pred = pipeline.predict(sample)[0]
    proba = pipeline.predict_proba(sample)[0, 1]
    actual = y_test.iloc[0]
    print(f"\n--- Sanity check ---")
    print(f"  Predicted: {pred} (prob={proba:.3f})")
    print(f"  Actual   : {actual}")


if __name__ == "__main__":
    main()
