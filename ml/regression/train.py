"""
Train a Random Forest Regressor on the California Housing dataset.
Saves model artifact to ml/regression/model.joblib and metrics to metrics.json.

Run: python ml/regression/train.py
"""
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target  # median house value in 100k USD
    print(f"Features: {list(X.columns)}")
    print(f"Shape: X={X.shape}, y={y.shape}")

    # --- Quick EDA summary ---
    print("\n--- Feature Summary ---")
    print(X.describe().T[["mean", "std", "min", "max"]])
    print(f"\nTarget mean: {y.mean():.3f} ({y.mean()*100000:,.0f} USD)")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # --- Pipeline: standardize + RF ---
    # Pipeline keeps preprocessing + model bundled — easier deployment
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print("\nTraining Random Forest...")
    pipeline.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "model": "RandomForestRegressor",
        "dataset": "California Housing",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "feature_names": list(X.columns),
    }
    print("\n--- Metrics ---")
    print(f"  RMSE : {rmse:.4f}  ({rmse*100000:,.0f} USD)")
    print(f"  MAE  : {mae:.4f}  ({mae*100000:,.0f} USD)")
    print(f"  R2   : {r2:.4f}")

    # --- Feature importance ---
    rf = pipeline.named_steps["rf"]
    importances = sorted(
        zip(X.columns, rf.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("\n--- Feature Importance ---")
    for feat, imp in importances:
        print(f"  {feat:12s}: {imp:.4f}")
    metrics["feature_importance"] = {f: float(i) for f, i in importances}

    # --- Save artifacts ---
    model_path = OUT_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    # --- Quick sanity check ---
    sample = X_test.iloc[0:1]
    pred = pipeline.predict(sample)[0]
    actual = y_test.iloc[0]
    print(f"\n--- Sanity check ---")
    print(f"  Sample input: {sample.to_dict('records')[0]}")
    print(f"  Predicted: {pred:.3f} ({pred*100000:,.0f} USD)")
    print(f"  Actual   : {actual:.3f} ({actual*100000:,.0f} USD)")


if __name__ == "__main__":
    main()
