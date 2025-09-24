import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, confusion_matrix
)

from src.model_utils import load_data, get_feature_columns, build_preprocessor

def drop_unnecessary_columns(df):
    """Drop index/id columns that are not features."""
    for col in ['Unnamed: 0', 'index', 'id']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training data")
    parser.add_argument("--test", type=str, required=True, help="Path to test data")
    parser.add_argument("--out", type=str, required=True, help="Output model path")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    args = parser.parse_args()

    # Load train & test
    train_df = load_data(args.train)
    test_df = load_data(args.test)

    # Drop unnecessary columns
    train_df = drop_unnecessary_columns(train_df)
    test_df = drop_unnecessary_columns(test_df)

    # Extract features & target
    X_train, y_train, num_cols, cat_cols = get_feature_columns(train_df)
    X_test, y_test, _, _ = get_feature_columns(test_df)

    # Build pipeline
    preproc = build_preprocessor(num_cols, cat_cols)
    model = Pipeline([
        ("preprocessor", preproc),
        ("clf", RandomForestClassifier(n_estimators=args.n_estimators, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds),
        "log_loss": log_loss(y_test, probs)
    }

    cm = confusion_matrix(y_test, preds)

    # Save model + metrics
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, args.out)

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    np.save("models/confusion_matrix.npy", cm)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params({"n_estimators": args.n_estimators})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random-forest-model")

    print("✅ Training complete. Model & metrics saved.")

if __name__ == "__main__":
    main()
