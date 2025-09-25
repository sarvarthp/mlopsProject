import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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

def evaluate_model(model, X_test, y_test):
    """Return metrics and confusion matrix for a model"""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds),
        "log_loss": log_loss(y_test, probs)
    }
    cm = confusion_matrix(y_test, preds)
    return metrics, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training data")
    parser.add_argument("--test", type=str, required=True, help="Path to test data")
    parser.add_argument("--out-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--rf-estimators", type=int, default=100, help="Random Forest n_estimators")
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

    # Preprocessing pipeline
    preproc = build_preprocessor(num_cols, cat_cols)

    # Define models
    models = {
        "RandomForest": Pipeline([
            ("preprocessor", preproc),
            ("clf", RandomForestClassifier(n_estimators=args.rf_estimators, random_state=42))
        ]),
        "DecisionTree": Pipeline([
            ("preprocessor", preproc),
            ("clf", DecisionTreeClassifier(random_state=42))
        ])
    }

    Path(args.out_dir).mkdir(exist_ok=True)

    all_metrics = {}
    all_conf_matrices = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics, cm = evaluate_model(model, X_test, y_test)

        # Save model
        joblib.dump(model, Path(args.out_dir) / f"{name.lower()}_model.pkl")
        all_metrics[name] = metrics
        all_conf_matrices[name] = cm

        # Log to MLflow
        with mlflow.start_run(run_name=name):
            if name == "RandomForest":
                mlflow.log_param("n_estimators", args.rf_estimators)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"{name.lower()}_model")

    # Save metrics and confusion matrices
    with open(Path(args.out_dir) / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    np.save(Path(args.out_dir) / "confusion_matrices.npy", all_conf_matrices)

    print("✅ Training complete. Models & metrics saved.")

if __name__ == "__main__":
    main()
