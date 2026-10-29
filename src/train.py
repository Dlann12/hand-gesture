"""
Model training pipeline for Hand Gesture Recognition.
Trains Random Forest Classifier, compares against SVM and MLP,
evaluates on held-out test data, and saves model artifacts and reports.
"""

import argparse
import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import yaml

from src.evaluate import (
    calculate_metrics,
    generate_classification_report_text,
    plot_confusion_matrix
)
from src.preprocess import (
    LANDMARK_FEATURE_NAMES,
    load_dataset,
    prepare_features_and_labels,
    split_dataset
)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train Random Forest, SVM, and MLP classifiers and compare performance on validation set.
    """
    candidate_models = {
        "RandomForest_Default": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "RandomForest_Deep": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=10.0,
            probability=True,
            random_state=random_state
        ),
        "MLP_2Layers": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=400,
            early_stopping=True,
            random_state=random_state
        )
    }

    comparison_records = []
    trained_models = {}

    print("\n--- Training and Comparing Candidate Models ---")
    for name, clf in candidate_models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        val_pred = clf.predict(X_val)
        val_acc = float(np.mean(val_pred == y_val))
        train_pred = clf.predict(X_train)
        train_acc = float(np.mean(train_pred == y_train))

        print(f"[{name}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {train_time:.2f}s")

        trained_models[name] = clf
        comparison_records.append({
            "model_name": name,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "training_time_sec": round(train_time, 3)
        })

    comparison_df = pd.DataFrame(comparison_records).sort_values(
        by="val_accuracy", ascending=False
    ).reset_index(drop=True)

    return trained_models, comparison_df


def main():
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = config["data"]["dataset_path"]
    reports_dir = Path(config["reports"]["report_dir"])
    models_dir = Path(config["model"]["save_path"]).parent

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = load_dataset(dataset_path)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples across {df['label'].nunique()} classes.")

    # Split dataset
    train_size = config["data"]["train_size"]
    val_size = config["data"]["val_size"]
    test_size = config["data"]["test_size"]
    seed = config["data"]["random_state"]

    print(f"Splitting dataset: Train={train_size*100:.0f}%, Val={val_size*100:.0f}%, Test={test_size*100:.0f}%...")
    train_df, val_df, test_df = split_dataset(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=seed
    )

    print(f"Subset sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Preprocess & normalize features
    X_train, y_train = prepare_features_and_labels(train_df, normalize=True)
    X_val, y_val = prepare_features_and_labels(val_df, normalize=True)
    X_test, y_test = prepare_features_and_labels(test_df, normalize=True)

    # Train and compare candidate models
    trained_models, comparison_df = train_and_compare_models(
        X_train, y_train, X_val, y_val, random_state=seed
    )

    # Save comparison table
    comparison_csv_path = config["reports"]["comparison_csv"]
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_csv_path}")

    # Best model selection (Random Forest as PRD core model)
    best_rf_name = "RandomForest_Default"
    if "RandomForest_Deep" in trained_models and comparison_df.loc[
        comparison_df["model_name"] == "RandomForest_Deep", "val_accuracy"
    ].values[0] > comparison_df.loc[
        comparison_df["model_name"] == "RandomForest_Default", "val_accuracy"
    ].values[0]:
        best_rf_name = "RandomForest_Deep"

    best_model = trained_models[best_rf_name]
    print(f"\nSelected primary model: {best_rf_name}")

    # Retrain on combined Train + Val for optimal production weights
    print("Fitting selected model on combined (Train + Val) dataset...")
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_model.fit(X_train_val, y_train_val)

    # Evaluate on unseen Test Set
    print("\n--- Evaluating Primary Model on Held-Out Test Set ---")
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)

    class_names = [config["labels"][i] for i in sorted(config["labels"].keys())]
    test_metrics = calculate_metrics(y_test, y_test_pred, class_names=class_names)

    print(f"Test Accuracy:       {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro Precision: {test_metrics['precision_macro'] * 100:.2f}%")
    print(f"Test Macro Recall:    {test_metrics['recall_macro'] * 100:.2f}%")
    print(f"Test Macro F1-Score:  {test_metrics['f1_macro'] * 100:.2f}%")

    # Generate and save reports
    rep_text = generate_classification_report_text(
        y_test,
        y_test_pred,
        class_names=class_names,
        save_path=config["reports"]["classification_report"]
    )
    print("\nClassification Report:\n", rep_text)

    # Plot & Save Confusion Matrix
    cm_array = np.array(test_metrics["confusion_matrix"])
    cm_path = config["reports"]["confusion_matrix_img"]
    plot_confusion_matrix(
        cm_array,
        class_names=[f"L{i}" for i in range(len(class_names))],
        save_path=cm_path,
        title=f"Hand Gesture Confusion Matrix ({best_rf_name})"
    )
    print(f"Confusion matrix image saved to: {cm_path}")

    # Save metrics JSON
    with open(config["reports"]["metrics_json"], "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save model artifact
    model_save_path = config["model"]["save_path"]
    joblib.dump(best_model, model_save_path)
    print(f"Trained model saved to: {model_save_path}")

    # Build and save rich metadata.json
    metadata = {
        "model_name": "HandGestureClassifier",
        "algorithm": "RandomForestClassifier",
        "variant": best_rf_name,
        "created_at": datetime.datetime.now().isoformat(),
        "n_classes": len(config["labels"]),
        "class_labels": config["labels"],
        "class_gestures": config["gestures"],
        "feature_count": len(LANDMARK_FEATURE_NAMES),
        "feature_names": LANDMARK_FEATURE_NAMES,
        "normalization": config["normalization"],
        "hyperparameters": best_model.get_params(),
        "dataset": {
            "total_samples": total_samples,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "class_distribution": df["label"].value_counts().to_dict()
        },
        "evaluation": {
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "per_class": test_metrics["per_class"]
        },
        "model_comparison": comparison_df.to_dict(orient="records")
    }

    metadata_path = config["model"]["metadata_path"]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")
    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
