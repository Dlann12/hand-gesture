"""
Model evaluation module for hand gesture classification.
Computes accuracy, precision, recall, f1-score, and renders confusion matrix plots.
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Headless mode for server/CLI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Model predicted labels.
        class_names: Human-readable names for classes.

    Returns:
        Dictionary of calculated metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")

    # Per-class metrics
    labels = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    for label in sorted(labels):
        name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        mask_true = (y_true == label)
        mask_pred = (y_pred == label)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        support = int(np.sum(mask_true))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[str(label)] = {
            "name": name,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "per_class": per_class
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Generate and optionally save a publication-ready confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)

    data = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    fmt = ".2%" if normalize else "d"

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.8,
        linecolor="#f0f2f6"
    )

    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    return fig


def generate_classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate scikit-learn text classification report and optionally save to file.
    """
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=====================================================\n")
            f.write("      HAND GESTURE RECOGNITION EVALUATION REPORT      \n")
            f.write("=====================================================\n\n")
            f.write(report_text)

    return report_text
